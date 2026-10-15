"""Contract and HTTP checks share the same real SQLite service as the UI."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError

from data_story.contracts import Failure, ProjectCreate, ProjectView, Success
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]
PROJECT_RESULT = TypeAdapter(Success[ProjectView] | Failure)


class ContractAndApiTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.store = ProjectStore(Path(self.temporary.name) / "projects.sqlite3")
        self.service = ProjectService(self.store)
        application = FastAPI()
        application.include_router(router(self.service))
        install_error_handlers(application)
        self.client = TestClient(application)
        self.addCleanup(self.client.close)

    def test_schema_is_strict_and_exposes_limits(self):
        for raw in ({"name": 5}, {"name": " "}, {"name": "Дані", "extra": True}):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                ProjectCreate.model_validate(raw)
        schema = ProjectCreate.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["name"]["maxLength"], 120)

    def test_statistics_pages_views_and_journal_over_http(self):
        project = self.store.create("Статистика")
        dataset = self.service.datasets.add(
            project.id, "test.csv", b"x,y,s\n1,2,A\n2,4,B\n3,6,A\n4,,B\n"
        )
        url = f"/api/projects/{project.id}/statistics"
        query = {"dataset_id": dataset.id, "limit": 1}
        first = self.client.post(url, json=query)
        self.assertEqual(first.status_code, 200, first.text)
        data = first.json()["data"]
        self.assertEqual((len(data["rows"]), data["total"], data["row_count"]), (1, 3, 4))
        self.assertEqual(data["rows"][0]["values"]["mean"], 2.5)
        second = self.client.post(url, json={**query, "start": 1}).json()["data"]
        self.assertEqual(second["rows"][0]["name"], "y")
        for view in ("missing", "correlations"):
            response = self.client.post(url, json={**query, "view": view})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()["data"]
            if view == "missing":
                self.assertEqual(result["rows_with_missing"], 1)
            else:
                self.assertEqual([c["name"] for c in result["fields"]], ["x", "y"])
                self.assertEqual(result["rows"][0]["counts"]["y"], 3)
                self.assertAlmostEqual(result["rows"][0]["values"]["y"], 1)
                self.assertIn("s", result["excluded"])
        events = self.client.get(f"/api/projects/{project.id}/events")
        self.assertEqual(events.status_code, 200, events.text)
        calculations = [
            e for e in events.json()["data"] if e["operation"] == "statistics.calculated"
        ]
        self.assertEqual(len(calculations), 4)
        self.assertEqual(calculations[-1]["payload"]["dataset_name"], "test.csv")

    def test_statistics_metadata_refresh_validation_and_isolation(self):
        project = self.store.create("Один")
        other = self.store.create("Інший")
        dataset = self.service.datasets.add(project.id, "test.csv", b"x,y\n0,1\n1,2\n2,3\n")
        url = f"/api/projects/{project.id}/statistics"
        query = {"dataset_id": dataset.id}
        for invalid in ({"limit": 51}, {"start": -1}, {"view": "invalid"}, {"extra": 1}):
            self.assertEqual(self.client.post(url, json={**query, **invalid}).status_code, 422)
        denied = self.client.post(f"/api/projects/{other.id}/statistics", json=query)
        self.assertEqual(denied.status_code, 422)
        self.assertNotIn("rows", denied.json())
        meta = self.service.metadata_store.get(dataset.id).document
        meta.columns[0].missing_codes = ["0"]
        self.service.metadata_store.save(dataset.id, meta.model_dump(), confirm=True)
        response = self.client.post(url, json=query).json()["data"]
        self.assertEqual(response["metadata_state"], "confirmed")
        self.assertEqual(response["rows"][0]["values"]["mean"], 1.5)
        self.assertEqual(
            self.client.post(url, json={**query, "view": "correlations", "fields": []}).json()[
                "data"
            ]["total"],
            0,
        )
        path = self.service.datasets.files_root / dataset.stored_path
        path.write_bytes(b"broken")
        error = self.client.post(url, json=query)
        self.assertEqual(error.status_code, 422)
        self.assertNotIn(str(path), error.text)
        self.assertEqual(self.client.get(f"/api/projects/{project.id}/events").status_code, 200)

    def test_http_and_ui_service_return_same_validated_project(self):
        response = self.client.post("/api/projects", json={"name": "  Дані України  "})
        self.assertEqual(response.status_code, 200)
        result = PROJECT_RESULT.validate_python(response.json())
        self.assertIsInstance(result, Success)
        project_id = response.json()["data"]["id"]
        self.assertEqual(
            self.client.get("/api/projects/" + project_id).json(),
            self.service.get(project_id).model_dump(mode="json"),
        )
        self.assertEqual(response.json()["data"]["name"], "Дані України")

    def test_bad_json_unknown_fields_and_empty_name_have_same_error_envelope(self):
        responses = [
            self.client.post("/api/projects", json={"name": ""}),
            self.client.post("/api/projects", json={"name": "Дані", "unknown": 1}),
            self.client.post(
                "/api/projects", content="{", headers={"Content-Type": "application/json"}
            ),
        ]
        for response in responses:
            with self.subTest(body=response.text):
                self.assertEqual(response.status_code, 422)
                failure = Failure.model_validate(response.json())
                self.assertEqual(failure.error.code, "invalid_input")
        self.assertEqual(self.store.list_projects(), [])

    def test_invalid_and_unknown_ids_are_different_failures(self):
        invalid = self.client.get("/api/projects/not-a-uuid")
        missing = self.client.get("/api/projects/" + str(uuid4()))
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(missing.json()["error"]["code"], "not_found")

    def test_storage_failure_is_not_success_and_does_not_leak_details(self):
        with patch.object(
            self.store, "create", side_effect=sqlite3.OperationalError("private disk path")
        ):
            response = self.client.post("/api/projects", json={"name": "Дані"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "storage_error")
        self.assertNotIn("private disk path", response.text)
        self.assertEqual(self.store.list_projects(), [])

    def test_invalid_output_is_internal_error_not_user_input_error(self):
        with patch.object(self.store, "list_projects", return_value=[{"id": "bad"}]):
            response = self.client.get("/api/projects")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_error")

    def test_http_open_and_event_scoping(self):
        first = self.store.create("Перший")
        second = self.store.create("Другий")
        self.client.post(f"/api/projects/{first.id}/open")
        first_events = self.client.get(f"/api/projects/{first.id}/events").json()["data"]
        second_events = self.client.get(f"/api/projects/{second.id}/events").json()["data"]
        self.assertEqual(len(first_events), 2)
        self.assertEqual(len(second_events), 1)
        self.assertTrue(all(event["project_id"] == first.id for event in first_events))

    def test_project_type_is_returned_and_disabled_type_is_rejected(self):
        created = self.client.post("/api/projects", json={"name": "Ціни"}).json()["data"]
        self.assertEqual(created["project_type"], "regression")
        refused = self.client.post(
            "/api/projects", json={"name": "Групи", "project_type": "clustering"}
        )
        self.assertEqual(refused.status_code, 422)
        self.assertIn("ще недоступний", refused.text)
        types = self.client.get("/api/project-types").json()["data"]
        self.assertEqual([(t["key"], t["enabled"]) for t in types][0], ("regression", True))

    def test_instruction_http_round_trip(self):
        url = "/api/project-types/regression/instruction"
        self.assertTrue(self.client.get(url).json()["data"]["is_default"])
        saved = self.client.put(url, json={"text": "Новий промпт"}).json()["data"]
        self.assertEqual((saved["text"], saved["is_default"]), ("Новий промпт", False))
        self.assertEqual(self.client.put(url, json={"text": " "}).status_code, 422)
        self.assertTrue(self.client.delete(url).json()["data"]["is_default"])
        missing = self.client.get("/api/project-types/unknown/instruction")
        self.assertEqual(missing.status_code, 404)

    def test_dataset_upload_list_view_and_preview_over_http(self):
        project = self.store.create("Дані")
        content = "x;y\n1;2\n3;\n".encode()
        uploaded = self.client.post(
            f"/api/projects/{project.id}/datasets", files={"file": ("дані.csv", content)}
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        data = uploaded.json()["data"]
        self.assertEqual((data["row_count"], data["column_count"]), (2, 2))
        self.assertEqual(data["source"]["delimiter"], ";")
        self.assertEqual([c["missing"] for c in data["columns"]], [0, 1])
        listed = self.client.get(f"/api/projects/{project.id}/datasets").json()["data"]
        self.assertEqual([d["id"] for d in listed], [data["id"]])
        preview = self.client.get(f"/api/datasets/{data['id']}/preview").json()["data"]
        self.assertEqual(preview["rows"], [["1", "2"], ["3", None]])
        url = f"/api/datasets/{data['id']}/rows"
        query = {"start": 0, "limit": 10, "sort": [{"colId": "c0", "sort": "desc"}], "filter": {}}
        rows = self.client.post(url, json=query).json()["data"]
        self.assertEqual(
            rows, {"rows": [{"c0": 3, "c1": None}, {"c0": 1, "c1": 2}], "total": 2, "all_rows": 2}
        )
        self.assertEqual(self.client.post(url, json={**query, "limit": 501}).status_code, 422)
        bad = {**query, "filter": {"c0": {"filterType": "text", "type": "contains", "filter": "1"}}}
        self.assertEqual(self.client.post(url, json=bad).status_code, 422)
        missing = self.client.post(f"/api/datasets/{uuid4()}/rows", json=query)
        self.assertEqual(missing.status_code, 404)
        events = self.client.get(f"/api/projects/{project.id}/events").json()["data"]
        self.assertEqual(events[-1]["operation"], "dataset.uploaded")

    def test_dataset_copy_over_http(self):
        source = self.store.create("Джерело")
        target = self.store.create("Ціль")
        content = "x;y\n1;2\n".encode()
        uploaded = self.client.post(
            f"/api/projects/{source.id}/datasets", files={"file": ("a.csv", content)}
        ).json()["data"]
        found = self.client.get(f"/api/projects/{target.id}/dataset-candidates").json()["data"]
        self.assertEqual([c["dataset"]["id"] for c in found], [uploaded["id"]])
        self.assertEqual((found[0]["project_name"], found[0]["metadata"]), ("Джерело", "none"))
        copied = self.client.post(
            f"/api/projects/{target.id}/datasets/copy", json={"dataset_id": uploaded["id"]}
        )
        self.assertEqual(copied.status_code, 200, copied.text)
        self.assertEqual(copied.json()["data"]["copied_from"], uploaded["id"])
        same = self.client.post(
            f"/api/projects/{source.id}/datasets/copy", json={"dataset_id": uploaded["id"]}
        )
        self.assertEqual(same.status_code, 422)
        missing = self.client.post(
            f"/api/projects/{target.id}/datasets/copy", json={"dataset_id": str(uuid4())}
        )
        self.assertEqual(missing.status_code, 404)
        events = self.client.get(f"/api/projects/{target.id}/events").json()["data"]
        self.assertEqual(events[-1]["operation"], "dataset.copied")

    def test_bad_dataset_upload_is_invalid_input_and_unknown_ids_are_not_found(self):
        project = self.store.create("Дані")
        refused = self.client.post(
            f"/api/projects/{project.id}/datasets", files={"file": ("notes.txt", b"a\n1\n")}
        )
        self.assertEqual(refused.status_code, 422)
        self.assertIn("лише файли CSV", refused.json()["error"]["message"])
        missing_project = self.client.post(
            f"/api/projects/{uuid4()}/datasets", files={"file": ("a.csv", b"a\n1\n")}
        )
        self.assertEqual(missing_project.status_code, 404)
        self.assertEqual(self.client.get(f"/api/datasets/{uuid4()}").status_code, 404)

    def test_metadata_round_trip_over_http(self):
        project = self.store.create("Дані")
        content = "x;колір\n1;червоний\n2;синій\n3;червоний\n".encode()
        dataset = self.client.post(
            f"/api/projects/{project.id}/datasets", files={"file": ("d.csv", content)}
        ).json()["data"]
        url = f"/api/datasets/{dataset['id']}/metadata"
        draft = self.client.get(url).json()["data"]
        self.assertFalse(draft["saved"])
        self.assertEqual([c["kind"] for c in draft["document"]["columns"]], ["number", "category"])
        document = draft["document"]
        document["columns"][0].update(label="Ікс", role="target", range_max=2)
        saved = self.client.put(url, json={"document": document, "confirm": True})
        self.assertEqual(saved.status_code, 200, saved.text)
        body = saved.json()["data"]
        self.assertTrue(body["saved"])
        self.assertIsNotNone(body["confirmed_at"])
        self.assertTrue(any("3 > 2" in i["message"] for i in body["issues"]))
        document["columns"] = document["columns"][:1]
        self.assertEqual(self.client.put(url, json={"document": document}).status_code, 422)
        text = self.client.post(url + "/sources/text", json={"text": "x — номер"})
        self.assertEqual(text.status_code, 200, text.text)
        upload = self.client.post(
            url + "/sources/file", files={"file": ("опис.txt", "колір — колір".encode())}
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        refused = self.client.post(url + "/sources/file", files={"file": ("a.pdf", b"%PDF")})
        self.assertEqual(refused.status_code, 422)
        sources = self.client.get(url + "/sources").json()["data"]
        self.assertEqual([s["kind"] for s in sources], ["text", "file"])

    def test_openapi_uses_shared_request_and_response_schemas(self):
        specification = self.client.get("/openapi.json").json()
        schemas = specification["components"]["schemas"]
        self.assertIn("ProjectCreate", schemas)
        self.assertIn("ProjectView", schemas)
        self.assertIn("Failure", schemas)


if __name__ == "__main__":
    unittest.main()
