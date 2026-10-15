"""Preparation API, provenance, stale preview rejection, isolation and persistence."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.contracts import Failure, Success
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.storage import (
    _V1,
    _V2,
    _V3,
    _V4,
    _V5,
    _V6,
    _V7,
    SCHEMA_VERSION,
    connect,
    ensure_schema,
)
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]


class PreparationStoreTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "tmp" / "tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "data.sqlite3"
        self.store = ProjectStore(self.database)
        self.service = ProjectService(self.store)
        self.project = self.store.create("Preparation")
        self.dataset = self.service.datasets.add(self.project.id, "test.csv", b"x,s\n1,A\n,B\n3,\n")
        self.query = {"dataset_id": self.dataset.id, "rules": [{"field": "x", "action": "drop"}]}

    def test_preview_one_audit_and_source_unchanged(self):
        before = len(self.store.events(self.project.id))
        result = self.service.preparation_preview(self.project.id, self.query)
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data["positions"], [0, 2])
        self.assertEqual(result.data["report"]["rows_removed"], 1)
        self.assertEqual(len(self.store.events(self.project.id)), before + 1)
        event = self.store.events(self.project.id)[-1]
        self.assertEqual(event.operation, "preparation.previewed")
        self.assertEqual(event.payload["status"], "success")
        self.assertEqual(self.service.datasets.get(self.dataset.id).row_count, 3)
        self.assertEqual(self.service.preparation_plans(self.project.id).data, [])

    def test_save_persists_recipe_positions_and_single_event_after_restart(self):
        preview = self.service.preparation_preview(self.project.id, self.query).data
        before = len(self.store.events(self.project.id))
        result = self.service.preparation_save(
            self.project.id, {**self.query, "preview_token": preview["preview_token"]}
        )
        self.assertIsInstance(result, Success)
        self.assertEqual(len(self.store.events(self.project.id)), before + 1)
        reopened = ProjectService(ProjectStore(self.database))
        plan = reopened.preparation_plans(self.project.id).data[0]
        self.assertEqual(plan["id"], result.data["plan_id"])
        self.assertEqual(plan["document"]["retained_positions"], [0, 2])
        self.assertEqual(plan["document"]["sha256"], self.dataset.sha256)
        self.assertEqual(
            plan["document"]["rules"], [{"field": "x", "action": "drop", "value": None}]
        )

    def test_stale_rules_and_metadata_rejected_without_plan(self):
        preview = self.service.preparation_preview(self.project.id, self.query).data
        changed = {
            **self.query,
            "rules": [{"field": "s", "action": "drop"}],
            "preview_token": preview["preview_token"],
        }
        self.assertIsInstance(self.service.preparation_save(self.project.id, changed), Failure)
        meta = self.service.metadata_store.get(self.dataset.id)
        doc = meta.document.model_dump()
        doc["columns"][0]["description"] = "Updated"
        self.service.metadata_store.save(self.dataset.id, doc)
        self.assertIsInstance(
            self.service.preparation_save(
                self.project.id, {**self.query, "preview_token": preview["preview_token"]}
            ),
            Failure,
        )
        self.assertEqual(self.service.preparation_plans(self.project.id).data, [])
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "error")

    def test_isolation_and_changed_source_error_logged(self):
        other = self.store.create("Other")
        self.assertIsInstance(self.service.preparation_preview(other.id, self.query), Failure)
        self.assertEqual(self.store.events(other.id)[-1].payload["status"], "error")
        path = self.service.datasets.files_root / self.dataset.stored_path
        path.write_bytes(b"changed")
        self.assertIsInstance(
            self.service.preparation_preview(self.project.id, self.query), Failure
        )
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "error")

    def test_constants_kept_in_recipe_but_not_audit(self):
        raw = {
            "dataset_id": self.dataset.id,
            "rules": [{"field": "s", "action": "constant", "value": "private-note"}],
        }
        result = self.service.preparation_preview(self.project.id, raw)
        self.assertIsInstance(result, Success)
        self.service.preparation_save(
            self.project.id, {**raw, "preview_token": result.data["preview_token"]}
        )
        events = [
            e for e in self.store.events(self.project.id) if e.operation.startswith("preparation.")
        ]
        self.assertNotIn("private-note", json.dumps([e.payload for e in events]))
        self.assertEqual(
            self.service.preparation_plans(self.project.id).data[0]["document"]["rules"][0][
                "value"
            ],
            "private-note",
        )

    def test_save_and_event_atomic_on_event_failure(self):
        preview = self.service.preparation_preview(self.project.id, self.query).data
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            response = self.service.preparation_save(
                self.project.id, {**self.query, "preview_token": preview["preview_token"]}
            )
        self.assertIsInstance(response, Failure)
        self.assertEqual(self.service.preparation_plans(self.project.id).data, [])

    def test_metadata_change_during_save_rejected(self):
        preview = self.service.preparation_preview(self.project.id, self.query).data
        get = self.service.preparation_store.metadata.get

        def changed_after_read(dataset_id):
            snapshot = get(dataset_id)
            document = snapshot.document.model_dump()
            document["dataset"]["description"] = "Changed during calculation"
            self.service.metadata_store.save(dataset_id, document)
            return snapshot

        with patch.object(
            self.service.preparation_store.metadata, "get", side_effect=changed_after_read
        ):
            response = self.service.preparation_save(
                self.project.id, {**self.query, "preview_token": preview["preview_token"]}
            )
        self.assertIsInstance(response, Failure)
        self.assertEqual(self.service.preparation_plans(self.project.id).data, [])
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "error")

    def test_v7_migration_keeps_existing_project(self):
        database = Path(self.temporary.name) / "v7.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.executescript(
                _V1 + _V2 + _V3 + _V4 + _V5 + _V6 + _V7 + "PRAGMA user_version=7;"
            )
            connection.execute(
                "INSERT INTO projects VALUES ('old', 'Existing', '', 'date', 'regression', '')"
            )
        connection.close()
        ensure_schema(database)
        with connect(database) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
            )
            self.assertEqual(
                connection.execute("SELECT name FROM projects").fetchone()[0], "Existing"
            )
            self.assertEqual(
                connection.execute("SELECT count(*) FROM preparation_plans").fetchone()[0], 0
            )

    def test_http_preview_save_list_and_validation(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        prefix = f"/api/projects/{self.project.id}/preparation"
        with TestClient(app) as client:
            preview = client.post(prefix + "/preview", json=self.query)
            self.assertEqual(preview.status_code, 200)
            request = {**self.query, "preview_token": preview.json()["data"]["preview_token"]}
            self.assertEqual(client.post(prefix + "/plans", json=request).status_code, 200)
            self.assertEqual(len(client.get(prefix + "/plans").json()["data"]), 1)
            self.assertEqual(client.post(prefix + "/plans", json=self.query).status_code, 422)
            self.assertEqual(
                client.post(
                    prefix + "/preview",
                    json={**self.query, "rules": [{"field": "x", "action": "constant"}]},
                ).status_code,
                422,
            )
