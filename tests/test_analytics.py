"""Persistent analytical storage: restart, isolation, immutable versions and service links."""

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.analytics import AnalyticsError, AnalyticsStore
from data_story.contracts import Failure, Success
from data_story.datasets import DatasetError, DatasetStore
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "tmp/tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AnalyticsStore(self.root / "analytics.duckdb")
        self.project = str(uuid4())
        self.frame = pd.DataFrame(
            {"district": ["North", "South", "North"], "value": [2.0, None, 4.0]}
        )

    def put(self, source="original", frame=None):
        return self.store.put(
            self.project,
            "dataset",
            "dataset-id",
            self.frame if frame is None else frame,
            source=source,
        )

    def test_restart_persists_exact_values_nulls_and_names(self):
        saved = self.put()
        reopened = AnalyticsStore(self.store.database)
        restored = reopened.read(self.project, "dataset", "dataset-id", saved["fingerprint"])
        pd.testing.assert_frame_equal(restored, self.frame)
        self.assertEqual(reopened.catalog(self.project)[0]["row_count"], 3)
        self.assertTrue(self.put()["reused"])
        self.assertEqual(len(reopened.catalog(self.project)), 1)

    def test_versions_never_replace_old_results(self):
        old = self.put()
        changed = self.frame.copy()
        changed.loc[0, "value"] = 10
        new = self.put(frame=changed)
        metadata_version = self.put(source="metadata changed", frame=changed)
        self.assertEqual(len(self.store.catalog(self.project)), 3)
        self.assertNotEqual(new["fingerprint"], metadata_version["fingerprint"])
        pd.testing.assert_frame_equal(
            self.store.read(self.project, "dataset", "dataset-id", old["fingerprint"]), self.frame
        )

    def test_other_project_cannot_read_or_discover_a_table(self):
        saved = self.put()
        other = str(uuid4())
        self.assertEqual(self.store.catalog(other), [])
        with self.assertRaises(AnalyticsError):
            self.store.read(other, "dataset", "dataset-id", saved["fingerprint"])
        foreign = self.store.put(other, "dataset", "dataset-id", self.frame, source="original")
        self.assertNotEqual(saved["table"], foreign["table"])

    def test_empty_catalog_does_not_create_database(self):
        self.assertEqual(self.store.catalog(self.project), [])
        self.assertFalse(self.store.database.exists())

    def test_invalid_identifiers_and_case_collisions_do_not_write(self):
        with self.assertRaises(AnalyticsError):
            self.store.put('x"; DROP TABLE projects;--', "dataset", "x", self.frame, source="x")
        with self.assertRaises(AnalyticsError):
            self.put(frame=pd.DataFrame({"X": [1], "x": [2]}))
        self.assertFalse(self.store.database.exists())
        saved = self.store.put(
            self.project, "dataset", 'x"; DROP TABLE y;--', self.frame, source="x"
        )
        self.assertNotIn("DROP", saved["table"])
        self.assertEqual(len(self.store.catalog(self.project)), 1)

    def test_two_instances_serialize_writes_without_losing_tables(self):
        def write(index):
            return AnalyticsStore(self.store.database).put(
                self.project, "statistics", str(index), pd.DataFrame({"n": [index]}), source="v1"
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(write, range(4)))
        self.assertEqual(len(self.store.catalog(self.project)), 4)
        self.assertEqual(sum(r["row_count"] for r in rows), 4)

    def test_service_verifies_snapshot_and_preserves_current_statistics(self):
        service = ProjectService(ProjectStore(self.root / "data.sqlite3"))
        project = service.store.create("Analytics")
        dataset = service.datasets.add(project.id, "numbers.csv", b"group,value\nA,2\nB,4\nA,6\n")
        before = service.statistics_store.calculate(project.id, dataset.id, "descriptive")
        imported = service.analytics_sync(project.id, dataset.id)
        self.assertIsInstance(imported, Success)
        after = service.statistics_store.calculate(project.id, dataset.id, "descriptive")
        self.assertEqual(before, after)
        self.assertEqual(service.datasets.get(dataset.id), dataset)
        self.assertIsInstance(service.events(project.id), Success)
        result = service.analytics_sync(project.id, dataset.id)
        self.assertTrue(result.data["reused"])
        other = service.store.create("Other")
        self.assertIsInstance(service.analytics_sync(other.id, dataset.id), Failure)
        self.assertEqual(service.analytics_catalog(other.id).data, [])
        path = service.datasets.files_root / dataset.stored_path
        path.write_bytes(b"group,value\nA,999\n")
        self.assertIsInstance(service.analytics_sync(project.id, dataset.id), Failure)
        self.assertEqual(
            len([r for r in service.analytics_catalog(project.id).data if r["kind"] == "dataset"]),
            1,
        )

    def test_api_returns_persisted_catalog_and_refuses_unknown_projects(self):
        service = ProjectService(ProjectStore(self.root / "data.sqlite3"))
        project = service.store.create("API")
        dataset = service.datasets.add(project.id, "test.csv", b"n\n1\n2\n")
        app = FastAPI()
        app.include_router(router(service))
        install_error_handlers(app)
        with TestClient(app) as client:
            response = client.post(f"/api/projects/{project.id}/analytics/datasets/{dataset.id}")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["row_count"], 2)
            self.assertEqual(
                len(client.get(f"/api/projects/{project.id}/analytics").json()["data"]), 1
            )
            self.assertEqual(client.get(f"/api/projects/{uuid4()}/analytics").status_code, 404)
            self.assertEqual(client.get("/api/projects/invalid/analytics").status_code, 422)

    def test_automatic_import_restart_never_reparses_csv(self):
        service = ProjectService(ProjectStore(self.root / "data.sqlite3"))
        project = service.store.create("Persistent input")
        dataset = service.datasets.add(project.id, "values.csv", b"label,n\nA,2\nB,\nC,4\n")
        original = service.datasets.frame(dataset.id)
        restarted = DatasetStore(service.store.database)
        with patch("data_story.datasets.read_table", side_effect=AssertionError("CSV reparsed")):
            restored = restarted.frame(dataset.id)
            preview = restarted.preview(dataset.id)
        pd.testing.assert_frame_equal(original, restored)
        self.assertEqual(preview.rows, [["A", "2"], ["B", None], ["C", "4"]])
        self.assertEqual(len(restarted.analytics.catalog(project.id)), 1)

    def test_cached_grid_still_rejects_changed_original(self):
        service = ProjectService(ProjectStore(self.root / "data.sqlite3"))
        project = service.store.create("Cached input")
        dataset = service.datasets.add(project.id, "values.csv", b"n\n1\n2\n")
        service.datasets._typed(dataset.id)
        (service.datasets.files_root / dataset.stored_path).write_bytes(b"n\n9\n")
        with self.assertRaises(DatasetError):
            service.datasets._typed(dataset.id)

    def test_failed_upload_leaves_no_analytical_or_sqlite_dataset(self):
        service = ProjectService(ProjectStore(self.root / "data.sqlite3"))
        project = service.store.create("Rollback")
        with patch.object(ProjectStore, "_event", side_effect=RuntimeError("audit failed")):
            with self.assertRaises(RuntimeError):
                service.datasets.add(project.id, "values.csv", b"n\n1\n2\n")
        self.assertEqual(service.datasets.list_datasets(project.id), [])
        self.assertEqual(service.analytics.catalog(project.id), [])


if __name__ == "__main__":
    unittest.main()
