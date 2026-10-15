"""Stage 08b: monitoring rows are local, honest about missing parts and logged without secrets."""

import json
import tempfile
import unittest
from pathlib import Path

import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story import embeddings
from data_story.contracts import Success
from data_story.monitoring import Monitor, lock_versions
from data_story.storage import SCHEMA_VERSION, connect
from data_story.web import install_error_handlers, router

KEYS = {"server", "database", "files", "packages", "assets", "storage", "matrices", "knowledge",
        "openai_access", "errors", "duckdb", "reports"}  # fmt: skip
SERVICES = {"server", "database", "duckdb", "knowledge", "reports", "openai_access"}


class MonitoringTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def rows(self, result=None):
        result = result or self.service.monitoring().data
        return {row["key"]: row for row in result["rows"]}

    def test_all_parts_reported_and_logged(self):
        result = self.service.monitoring()
        self.assertIsInstance(result, Success)
        rows = self.rows(result.data)
        self.assertEqual(set(rows), KEYS)
        # The main view lists only service parts; project-wide details are grouped apart.
        self.assertEqual({k for k, r in rows.items() if r["group"] == "service"}, SERVICES)
        self.assertEqual(sum(result.data["summary"].values()), len(SERVICES))
        self.assertEqual(rows["database"]["state"], "ok")
        self.assertIn(f"Схема v{SCHEMA_VERSION}", rows["database"]["detail"])
        self.assertEqual(rows["assets"]["state"], "ok")
        self.assertEqual(rows["knowledge"]["state"], "warning" if embeddings.status() else "ok")
        self.assertIn(rows["openai_access"]["state"], ("not_configured", "warning"))
        self.assertIn(rows["duckdb"]["state"], ("ok", "error"))
        self.assertIn("МБ", rows["storage"]["detail"])
        with connect(self.database) as connection:
            payload = json.loads(
                connection.execute(
                    "SELECT payload FROM config_events WHERE subject='monitoring'"
                ).fetchone()[0]
            )
        self.assertEqual(payload["states"]["database"], "ok")
        self.assertEqual(
            self.service.knowledge.events(), []
        )  # the index check does not log searches

    def test_missing_dataset_file_and_broken_matrix_are_flagged(self):
        lines = ["x,y"] + [f"{i},{2 * i}" for i in range(12)]
        dataset = self.service.datasets.add(self.project.id, "m.csv", "\n".join(lines).encode())
        run = self.service.training_prepare(
            self.project.id, {"dataset_id": dataset.id, "target": "y", "features": ["x"]}
        ).data
        self.assertEqual(self.rows()["matrices"]["state"], "ok")
        (self.service.datasets.files_root / run["document"]["matrix_path"]).write_bytes(b"x")
        self.assertEqual(self.rows()["matrices"]["state"], "error")
        (self.service.datasets.files_root / self.dataset.stored_path).unlink()
        files = self.rows()["files"]
        self.assertEqual(files["state"], "warning")
        self.assertIn("відсутніх файлів 1", files["detail"])

    def test_key_states_without_exposing_key(self):
        folder = Path(tempfile.mkdtemp(dir=self.temporary.name))
        env = folder / ".env"
        self.service.ai_settings.env_file = env
        self.assertEqual(self.rows()["openai_access"]["state"], "not_configured")
        secret = "sk-mon-" + "k" * 30 + "TAIL"
        env.write_text(f"OPENAI_API_KEY={secret}\n", encoding="utf-8")
        result = self.service.monitoring().data
        # With a key but no access check yet: still "not configured", and no key is shown.
        self.assertEqual(self.rows(result)["openai_access"]["state"], "not_configured")
        self.assertIn("Ще не перевірявся", self.rows(result)["openai_access"]["detail"])
        dumped = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(secret, dumped)
        self.assertNotIn("TAIL", dumped)

    def test_package_versions_against_lock(self):
        folder = Path(tempfile.mkdtemp(dir=self.temporary.name))
        lock = folder / "requirements.lock"
        lock.write_text("pandas==0.0.1\n", encoding="utf-8")
        self.assertEqual(lock_versions(lock), {"pandas": "0.0.1"})
        row = Monitor(self.service, lock=lock).packages()
        self.assertEqual(row["state"], "warning")
        self.assertIn("pandas", row["detail"])

    def test_duckdb_checks_actual_storage_and_missing_tables(self):
        monitor = Monitor(self.service)
        row = monitor.duckdb()
        self.assertEqual(row["state"], "ok")
        self.assertIn("таблиць", row["detail"])
        table = self.service.analytics.catalog(self.project.id)[0]
        with self.service.analytics._connection() as connection:
            schema, name = table["table"].split(".")
            connection.execute(f'DROP TABLE "{schema}"."{name}"')
        self.assertEqual(monitor.duckdb()["state"], "error")

    def test_duckdb_empty_and_unreadable_do_not_hide_other_services(self):
        database = Path(self.temporary.name) / "unreadable.duckdb"
        self.service.analytics.database = database
        monitor = Monitor(self.service)
        self.assertEqual(monitor.duckdb()["state"], "ok")
        self.assertFalse(database.exists())
        database.write_bytes(b"invalid database")
        self.assertEqual(monitor.duckdb()["state"], "error")
        self.assertEqual(self.rows()["database"]["state"], "ok")

    def test_one_failing_check_does_not_hide_others(self):
        monitor = Monitor(self.service)
        monitor.knowledge = lambda: 1 / 0  # type: ignore[method-assign]
        rows = {r["key"]: r for r in monitor.run()["rows"]}
        self.assertEqual(rows["<lambda>"]["state"], "error")
        self.assertEqual(rows["database"]["state"], "ok")

    def test_http(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            response = client.get("/api/monitoring")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json()["data"]["rows"]), len(KEYS))
