"""SQL input, project boundaries, resource limits, reviewed summaries and model results."""

import unittest
from unittest.mock import patch

import numpy as np
import test_assistant as assistant_fixtures
import test_model as model_fixtures
import test_statistics as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router


class AnalyticalQueryTests(unittest.TestCase):
    setUp = fixtures.StatisticsStoreTests.setUp

    def service(self):
        return ProjectService(self.projects)

    def alias(self):
        return "d_" + self.dataset.id.replace("-", "")

    def test_aggregate_cte_exact_cache_and_audit(self):
        service = self.service()
        sql = (
            f"WITH a AS (SELECT * FROM {self.alias()}) SELECT avg(x) AS mean, count(*) AS n FROM a"
        )
        first = service.analytics_query(self.project.id, {"sql": sql})
        self.assertIsInstance(first, Success, getattr(first, "error", None))
        self.assertEqual(first.data["rows"], [[1, 3]])
        self.assertFalse(first.data["reused"])
        second = self.service().analytics_query(self.project.id, {"sql": sql})
        self.assertTrue(second.data["reused"])
        self.assertEqual(first.data["rows"], second.data["rows"])
        event = self.projects.events(self.project.id)[-1]
        self.assertEqual(event.operation, "analytics.queried")
        self.assertTrue(event.payload["reused"])

    def test_normalized_missing_codes_roles_and_metadata_change(self):
        service = self.service()
        sql = f"SELECT avg(x) FROM {self.alias()}"
        self.assertEqual(service.analytics_query(self.project.id, {"sql": sql}).data["rows"], [[1]])
        document = service.metadata_store.get(self.dataset.id).document
        document.columns[0].missing_codes = ["0"]
        document.columns[1].role = "ignore"
        service.metadata_store.save(self.dataset.id, document.model_dump(), confirm=True)
        schema = service.analytics_schema(self.project.id).data
        self.assertEqual(schema[0]["columns"], ["x"])
        result = service.analytics_query(self.project.id, {"sql": sql})
        self.assertEqual(result.data["rows"], [[1.5]])
        self.assertFalse(result.data["reused"])
        self.assertEqual(service.datasets.frame(self.dataset.id)["x"].tolist(), [0, 1, 2])

    def test_boolean_nulls_preserve_true_false_and_missing(self):
        service = self.service()
        dataset = service.datasets.add(
            self.project.id, "booleans.csv", b"flag,value\ntrue,1\nfalse,2\n,3\n"
        )
        document = service.metadata_store.get(dataset.id).document
        document.columns[0].kind = "bool"
        service.metadata_store.save(dataset.id, document.model_dump(), confirm=True)
        alias = "d_" + dataset.id.replace("-", "")
        result = service.analytics_query(
            self.project.id, {"sql": f"SELECT flag FROM {alias} ORDER BY value"}
        )
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        self.assertEqual(result.data["rows"], [[True], [False], [None]])

    def test_multiple_statements_writes_files_and_other_projects_refused(self):
        service = self.service()
        other = self.projects.create("Other")
        foreign = self.datasets.add(other.id, "private.csv", b"secret\n42\n")
        path = self.datasets.files_root / foreign.stored_path
        for sql in (
            "SELECT 1; SELECT 2",
            "DELETE FROM " + self.alias(),
            "CREATE TABLE x AS SELECT 1",
            "ATTACH ':memory:' AS other",
            f"SELECT * FROM d_{foreign.id.replace('-', '')}",
            f"SELECT * FROM read_csv('{path.as_posix()}')",
            f"SELECT * FROM read_blob('{path.as_posix()}')",
            "SELECT * FROM _analytics_catalog",
            "SELECT * FROM frame",
            "SELECT * FROM sqlite_scan('data.sqlite3', 'projects')",
        ):
            with self.subTest(sql=sql):
                self.assertIsInstance(
                    service.analytics_query(self.project.id, {"sql": sql}), Failure
                )
        self.assertEqual(self.datasets.frame(self.dataset.id)["x"].tolist(), [0, 1, 2])
        self.assertEqual(len(service.analytics_schema(other.id).data), 1)
        self.assertNotIn("secret", service.analytics_schema(self.project.id).data[0]["columns"])

    def test_limits_timeout_and_syntax_errors_are_audited(self):
        service = self.service()
        limited = service.analytics_query(
            self.project.id, {"sql": f"SELECT * FROM {self.alias()}", "limit": 1}
        )
        self.assertEqual(limited.data["row_count"], 1)
        self.assertTrue(limited.data["truncated"])
        self.assertIsInstance(
            service.analytics_query(self.project.id, {"sql": "SELECT 1", "limit": 10001}), Failure
        )
        self.assertIsInstance(
            service.analytics_query(self.project.id, {"sql": "SELEC wrong"}), Failure
        )
        with patch("data_story.analytical_queries.SQL_SECONDS", 1.0):
            slow = service.analytics_query(
                self.project.id,
                {
                    "sql": "SELECT sum(a.i * b.i) FROM range(1000000) a(i) CROSS JOIN range(1000000) b(i)"
                },
            )
        self.assertIsInstance(slow, Failure)
        self.assertEqual(self.projects.events(self.project.id)[-1].payload["status"], "error")

    def test_reviewed_sql_summary_becomes_normal_dataset_for_charts(self):
        service = self.service()
        raw = {
            "sql": f"SELECT x, y FROM {self.alias()} ORDER BY x",
            "limit": 100,
            "name": "SQL result",
            "explanation": "Зведення для перевірки графіків.",
        }
        before = len(service.datasets.list_datasets(self.project.id))
        draft = run_tool(service, self.project.id, "propose_sql_summary", raw)
        self.assertIsInstance(draft, Success, getattr(draft, "error", None))
        self.assertEqual(len(service.datasets.list_datasets(self.project.id)), before)
        applied = service.assistant_apply_draft(self.project.id, draft.data["proposal"])
        self.assertIsInstance(applied, Success, getattr(applied, "error", None))
        dataset_id = applied.data["dataset_id"]
        self.assertEqual(service.datasets.frame(dataset_id)["x"].tolist(), [0, 1, 2])
        self.assertIn("sql_sources", service.datasets.get(dataset_id).profile)
        chart = service.scatter(self.project.id, {"dataset_id": dataset_id, "x": "x", "y": "y"})
        self.assertIsInstance(chart, Success)
        self.assertEqual(chart.data.n, 3)

    def test_stale_foreign_and_truncated_proposals_do_not_create_datasets(self):
        service = self.service()
        raw = {
            "sql": f"SELECT x FROM {self.alias()}",
            "limit": 1,
            "name": "Limited",
            "explanation": "Тест.",
        }
        proposal = run_tool(service, self.project.id, "propose_sql_summary", raw).data["proposal"]
        self.assertIsInstance(service.assistant_apply_draft(self.project.id, proposal), Failure)
        other = self.projects.create("Other")
        self.assertIsInstance(service.assistant_apply_draft(other.id, proposal), Failure)
        document = service.metadata_store.get(self.dataset.id).document
        document.columns[0].missing_codes = ["0"]
        service.metadata_store.save(self.dataset.id, document.model_dump())
        self.assertIsInstance(service.assistant_apply_draft(self.project.id, proposal), Failure)
        self.assertEqual(len(service.datasets.list_datasets(self.project.id)), 1)

    def test_http_and_tool_share_query_contract(self):
        service = self.service()
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router(service))
        base = f"/api/projects/{self.project.id}/analytics"
        with TestClient(app) as client:
            self.assertEqual(client.get(base + "/schema").status_code, 200)
            response = client.post(
                base + "/query", json={"sql": f"SELECT sum(x) FROM {self.alias()}"}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["rows"], [[3]])
            self.assertEqual(
                client.post(base + "/query", json={"sql": "DROP TABLE x"}).status_code, 422
            )


class ModelAnalyticsTests(unittest.TestCase):
    setUp = model_fixtures.ModelStoreTests.setUp
    _run = model_fixtures.ModelStoreTests._run

    def test_model_tables_predictions_restart_and_cached_chart(self):
        run_id = self._run()
        model = self.service.model_train(self.project.id, {"training_run_id": run_id}).data
        tables = {
            t["kind"]: t
            for t in self.service.analytics.catalog(self.project.id)
            if t["object_id"] == model["id"]
        }
        self.assertEqual(set(tables), {"coefficients", "model_metrics", "predictions", "residuals"})
        prediction = self.service.analytics.read(
            self.project.id, "predictions", model["id"], tables["predictions"]["fingerprint"]
        )
        residuals = self.service.analytics.read(
            self.project.id, "residuals", model["id"], tables["residuals"]["fingerprint"]
        )
        np.testing.assert_allclose(
            prediction["actual"] - prediction["predicted"], residuals["residual"]
        )
        self.assertEqual(len(prediction), 20)
        first = self.service.model_check(
            self.project.id, {"model_id": model["id"], "chart": "actual"}
        )
        reopened = ProjectService(self.store)
        with patch("data_story.model_store.diagnostics", side_effect=AssertionError("recomputed")):
            second = reopened.model_check(
                self.project.id, {"model_id": model["id"], "chart": "actual"}
            )
        self.assertIsInstance(second, Success)
        self.assertEqual(second.data["n"], first.data["n"])
        self.assertIn("<svg", second.data["svg"])
        aliases = reopened.analytics_schema(self.project.id).data
        table = next(t for t in aliases if t["kind"] == "residuals")
        query = reopened.analytics_query(
            self.project.id,
            {"sql": f"SELECT avg(abs(residual)) FROM {table['alias']} WHERE split='test'"},
        )
        self.assertIsInstance(query, Success)
        self.assertAlmostEqual(
            query.data["rows"][0][0], model["document"]["metrics"]["test"]["mae"]
        )


class AssistantSqlTests(unittest.TestCase):
    setUp = assistant_fixtures.AssistantTests.setUp
    fake = assistant_fixtures.AssistantTests.fake

    def test_model_authors_sql_service_executes_in_explicit_chat_action(self):
        dataset = self.service.datasets.add(self.project.id, "values.csv", b"x\n1\n3\n")
        self.fake(
            [
                assistant_fixtures.tool_reply("analytics_schema", {}),
                assistant_fixtures.tool_reply(
                    "query_analytics",
                    {"sql": f"SELECT avg(x) FROM d_{dataset.id.replace('-', '')}"},
                    "call_2",
                ),
                assistant_fixtures.text_reply("Середнє — 2."),
            ]
        )
        result = self.service.assistant_ask(
            self.project.id, {"block": "eda", "mode": "advise", "question": "Яке середнє x?"}
        )
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        self.assertEqual(
            [t["name"] for t in result.data["tools"]], ["analytics_schema", "query_analytics"]
        )
        self.assertIn("2", self.requests[2]["input"][-1]["output"])
