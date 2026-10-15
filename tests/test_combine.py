"""Joins and summaries (DuckDB): recipe validation, generated SQL, execution and provenance."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.combine import Plan, Recipe, SummaryError
from data_story.contracts import Failure, Success
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]
HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None

SALES = b"id,suburb,type,price,rooms\n1,North,h,100,2\n2,North,u,80,1\n3,South,h,200,3\n4,South,h,,4\n5,East,u,90,2\n"
AREAS = b"suburb,council,distance\nNorth,Alpha,5\nSouth,Beta,12\nWest,Gamma,20\n"


class CombineTests(unittest.TestCase):
    def setUp(self):
        folder = ROOT / "tmp" / "tests"
        folder.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=folder)
        self.addCleanup(self.temporary.cleanup)
        self.store = ProjectStore(Path(self.temporary.name) / "data.sqlite3")
        self.service = ProjectService(self.store)
        self.project = self.store.create("Combine")
        self.sales = self.service.datasets.add(self.project.id, "sales.csv", SALES)
        self.areas = self.service.datasets.add(self.project.id, "areas.csv", AREAS)

    def plan(self, **recipe):
        recipe.setdefault("dataset_id", str(self.sales.id))
        return self.service.summaries.plan(str(self.project.id), Recipe.model_validate(recipe))

    def test_sql_is_generated_with_quoted_names_and_parameters(self):
        plan = self.plan(
            join={"dataset_id": str(self.areas.id), "left_key": "suburb", "right_key": "suburb"},
            filters=[{"field": "type", "op": "eq", "value": "h'; DROP TABLE a; --"}],
            group_by=["council"],
            aggregates=[{"func": "avg", "field": "price"}, {"func": "count"}],
            order_by="avg_price",
            descending=True,
        )
        self.assertIn('INNER JOIN b ON a."suburb" = b."suburb"', plan.sql)
        self.assertIn('WHERE a."type" = ?', plan.sql)
        self.assertIn('AVG(a."price") AS "avg_price"', plan.sql)
        self.assertIn('GROUP BY b."council"', plan.sql)
        self.assertEqual(plan.params, ["h'; DROP TABLE a; --"])
        self.assertNotIn("DROP TABLE a; --'", plan.sql)
        self.assertIn("'h''; DROP TABLE a; --'", plan.literal_sql())
        # The right key with the same name is not duplicated.
        self.assertNotIn("suburb_2", plan.columns)

    def test_invalid_recipes_are_refused(self):
        cases = [
            {"filters": [{"field": "missing", "op": "eq", "value": "x"}]},
            {"aggregates": [{"func": "avg", "field": "suburb"}]},
            {"filters": [{"field": "suburb", "op": "gt", "value": "A"}]},
            {"filters": [{"field": "price", "op": "eq", "value": "дорого"}]},
            {"order_by": "nothing", "aggregates": [{"func": "count"}]},
            {
                "join": {
                    "dataset_id": str(self.areas.id),
                    "left_key": "price",
                    "right_key": "suburb",
                }
            },
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(SummaryError):
                self.plan(**case)
        for raw in [
            {"dataset_id": str(self.sales.id), "group_by": ["suburb"]},
            {"dataset_id": str(self.sales.id), "sql": "SELECT 1"},
            {
                "dataset_id": str(self.sales.id),
                "filters": [{"field": "x", "op": "in", "value": "a"}],
            },
        ]:
            with self.subTest(raw=raw):
                self.assertIsInstance(
                    self.service.summary_preview(str(self.project.id), raw), Failure
                )

    def test_foreign_dataset_is_refused(self):
        other = self.store.create("Other")
        foreign = self.service.datasets.add(other.id, "f.csv", AREAS)
        result = self.service.summary_preview(
            str(self.project.id),
            {"dataset_id": str(self.sales.id),
             "join": {"dataset_id": str(foreign.id), "left_key": "suburb", "right_key": "suburb"}},
        )  # fmt: skip
        self.assertIsInstance(result, Failure)

    def test_output_names_avoid_clashes(self):
        plan = self.plan(
            join={"dataset_id": str(self.sales.id), "left_key": "id", "right_key": "id",
                  "columns": ["price"]},
        )  # fmt: skip
        self.assertEqual(plan.columns["price_2"], ("b", "price"))
        self.assertIsInstance(plan, Plan)


@unittest.skipUnless(HAS_DUCKDB, "DuckDB is not installed in this environment")
class CombineExecutionTests(unittest.TestCase):
    setUp = CombineTests.setUp

    def test_preview_runs_in_duckdb_and_logs(self):
        result = self.service.summary_preview(
            str(self.project.id),
            {
                "dataset_id": str(self.sales.id),
                "join": {"dataset_id": str(self.areas.id), "left_key": "suburb",
                         "right_key": "suburb", "how": "left"},
                "group_by": ["suburb"],
                "aggregates": [{"func": "count"}, {"func": "avg", "field": "price"},
                               {"func": "max", "field": "distance"}],
                "order_by": "suburb",
                "descending": False,
            },
        )  # fmt: skip
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        data = result.data
        self.assertEqual(data["columns"], ["suburb", "count_rows", "avg_price", "max_distance"])
        rows = {r[0]: r[1:] for r in data["rows"]}
        self.assertEqual(rows["North"], [2, 90.0, 5.0])
        self.assertEqual(rows["South"][0], 2)
        self.assertEqual(rows["South"][1], 200.0)  # the missing price is ignored by AVG
        self.assertIsNone(rows["East"][2])  # LEFT JOIN keeps East without a council
        self.assertEqual(data["row_count"], 3)
        self.assertEqual(self.store.events(self.project.id)[-1].operation, "summary.previewed")

    def test_save_creates_independent_dataset_with_provenance(self):
        recipe = {
            "dataset_id": str(self.sales.id),
            "filters": [{"field": "rooms", "op": "ge", "value": 2}],
            "group_by": ["type"],
            "aggregates": [{"func": "median", "field": "price", "name": "Медіана ціни"}],
        }
        preview = self.service.summary_preview(str(self.project.id), recipe).data
        stale = self.service.summary_create(
            str(self.project.id),
            {"recipe": {**recipe, "limit": 1}, "preview_token": preview["preview_token"]},
        )
        self.assertIsInstance(stale, Failure)
        saved = self.service.summary_create(
            str(self.project.id),
            {"recipe": recipe, "preview_token": preview["preview_token"], "name": "Ціни за типом"},
        )
        self.assertIsInstance(saved, Success, getattr(saved, "error", None))
        dataset = self.service.datasets.get(str(saved.data.id))
        self.assertEqual(dataset.name, "Ціни за типом.csv")
        self.assertEqual(dataset.row_count, 2)
        self.assertEqual(dataset.profile["engine"], "duckdb")
        self.assertIn(
            dataset.id,
            {r["object_id"] for r in self.service.analytics.catalog(str(self.project.id))},
        )
        self.assertIn("MEDIAN", dataset.profile["summary_sql"])
        self.assertEqual(dataset.profile["summary_sources"][0]["id"], str(self.sales.id))
        metadata = self.service.metadata(str(dataset.id)).data.document
        column = next(c for c in metadata.columns if c.name == "Медіана ціни")
        self.assertIn("Медіана", column.description)
        self.assertEqual(self.service.datasets.get(str(self.sales.id)).row_count, 5)
        self.assertEqual(self.store.events(self.project.id)[-1].operation, "summary.created")
        # The new dataset is a normal dataset: statistics work on it.
        stats = self.service.statistics(
            str(self.project.id), {"dataset_id": str(dataset.id), "view": "descriptive"}
        )
        self.assertIsInstance(stats, Success, getattr(stats, "error", None))

    def test_http_routes(self):
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router(self.service))
        base = f"/api/projects/{self.project.id}/summaries"
        with TestClient(app) as client:
            fields = client.get(f"{base}/fields/{self.areas.id}")
            self.assertEqual(fields.status_code, 200)
            self.assertEqual(
                [f["name"] for f in fields.json()["data"]], ["suburb", "council", "distance"]
            )
            preview = client.post(
                f"{base}/preview",
                json={
                    "dataset_id": str(self.areas.id),
                    "filters": [{"field": "distance", "op": "lt", "value": 15}],
                },
            )
            self.assertEqual(preview.status_code, 200)
            self.assertEqual(preview.json()["data"]["row_count"], 2)
            self.assertEqual(
                client.post(f"{base}/preview", json={"sql": "SELECT 1"}).status_code, 422
            )
