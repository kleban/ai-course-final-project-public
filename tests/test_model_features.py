"""History 143: one split from all fields, model fields chosen by hand or stepwise."""

import sqlite3
import unittest

import numpy as np
import test_preparation_store as fixtures

from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.model import ModelError
from data_story.stepwise import fit_stats, stepwise
from data_story.storage import SCHEMA_VERSION, connect, ensure_schema


def noisy_rows(rows=80, seed=7):
    """y depends on x1, x2 and category g; ``noise`` is unrelated to y."""
    rng = np.random.default_rng(seed)
    lines = ["x1,x2,noise,g,y"]
    for i in range(rows):
        x1, x2, noise = rng.normal(size=3)
        g = "B" if i % 3 == 0 else "A"
        y = 3 + 2 * x1 - 1.5 * x2 + (2.5 if g == "B" else 0) + rng.normal(scale=0.3)
        lines.append(f"{x1:.6f},{x2:.6f},{noise:.6f},{g},{y:.6f}")
    return "\n".join(lines).encode()


class StepwiseUnitTests(unittest.TestCase):
    def test_fit_stats_match_textbook_formulas(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(50, 2))
        y = 1 + x @ [2.0, -1.0] + rng.normal(size=50)
        stats = fit_stats(x, y)
        design = np.column_stack([np.ones(50), x])
        beta = np.linalg.solve(design.T @ design, design.T @ y)
        rss = float(np.sum((y - design @ beta) ** 2))
        r2 = 1 - rss / float(np.sum((y - y.mean()) ** 2))
        self.assertAlmostEqual(stats["r2"], r2)
        self.assertAlmostEqual(stats["adj_r2"], 1 - (1 - r2) * 49 / 47)
        self.assertAlmostEqual(stats["aic"], 50 * np.log(rss / 50) + 2 * 3)
        self.assertAlmostEqual(stats["bic"], 50 * np.log(rss / 50) + np.log(50) * 3)

    def test_drops_noise_and_moves_category_as_one_group(self):
        rng = np.random.default_rng(3)
        n = 120
        x1, x2, noise = rng.normal(size=(3, n))
        level = rng.integers(0, 3, size=n)
        dummies = np.column_stack([level == 1, level == 2]).astype(float)
        y = 1 + 2 * x1 - x2 + dummies @ [3.0, -2.0] + rng.normal(scale=0.2, size=n)
        x = np.column_stack([x1, x2, noise, dummies])
        columns = {"x1": [0], "x2": [1], "noise": [2], "g": [3, 4]}
        for criterion in ("adj_r2", "aic", "bic"):
            with self.subTest(criterion=criterion):
                result = stepwise(x, y, columns, criterion)
                self.assertTrue({"x1", "x2", "g"} <= set(result["features"]))
                self.assertEqual(result["steps"][0]["action"], "start")
                last = result["steps"][-1]
                self.assertEqual(last["action"], "stop")
                self.assertEqual(last["candidates"][0]["action"], "none")
                first = result["steps"][1]["candidates"]
                self.assertEqual((first[0]["action"], len(first)), ("add", 5))
                values = [s[criterion] for s in result["steps"]]
                better = sorted(values, reverse=criterion != "adj_r2")
                self.assertEqual(values, better)
        self.assertEqual(stepwise(x, y, columns, "bic")["features"], ["x1", "x2", "g"])

    def test_unknown_criterion_and_useless_fields_refused(self):
        x = np.arange(20, dtype=float).reshape(-1, 1)
        with self.assertRaises(ModelError):
            stepwise(x, x.ravel(), {"x": [0]}, "r2")
        y = np.tile([1.0, -1.0], 10)
        with self.assertRaises(ModelError):
            stepwise(x, y, {"x": [0]}, "bic")


class ModelFeatureTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def _split(self):
        dataset = self.service.datasets.add(self.project.id, "noisy.csv", noisy_rows())
        result = self.service.training_prepare(
            self.project.id,
            {
                "dataset_id": dataset.id,
                "target": "y",
                "features": ["x1", "x2", "noise", "g"],
                "encoding": "onehot",
            },  # fmt: skip
        )
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        return result.data["id"]

    def test_several_models_on_one_split(self):
        run = self._split()
        project = self.project.id
        full = self.service.model_train(project, {"training_run_id": run}).data
        small = self.service.model_train(
            project, {"training_run_id": run, "features": ["g", "x1"]}
        ).data
        self.assertFalse(small["reused"])
        doc = small["document"]
        self.assertEqual(doc["choices"]["features"], ["x1", "g"])
        self.assertEqual([c["name"] for c in doc["coefficients"]], ["x1", "g = B"])
        self.assertEqual(doc["split_features"], ["x1", "x2", "noise", "g"])
        self.assertEqual(doc["selection"], {"method": "manual"})
        self.assertEqual(set(doc["transforms"]), {"g"})
        self.assertEqual(len(full["document"]["coefficients"]), 4)
        again = self.service.model_train(project, {"training_run_id": run, "features": ["x1", "g"]})
        self.assertTrue(again.data["reused"])
        self.assertEqual(again.data["id"], small["id"])
        self.assertEqual(len(self.service.model_runs(project).data), 2)
        check = self.service.model_check(
            project, {"model_id": small["id"], "chart": "regression", "feature": "x1"}
        )
        self.assertIsInstance(check, Success, getattr(check, "error", None))
        self.assertEqual(check.data["n"], doc["test_count"])
        stored = self.service.model_run(project, small["id"]).data["document"]
        self.assertEqual(stored["choices"]["features"], ["x1", "g"])
        for bad in (["missing"], []):
            with self.subTest(features=bad):
                refused = self.service.model_train(
                    project, {"training_run_id": run, "features": bad}
                )
                self.assertIsInstance(refused, Failure)

    def test_stepwise_through_service_and_assistant(self):
        run = self._split()
        project = self.project.id
        result = self.service.model_stepwise(project, {"training_run_id": run, "criterion": "bic"})
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        selection = result.data["selection"]
        self.assertEqual(selection["features"], ["x1", "x2", "g"])
        self.assertEqual(selection["part"], "train")
        doc = result.data["document"]
        self.assertEqual(doc["choices"]["features"], ["x1", "x2", "g"])
        self.assertEqual(doc["selection"]["criterion"], "bic")
        self.assertGreater(doc["metrics"]["test"]["r2"], 0.9)
        manual = self.service.model_train(
            project, {"training_run_id": run, "features": ["x1", "x2", "g"]}
        ).data
        self.assertTrue(manual["reused"])
        self.assertIsInstance(
            self.service.model_stepwise(project, {"training_run_id": run, "criterion": "r2"}),
            Failure,
        )
        tool = run_tool(
            self.service, project, "stepwise_linear_regression",
            {"training_run_id": run, "criterion": "adj_r2"},
        )  # fmt: skip
        self.assertIsInstance(tool, Success)
        listed = run_tool(self.service, project, "list_training_runs", {}).data
        self.assertGreaterEqual(len(listed[0]["models"]), 1)

    def test_v16_unique_models_table_upgrades_and_keeps_conclusions(self):
        run = self._split()
        model = self.service.model_train(self.project.id, {"training_run_id": run}).data
        with connect(self.database) as c:
            c.execute(
                "INSERT INTO conclusions VALUES ('c1', ?, ?, 't', 't', 1, 'draft', '{}')",
                (self.project.id, model["id"]),
            )
        raw = sqlite3.connect(self.database)
        raw.executescript(
            """
            PRAGMA foreign_keys = OFF;
            BEGIN;
            CREATE TABLE old_models (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id),
                training_run_id TEXT NOT NULL UNIQUE REFERENCES training_runs(id),
                created_at TEXT NOT NULL,
                document TEXT NOT NULL
            );
            INSERT INTO old_models SELECT * FROM model_runs;
            DROP TABLE model_runs;
            ALTER TABLE old_models RENAME TO model_runs;
            CREATE INDEX model_runs_project ON model_runs(project_id, created_at);
            PRAGMA user_version = 16;
            COMMIT;
            """
        )
        raw.close()
        ensure_schema(self.database)
        with connect(self.database) as c:
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, 17)
            self.assertEqual(c.execute("SELECT count(*) FROM model_runs").fetchone()[0], 1)
            self.assertIsNone(c.execute("PRAGMA foreign_key_check").fetchone())
            self.assertEqual(c.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        second = self.service.model_train(
            self.project.id, {"training_run_id": run, "features": ["x1"]}
        )
        self.assertIsInstance(second, Success, getattr(second, "error", None))
        self.assertEqual(
            self.service.model_run(self.project.id, model["id"]).data["id"], model["id"]
        )
