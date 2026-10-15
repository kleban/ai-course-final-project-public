"""Stage 06: LinearRegression on saved splits — known answers, refusals, persistence, API."""

import json
import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.analysis_blocks import BLOCKS
from data_story.assistant_tools import TOOLS, run_tool
from data_story.contracts import Failure, Success
from data_story.model import (
    ModelError,
    check_matrices,
    explain,
    stars,
    train_linear_regression,
)
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
    _V8,
    _V9,
    _V10,
    SCHEMA_VERSION,
    connect,
    ensure_schema,
)
from data_story.web import install_error_handlers, router


def arrays(x_train, y_train, x_test, y_test):
    return {
        "x_train": np.asarray(x_train, dtype=float),
        "y_train": np.asarray(y_train, dtype=float),
        "x_test": np.asarray(x_test, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
    }


def expected(data, names):
    return {
        "train_count": len(data["y_train"]),
        "test_count": len(data["y_test"]),
        "feature_count": data["x_train"].shape[1],
        "feature_names": names,
    }


class ModelCoreTests(unittest.TestCase):
    def test_known_noise_free_coefficients_recovered_exactly(self):
        rng = np.random.default_rng(7)
        x = rng.uniform(-10, 10, size=(40, 2))
        y = 3 + 2 * x[:, 0] - 1.5 * x[:, 1]
        data = arrays(x[:28], y[:28], x[28:], y[28:])
        result = train_linear_regression(data, ["x1", "x2"])
        values = [c["value"] for c in result["coefficients"]]
        np.testing.assert_allclose(values, [2, -1.5], atol=1e-9)
        self.assertAlmostEqual(result["intercept"], 3, places=9)
        self.assertAlmostEqual(result["metrics"]["test"]["r2"], 1, places=12)
        self.assertLess(result["metrics"]["test"]["mae"], 1e-9)
        self.assertLess(result["metrics"]["test"]["rmse"], 1e-9)
        self.assertEqual(result["rank"], 3)
        self.assertEqual([c["name"] for c in result["coefficients"]], ["x1", "x2"])

    def test_noisy_example_matches_independent_lstsq_and_hand_metrics(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(60, 3))
        y = 1.2 + x @ np.array([0.5, -2.0, 3.0]) + rng.normal(scale=0.7, size=60)
        data = arrays(x[:42], y[:42], x[42:], y[42:])
        result = train_linear_regression(data, ["a", "b", "c"])
        design = np.column_stack([np.ones(42), x[:42]])
        solution = np.linalg.lstsq(design, y[:42], rcond=None)[0]
        self.assertAlmostEqual(result["intercept"], solution[0], places=10)
        np.testing.assert_allclose(
            [c["value"] for c in result["coefficients"]], solution[1:], atol=1e-10
        )
        predicted = solution[0] + x[42:] @ solution[1:]
        error = y[42:] - predicted
        r2 = 1 - (error**2).sum() / ((y[42:] - y[42:].mean()) ** 2).sum()
        test = result["metrics"]["test"]
        self.assertAlmostEqual(test["r2"], r2, places=10)
        self.assertAlmostEqual(test["mae"], np.abs(error).mean(), places=10)
        self.assertAlmostEqual(test["rmse"], np.sqrt((error**2).mean()), places=10)
        baseline = np.abs(y[42:] - y[:42].mean()).mean()
        self.assertAlmostEqual(result["metrics"]["baseline_mae"], baseline, places=10)
        self.assertTrue(any("Тестова частина мала" in w for w in result["warnings"]))

    def test_fit_never_sees_test_values(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(30, 2))
        y = x @ np.array([1.0, 2.0]) + rng.normal(size=30)
        first = train_linear_regression(arrays(x[:20], y[:20], x[20:], y[20:]), ["a", "b"])
        second = train_linear_regression(arrays(x[:20], y[:20], x[20:], y[20:] * 100), ["a", "b"])
        self.assertEqual(first["coefficients"], second["coefficients"])
        self.assertEqual(first["intercept"], second["intercept"])
        self.assertNotEqual(first["metrics"]["test"], second["metrics"]["test"])

    def test_full_one_hot_collinearity_warns_but_predicts(self):
        cat = np.array([0, 1] * 10)
        x = np.column_stack([np.arange(20.0), cat == 0, cat == 1]).astype(float)
        y = 3 + 2 * x[:, 0] + 4 * x[:, 2]
        result = train_linear_regression(
            arrays(x[:14], y[:14], x[14:], y[14:]), ["x", "cat = A", "cat = B"]
        )
        values = [c["value"] for c in result["coefficients"]]
        self.assertAlmostEqual(values[0], 2, places=8)
        self.assertAlmostEqual(values[2] - values[1], 4, places=8)
        self.assertAlmostEqual(result["metrics"]["test"]["r2"], 1, places=10)
        self.assertEqual(result["rank"], 3)
        self.assertTrue(any("лінійно залежні" in w for w in result["warnings"]))

    def test_constant_test_target_leaves_r2_undefined(self):
        x = np.arange(12.0).reshape(-1, 1)
        y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 5, 5], dtype=float)
        result = train_linear_regression(arrays(x[:9], y[:9], x[9:], y[9:]), ["x"])
        test = result["metrics"]["test"]
        self.assertIsNone(test["r2"])
        self.assertIn("однакові", test["r2_reason"])
        self.assertTrue(np.isfinite(test["mae"]))
        self.assertIn("однакові", explain({**result, "choices": {"target": "y"}})[0])
        json.dumps(result, allow_nan=False)

    def test_too_few_training_rows_and_unstable_values_refused(self):
        x = np.arange(6.0).reshape(-1, 2)
        with self.assertRaises(ModelError):
            train_linear_regression(arrays(x[:3], [1, 2, 3], x[:2], [1, 2]), ["a", "b"])
        big = np.full((10, 1), 1e308)
        big[::2] = -1e308
        with self.assertRaises(ModelError):
            train_linear_regression(arrays(big[:7], big[:7, 0], big[7:], [1, 2, 3]), ["a"])

    def test_p_values_match_published_r_summary_for_mtcars(self):
        # R: summary(lm(mpg ~ wt, data = mtcars)) — published, independent of our code.
        mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2, 17.8, 16.4, 17.3,
               15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9, 21.5, 15.5, 15.2, 13.3, 19.2, 27.3,
               26.0, 30.4, 15.8, 19.7, 15.0, 21.4]  # fmt: skip
        wt = [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440, 3.440,
              4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835, 2.465, 3.520,
              3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780]  # fmt: skip
        x = np.array(wt).reshape(-1, 1)
        result = train_linear_regression(arrays(x, mpg, x[:2], mpg[:2]), ["wt"])
        intercept, slope = result["intercept_inference"], result["coefficients"][0]
        self.assertAlmostEqual(result["intercept"], 37.2851, places=4)
        self.assertAlmostEqual(intercept["std_error"], 1.8776, places=4)
        self.assertAlmostEqual(intercept["t"], 19.858, places=3)
        self.assertLess(intercept["p_value"], 2e-16)
        self.assertAlmostEqual(slope["value"], -5.3445, places=4)
        self.assertAlmostEqual(slope["std_error"], 0.5591, places=4)
        self.assertAlmostEqual(slope["t"], -9.559, places=3)
        self.assertAlmostEqual(slope["p_value"] * 1e10, 1.29, places=2)
        self.assertEqual((slope["stars"], intercept["stars"]), ("***", "***"))
        self.assertEqual(result["inference"]["df_residual"], 30)
        self.assertAlmostEqual(result["inference"]["sigma"], 3.046, places=3)

    def test_significance_stars_like_r(self):
        cases = [(0.0005, "***"), (0.005, "**"), (0.03, "*"), (0.07, "."), (0.5, ""), (None, "")]
        for value, mark in cases:
            self.assertEqual(stars(value), mark)

    def test_noise_feature_not_significant_and_collinear_undefined(self):
        rng = np.random.default_rng(21)
        x = np.column_stack([rng.normal(size=200), rng.normal(size=200)])
        y = 2 + 3 * x[:, 0] + rng.normal(size=200)
        result = train_linear_regression(arrays(x, y, x[:5], y[:5]), ["signal", "noise"])
        signal, noise = result["coefficients"]
        self.assertEqual(signal["stars"], "***")
        self.assertGreater(noise["p_value"], 0.05)
        self.assertIn("signal", explain({**result, "choices": {"target": "y"}})[3])
        cat = np.array([0, 1] * 10)
        dummies = np.column_stack([np.arange(20.0), cat == 0, cat == 1]).astype(float)
        target = 3 + 2 * dummies[:, 0] + 4 * dummies[:, 2] + rng.normal(size=20)
        trap = train_linear_regression(
            arrays(dummies, target, dummies[:3], target[:3]), ["x", "cat = A", "cat = B"]
        )
        self.assertIsNotNone(trap["coefficients"][0]["p_value"])
        self.assertIsNone(trap["coefficients"][1]["p_value"])
        self.assertIsNone(trap["intercept_inference"]["p_value"])
        self.assertIn("лінійну залежність", trap["coefficients"][2]["reason"])

    def test_perfect_fit_hides_p_values(self):
        x = np.arange(20.0).reshape(-1, 1)
        result = train_linear_regression(
            arrays(x[:14], 2 * x[:14, 0], x[14:], 2 * x[14:, 0]), ["x"]
        )
        self.assertIsNone(result["coefficients"][0]["p_value"])
        self.assertIn("ненадійні", result["inference"]["reason"])

    def test_damaged_or_mismatching_matrices_refused(self):
        x = np.arange(20.0).reshape(-1, 1)
        good = arrays(x[:7], x[:7, 0], x[7:10], x[7:10, 0])
        check_matrices(good, expected(good, ["x"]))
        cases = [
            {k: v for k, v in good.items() if k != "y_test"},
            {**good, "extra": good["y_test"]},
            {**good, "x_train": good["x_train"].ravel()},
            {**good, "y_train": good["y_train"].astype(bool)},
            {**good, "x_test": np.where(good["x_test"] > 8, np.nan, good["x_test"])},
            {**good, "x_train": np.vstack([good["x_train"], [[1.0]]])},
        ]
        for case in cases:
            with self.assertRaises(ModelError):
                check_matrices(case, expected(good, ["x"]))
        with self.assertRaises(ModelError):
            check_matrices(good, expected(good, ["x", "z"]))

    def test_explanation_compares_baseline(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(50, 1))
        y = 4 * x[:, 0] + rng.normal(scale=0.1, size=50)
        result = train_linear_regression(arrays(x[:35], y[:35], x[35:], y[35:]), ["x"])
        lines = explain({**result, "choices": {"target": "Price"}})
        self.assertTrue(lines[0].startswith("R² = 0,99"))
        self.assertIn("зменшує середню помилку", lines[-1])


class ModelStoreTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def _run(self, project_id=None, rows=20):
        project_id = project_id or self.project.id
        lines = ["x1,x2,y"] + [
            f"{i},{(i * 7) % 5},{3 + 2 * i - 1.5 * ((i * 7) % 5)}" for i in range(rows)
        ]
        dataset = self.service.datasets.add(project_id, "model.csv", "\n".join(lines).encode())
        raw = {"dataset_id": dataset.id, "target": "y", "features": ["x1", "x2"]}
        result = self.service.training_prepare(project_id, raw)
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        return result.data["id"]

    def test_known_answer_through_service_persisted_with_provenance(self):
        run_id = self._run()
        before = len(self.store.events(self.project.id))
        result = self.service.model_train(self.project.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Success)
        doc = result.data["document"]
        np.testing.assert_allclose([c["value"] for c in doc["coefficients"]], [2, -1.5], atol=1e-8)
        self.assertAlmostEqual(doc["intercept"], 3, places=8)
        self.assertAlmostEqual(doc["metrics"]["test"]["r2"], 1, places=10)
        self.assertEqual((doc["train_count"], doc["test_count"]), (14, 6))
        training = self.service.training_runs(self.project.id).data[0]["document"]
        self.assertEqual(doc["matrix_sha256"], training["matrix_sha256"])
        self.assertEqual(doc["source_sha256"], training["sha256"])
        self.assertEqual(doc["choices"], training["choices"])
        self.assertEqual(doc["estimator"], "sklearn.linear_model.LinearRegression")
        self.assertTrue(doc["params"]["fit_intercept"])
        events = self.store.events(self.project.id)
        self.assertEqual(len(events), before + 1)
        event = events[-1]
        self.assertEqual((event.operation, event.payload["status"]), ("model.trained", "success"))
        self.assertEqual(event.payload["training_run_id"], run_id)
        self.assertNotIn("coefficients", event.payload)
        journal = self.service.events(self.project.id)
        self.assertIsInstance(journal, Success)
        self.assertEqual(journal.data[-1].operation, "model.trained")
        reopened = ProjectService(ProjectStore(self.database))
        stored = reopened.model_runs(self.project.id).data
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["document"], doc)
        self.assertEqual(
            reopened.model_run(self.project.id, result.data["id"]).data["document"], doc
        )

    def test_category_gets_its_own_p_value_against_baseline(self):
        lines = ["x,cat,y"] + [
            f"{i},{'B' if i % 3 == 0 else 'A'},{3 + 2 * i + (4 if i % 3 == 0 else 0) + (i % 5) * 0.3}"
            for i in range(40)
        ]
        dataset = self.service.datasets.add(self.project.id, "cat.csv", "\n".join(lines).encode())
        run = self.service.training_prepare(
            self.project.id,
            {
                "dataset_id": dataset.id,
                "target": "y",
                "features": ["x", "cat"],
                "encoding": "onehot",
            },
        ).data["id"]
        doc = self.service.model_train(self.project.id, {"training_run_id": run}).data["document"]
        names = [c["name"] for c in doc["coefficients"]]
        self.assertEqual(names, ["x", "cat = B"])
        self.assertEqual(doc["transforms"]["cat"]["baseline"], "A")
        self.assertEqual(doc["rank"], 3)
        self.assertFalse(any("лінійно залежні" in w for w in doc["warnings"]))
        shift = doc["coefficients"][1]
        self.assertAlmostEqual(shift["value"], 4, delta=0.5)
        self.assertEqual(shift["stars"], "***")
        self.assertIsNotNone(doc["intercept_inference"]["p_value"])

    def test_repeat_returns_saved_result_without_new_event(self):
        run_id = self._run()
        first = self.service.model_train(self.project.id, {"training_run_id": run_id}).data
        count = len(self.store.events(self.project.id))
        second = self.service.model_train(self.project.id, {"training_run_id": run_id}).data
        self.assertTrue(second["reused"])
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(self.store.events(self.project.id)), count)

    def test_isolation_and_unknown_ids(self):
        run_id = self._run()
        other = self.store.create("Other")
        result = self.service.model_train(other.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Failure)
        self.assertEqual(self.service.model_runs(other.id).data, [])
        self.assertEqual(self.store.events(other.id)[-1].payload["status"], "error")
        model = self.service.model_train(self.project.id, {"training_run_id": run_id}).data
        self.assertIsInstance(self.service.model_run(other.id, model["id"]), Failure)
        self.assertIsInstance(
            self.service.model_train(self.project.id, {"training_run_id": "bad"}), Failure
        )

    def test_corrupted_matrix_file_refused_and_logged(self):
        run_id = self._run()
        doc = self.service.training_runs(self.project.id).data[0]["document"]
        path = self.service.datasets.files_root / doc["matrix_path"]
        path.write_bytes(path.read_bytes()[:-10])
        result = self.service.model_train(self.project.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Failure)
        self.assertIn("пошкоджений", result.error.message)
        event = self.store.events(self.project.id)[-1]
        self.assertEqual((event.operation, event.payload["status"]), ("model.trained", "error"))
        self.assertEqual(self.service.model_runs(self.project.id).data, [])
        self.assertFalse(
            [
                t
                for t in self.service.analytics.catalog(self.project.id)
                if t["kind"] in {"coefficients", "model_metrics", "predictions", "residuals"}
            ]
        )

    def test_matrix_shape_mismatch_with_valid_hash_refused(self):
        run_id = self._run()
        with connect(self.database) as connection:
            row = connection.execute("SELECT document FROM training_runs").fetchone()
            doc = json.loads(row["document"])
            doc["report"]["feature_count"] = 3
            connection.execute("UPDATE training_runs SET document = ?", (json.dumps(doc),))
        result = self.service.model_train(self.project.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Failure)
        self.assertIn("не збігаються", result.error.message)

    def test_too_small_split_refused(self):
        run_id = self._run(rows=5)
        result = self.service.model_train(self.project.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Failure)
        self.assertIn("Замало навчальних рядків", result.error.message)

    def test_atomic_rollback_on_audit_failure(self):
        run_id = self._run()
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            result = self.service.model_train(self.project.id, {"training_run_id": run_id})
        self.assertIsInstance(result, Failure)
        self.assertEqual(result.error.code, "storage_error")
        self.assertEqual(self.service.model_runs(self.project.id).data, [])

    def test_http_train_list_get_and_errors(self):
        run_id = self._run()
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        route = f"/api/projects/{self.project.id}/models"
        with TestClient(app) as client:
            response = client.post(route, json={"training_run_id": run_id})
            self.assertEqual(response.status_code, 200, response.text)
            model_id = response.json()["data"]["id"]
            self.assertEqual(len(client.get(route).json()["data"]), 1)
            self.assertEqual(client.get(f"{route}/{model_id}").status_code, 200)
            self.assertEqual(client.post(route, json={"training_run_id": "x"}).status_code, 422)
            self.assertEqual(client.post(route, json={}).status_code, 422)
            missing = "00000000-0000-4000-8000-000000000000"
            self.assertEqual(client.get(f"{route}/{missing}").status_code, 404)
            self.assertEqual(client.get(f"/api/projects/{missing}/models").status_code, 404)

    def test_assistant_tools_reuse_service(self):
        run_id = self._run()
        schemas = [tool.schema() for tool in TOOLS.values()]
        json.dumps(schemas)
        self.assertEqual(
            [s["name"] for s in schemas],
            [
                "propose_project_description",
                "propose_metadata",
                "propose_eda_chart",
                "list_training_runs",
                "train_linear_regression",
                "stepwise_linear_regression",
                "explain_model_result",
                "search_knowledge",
                "build_model_chart",
                "analytics_schema",
                "query_analytics",
                "propose_sql_summary",
            ],
        )
        self.assertEqual({t.mode for t in TOOLS.values()}, {"advise", "execute", "explain"})
        training_schema = next(s for s in schemas if s["name"] == "train_linear_regression")
        self.assertIn("training_run_id", training_schema["parameters"]["properties"])
        listed = run_tool(self.service, self.project.id, "list_training_runs", {}).data
        self.assertEqual(listed[0]["models"], [])
        trained = run_tool(
            self.service, self.project.id, "train_linear_regression", {"training_run_id": run_id}
        )
        self.assertIsInstance(trained, Success)
        explained = run_tool(
            self.service, self.project.id, "explain_model_result", {"model_id": trained.data["id"]}
        ).data
        self.assertTrue(explained["explanation"][0].startswith("R² = 1"))
        self.assertEqual(explained["provenance"]["training_run_id"], run_id)
        self.assertIsInstance(run_tool(self.service, self.project.id, "nope", {}), Failure)

    def test_model_block_available_in_navigation(self):
        self.assertTrue(BLOCKS["model"].available)
        self.assertEqual(BLOCKS["model"].view, "model")


class ModelMigrationTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def test_v10_database_upgrades_to_v11_with_data_kept(self):
        database = self.database.with_name("old.sqlite3")
        with connect(database) as connection:
            connection.executescript(
                _V1
                + _V2
                + _V3
                + _V4
                + _V5
                + _V6
                + _V7
                + _V8
                + _V9
                + _V10
                + "INSERT INTO projects VALUES "
                "('p', 'Old', '', '2026-01-01T00:00:00+00:00', 'regression', '');"
                "PRAGMA user_version=10;"
            )
        ensure_schema(database)
        with connect(database) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
            )
            self.assertGreaterEqual(SCHEMA_VERSION, 11)
            self.assertEqual(connection.execute("SELECT name FROM projects").fetchone()[0], "Old")
            connection.execute("SELECT * FROM model_runs").fetchall()
