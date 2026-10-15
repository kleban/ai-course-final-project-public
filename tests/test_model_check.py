"""Model check block: regression line by feature, errors, prediction vs actual on test rows."""

import copy
import json
import unittest

import numpy as np
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.analysis_blocks import BLOCKS
from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.model import ModelError, train_linear_regression
from data_story.model_check import diagnostics, equation
from data_story.web import install_error_handlers, router


def fitted(x_train, y_train, x_test, y_test, names, features=None):
    arrays = {
        "x_train": np.asarray(x_train, float),
        "y_train": np.asarray(y_train, float),
        "x_test": np.asarray(x_test, float),
        "y_test": np.asarray(y_test, float),
    }
    document = train_linear_regression(arrays, names)
    document["choices"] = {"target": "y", "features": features or names}
    return document, arrays


class ModelCheckCoreTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        x = np.column_stack([rng.uniform(0, 10, 60), rng.uniform(0, 5, 60)])
        y = 3 + 2 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(scale=0.5, size=60)
        self.x, self.y = x, y
        self.document, self.arrays = fitted(x[:42], y[:42], x[42:], y[42:], ["x1", "x2"])

    def test_regression_line_holds_other_features_at_train_means(self):
        view = diagnostics(self.document, self.arrays, "regression", "x1")
        coef = {c["name"]: c["value"] for c in self.document["coefficients"]}
        base = self.document["intercept"] + coef["x2"] * self.x[:42, 1].mean()
        (x1, y1), (x2, y2) = view["lines"][0]["points"]
        self.assertAlmostEqual(y1, base + coef["x1"] * x1, places=9)
        self.assertAlmostEqual((y2 - y1) / (x2 - x1), coef["x1"], places=9)
        self.assertAlmostEqual(coef["x1"], 2, delta=0.1)
        self.assertEqual(view["points"][0], [self.x[42, 0], self.y[42]])
        self.assertEqual(len(view["points"]), 18)
        self.assertEqual(view["features"], ["x1", "x2"])
        self.assertEqual(diagnostics(self.document, self.arrays, "regression")["feature"], "x1")
        self.assertEqual(
            diagnostics(self.document, self.arrays, "regression", "x2")["x_label"], "x2"
        )

    def test_residuals_and_actual_match_independent_prediction(self):
        coef = np.array([c["value"] for c in self.document["coefficients"]])
        predicted = self.document["intercept"] + self.x[42:] @ coef
        residuals = diagnostics(self.document, self.arrays, "residuals")
        np.testing.assert_allclose([p[1] for p in residuals["points"]], self.y[42:] - predicted)
        self.assertEqual(residuals["lines"][0]["kind"], "zero")
        actual = diagnostics(self.document, self.arrays, "actual")
        np.testing.assert_allclose([p[1] for p in actual["points"]], predicted)
        (a, b), (c, d) = actual["lines"][0]["points"]
        self.assertEqual((a, c), (b, d))

    def test_equation_text_and_tex(self):
        text, tex = equation(self.document)
        self.assertTrue(text.startswith("ŷ(y) = "))
        self.assertIn("·x1", text)
        self.assertIn("\\hat y =", tex)
        self.assertIn("\\text{x2}", tex)

    def test_refusals(self):
        with self.assertRaises(ModelError):
            diagnostics(self.document, self.arrays, "regression", "missing")
        with self.assertRaises(ModelError):
            diagnostics(self.document, self.arrays, "pie")
        broken = copy.deepcopy(self.document)
        broken["metrics"]["test"]["mae"] += 1
        with self.assertRaises(ModelError):
            diagnostics(broken, self.arrays, "residuals")
        damaged = {**self.arrays, "x_test": self.arrays["x_test"][:, :1]}
        with self.assertRaises(ModelError):
            diagnostics(self.document, damaged, "residuals")

    def test_category_only_model_has_no_regression_line(self):
        cat = np.array([[1.0, 0.0], [0.0, 1.0]] * 15)
        y = 5 + 3 * cat[:, 1] + np.linspace(0, 1, 30)
        document, arrays = fitted(
            cat[:20, :1], y[:20], cat[20:, :1], y[20:], ["c = B"], features=["c"]
        )
        with self.assertRaises(ModelError):
            diagnostics(document, arrays, "regression")
        self.assertEqual(diagnostics(document, arrays, "residuals")["n"], 10)


class ModelCheckServiceTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def _model(self):
        lines = ["x1,x2,y"] + [
            f"{i},{(i * 7) % 5},{3 + 2 * i - 1.5 * ((i * 7) % 5) + (i % 3) * 0.4}"
            for i in range(30)
        ]
        dataset = self.service.datasets.add(self.project.id, "check.csv", "\n".join(lines).encode())
        run = self.service.training_prepare(
            self.project.id, {"dataset_id": dataset.id, "target": "y", "features": ["x1", "x2"]}
        ).data["id"]
        return self.service.model_train(self.project.id, {"training_run_id": run}).data["id"]

    def test_chart_svg_audit_and_switching(self):
        model_id = self._model()
        result = self.service.model_check(
            self.project.id, {"model_id": model_id, "chart": "regression", "feature": "x2"}
        )
        self.assertIsInstance(result, Success)
        self.assertIn('class="diagnostic-model"', result.data["svg"])
        self.assertEqual(result.data["feature"], "x2")
        self.assertNotIn("points", result.data)
        event = self.store.events(self.project.id)[-1]
        self.assertEqual((event.operation, event.payload["status"]), ("model.checked", "success"))
        self.assertEqual(event.payload["feature"], "x2")
        for chart, kind in (("residuals", "zero"), ("actual", "diagonal")):
            svg = self.service.model_check(
                self.project.id, {"model_id": model_id, "chart": chart}
            ).data["svg"]
            self.assertIn(f'class="diagnostic-{kind}"', svg)
        self.assertIsInstance(self.service.events(self.project.id), Success)

    def test_errors_logged_and_isolated(self):
        model_id = self._model()
        bad = self.service.model_check(
            self.project.id, {"model_id": model_id, "chart": "regression", "feature": "zzz"}
        )
        self.assertIsInstance(bad, Failure)
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "error")
        other = self.store.create("Other")
        self.assertEqual(
            self.service.model_check(other.id, {"model_id": model_id}).error.code, "not_found"
        )
        doc = self.service.training_runs(self.project.id).data[0]["document"]
        path = self.service.datasets.files_root / doc["matrix_path"]
        path.write_bytes(b"broken")
        self.assertIsInstance(
            self.service.model_check(self.project.id, {"model_id": model_id}), Failure
        )

    def test_http_and_assistant_tool(self):
        model_id = self._model()
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        route = f"/api/projects/{self.project.id}/models/check"
        with TestClient(app) as client:
            ok = client.post(route, json={"model_id": model_id, "chart": "actual"})
            self.assertEqual(ok.status_code, 200, ok.text)
            self.assertEqual(
                client.post(route, json={"model_id": model_id, "chart": "pie"}).status_code, 422
            )
        tool = run_tool(
            self.service,
            self.project.id,
            "build_model_chart",
            {"model_id": model_id, "chart": "residuals"},
        )
        self.assertIsInstance(tool, Success)
        self.assertNotIn("svg", tool.data)
        json.dumps(tool.data)
        self.assertTrue(BLOCKS["validation"].available)
