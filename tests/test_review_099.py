"""Numerical and export regressions discovered in review 098; no external AI."""

import json
import unittest
from unittest.mock import patch

import numpy as np
import test_reports as report_fixtures
from scipy import stats

from data_story import embeddings
from data_story.chart_configuration import ChartConfiguration, ChartOptions
from data_story.conclusions import facts
from data_story.contracts import Success
from data_story.knowledge import METHOD
from data_story.model import format_p_value, stars, train_linear_regression
from data_story.storage import connect


class NumericalReviewTests(unittest.TestCase):
    def test_large_offset_has_analytical_standard_errors_and_p_values(self):
        x = np.arange(100, dtype=float) + 1_000_000
        y = 3 + 2 * x + np.sin(np.arange(100))
        arrays = {"x_train": x[:, None], "y_train": y, "x_test": x[:20, None], "y_test": y[:20]}
        result = train_linear_regression(arrays, ["x"])
        slope = result["coefficients"][0]
        residual = y - result["intercept"] - slope["value"] * x
        sigma2 = np.sum(residual**2) / 98
        sxx = np.sum((x - x.mean()) ** 2)
        expected = [np.sqrt(sigma2 * (1 / len(x) + x.mean() ** 2 / sxx)), np.sqrt(sigma2 / sxx)]
        actual = [result["intercept_inference"]["std_error"], slope["std_error"]]
        np.testing.assert_allclose(actual, expected, rtol=1e-7)
        p = 2 * stats.t.sf(abs(result["intercept"] / expected[0]), 98)
        self.assertAlmostEqual(result["intercept_inference"]["p_value"], p, places=7)
        self.assertIsNotNone(slope["p_value"])

    def test_rounding_preserves_all_significance_thresholds(self):
        for threshold in [0.001, 0.01, 0.05, 0.1]:
            for p in [threshold * 0.99, threshold, threshold * 1.01]:
                formatted = format_p_value(p)
                if formatted.startswith("<"):
                    self.assertLess(p, 0.001)
                else:
                    self.assertEqual(stars(float(formatted.replace(",", "."))), stars(p))
        self.assertNotEqual(format_p_value(0.049), format_p_value(0.051))


class IntegrationReviewTests(unittest.TestCase):
    setUp = report_fixtures.ReportTests.setUp
    _run = report_fixtures.ReportTests._run
    create = report_fixtures.ReportTests.create

    def test_inference_failure_falls_back_and_logs_search(self):
        self.addCleanup(embeddings.reset)
        with patch.object(
            self.service.knowledge, "_semantic", side_effect=RuntimeError("bad allocation")
        ):
            result = self.service.knowledge_search({"query": "що таке вільний член"})
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data["method"], METHOD)
        self.assertIn("bad allocation", result.data["semantic_reason"])
        self.assertTrue(result.data["hits"])
        self.assertEqual(self.service.knowledge.events()[0]["method"], METHOD)
        self.assertIsNone(embeddings._state["embedder"])

    def test_query_inference_failure_uses_lexical_hits(self):
        class Encoder:
            def encode(self, *args):
                raise MemoryError("query allocation")

        self.addCleanup(embeddings.reset)
        with patch.object(
            self.service.knowledge, "_semantic", return_value=(Encoder(), np.ones((1, 1)))
        ):
            result = self.service.knowledge.search("що таке вільний член")
        self.assertEqual(result["method"], METHOD)
        self.assertIn("query allocation", result["semantic_reason"])
        self.assertTrue(result["hits"])
        self.assertEqual(result["book_hits"], [])

    def test_report_caption_uses_actual_theme(self):
        options = ChartOptions(histogram_median=False, missing_order="source")
        self.service.chart_config.save(ChartConfiguration(theme="custom", custom=options))
        report = self.create()
        figures = {f["file"]: f for f in report["snapshot"]["figures"]}
        self.assertNotIn("медіана", figures["distribution.svg"]["explanation"].casefold())
        self.assertIn("у порядку стовпців", figures["missing.svg"]["explanation"])
        svg = self.service.reports.figure(
            self.project.id, report["id"], "distribution.svg"
        ).read_text()
        self.assertNotIn('class="median-line"', svg)

    def test_legacy_statistics_hidden_without_overwriting_saved_model(self):
        report = self.create()
        model_id = report["snapshot"]["model_id"]
        with connect(self.database) as connection:
            doc = json.loads(
                connection.execute(
                    "SELECT document FROM model_runs WHERE id=?", (model_id,)
                ).fetchone()[0]
            )
            doc["inference"].pop("version")
            connection.execute(
                "UPDATE model_runs SET document=? WHERE id=?", (json.dumps(doc), model_id)
            )
        view = self.service.model_store.get(self.project.id, model_id)["document"]
        self.assertIsNone(view["intercept_inference"]["p_value"])
        self.assertIn("попереднім методом", view["inference"]["reason"])
        with connect(self.database) as connection:
            stored = json.loads(
                connection.execute(
                    "SELECT document FROM model_runs WHERE id=?", (model_id,)
                ).fetchone()[0]
            )
        self.assertEqual(stored, doc)

    def test_conclusion_p_values_are_not_rounded_a_second_time(self):
        report = self.create()
        model = report["snapshot"]["model"]
        model["coefficients"][0]["p_value"] = 0.049
        model["coefficients"][1]["p_value"] = 0.051
        result = facts(model)
        self.assertEqual(result["coefficients"][0]["p_value"], "0,049")
        self.assertNotIn(model["coefficients"][1]["name"], result["significant_p_lt_0_05"])
