"""Histogram: full range, [a, b) bins with the maximum included, n from stage 03 rules."""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data_story.chart_configuration import ChartConfiguration, ChartOptions, preset
from data_story.charts import BinningError, histogram_bins, nice_ticks, number_text
from data_story.contracts import Failure, HistogramQuery, Success
from data_story.metadata import MetadataStore
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "statistics-melbourne-reference.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


class BinningTests(unittest.TestCase):
    def test_count_mode_edges_and_maximum(self):
        values = [0.0, 1.0, 2.0, 2.5, 5.0, 9.999, 10.0]
        result = histogram_bins(values, 5)
        self.assertEqual(result["edges"][0], 0.0)
        self.assertEqual(result["edges"][-1], 10.0)
        self.assertEqual(len(result["counts"]), 5)
        self.assertEqual(sum(result["counts"]), len(values))
        # 2.0 lies on the edge [2, 4): it belongs to the right interval; max in the last one.
        self.assertEqual(result["counts"], [2, 2, 1, 0, 2])
        self.assertFalse(result["constant"])

    def test_width_mode_covers_full_range(self):
        result = histogram_bins([1.0, 2.0, 7.0], width=2.5)
        self.assertEqual(result["mode"], "width")
        self.assertEqual(result["edges"], [1.0, 3.5, 6.0, 8.5])
        self.assertEqual(result["counts"], [2, 0, 1])
        exact = histogram_bins([0.0, 10.0], width=5)
        self.assertEqual(exact["edges"], [0.0, 5.0, 10.0])
        self.assertEqual(exact["counts"], [1, 1])

    def test_floating_edges_never_lose_values(self):
        values = [0.1 * i for i in range(1001)]
        for bins in (5, 7, 13, 60):
            self.assertEqual(sum(histogram_bins(values, bins)["counts"]), len(values))
        self.assertEqual(sum(histogram_bins(values, width=1.3)["counts"]), len(values))

    def test_empty_constant_and_invalid_settings(self):
        empty = histogram_bins([])
        self.assertEqual(empty["counts"], [])
        self.assertIn("Немає придатних", empty["reason"])
        constant = histogram_bins([4.0, 4.0, 4.0], 20)
        self.assertTrue(constant["constant"])
        self.assertEqual(constant["counts"], [3])
        self.assertIn("однакові", constant["reason"])
        with self.assertRaises(BinningError):
            histogram_bins([0.0, 1.0], 4)
        with self.assertRaises(BinningError):
            histogram_bins([0.0, 1000.0], width=1)
        with self.assertRaises(BinningError):
            histogram_bins([0.0, 1.0], width=0)

    def test_number_text_and_ticks(self):
        self.assertEqual(number_text(1330000.0, 2), "1 330 000")
        self.assertEqual(number_text(0.5, 2), "0.50")
        self.assertEqual(number_text(0.0001, 2), "0.0001")
        ticks = nice_ticks(85000, 9000000)
        self.assertTrue(all(85000 <= t <= 9000000 for t in ticks))
        self.assertGreaterEqual(len(ticks), 3)

    def test_query_accepts_one_mode_only(self):
        base = {"dataset_id": "00000000-0000-0000-0000-000000000001", "field": "x"}
        HistogramQuery.model_validate({**base, "bins": 5})
        HistogramQuery.model_validate({**base, "bin_width": 0.5})
        for bad in ({"bins": 5, "bin_width": 1}, {"bins": 4}, {"bins": 61}, {"bin_width": 0}):
            with self.assertRaises(ValidationError):
                HistogramQuery.model_validate({**base, **bad})


class HistogramServiceTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "data.sqlite3"
        self.store = ProjectStore(self.database)
        self.service = ProjectService(self.store)
        self.project = self.store.create("Histogram")
        self.dataset = self.service.datasets.add(
            self.project.id,
            "test.csv",
            b"id,x,k,c\n1,1,5,a\n2,2,5,b\n3,-99,5,c\n4,,5,d\n5,4,5,e\n6,10,5,f\n",
        )

        self.metadata = MetadataStore(self.database, self.service.datasets)
        document = self.metadata.get(self.dataset.id).document.model_dump()
        for column in document["columns"]:
            if column["name"] == "x":
                column["missing_codes"] = ["-99"]
            if column["name"] == "id":
                column["role"] = "id"
        self.metadata.save(self.dataset.id, document)

    def histogram(self, **query):
        return self.service.histogram(self.project.id, {"dataset_id": self.dataset.id, **query})

    def test_missing_codes_counts_and_single_event(self):
        before = len(self.store.events(self.project.id))
        response = self.histogram(field="x", bins=5)
        self.assertIsInstance(response, Success, response)
        data = response.data
        self.assertEqual((data.n, data.missing, data.row_count), (4, 2, 6))
        self.assertEqual(sum(data.counts), data.n)
        self.assertEqual((data.edges[0], data.edges[-1]), (1.0, 10.0))
        self.assertEqual(data.median, 3.0)
        self.assertTrue(data.small_sample)
        self.assertTrue(any("Малий набір" in w for w in data.warnings))
        self.assertIn('class="median-line"', data.svg)
        self.assertIn("[1; 2.80)", data.svg)
        self.assertIn("[8.20; 10]", data.svg)
        events = self.store.events(self.project.id)[before:]
        self.assertEqual([e.operation for e in events], ["chart.built"])
        payload = events[0].payload
        self.assertEqual(
            (payload["chart"], payload["fields"], payload["value_count"], payload["bin_count"]),
            ("histogram", ["x"], 4, 5),
        )
        self.assertEqual(payload["bin_counts"], data.counts)
        self.assertNotIn("values", payload)

    def test_options_change_presentation_only(self):
        baseline = self.histogram(field="x", bins=5).data
        for options in (
            preset("academic"),
            ChartOptions(histogram_median=False, axis_labels=False, grid_y=False),
            ChartOptions(histogram_opacity=0.3, accent="#3A9D8F", frame=True),
        ):
            data = self.histogram(field="x", bins=5, options=options.model_dump()).data
            self.assertEqual(
                (data.counts, data.edges, data.n), (baseline.counts, baseline.edges, baseline.n)
            )
        hidden = self.histogram(
            field="x", bins=5, options=ChartOptions(histogram_median=False).model_dump()
        ).data
        self.assertNotIn("median-line", hidden.svg)
        self.assertEqual(self.service.chart_config.get(), ChartConfiguration())

    def test_default_bins_from_configuration_and_width(self):
        self.assertEqual(len(self.histogram(field="x").data.counts), 20)
        self.service.save_chart_configuration(
            {"theme": "custom", "custom": ChartOptions(histogram_bins=8).model_dump()}
        )
        self.assertEqual(len(self.histogram(field="x").data.counts), 8)
        data = self.histogram(field="x", bin_width=4).data
        self.assertEqual((data.mode, data.edges), ("width", [1.0, 5.0, 9.0, 13.0]))

    def test_constant_excluded_and_failing_requests(self):
        constant = self.histogram(field="k").data
        self.assertTrue(constant.constant)
        self.assertEqual(constant.counts, [6])
        self.assertIn("однакові", constant.reason)
        for field in ("c", "unknown", "id"):
            self.assertIsInstance(self.histogram(field=field), Failure, field)
        before = len(self.store.events(self.project.id))
        failure = self.histogram(field="x", bin_width=0.001)
        self.assertIsInstance(failure, Failure)
        self.assertIn("інтервалів", failure.error.message)
        event = self.store.events(self.project.id)[before]
        self.assertEqual((event.operation, event.payload["status"]), ("chart.built", "error"))
        other = self.store.create("Other")
        response = self.service.histogram(other.id, {"dataset_id": self.dataset.id, "field": "x"})
        self.assertIsInstance(response, Failure)

    def test_melbourne_matches_stage_03_reference(self):
        dataset = self.service.datasets.add(
            self.project.id, "melb.csv", (ROOT / "samples" / "melb_data.csv").read_bytes()
        )
        for field in ("Price", "BuildingArea", "Rooms"):
            expected = REFERENCE["descriptive"][field]
            data = self.service.histogram(
                self.project.id, {"dataset_id": dataset.id, "field": field}
            ).data
            self.assertEqual(data.n, expected["count"], field)
            self.assertEqual(sum(data.counts), expected["count"], field)
            self.assertEqual(data.median, expected["median"], field)
            self.assertEqual((data.edges[0], data.edges[-1]), (expected["min"], expected["max"]))
            self.assertGreater(data.counts[-1], 0, field)  # the maximum is counted
            self.assertFalse(data.small_sample)
            json.dumps(data.model_dump(mode="json"), allow_nan=False)
        self.assertEqual(data.missing, 0)

    def test_api_route_and_validation(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        url = f"/api/projects/{self.project.id}/charts/histogram"
        query = {"dataset_id": self.dataset.id, "field": "x"}
        with TestClient(app) as client:
            response = client.post(url, json={**query, "bins": 6})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(len(response.json()["data"]["counts"]), 6)
            for bad in ({"bins": 3}, {"bins": 5, "bin_width": 1}, {"field": ""}, {"extra": 1}):
                self.assertEqual(client.post(url, json={**query, **bad}).status_code, 422, bad)
            self.assertEqual(client.post(url, json={**query, "field": "c"}).status_code, 422)


class ConfigurationCompatibilityTests(unittest.TestCase):
    def test_saved_050_custom_theme_without_new_fields_still_loads(self):
        old = {
            "theme": "custom",
            "custom": {
                k: v
                for k, v in ChartOptions(positive="#123456").model_dump().items()
                if not k.startswith("histogram")
                and k not in ("accent", "axis_labels", "grid_x", "grid_y")
            },
        }
        loaded = ChartConfiguration.model_validate_json(json.dumps(old))
        self.assertEqual(loaded.effective().positive, "#123456")
        self.assertEqual(loaded.effective().histogram_bins, 20)
        self.assertTrue(preset("academic").grid_x)
        self.assertFalse(preset("storytelling").grid_x)
        for bad in ({"histogram_bins": 4}, {"histogram_opacity": 0.1}, {"accent": "blue"}):
            with self.assertRaises(ValidationError):
                ChartOptions.model_validate(bad)


class AxisTests(unittest.TestCase):
    def test_count_axis_ticks_are_whole_numbers(self):
        for top in (1, 2, 3, 7, 13580):
            ticks = nice_ticks(0, top, 5, integer=True)
            self.assertTrue(all(float(t).is_integer() for t in ticks), (top, ticks))
            self.assertEqual(len(set(ticks)), len(ticks))
