"""Scatter: every pairwise complete row is a point; r and n equal stage 03 Pearson."""

import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data_story.chart_configuration import ChartOptions
from data_story.contracts import Failure, ScatterQuery, Success
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]


class ScatterTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        database = Path(self.temporary.name) / "data.sqlite3"
        self.store = ProjectStore(database)
        self.service = ProjectService(self.store)
        self.project = self.store.create("Scatter")
        self.dataset = self.service.datasets.add(
            self.project.id, "t.csv", b"x,y,k,c\n1,2,5,a\n2,4,5,b\n3,,5,c\n4,9,5,d\n,1,5,e\n"
        )

    def scatter(self, **query):
        return self.service.scatter(self.project.id, {"dataset_id": self.dataset.id, **query})

    def test_pairs_match_pearson_and_single_event(self):
        before = len(self.store.events(self.project.id))
        data = self.scatter(x="x", y="y").data
        baseline = self.service.statistics_store.calculate(
            self.project.id, self.dataset.id, "correlations", ["x", "y"]
        )
        self.assertEqual(data.n, baseline["n"][0][1])
        self.assertEqual(data.n, 3)
        self.assertAlmostEqual(data.r, baseline["r"][0][1])
        self.assertEqual(data.svg.count("<circle"), 3)
        self.assertEqual((data.x_range, data.y_range), ([1.0, 4.0], [2.0, 9.0]))
        events = self.store.events(self.project.id)[before:]
        self.assertEqual([e.operation for e in events], ["chart.built", "statistics.calculated"])
        self.assertEqual(events[0].payload["chart"], "scatter")
        self.assertNotIn("pairs", events[0].payload)
        json.dumps(data.model_dump(mode="json"), allow_nan=False)

    def test_options_presentation_only_and_constant_axis(self):
        base = self.scatter(x="x", y="y").data
        styled = self.scatter(
            x="x",
            y="y",
            options=ChartOptions(
                scatter_size=6, scatter_opacity=0.2, scatter_outline=True
            ).model_dump(),
        ).data
        self.assertEqual((styled.n, styled.r), (base.n, base.r))
        self.assertIn('r="6"', styled.svg)
        self.assertIn('fill-opacity="0.2"', styled.svg)
        constant = self.scatter(x="x", y="k").data
        self.assertIsNone(constant.r)
        self.assertIn("нульову варіацію", constant.reason)
        self.assertEqual(constant.svg.count("<circle"), constant.n)

    def test_invalid_requests(self):
        for query in ({"x": "x", "y": "c"}, {"x": "x", "y": "missing"}):
            self.assertIsInstance(self.scatter(**query), Failure)
        with self.assertRaises(ValidationError):
            ScatterQuery.model_validate({"dataset_id": self.dataset.id, "x": "x", "y": "x"})
        for bad in ({"scatter_size": 0}, {"scatter_opacity": 0.01}):
            with self.assertRaises(ValidationError):
                ChartOptions.model_validate(bad)

    def test_melbourne_price_rooms(self):
        dataset = self.service.datasets.add(
            self.project.id, "melb.csv", (ROOT / "samples" / "melb_data.csv").read_bytes()
        )
        response = self.service.scatter(
            self.project.id, {"dataset_id": dataset.id, "x": "Rooms", "y": "Price"}
        )
        self.assertIsInstance(response, Success)
        self.assertEqual(response.data.n, 13580)
        self.assertAlmostEqual(response.data.r, 0.4966336761865301)
        self.assertEqual(response.data.svg.count("<circle"), 13580)
        self.assertEqual(response.data.y_range, [85000.0, 9000000.0])

    def test_api_route(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        url = f"/api/projects/{self.project.id}/charts/scatter"
        with TestClient(app) as client:
            ok = client.post(url, json={"dataset_id": self.dataset.id, "x": "x", "y": "y"})
            self.assertEqual(ok.status_code, 200, ok.text)
            same = client.post(url, json={"dataset_id": self.dataset.id, "x": "x", "y": "x"})
            self.assertEqual(same.status_code, 422)
