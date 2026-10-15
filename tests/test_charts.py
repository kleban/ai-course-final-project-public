"""Chart profiles persist, presentation preserves statistics, and APIs enforce bounds."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data_story.chart_configuration import (
    ChartConfiguration,
    ChartConfigurationStore,
    ChartOptions,
    preset,
)
from data_story.charts import color, contrast_warnings
from data_story.contracts import Failure, Success
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.storage import _V1, _V2, _V3, _V4, _V5, SCHEMA_VERSION, connect
from data_story.web import install_error_handlers, router

ROOT = Path(__file__).resolve().parents[1]


class ChartTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "data.sqlite3"
        self.store = ProjectStore(self.database)
        self.service = ProjectService(self.store)
        self.project = self.store.create("Charts")
        self.dataset = self.service.datasets.add(
            self.project.id, "test.csv", b"x,y,c,z\n1,2,1,5\n2,4,1,\n3,6,1,\n4,,1,\n"
        )

    def test_configuration_persistence_presets_reset_and_validation(self):
        store = ChartConfigurationStore(self.database)
        self.assertEqual(store.get().theme, "storytelling")
        custom = ChartOptions(positive="#123456", numbers=False)
        store.save(ChartConfiguration(theme="custom", custom=custom))
        reopened = ChartConfigurationStore(self.database)
        self.assertEqual(reopened.get().effective(), custom)
        reopened.save(ChartConfiguration(theme="academic", custom=custom))
        self.assertFalse(reopened.get().effective().lower_triangle)
        self.assertEqual(reopened.get().custom, custom)
        self.assertEqual(preset("storytelling").positive, "#1B4F8A")
        self.assertEqual(reopened.reset(), ChartConfiguration())
        self.assertEqual(len(reopened.events()), 3)
        for raw in (
            {"theme": "custom"},
            {"theme": "x"},
            {"custom": {"font": "url(x)"}},
            {"custom": {"positive": "red"}},
            {"custom": {"threshold": 2}},
            {"custom": {"label_size": 3}},
            {"custom": {"frame": "false"}},
        ):
            with self.assertRaises(ValidationError):
                ChartConfiguration.model_validate(raw)

    def test_migration_from_v5_preserves_existing_rows(self):
        database = Path(self.temporary.name) / "v5.sqlite3"
        with sqlite3.connect(database) as connection:
            connection.executescript(f"{_V1 + _V2 + _V3 + _V4 + _V5} PRAGMA user_version=5;")
            connection.execute(
                "INSERT INTO projects VALUES ('old','Existing','purpose','date','regression')"
            )
        connection.close()
        ChartConfigurationStore(database)
        with connect(database) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
            )
            self.assertEqual(
                connection.execute("SELECT name FROM projects").fetchone()[0], "Existing"
            )

    def test_heatmap_preserves_matrix_and_safe_accessible_rendering(self):
        query = {"dataset_id": self.dataset.id, "fields": ["x", "y", "c", "z"]}
        baseline = self.service.statistics_store.calculate(
            self.project.id, self.dataset.id, "correlations", query["fields"]
        )
        events_before = len(self.store.events(self.project.id))
        for options in (
            preset("storytelling"),
            preset("academic"),
            ChartOptions(numbers=False, threshold_enabled=True, threshold=0.9),
        ):
            response = self.service.heatmap(
                self.project.id,
                {**query, "options": options.model_dump(), "title": '<script>alert("x")</script>'},
            )
            self.assertIsInstance(response, Success)
            data = response.data
            self.assertEqual(
                (data.r, data.n, data.reasons), (baseline["r"], baseline["n"], baseline["reasons"])
            )
            self.assertIn("&lt;script&gt;", data.svg)
            self.assertNotIn("<script>", data.svg)
            self.assertIn('tabindex="0"', data.svg)
            self.assertIn("нульову варіацію", data.svg)
            self.assertIn("n=3", data.svg)
            json.dumps(data.model_dump(mode="json"), allow_nan=False)
        events = self.store.events(self.project.id)[events_before:]
        self.assertEqual([e.operation for e in events], ["chart.built"] * 3)
        self.assertIn("options", events[-1].payload)
        self.assertEqual(self.service.chart_config.get(), ChartConfiguration())
        other = self.store.create("Other")
        self.assertIsInstance(self.service.heatmap(other.id, query), Failure)
        self.assertEqual(self.store.events(other.id)[-1].payload["status"], "error")

    def test_melbourne_reference_and_palette(self):
        dataset = self.service.datasets.add(
            self.project.id, "melb.csv", (ROOT / "samples" / "melb_data.csv").read_bytes()
        )
        response = self.service.heatmap(
            self.project.id, {"dataset_id": dataset.id, "fields": ["Price", "Rooms"]}
        )
        self.assertIsInstance(response, Success)
        self.assertAlmostEqual(response.data.r[0][1], 0.4966336761865301)
        self.assertEqual(response.data.n[0][1], 13580)
        options = ChartOptions()
        for value, expected in (
            (-1, options.negative),
            (0, options.neutral),
            (1, options.positive),
        ):
            self.assertEqual(color(value, options).lower(), expected.lower())
        self.assertTrue(contrast_warnings(ChartOptions(text="#FFFFFF")))

    def test_api_configuration_heatmap_bounds_and_errors(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            route = "/api/configuration/charts"
            self.assertEqual(client.get(route).json()["data"]["theme"], "storytelling")
            self.assertEqual(client.put(route, json={"theme": "academic"}).status_code, 200)
            url = f"/api/projects/{self.project.id}/charts/heatmap"
            query = {"dataset_id": self.dataset.id, "fields": ["x", "y"]}
            response = client.post(url, json=query)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(response.json()["data"]["options"]["lower_triangle"])
            for fields in (["x"], ["x"] * 13, ["x", "x"], ["x", "unknown"]):
                self.assertEqual(
                    client.post(url, json={**query, "fields": fields}).status_code, 422
                )
            self.assertEqual(
                client.put(route, json={"custom": {"positive": "bad"}}).status_code, 422
            )
            self.assertEqual(client.post(route + "/reset").json()["data"]["theme"], "storytelling")
            path = self.service.datasets.files_root / self.dataset.stored_path
            path.write_text("changed", encoding="utf-8")
            self.assertEqual(client.post(url, json=query).status_code, 422)
