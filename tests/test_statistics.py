"""Stage 03: independent numbers, metadata semantics, edge cases and provenance."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_story.dataset_copy import DatasetCopier
from data_story.datasets import DatasetStore, profile, read_table
from data_story.metadata import Metadata, MetadataStore, skeleton
from data_story.projects import ProjectStore
from data_story.statistics import StatisticsError, StatisticsStore, calculate

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "statistics-melbourne-reference.json"
sys.path.insert(0, str(ROOT / "tools"))
from statistics_reference import reference  # noqa: E402


def snapshot(frame):
    described = {"columns": profile(frame)}
    meta = Metadata("test", skeleton(frame, described), False, None, None)
    return described, meta


def by_name(result):
    return {c["name"]: c for c in result["columns"]}


class CalculationTests(unittest.TestCase):
    def test_counts_quartiles_small_samples_empty_and_constant(self):
        for values in ([], [2], [2, 4], [2, 2, 2]):
            frame = pd.DataFrame({"x": pd.Series(values, dtype=float)})
            p, meta = snapshot(frame)
            result = calculate(frame, p, meta, "descriptive")
            stats = result["columns"][0]["statistics"]
            self.assertEqual(stats["count"], len(values))
            self.assertEqual(stats["std"] is None, len(values) < 2)
            json.dumps(result, allow_nan=False)
            if len(values) == 2:
                self.assertEqual(stats["q1"], 2.5)
                self.assertEqual(stats["q3"], 3.5)
                self.assertAlmostEqual(stats["std"], math.sqrt(2))
            missing = calculate(frame, p, meta, "missing")
            self.assertEqual(missing["rows_with_missing_percent"], 0 if values else None)

    def test_codes_zero_infinities_and_invalid_rule(self):
        frame = pd.DataFrame(
            {"x": [0, 1, -99, None, math.inf, -math.inf], "s": [" NA ", "na", "A", None, "A", "B"]}
        )
        p, meta = snapshot(frame)
        meta.document.columns[0].missing_codes = ["-99", "-99.0"]
        meta.document.columns[1].missing_codes = ["NA"]
        result = calculate(frame, p, meta, "missing")
        x = by_name(result)["x"]
        self.assertEqual(
            (x["base_missing"], x["additional_missing"], x["invalid_infinite"]), (1, 1, 2)
        )
        self.assertEqual(result["rows_with_missing"], 3)
        self.assertEqual(
            by_name(calculate(frame, p, meta, "descriptive"))["x"]["statistics"]["mean"], 0.5
        )
        meta.document.columns[0].missing_codes = ["bad"]
        result = calculate(frame, p, meta, "missing")
        self.assertIsNone(result["rows_with_missing"])
        self.assertIsNone(by_name(result)["x"]["additional_missing"])
        self.assertIsNotNone(by_name(result)["x"]["error"])
        self.assertEqual(by_name(result)["s"]["missing"], 2)
        json.dumps(result, allow_nan=False)

    def test_category_ties_and_numeric_category(self):
        frame = pd.DataFrame({"x": [2, 1, 2, 1], "b": [True, False, True, False]})
        p, meta = snapshot(frame)
        meta.document.columns[0].kind = "category"
        stats = by_name(calculate(frame, p, meta, "descriptive"))["x"]["statistics"]
        self.assertEqual((stats["top"], stats["frequency"], stats["tied"]), ("1", 2, True))
        self.assertNotIn("mean", stats)
        self.assertEqual(calculate(frame, p, meta, "correlations")["fields"], [])

    def test_pairwise_n_diagonal_reasons_selection_and_roles(self):
        frame = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, None],
                "y": [2, None, 6, 8, 10],
                "c": [1] * 5,
                "z": [0, 1, None, None, None],
            }
        )
        p, meta = snapshot(frame)
        result = calculate(frame, p, meta, "correlations")
        self.assertEqual(result["n"][0], [4, 3, 4, 2])
        self.assertAlmostEqual(result["r"][0][1], 1)
        self.assertIsNone(result["r"][2][2])
        self.assertIsNotNone(result["reasons"][2][2])
        self.assertIsNone(result["r"][3][3])
        self.assertEqual(result["n"], [list(row) for row in zip(*result["n"])])
        meta.document.columns[0].role = "target"
        meta.document.columns[1].role = "ignore"
        meta.document.columns[2].role = "id"
        result = calculate(frame, p, meta, "correlations", ["z", "x"])
        self.assertEqual(result["fields"], ["z", "x"])
        self.assertIn("y", result["excluded"])
        with self.assertRaises(StatisticsError):
            calculate(frame, p, meta, "correlations", ["x", "x"])
        with self.assertRaises(StatisticsError):
            calculate(frame, p, meta, "correlations", ["y", "x"])
        json.dumps(result, allow_nan=False)

    def test_dates_time_ambiguity_and_type_mismatch(self):
        frame = pd.DataFrame(
            {
                "d": ["03/04/2020", "04/05/2020"],
                "t": ["2020-01-01 12:30", "2020-01-01 09:15"],
                "s": ["one", "two"],
            }
        )
        p, meta = snapshot(frame)
        meta.document.columns[2].kind = "number"
        result = by_name(calculate(frame, p, meta, "descriptive"))
        self.assertTrue(result["d"]["date_ambiguous"])
        self.assertEqual(result["d"]["statistics"]["first"], "2020-04-03")
        self.assertEqual(result["t"]["statistics"]["first"], "2020-01-01T09:15")
        self.assertIsNotNone(result["s"]["reason"])
        meta.document.columns[2].kind = "date"
        self.assertIsNotNone(by_name(calculate(frame, p, meta, "descriptive"))["s"]["reason"])

    def test_melbourne_against_independent_csv_control(self):
        path = ROOT / "samples" / "melb_data.csv"
        control = reference(path)
        self.assertEqual(
            control,
            json.loads(FIXTURE.read_text(encoding="utf-8")),
        )
        frame, _ = read_table(path.read_bytes(), "csv")
        p, meta = snapshot(frame)
        meta.document.columns[frame.columns.get_loc("Postcode")].kind = "category"
        meta.document.columns[frame.columns.get_loc("Price")].role = "target"
        meta.document.columns[frame.columns.get_loc("Lattitude")].role = "ignore"
        meta.document.columns[frame.columns.get_loc("Longtitude")].role = "id"
        result = by_name(calculate(frame, p, meta, "descriptive"))
        self.assertEqual((len(frame), len(frame.columns)), (13580, 21))
        for name, expected in control["descriptive"].items():
            for key, value in expected.items():
                self.assertTrue(
                    math.isclose(
                        result[name]["statistics"][key], value, rel_tol=1e-9, abs_tol=1e-9
                    ),
                    (name, key),
                )
        missing = calculate(frame, p, meta, "missing")
        for name, count in {
            "BuildingArea": 6450,
            "YearBuilt": 5375,
            "CouncilArea": 1369,
            "Car": 62,
        }.items():
            self.assertEqual(by_name(missing)[name]["missing"], count)
            self.assertAlmostEqual(by_name(missing)[name]["missing_percent"], 100 * count / 13580)
        self.assertEqual({c["name"]: c["missing"] for c in missing["columns"]}, control["missing"])
        self.assertEqual(missing["rows_with_missing"], control["rows_with_missing"])
        self.assertEqual(result["Date"]["statistics"]["first"], "2016-01-28")
        self.assertEqual(result["Date"]["statistics"]["last"], "2017-09-23")
        pair = calculate(frame, p, meta, "correlations", ["Price", "Rooms"])
        self.assertAlmostEqual(pair["r"][0][1], control["Price_Rooms"]["r"], delta=1e-9)
        self.assertEqual(pair["n"][0][1], control["Price_Rooms"]["n"])
        for excluded in ("Postcode", "Date", "Lattitude", "Longtitude"):
            self.assertIn(excluded, pair["excluded"])


class StatisticsStoreTests(unittest.TestCase):
    def setUp(self):
        root = ROOT / "tmp" / "tests"
        root.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(directory.cleanup)
        self.datasets = DatasetStore(Path(directory.name) / "test.sqlite3")
        self.projects = ProjectStore(self.datasets.database)
        self.project = self.projects.create("Статистика")
        self.dataset = self.datasets.add(self.project.id, "test.csv", b"x,y\n0,1\n1,2\n2,3\n")
        self.store = StatisticsStore(self.datasets)

    def test_fresh_metadata_restart_confirmation_and_single_event(self):
        result = self.store.calculate(self.project.id, self.dataset.id, "descriptive")
        self.assertEqual(result["metadata_state"], "none")
        self.assertEqual(by_name(result)["x"]["statistics"]["mean"], 1)
        meta = self.store.metadata.get(self.dataset.id).document
        meta.columns[0].missing_codes = ["0"]
        self.store.metadata.save(self.dataset.id, meta.model_dump(), confirm=True)
        reopened = StatisticsStore(DatasetStore(self.datasets.database))
        result = reopened.calculate(self.project.id, self.dataset.id, "descriptive")
        self.assertEqual(result["metadata_state"], "confirmed")
        self.assertEqual(result["warnings"], [])
        self.assertEqual(by_name(result)["x"]["statistics"]["mean"], 1.5)
        self.assertEqual(result["sha256"], self.dataset.sha256)
        events = [
            e
            for e in self.projects.events(self.project.id)
            if e.operation == "statistics.calculated"
        ]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[-1].payload["status"], "success")
        self.assertIsNotNone(events[-1].payload["metadata_updated_at"])
        self.assertGreaterEqual(events[-1].payload["duration_ms"], 0)
        self.store.metadata.save(self.dataset.id, meta.model_dump())
        self.assertEqual(
            self.store.calculate(self.project.id, self.dataset.id, "missing")["metadata_state"],
            "draft",
        )

    def test_isolation_copy_missing_and_changed_file_errors(self):
        other = self.projects.create("Інший")
        with self.assertRaises(StatisticsError):
            self.store.calculate(other.id, self.dataset.id, "missing")
        copied = DatasetCopier(self.datasets).copy(self.dataset.id, other.id)
        meta = MetadataStore(self.datasets.database).get(copied.id).document
        meta.columns[0].missing_codes = ["0"]
        self.store.metadata.save(copied.id, meta.model_dump())
        self.assertEqual(
            self.store.calculate(other.id, copied.id, "missing")["rows_with_missing"], 1
        )
        self.assertEqual(
            self.store.calculate(self.project.id, self.dataset.id, "missing")["rows_with_missing"],
            0,
        )
        path = self.datasets.files_root / copied.stored_path
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(StatisticsError, "пошкоджений"):
            self.store.calculate(other.id, copied.id, "missing")
        path.unlink()
        with self.assertRaisesRegex(StatisticsError, "відсутній"):
            self.store.calculate(other.id, copied.id, "missing")
        with self.assertRaises(StatisticsError):
            self.store.calculate(other.id, "unknown", "missing")
        events = [
            e for e in self.projects.events(other.id) if e.operation == "statistics.calculated"
        ]
        self.assertEqual(len(events), 5)
        self.assertEqual(
            [e.payload["status"] for e in events], ["error", "success", "error", "error", "error"]
        )
