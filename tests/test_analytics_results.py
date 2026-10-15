"""Persistent computed results: exact reuse, metadata invalidation and numerical chart data."""

import unittest
from unittest.mock import patch

import test_statistics as fixtures

from data_story.contracts import Success
from data_story.datasets import DatasetStore
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.statistics import StatisticsError, StatisticsStore


class PersistentResultsTests(unittest.TestCase):
    setUp = fixtures.StatisticsStoreTests.setUp

    def test_restart_reuses_result_and_returns_independent_objects(self):
        expected = self.store.calculate(self.project.id, self.dataset.id, "descriptive")
        expected["columns"][0]["label"] = "mutated caller copy"
        restarted = StatisticsStore(DatasetStore(self.datasets.database))
        with patch("data_story.statistics.calculate", side_effect=AssertionError("recomputed")):
            result = restarted.calculate(self.project.id, self.dataset.id, "descriptive")
        self.assertNotEqual(result["columns"][0]["label"], "mutated caller copy")
        self.assertEqual(result["columns"][0]["statistics"]["mean"], 1)
        self.assertTrue(self.projects.events(self.project.id)[-1].payload["reused"])

    def test_metadata_roles_and_missing_codes_invalidate_cache(self):
        first = self.store.calculate(self.project.id, self.dataset.id, "descriptive")
        metadata = self.store.metadata.get(self.dataset.id).document
        metadata.columns[0].missing_codes = ["0"]
        metadata.columns[1].role = "ignore"
        self.store.metadata.save(self.dataset.id, metadata.model_dump(), confirm=True)
        second = self.store.calculate(self.project.id, self.dataset.id, "descriptive")
        self.assertEqual(second["columns"][0]["statistics"]["mean"], 1.5)
        self.assertIsNone(second["columns"][1]["statistics"])
        self.assertEqual(first["columns"][0]["statistics"]["mean"], 1)
        tables = [
            r for r in self.datasets.analytics.catalog(self.project.id) if r["kind"] == "statistics"
        ]
        self.assertEqual(len(tables), 2)

    def test_chart_style_reuses_bins_but_changed_bin_count_does_not(self):
        service = ProjectService(ProjectStore(self.datasets.database))
        query = {"dataset_id": self.dataset.id, "field": "x", "bins": 5}
        first = service.histogram(self.project.id, query)
        self.assertIsInstance(first, Success)
        with patch("data_story.service.histogram_bins", side_effect=AssertionError("rebinned")):
            second = service.histogram(self.project.id, {**query, "title": "New title"})
        self.assertIsInstance(second, Success)
        self.assertEqual(second.data.counts, first.data.counts)
        self.assertNotEqual(second.data.svg, first.data.svg)
        third = service.histogram(self.project.id, {**query, "bins": 6})
        self.assertIsInstance(third, Success)
        self.assertEqual(len(third.data.counts), 6)
        bins = [
            r
            for r in service.analytics.catalog(self.project.id)
            if r["kind"] == "chart_data" and r["object_id"].endswith(".bins")
        ]
        self.assertEqual(len(bins), 2)
        stored = service.analytics.read(
            self.project.id, "chart_data", bins[0]["object_id"], bins[0]["fingerprint"]
        )
        self.assertEqual(stored["count"].sum(), 3)

    def test_cache_never_hides_a_damaged_original(self):
        self.store.calculate(self.project.id, self.dataset.id, "missing")
        (self.datasets.files_root / self.dataset.stored_path).write_bytes(b"x,y\n99,99\n")
        with self.assertRaisesRegex(StatisticsError, "пошкоджений"):
            self.store.calculate(self.project.id, self.dataset.id, "missing")

    def test_correlations_pairs_and_histogram_values_use_distinct_keys(self):
        first = self.store.calculate(self.project.id, self.dataset.id, "correlations", ["x", "y"])
        second = self.store.calculate(
            self.project.id, self.dataset.id, "correlations", ["x", "y"], with_pairs=True
        )
        self.assertNotIn("pairs", first)
        self.assertEqual(second["pairs"], [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
        tables = [
            r
            for r in self.datasets.analytics.catalog(self.project.id)
            if r["object_id"].endswith(".pairs")
        ]
        self.assertEqual(len(tables), 1)
