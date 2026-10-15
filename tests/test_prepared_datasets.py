"""Prepared versions preserve values, provenance, metadata, atomicity and originals."""

import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd
import test_preparation_store as fixtures

from data_story.contracts import Failure, Success
from data_story.datasets import read_table
from data_story.prepared_datasets import csv_snapshot
from data_story.projects import ProjectStore


class PreparedDatasetTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def _plan(self, query=None):
        raw = query or self.query
        preview = self.service.preparation_preview(self.project.id, raw).data
        return self.service.preparation_save(
            self.project.id, {**raw, "preview_token": preview["preview_token"]}
        ).data["plan_id"]

    def test_applied_dataset_values_metadata_provenance_and_idempotence(self):
        plan = self._plan()
        before = (self.service.datasets.files_root / self.dataset.stored_path).read_bytes()
        result = self.service.preparation_apply(self.project.id, plan)
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data.row_count, 2)
        output = self.service.datasets.get(str(result.data.id))
        self.assertEqual(output.profile["prepared_from"], self.dataset.id)
        self.assertEqual(output.profile["preparation_plan_id"], plan)
        self.assertEqual(
            {r["object_id"] for r in self.service.analytics.catalog(self.project.id)},
            {self.dataset.id, output.id},
        )
        self.assertIsNone(self.service.metadata_store.get(output.id).confirmed_at)
        self.assertEqual(
            (self.service.datasets.files_root / self.dataset.stored_path).read_bytes(), before
        )
        self.assertEqual(
            self.service.preparation_apply(self.project.id, plan).data.id, result.data.id
        )
        self.assertEqual(len(self.service.datasets.list_datasets(self.project.id)), 2)
        self.assertEqual(
            sum(e.operation == "preparation.applied" for e in self.store.events(self.project.id)), 1
        )

    def test_stale_plan_and_other_project_rejected(self):
        plan = self._plan()
        other = self.store.create("Other")
        self.assertIsInstance(self.service.preparation_apply(other.id, plan), Failure)
        metadata = self.service.metadata_store.get(self.dataset.id).document.model_dump()
        metadata["columns"][0]["role"] = "ignore"
        self.service.metadata_store.save(self.dataset.id, metadata)
        self.assertIsInstance(self.service.preparation_apply(self.project.id, plan), Failure)
        self.assertEqual(len(self.service.datasets.list_datasets(self.project.id)), 1)

    def test_atomic_rollback_and_fresh_folder_cleanup(self):
        plan = self._plan()
        root = self.service.datasets.files_root / self.project.id / "datasets"
        before = set(root.iterdir())
        before_catalog = self.service.analytics.catalog(self.project.id)
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            self.assertIsInstance(self.service.preparation_apply(self.project.id, plan), Failure)
        self.assertEqual(set(root.iterdir()), before)
        self.assertEqual(len(self.service.datasets.list_datasets(self.project.id)), 1)
        self.assertEqual(self.service.analytics.catalog(self.project.id), before_catalog)

    def test_csv_preserves_na_text_empty_text_numeric_strings_bool_and_precision(self):
        frame = pd.DataFrame(
            {
                "s": ["NA", "", "001", None],
                "b": pd.Series([True, False, None, True], dtype="boolean"),
                "x": [1.2345678901234567, 0.1, None, 1.0],
            }
        )
        content, meta = csv_snapshot(frame)
        restored, _ = read_table(content, "csv", meta=meta)
        pd.testing.assert_frame_equal(frame, restored, check_dtype=False, check_exact=True)
