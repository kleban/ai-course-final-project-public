"""Split coverage, no leakage, encoding semantics and persisted model-ready matrices."""

import sqlite3
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data_story.contracts import Failure, Success
from data_story.datasets import profile
from data_story.metadata import Metadata, skeleton
from data_story.preparation import PreparationError
from data_story.projects import ProjectStore
from data_story.service import ProjectService
from data_story.training_data import TrainingChoices, prepare_training
from data_story.web import install_error_handlers, router


def snapshot(frame):
    return Metadata("test", skeleton(frame, {"columns": profile(frame)}), False, None, None)


class TrainingCoreTests(unittest.TestCase):
    def test_default_split_coverage_repeatability_and_changed_seed(self):
        frame = pd.DataFrame({"x": range(10), "y": range(10)})
        before = frame.copy(deep=True)
        metadata = snapshot(frame)
        choices = TrainingChoices(target="y", features=["x"])
        result = prepare_training(frame, metadata, choices)
        self.assertEqual((len(result.train_positions), len(result.test_positions)), (7, 3))
        self.assertFalse(set(result.train_positions) & set(result.test_positions))
        self.assertEqual(sorted(result.train_positions + result.test_positions), list(range(10)))
        self.assertEqual(
            prepare_training(frame, metadata, choices).train_positions, result.train_positions
        )
        self.assertNotEqual(
            prepare_training(
                frame, metadata, choices.model_copy(update={"seed": 43})
            ).train_positions,
            result.train_positions,
        )
        pd.testing.assert_frame_equal(frame, before)

    def test_mean_imputer_uses_train_only(self):
        frame = pd.DataFrame(
            {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], "y": range(10)}
        )
        choices = TrainingChoices(target="y", features=["x"], numeric_missing="mean")
        split = prepare_training(frame, snapshot(frame), choices)
        frame.loc[split.test_positions[0], "x"] = 10000
        frame.loc[split.test_positions[1], "x"] = np.nan
        result = prepare_training(frame, snapshot(frame), choices)
        expected = float(frame.iloc[result.train_positions]["x"].mean())
        self.assertEqual(result.report["transforms"]["x"]["fill"], expected)
        self.assertNotEqual(expected, float(frame["x"].mean()))
        self.assertIn(expected, result.x_test[:, 0])

    def test_category_vocabulary_train_only_baseline_and_unknown_test_warning(self):
        frame = pd.DataFrame({"x": range(10), "cat": ["A", "B"] * 5, "y": range(10)})
        metadata = snapshot(frame)
        metadata.document.columns[1].kind = "category"
        split = prepare_training(frame, metadata, TrainingChoices(target="y", features=["x"]))
        frame.loc[split.test_positions[0], "cat"] = "OnlyTest"
        result = prepare_training(
            frame, metadata, TrainingChoices(target="y", features=["cat"], encoding="onehot")
        )
        transform = result.report["transforms"]["cat"]
        self.assertEqual(transform["categories"], ["A", "B"])
        self.assertEqual(transform["baseline"], "A")
        self.assertEqual(transform["unknown_test_rows"], 1)
        self.assertEqual(result.report["feature_names"], ["cat = B"])  # baseline column dropped
        self.assertEqual(result.x_test.shape[1], 1)
        self.assertEqual(result.x_test[0, 0], 0)  # unknown → like the baseline
        self.assertTrue(any("базової категорії" in w for w in result.report["warnings"]))

    def test_single_category_feature_refused(self):
        frame = pd.DataFrame({"x": range(10), "cat": ["A"] * 10, "y": range(10)})
        metadata = snapshot(frame)
        metadata.document.columns[1].kind = "category"
        with self.assertRaises(PreparationError):
            prepare_training(
                frame, metadata, TrainingChoices(target="y", features=["cat"], encoding="onehot")
            )

    def test_missing_codes_median_and_categorical_mode(self):
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, -99.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "cat": ["A", "A", "?", "A", "B", "B", "A", "A", "B", "A"],
                "y": range(10),
            }
        )
        metadata = snapshot(frame)
        metadata.document.columns[0].missing_codes = ["-99"]
        metadata.document.columns[1].kind = "category"
        metadata.document.columns[1].missing_codes = ["?"]
        result = prepare_training(
            frame,
            metadata,
            TrainingChoices(
                target="y",
                features=["x", "cat"],
                numeric_missing="median",
                category_missing="most_frequent",
                encoding="onehot",
            ),
        )
        train_values = frame.iloc[result.train_positions]["x"].replace(-99, np.nan)
        self.assertEqual(result.report["transforms"]["x"]["fill"], float(train_values.median()))
        self.assertNotIn("?", result.report["transforms"]["cat"]["categories"])
        self.assertTrue(np.isfinite(result.x_train).all())

    def test_reject_missing_target_features_infinite_unselected_columns_safe(self):
        frame = pd.DataFrame({"x": range(10), "y": range(10), "unused": [None] * 10})
        choices = TrainingChoices(target="y", features=["x"])
        self.assertEqual(prepare_training(frame, snapshot(frame), choices).x_train.shape, (7, 1))
        for field, value in (("x", np.nan), ("y", np.nan), ("x", np.inf)):
            altered = frame.astype({"x": float, "y": float})
            altered.loc[0, field] = value
            with self.assertRaises(PreparationError):
                prepare_training(altered, snapshot(altered), choices)

    def test_roles_types_and_no_implicit_encoding(self):
        frame = pd.DataFrame({"cat": ["A", "B"] * 5, "y": range(10)})
        metadata = snapshot(frame)
        metadata.document.columns[0].kind = "category"
        with self.assertRaises(PreparationError):
            prepare_training(frame, metadata, TrainingChoices(target="y", features=["cat"]))
        metadata.document.columns[0].role = "id"
        with self.assertRaises(PreparationError):
            prepare_training(
                frame, metadata, TrainingChoices(target="y", features=["cat"], encoding="onehot")
            )
        with self.assertRaises(ValidationError):
            TrainingChoices(target="y", features=["y"])

    def test_all_missing_train_and_oversized_categories(self):
        frame = pd.DataFrame({"x": range(10), "y": range(10)})
        choices = TrainingChoices(target="y", features=["x"], numeric_missing="median")
        split = prepare_training(frame, snapshot(frame), choices)
        frame = frame.astype({"x": float})
        frame.loc[split.train_positions, "x"] = np.nan
        with self.assertRaises(PreparationError):
            prepare_training(frame, snapshot(frame), choices)
        large = pd.DataFrame({"cat": [f"c{i}" for i in range(400)], "y": range(400)})
        metadata = snapshot(large)
        metadata.document.columns[0].kind = "category"
        metadata.document.columns[0].role = "feature"
        with self.assertRaises(PreparationError):
            prepare_training(
                large, metadata, TrainingChoices(target="y", features=["cat"], encoding="onehot")
            )


class TrainingStoreTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def _dataset(self):
        return self.service.datasets.add(
            self.project.id,
            "train.csv",
            b"x,y\n0,1\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17\n9,19\n",
        )

    def test_persistence_arrays_hash_and_isolation(self):
        dataset = self._dataset()
        raw = {"dataset_id": dataset.id, "target": "y", "features": ["x"]}
        result = self.service.training_prepare(self.project.id, raw)
        self.assertIsInstance(result, Success)
        reopened = ProjectService(ProjectStore(self.database))
        stored = reopened.training_runs(self.project.id).data[0]
        matrices = reopened.training_store.matrices(self.project.id, stored["id"])
        self.assertEqual(matrices["x_train"].shape, (7, 1))
        np.testing.assert_array_equal(matrices["y_train"], 2 * matrices["x_train"][:, 0] + 1)
        other = self.store.create("Other")
        self.assertIsInstance(self.service.training_prepare(other.id, raw), Failure)
        self.assertEqual(self.service.training_runs(other.id).data, [])
        path = self.service.datasets.files_root / stored["document"]["matrix_path"]
        path.write_bytes(b"changed")
        with self.assertRaises(PreparationError):
            reopened.training_store.matrices(self.project.id, stored["id"])

    def test_atomic_rollback(self):
        dataset = self._dataset()
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            result = self.service.training_prepare(
                self.project.id, {"dataset_id": dataset.id, "target": "y", "features": ["x"]}
            )
        self.assertIsInstance(result, Failure)
        self.assertEqual(self.service.training_runs(self.project.id).data, [])
        self.assertEqual(
            list((self.service.datasets.files_root / self.project.id / "training").iterdir()), []
        )

    def test_http_prepare_and_list(self):
        dataset = self._dataset()
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        route = f"/api/projects/{self.project.id}/training"
        with TestClient(app) as client:
            response = client.post(
                route, json={"dataset_id": dataset.id, "target": "y", "features": ["x"]}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["document"]["report"]["train_count"], 7)
            self.assertEqual(len(client.get(route).json()["data"]), 1)
            self.assertEqual(
                client.post(
                    route, json={"dataset_id": dataset.id, "target": "y", "features": ["y"]}
                ).status_code,
                422,
            )
