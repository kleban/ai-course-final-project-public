"""Explicit preparation choices, exact rows, source preservation and EDA parity."""

import math
import unittest

import pandas as pd
from pydantic import ValidationError

from data_story.datasets import profile
from data_story.metadata import Metadata, skeleton
from data_story.preparation import MissingRule, PreparationError, preview_missing
from data_story.statistics import calculate


def snapshot(frame):
    described = {"columns": profile(frame)}
    return described, Metadata("test", skeleton(frame, described), False, None, None)


class PreparationTests(unittest.TestCase):
    def test_boolean_and_large_integer_fill_preserve_types(self):
        frame = pd.DataFrame(
            {
                "x": pd.Series([2**60 + 1, None], dtype="Int64"),
                "b": pd.Series([True, None], dtype="boolean"),
            }
        )
        _, meta = snapshot(frame)
        result = preview_missing(
            frame,
            meta,
            [
                MissingRule(field="x", action="constant", value=2.0),
                MissingRule(field="b", action="constant", value=False),
            ],
        )
        self.assertEqual(result.frame["x"].tolist(), [2**60 + 1, 2])
        self.assertEqual(result.frame["b"].tolist(), [True, False])
        self.assertEqual(result.frame["x"].dtype, frame["x"].dtype)

    def test_no_choices_no_changes_and_independent_copy(self):
        frame = pd.DataFrame({"x": [1, None], "s": ["A", None]})
        _, meta = snapshot(frame)
        result = preview_missing(frame, meta, [])
        pd.testing.assert_frame_equal(result.frame, frame)
        result.frame.iloc[0, 0] = 99
        self.assertEqual(frame.iloc[0, 0], 1)
        self.assertEqual(result.retained_positions, [0, 1])
        self.assertTrue(result.report["warnings"])

    def test_union_of_drop_masks_and_rule_order(self):
        frame = pd.DataFrame({"x": [None, 1, None, 3], "s": [None, None, "A", "B"]})
        _, meta = snapshot(frame)
        rules = [MissingRule(field="x", action="drop"), MissingRule(field="s", action="drop")]
        result = preview_missing(frame, meta, rules)
        self.assertEqual(result.retained_positions, [3])
        self.assertEqual(result.report["rows_removed"], 3)
        pd.testing.assert_frame_equal(result.frame, preview_missing(frame, meta, rules[::-1]).frame)

    def test_fill_after_drop_and_unselected_fields_untouched(self):
        frame = pd.DataFrame({"x": [None, 1, 2], "s": [None, None, "B"], "z": [None] * 3})
        before = frame.copy(deep=True)
        _, meta = snapshot(frame)
        rules = [
            MissingRule(field="s", action="constant", value="Unknown"),
            MissingRule(field="x", action="drop"),
        ]
        result = preview_missing(frame, meta, rules)
        self.assertEqual(result.frame["s"].tolist(), ["Unknown", "B"])
        self.assertEqual(result.report["changes"][0]["filled"], 1)
        self.assertTrue(result.frame["z"].isna().all())
        pd.testing.assert_frame_equal(frame, before)

    def test_explicit_codes_match_statistics_and_infinity_is_not_missing(self):
        frame = pd.DataFrame({"x": [0, -99, None, math.inf], "s": [" NA ", "na", None, "A"]})
        p, meta = snapshot(frame)
        meta.document.columns[0].missing_codes = ["-99"]
        meta.document.columns[1].missing_codes = ["NA"]
        result = preview_missing(frame, meta, [MissingRule(field="x", action="drop")])
        stats = calculate(frame, p, meta, "missing")
        self.assertEqual(result.report["rows_removed"], stats["columns"][0]["missing"])
        self.assertEqual(result.retained_positions, [0, 3])
        self.assertTrue(math.isinf(result.frame.iloc[1]["x"]))
        text = preview_missing(frame, meta, [MissingRule(field="s", action="drop")])
        self.assertEqual(text.retained_positions, [1, 3])

    def test_invalid_codes_and_bad_fill_values_rejected_without_mutation(self):
        frame = pd.DataFrame({"x": [1, None]})
        _, meta = snapshot(frame)
        before = frame.copy(deep=True)
        for value in (math.inf, math.nan, "1", True):
            with self.assertRaises(PreparationError):
                preview_missing(
                    frame, meta, [MissingRule(field="x", action="constant", value=value)]
                )
        meta.document.columns[0].missing_codes = ["bad"]
        with self.assertRaises(PreparationError):
            preview_missing(frame, meta, [MissingRule(field="x", action="drop")])
        pd.testing.assert_frame_equal(frame, before)

    def test_target_fill_prohibited_but_explicit_drop_allowed(self):
        frame = pd.DataFrame({"y": [1, None]})
        _, meta = snapshot(frame)
        meta.document.columns[0].role = "target"
        with self.assertRaises(PreparationError):
            preview_missing(frame, meta, [MissingRule(field="y", action="constant", value=0.0)])
        self.assertEqual(
            preview_missing(
                frame, meta, [MissingRule(field="y", action="drop")]
            ).retained_positions,
            [0],
        )

    def test_empty_result_duplicate_unknown_excluded_and_column_mismatch(self):
        frame = pd.DataFrame({"x": [None]})
        _, meta = snapshot(frame)
        rule = MissingRule(field="x", action="drop")
        for rules in ([rule], [rule, rule], [MissingRule(field="absent", action="drop")]):
            with self.assertRaises(PreparationError):
                preview_missing(frame, meta, rules)
        meta.document.columns[0].role = "ignore"
        with self.assertRaises(PreparationError):
            preview_missing(frame, meta, [rule])
        with self.assertRaises(PreparationError):
            preview_missing(frame.rename(columns={"x": "z"}), meta, [])

    def test_duplicate_index_positions_and_nullable_fractional_constant(self):
        frame = pd.DataFrame({"x": pd.Series([1, None, -99], dtype="Int64")})
        frame.index = [7, 7, 9]
        _, meta = snapshot(frame)
        meta.document.columns[0].missing_codes = ["-99"]
        result = preview_missing(
            frame, meta, [MissingRule(field="x", action="constant", value=2.5)]
        )
        self.assertEqual(result.frame["x"].tolist(), [1, 2.5, 2.5])
        self.assertEqual(result.frame.index.tolist(), [7, 7, 9])
        self.assertEqual(result.retained_positions, [0, 1, 2])

    def test_fill_cannot_be_missing_code_and_numeric_category_keeps_numbers(self):
        frame = pd.DataFrame({"x": [1, -99, None]})
        _, meta = snapshot(frame)
        meta.document.columns[0].kind = "category"
        meta.document.columns[0].missing_codes = ["-99"]
        with self.assertRaises(PreparationError):
            preview_missing(frame, meta, [MissingRule(field="x", action="constant", value=-99.0)])
        with self.assertRaises(PreparationError):
            preview_missing(frame, meta, [MissingRule(field="x", action="constant", value="New")])
        result = preview_missing(
            frame, meta, [MissingRule(field="x", action="constant", value=2.0)]
        )
        self.assertEqual(result.frame["x"].tolist(), [1, 2, 2])

    def test_rule_contract_requires_deliberate_action_and_value(self):
        for raw in (
            {"field": "x"},
            {"field": "x", "action": "constant"},
            {"field": "x", "action": "drop", "value": 1.0},
            {"field": "x", "action": "automatic"},
            {"field": "x", "action": "drop", "unknown": True},
        ):
            with self.assertRaises(ValidationError):
                MissingRule.model_validate(raw)

    def test_melbourne_drop_known_missing_rows(self):
        from pathlib import Path

        from data_story.datasets import read_table

        path = Path(__file__).resolve().parents[1] / "samples" / "melb_data.csv"
        frame, _ = read_table(path.read_bytes(), "csv")
        p, meta = snapshot(frame)
        result = preview_missing(frame, meta, [MissingRule(field="BuildingArea", action="drop")])
        self.assertEqual(result.report["rows_removed"], 6450)
        self.assertEqual(result.report["rows_after"], 7130)
        self.assertEqual(
            calculate(frame, p, meta, "missing")["columns"][frame.columns.get_loc("BuildingArea")][
                "missing"
            ],
            6450,
        )
        self.assertEqual(int(frame["BuildingArea"].isna().sum()), 6450)
