"""Stage 02: upload, validation, safe storage, profile and preview of datasets."""

import hashlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_story.datasets import (
    MAX_BYTES,
    DatasetError,
    DatasetNotFound,
    DatasetStore,
    date_format_title,
    profile,
)
from data_story.projects import ProjectNotFound, ProjectStore

ROOT = Path(__file__).resolve().parents[1]
CSV = "площа;кімнат;район;ціна\n45,5;1;Центр;2100\n60;2;Поділ;\n81,2;3;Центр;3900\n".replace(
    ",", "."
).encode("utf-8")


def xlsx_bytes() -> bytes:
    buffer = io.BytesIO()
    pd.DataFrame({"x": [1, 2, 3]}).to_excel(buffer, sheet_name="Дані", index=False)
    return buffer.getvalue()


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        database = Path(self.directory.name) / "story.sqlite3"
        self.projects = ProjectStore(database)
        self.store = DatasetStore(database)
        self.project = self.projects.create("Ціни на житло")

    def stored_files(self) -> list[Path]:
        root = self.store.files_root
        return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []

    def test_csv_with_semicolon_is_profiled_stored_unchanged_and_logged(self):
        dataset = self.store.add(self.project.id, "ціни.csv", CSV)
        self.assertEqual((dataset.row_count, dataset.column_count), (3, 4))
        self.assertEqual(dataset.profile["delimiter"], ";")
        columns = {c["name"]: c for c in dataset.profile["columns"]}
        self.assertEqual(columns["площа"]["kind"], "number")
        self.assertEqual(columns["район"]["kind"], "text")
        self.assertEqual(columns["ціна"]["missing"], 1)
        self.assertEqual((columns["площа"]["min"], columns["площа"]["max"]), (45.5, 81.2))
        [path] = self.stored_files()
        self.assertEqual(path.read_bytes(), CSV)
        self.assertEqual(dataset.sha256, hashlib.sha256(CSV).hexdigest())
        event = self.projects.events(self.project.id)[-1]
        self.assertEqual(event.operation, "dataset.uploaded")
        self.assertEqual(event.payload["rows"], "3")
        preview = self.store.preview(dataset.id)
        self.assertEqual(preview.columns, ["площа", "кімнат", "район", "ціна"])
        self.assertEqual(preview.rows[1], ["60", "2", "Поділ", None])

    def test_windows_1251_csv_with_comma_is_read(self):
        content = "Регіон,Значення\nКиїв,1\nЛьвів,2\n".encode("cp1251")
        dataset = self.store.add(self.project.id, "регіони.csv", content)
        self.assertEqual(dataset.profile["encoding"], "cp1251")
        self.assertEqual(self.store.preview(dataset.id).rows[1], ["Львів", "2"])

    def test_only_csv_is_accepted_and_limit_is_5_mb(self):
        self.assertEqual(MAX_BYTES, 5 * 1024 * 1024)
        with self.assertRaisesRegex(DatasetError, "лише файли CSV"):
            self.store.add(self.project.id, "Книга.XLSX", xlsx_bytes())
        exact = b"a\n" + b"1\n" * ((MAX_BYTES - 2) // 2)
        self.assertEqual(len(exact), MAX_BYTES)
        self.assertEqual(self.store.add(self.project.id, "межа.csv", exact).size_bytes, MAX_BYTES)
        with self.assertRaisesRegex(DatasetError, "5 МБ"):
            self.store.add(self.project.id, "завеликий.csv", exact + b"1")

    def test_bad_files_are_rejected_without_rows_or_files(self):
        cases = [
            ("notes.txt", b"a,b\n1,2\n"),
            ("empty.csv", b""),
            ("blank.csv", b"   \n"),
            ("table.xlsx", b"PK\x03\x04 not really a workbook"),
            ("text.xlsx", b"a,b\n1,2\n"),
            ("big.csv", b"a\n" + b"1\n" * (MAX_BYTES // 2)),
            ("", b"a\n1\n"),
        ]
        for name, content in cases:
            with self.subTest(name=name), self.assertRaises(DatasetError):
                self.store.add(self.project.id, name, content)
        self.assertEqual(self.store.list_datasets(self.project.id), [])
        self.assertEqual(self.stored_files(), [])

    def test_unknown_project_and_unknown_dataset(self):
        with self.assertRaises(ProjectNotFound):
            self.store.add("00000000-0000-4000-8000-000000000000", "a.csv", CSV)
        with self.assertRaises(DatasetNotFound):
            self.store.get("00000000-0000-4000-8000-000000000000")
        self.assertEqual(self.stored_files(), [])

    def test_projects_are_isolated_and_same_file_twice_gives_two_datasets(self):
        other = self.projects.create("Інший")
        first = self.store.add(self.project.id, "a.csv", CSV)
        second = self.store.add(self.project.id, "a.csv", CSV)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(self.store.list_datasets(self.project.id)), 2)
        self.assertEqual(self.store.list_datasets(other.id), [])
        self.assertEqual(len(self.stored_files()), 2)

    def test_path_in_file_name_is_only_a_label(self):
        dataset = self.store.add(self.project.id, "..\\..\\evil.csv", CSV)
        self.assertEqual(dataset.name, "evil.csv")
        [path] = self.stored_files()
        self.assertTrue(path.resolve().is_relative_to(self.store.files_root))

    def test_journal_failure_rolls_back_dataset_and_removes_file(self):
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            with self.assertRaises(sqlite3.OperationalError):
                self.store.add(self.project.id, "a.csv", CSV)
        self.assertEqual(self.store.list_datasets(self.project.id), [])
        self.assertEqual(self.stored_files(), [])

    def test_demo_samples_upload_as_regression_ready_tables(self):
        name = "demo_apartments.csv"
        dataset = self.store.add(self.project.id, name, (ROOT / "samples" / name).read_bytes())
        self.assertEqual((dataset.row_count, dataset.column_count), (150, 7))
        kinds = {c["name"]: c["kind"] for c in dataset.profile["columns"]}
        self.assertEqual(kinds["ціна_тис_грн"], "number")
        self.assertEqual(kinds["район"], "text")


class PageTests(unittest.TestCase):
    """Server-side pages for the grid: only requested rows, filtered and sorted here."""

    @classmethod
    def setUpClass(cls):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        cls.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        database = Path(cls.directory.name) / "story.sqlite3"
        project = ProjectStore(database).create("Сторінки")
        cls.store = DatasetStore(database)
        content = (
            "дата;ціна;район;площа\n"
            "3/12/2016;100;Центр;45.5\n"
            "28/01/2016;;;\n"
            "15/06/2016;300;Поділ;60\n"
            "01/02/2017;200;центр;80\n"
        ).encode()
        cls.dataset = cls.store.add(project.id, "t.csv", content)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def page(self, **kwargs):
        return self.store.page(self.dataset.id, **kwargs)

    def test_rows_are_typed_and_sliced(self):
        page = self.page(start=1, limit=2)
        self.assertEqual((page.total, page.all_rows, len(page.rows)), (4, 4, 2))
        self.assertEqual(page.rows[0], {"c0": "2016-01-28", "c1": None, "c2": None, "c3": None})
        self.assertEqual(page.rows[1], {"c0": "2016-06-15", "c1": 300, "c2": "Поділ", "c3": 60})
        self.assertIsInstance(page.rows[1]["c1"], int)
        self.assertEqual(self.page(start=10, limit=5).rows, [])

    def test_sort_numbers_and_dates_with_missing_last(self):
        prices = [r["c1"] for r in self.page(sort=[{"colId": "c1", "sort": "desc"}]).rows]
        self.assertEqual(prices, [300, 200, 100, None])
        dates = [r["c0"] for r in self.page(sort=[{"colId": "c0", "sort": "asc"}]).rows]
        self.assertEqual(dates, ["2016-01-28", "2016-06-15", "2016-12-03", "2017-02-01"])

    def test_text_number_date_and_blank_filters(self):
        def total(model):
            return self.page(filters=model).total

        text = {"filterType": "text", "type": "contains", "filter": "ЦЕН"}
        self.assertEqual(total({"c2": text}), 2)
        self.assertEqual(total({"c2": {**text, "type": "notContains"}}), 2)
        number = {"filterType": "number", "type": "inRange", "filter": 100, "filterTo": 200}
        self.assertEqual(total({"c1": number}), 2)
        date = {"filterType": "date", "type": "greaterThan", "dateFrom": "2016-06-15 00:00:00"}
        self.assertEqual(total({"c0": date}), 2)
        self.assertEqual(total({"c1": {"filterType": "number", "type": "blank"}}), 1)
        either = {
            "filterType": "number",
            "operator": "OR",
            "conditions": [
                {"filterType": "number", "type": "lessThan", "filter": 150},
                {"filterType": "number", "type": "greaterThan", "filter": 250},
            ],
        }
        self.assertEqual(total({"c1": either}), 2)
        self.assertEqual(total({"c1": number, "c2": text}), 2)

    def test_bad_requests_are_refused(self):
        for kwargs in (
            {"limit": 0},
            {"limit": 501},
            {"start": -1},
            {"sort": [{"colId": "c9", "sort": "asc"}]},
            {"filters": {"c1": {"filterType": "text", "type": "contains", "filter": "x"}}},
            {"filters": {"c1": {"filterType": "number", "type": "like", "filter": 1}}},
            {"filters": {"c1": {"filterType": "number", "type": "equals", "filter": "x"}}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(DatasetError):
                self.page(**kwargs)

    def test_grid_options_use_server_pages(self):
        from data_story.data_grid import grid_options

        options = grid_options("abc", [("d", "date"), ("x", "number"), ("t", "text")])
        self.assertEqual(options["rowModelType"], "infinite")
        self.assertTrue(options["pagination"])
        self.assertNotIn("rowData", options)
        self.assertIn('"/api/datasets/abc/rows"', options[":datasource"])
        date, number, text = options["columnDefs"]
        self.assertEqual(date["filter"], "agDateColumnFilter")
        self.assertEqual(
            (number["filter"], number["type"]), ("agNumberColumnFilter", "rightAligned")
        )
        self.assertEqual(text["filter"], "agTextColumnFilter")
        self.assertIn("replace('.', ',')", number[":valueFormatter"])
        self.assertEqual(options["localeText"]["page"], "Сторінка")


def column(values: list, name: str = "дата") -> dict:
    [item] = profile(pd.DataFrame({name: pd.Series(values, dtype="str")}))
    return item


class DateRecognitionTests(unittest.TestCase):
    """Dates in CSV are recognised by explicit formats, without AI."""

    def test_common_formats_become_dates_with_range(self):
        cases = [
            (["2024-03-01", "2023-12-31"], "%Y-%m-%d", "2023-12-31", "2024-03-01"),
            (["01.03.2024", "31.12.2023"], "%d.%m.%Y", "2023-12-31", "2024-03-01"),
            (["3/12/2016", "28/01/2016"], "%d/%m/%Y", "2016-01-28", "2016-12-03"),
            (["12/31/2023", "3/1/2024"], "%m/%d/%Y", "2023-12-31", "2024-03-01"),
            (["2024-03-01 10:15", "2024-03-01 09:00"], "%Y-%m-%d %H:%M", None, None),
        ]
        for values, fmt, first, last in cases:
            with self.subTest(fmt=fmt):
                item = column(values)
                self.assertEqual((item["kind"], item["date_format"]), ("date", fmt))
                if first:
                    self.assertEqual((item["first"], item["last"]), (first, last))
        timed = column(["2024-03-01 10:15", "2024-03-01 09:00"])
        self.assertEqual((timed["first"], timed["last"]), ("2024-03-01T09:00", "2024-03-01T10:15"))

    def test_month_first_is_chosen_only_when_day_first_is_impossible(self):
        self.assertEqual(column(["12/31/2023", "3/1/2024"])["date_format"], "%m/%d/%Y")
        self.assertEqual(column(["12/31/2023", "3/1/2024"])["first"], "2023-12-31")
        ambiguous = column(["03/04/2020", "05/06/2020"])
        self.assertEqual(ambiguous["date_format"], "%d/%m/%Y")
        self.assertTrue(ambiguous["date_ambiguous"])
        self.assertNotIn("date_ambiguous", column(["28/01/2016"]))

    def test_missing_values_are_counted_and_do_not_block_recognition(self):
        item = column(["2024-01-02", None, " ", "2024-01-05"])
        self.assertEqual((item["kind"], item["missing"]), ("date", 1))

    def test_mixed_or_non_date_text_stays_text(self):
        for values in (
            ["2024-01-02", "02.01.2024"],
            ["2024-01-02", "не знаю"],
            ["Центр", "Поділ"],
            ["12", "13"],
            ["2024-13-01"],
        ):
            with self.subTest(values=values):
                item = column(values)
                self.assertEqual(item["kind"], "text")
                self.assertNotIn("date_format", item)

    def test_numbers_are_never_turned_into_dates(self):
        [item] = profile(pd.DataFrame({"рік": [2016, 2017]}))
        self.assertEqual(item["kind"], "number")

    def test_format_title_for_interface(self):
        self.assertEqual(date_format_title("%d/%m/%Y"), "ДД/ММ/РРРР")
        self.assertEqual(date_format_title("%Y-%m-%d %H:%M"), "РРРР-ММ-ДД гг:хх")

    def test_melbourne_demo_sample_profile(self):
        content = (ROOT / "samples" / "melb_data.csv").read_bytes()
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            database = Path(directory) / "story.sqlite3"
            project = ProjectStore(database).create("Мельбурн")
            dataset = DatasetStore(database).add(project.id, "melb_data.csv", content)
        self.assertEqual((dataset.row_count, dataset.column_count), (13580, 21))
        columns = {c["name"]: c for c in dataset.profile["columns"]}
        self.assertEqual(columns["Date"]["kind"], "date")
        self.assertEqual(columns["Date"]["date_format"], "%d/%m/%Y")
        self.assertEqual(
            (columns["Date"]["first"], columns["Date"]["last"]), ("2016-01-28", "2017-09-23")
        )
        self.assertEqual(columns["Price"]["kind"], "number")
        self.assertEqual(columns["Type"]["kind"], "text")
        missing = {n: c["missing"] for n, c in columns.items() if c["missing"]}
        self.assertEqual(
            missing, {"Car": 62, "BuildingArea": 6450, "YearBuilt": 5375, "CouncilArea": 1369}
        )


if __name__ == "__main__":
    unittest.main()
