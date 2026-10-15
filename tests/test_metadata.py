"""Stage 02b: data dictionary — skeleton from data, author edits, checks, raw descriptions."""

import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from data_story.datasets import DatasetNotFound, DatasetStore
from data_story.metadata import MetadataError, MetadataStore, issues
from data_story.projects import ProjectStore

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        database = Path(self.directory.name) / "story.sqlite3"
        self.projects = ProjectStore(database)
        self.datasets = DatasetStore(database)
        self.store = MetadataStore(database, self.datasets)
        self.project = self.projects.create("Ціни на житло")
        content = (ROOT / "samples" / "demo_apartments.csv").read_bytes()
        self.dataset = self.datasets.add(self.project.id, "demo.csv", content)

    def columns(self, metadata):
        return {c.name: c for c in metadata.document.columns}

    def test_skeleton_comes_from_data_and_is_not_saved_until_author_saves(self):
        metadata = self.store.get(self.dataset.id)
        self.assertFalse(metadata.saved)
        columns = self.columns(metadata)
        self.assertEqual(list(columns)[0], "площа_м2")
        self.assertEqual(columns["ціна_тис_грн"].kind, "number")
        self.assertEqual(columns["район"].kind, "category")
        self.assertEqual(
            sorted(c.value for c in columns["район"].categories),
            sorted(["Центр", "Поділ", "Оболонь", "Позняки", "Троєщина"]),
        )
        self.assertEqual(columns["рік_побудови"].observed.missing, 6)
        self.assertIsNotNone(columns["площа_м2"].observed.max)
        self.assertTrue(all(c.source == "data" for c in metadata.document.columns))

    def test_author_edit_is_saved_logged_and_marked_as_author(self):
        document = self.store.get(self.dataset.id).document.model_dump()
        document["dataset"]["description"] = "Синтетичні квартири"
        price = next(c for c in document["columns"] if c["name"] == "ціна_тис_грн")
        price.update(label="Ціна", unit="тис. грн", role="target", range_min=0)
        saved = self.store.save(self.dataset.id, document)
        reopened = MetadataStore(self.store.database).get(self.dataset.id)
        self.assertTrue(reopened.saved)
        self.assertIsNone(reopened.confirmed_at)
        columns = self.columns(reopened)
        self.assertEqual(
            (columns["ціна_тис_грн"].label, columns["ціна_тис_грн"].source), ("Ціна", "author")
        )
        self.assertEqual(columns["площа_м2"].source, "data")
        self.assertEqual(reopened.document.dataset.description, "Синтетичні квартири")
        self.assertEqual(saved.document, reopened.document)
        confirmed = self.store.save(self.dataset.id, document, confirm=True)
        self.assertIsNotNone(confirmed.confirmed_at)
        operations = [e.operation for e in self.projects.events(self.project.id)]
        self.assertEqual(operations[-2:], ["metadata.saved", "metadata.confirmed"])

    def test_observed_facts_cannot_be_overwritten_and_columns_must_match(self):
        document = self.store.get(self.dataset.id).document.model_dump()
        document["columns"][0]["observed"]["max"] = 1.0
        saved = self.store.save(self.dataset.id, document)
        self.assertGreater(saved.document.columns[0].observed.max, 1.0)
        broken = self.store.get(self.dataset.id).document.model_dump()
        broken["columns"] = broken["columns"][1:]
        with self.assertRaises(MetadataError):
            self.store.save(self.dataset.id, broken)
        bad_kind = self.store.get(self.dataset.id).document.model_dump()
        bad_kind["columns"][0]["kind"] = "colour"
        with self.assertRaises(MetadataError):
            self.store.save(self.dataset.id, bad_kind)

    def test_issues_find_range_category_role_and_missing_code_conflicts(self):
        document = self.store.get(self.dataset.id).document.model_copy(deep=True)
        columns = {c.name: c for c in document.columns}
        columns["площа_м2"].range_max = 50
        columns["район"].categories = columns["район"].categories[:-1] + [
            type(columns["район"].categories[0])(value="Інше")
        ]
        columns["район"].role = "target"
        columns["ціна_тис_грн"].role = "target"
        columns["район"].missing_codes = ["Центр"]
        messages = [(i.column, i.level) for i in issues(document)]
        self.assertIn(("", "warning"), messages)  # two targets
        self.assertIn(("площа_м2", "warning"), messages)
        self.assertIn(("район", "info"), messages)
        self.assertGreaterEqual(sum(1 for c, _ in messages if c == "район"), 3)

    def test_raw_text_and_files_are_kept_with_extracted_text(self):
        self.store.add_text(self.dataset.id, "  площа_м2 — загальна площа квартири  ")
        buffer = io.BytesIO()
        pd.DataFrame({"поле": ["район"], "опис": ["адміністративний район"]}).to_excel(
            buffer, index=False
        )
        self.store.add_file(self.dataset.id, "опис.xlsx", buffer.getvalue())
        self.store.add_file(self.dataset.id, "notes.md", "# Опис\nціна в тис. грн".encode("cp1251"))
        sources = self.store.sources(self.dataset.id)
        self.assertEqual([s.kind for s in sources], ["text", "file", "file"])
        self.assertEqual(sources[0].text, "площа_м2 — загальна площа квартири")
        self.assertIn("адміністративний район", sources[1].text)
        self.assertIn("тис. грн", sources[2].text)
        stored = self.datasets.files_root / sources[1].stored_path
        self.assertEqual(stored.read_bytes(), buffer.getvalue())
        self.assertEqual(
            self.projects.events(self.project.id)[-1].operation, "metadata.source_added"
        )

    def test_bad_descriptions_are_rejected_without_records(self):
        for name, content in [
            ("scan.pdf", b"%PDF"),
            ("empty.txt", b""),
            ("bad.json", b"{oops"),
            ("big.txt", b"a" * (2 * 1024 * 1024 + 1)),
            ("broken.xlsx", b"PK no"),
        ]:
            with self.subTest(name=name), self.assertRaises(MetadataError):
                self.store.add_file(self.dataset.id, name, content)
        for text in ("   ", None, "x" * 50_001):
            with self.subTest(text=str(text)[:5]), self.assertRaises(MetadataError):
                self.store.add_text(self.dataset.id, text)
        with self.assertRaises(DatasetNotFound):
            self.store.add_text("00000000-0000-4000-8000-000000000000", "текст")
        self.assertEqual(self.store.sources(self.dataset.id), [])

    def test_journal_failure_leaves_no_source_file(self):
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            with self.assertRaises(sqlite3.OperationalError):
                self.store.add_file(self.dataset.id, "a.txt", "опис".encode())
        self.assertEqual(self.store.sources(self.dataset.id), [])
        folder = (self.datasets.files_root / self.dataset.stored_path).parent / "metadata"
        self.assertEqual(list(folder.glob("*")) if folder.exists() else [], [])


if __name__ == "__main__":
    unittest.main()
