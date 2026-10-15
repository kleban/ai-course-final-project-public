"""Stage 02c: an existing dataset with its descriptions and metadata copied into a project."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_story.dataset_copy import DatasetCopier
from data_story.datasets import DatasetError, DatasetNotFound, DatasetStore
from data_story.metadata import MetadataStore
from data_story.projects import ProjectNotFound, ProjectStore
from data_story.storage import _V1, _V2, _V3, _V4, SCHEMA_VERSION, connect, ensure_schema

ROOT = Path(__file__).resolve().parents[1]
CSV = "дата;ціна;район\n3/12/2016;100;Центр\n28/01/2016;;Поділ\n".encode()


class DatasetCopyTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "story.sqlite3"
        self.projects = ProjectStore(self.database)
        self.datasets = DatasetStore(self.database)
        self.metadata = MetadataStore(self.database, self.datasets)
        self.copier = DatasetCopier(self.datasets)
        self.source_project = self.projects.create("Джерело")
        self.target = self.projects.create("Нова аналітика")
        self.source = self.datasets.add(self.source_project.id, "ціни.csv", CSV)
        self.metadata.add_text(self.source.id, "ціна — у доларах")
        self.metadata.add_file(
            self.source.id, "опис.md", "# Поля\nрайон — адміністративний".encode()
        )
        document = self.metadata.get(self.source.id).document.model_dump()
        document["columns"][1]["role"] = "target"
        document["columns"][1]["label"] = "Ціна"
        self.metadata.save(self.source.id, document, confirm=True)

    def files(self, project_id: str) -> list[Path]:
        root = self.datasets.files_root / project_id
        return sorted(p for p in root.rglob("*") if p.is_file()) if root.exists() else []

    def test_copy_is_independent_and_keeps_description_and_metadata(self):
        copied = self.copier.copy(self.source.id, self.target.id)
        self.assertNotEqual(copied.id, self.source.id)
        self.assertEqual(copied.project_id, self.target.id)
        self.assertEqual(copied.copied_from, self.source.id)
        self.assertEqual((copied.row_count, copied.sha256), (2, self.source.sha256))
        stored = self.datasets.get(copied.id)
        self.assertEqual(stored.copied_from, self.source.id)
        self.assertEqual(stored.profile, self.source.profile)
        original = self.datasets.files_root / self.source.stored_path
        copy = self.datasets.files_root / copied.stored_path
        self.assertNotEqual(original.resolve(), copy.resolve())
        self.assertEqual(copy.read_bytes(), CSV)
        self.assertEqual(len(self.files(self.target.id)), 2)  # data + description file

        meta = self.metadata.get(copied.id)
        self.assertTrue(meta.saved and meta.confirmed_at)
        self.assertEqual(meta.document.columns[1].role, "target")
        sources = self.metadata.sources(copied.id)
        self.assertEqual([s.name for s in sources], ["Вставлений текст", "опис.md"])
        self.assertTrue(
            all(s.stored_path is None or self.target.id in s.stored_path for s in sources)
        )

        # edits in the copy never reach the source
        document = meta.document.model_dump()
        document["columns"][1]["label"] = "Ціна продажу"
        self.metadata.save(copied.id, document)
        self.assertEqual(self.metadata.get(self.source.id).document.columns[1].label, "Ціна")
        self.assertTrue(self.metadata.get(self.source.id).confirmed_at)
        self.assertEqual(len(self.metadata.sources(self.source.id)), 2)

        events = self.projects.events(self.target.id)
        [event] = [e for e in events if e.operation == "dataset.copied"]
        self.assertEqual(event.payload["source_project"], "Джерело")
        self.assertEqual(event.payload["metadata"], "підтверджено")
        self.assertEqual(event.payload["descriptions"], "2")
        self.assertEqual(self.datasets.page(copied.id).total, 2)
        self.assertEqual(
            {r["object_id"] for r in self.datasets.analytics.catalog(self.target.id)}, {copied.id}
        )

    def test_copy_without_saved_metadata_stays_without_metadata(self):
        plain = self.datasets.add(self.source_project.id, "інший.csv", CSV)
        copied = self.copier.copy(plain.id, self.target.id)
        self.assertFalse(self.metadata.get(copied.id).saved)
        self.assertEqual(self.metadata.sources(copied.id), [])

    def test_candidates_list_other_projects_with_state(self):
        self.assertEqual(self.copier.candidates(self.source_project.id), [])
        [candidate] = self.copier.candidates(self.target.id)
        self.assertEqual(candidate.dataset.id, self.source.id)
        self.assertEqual((candidate.project_name, candidate.metadata), ("Джерело", "confirmed"))
        self.assertFalse(candidate.already_here)
        self.copier.copy(self.source.id, self.target.id)
        self.assertTrue(self.copier.candidates(self.target.id)[0].already_here)
        # the copy is now a candidate for the source project, without saved-state surprises
        [back] = self.copier.candidates(self.source_project.id)
        self.assertEqual(back.project_name, "Нова аналітика")

    def test_refusals_change_nothing(self):
        with self.assertRaises(DatasetError):
            self.copier.copy(self.source.id, self.source_project.id)
        with self.assertRaises(ProjectNotFound):
            self.copier.copy(self.source.id, "00000000-0000-4000-8000-000000000000")
        with self.assertRaises(DatasetNotFound):
            self.copier.copy("00000000-0000-4000-8000-000000000000", self.target.id)
        self.assertEqual(self.datasets.list_datasets(self.target.id), [])
        self.assertEqual(self.files(self.target.id), [])

    def test_storage_failure_rolls_back_rows_and_files(self):
        with patch.object(ProjectStore, "_event", side_effect=sqlite3.OperationalError("test")):
            with self.assertRaises(sqlite3.OperationalError):
                self.copier.copy(self.source.id, self.target.id)
        self.assertEqual(self.datasets.list_datasets(self.target.id), [])
        self.assertEqual(self.files(self.target.id), [])
        self.assertEqual(self.datasets.analytics.catalog(self.target.id), [])


class MigrationTests(unittest.TestCase):
    def test_v4_database_gets_copied_from_column(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            database = Path(directory) / "v4.sqlite3"
            with connect(database) as connection:
                connection.executescript(
                    f"BEGIN; {_V1 + _V2 + _V3 + _V4} PRAGMA user_version = 4; COMMIT;"
                )
                connection.execute(
                    "INSERT INTO projects VALUES ('p', 'Старий', '', '2026-01-01', 'regression')"
                )
                connection.execute(
                    "INSERT INTO datasets VALUES "
                    "('d', 'p', 'a.csv', 'csv', 'p/a.csv', 1, 'x', 1, 1, '{}', '2026-01-01')"
                )
            ensure_schema(database)
            with connect(database) as connection:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                row = connection.execute("SELECT * FROM datasets").fetchone()
            self.assertEqual(version, SCHEMA_VERSION)
            self.assertEqual((row["name"], row["copied_from"]), ("a.csv", None))
            self.assertEqual(DatasetStore(database).get("d").copied_from, None)


if __name__ == "__main__":
    unittest.main()
