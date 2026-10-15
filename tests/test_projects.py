"""Acceptance checks using only disposable databases inside this repository."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from data_story.projects import ProjectNotFound, ProjectStore

ROOT = Path(__file__).resolve().parents[1]


class ProjectStoreTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "story.sqlite3"
        self.store = ProjectStore(self.database)

    def test_first_start_and_restart_preserve_unicode_projects(self):
        self.assertEqual(self.store.list_projects(), [])
        project = self.store.create("  Пропуски в даних  ", "  Перевірити якість  ")
        reopened = ProjectStore(self.database)
        self.assertEqual(reopened.get(project.id), project)
        self.assertEqual(project.name, "Пропуски в даних")
        self.assertEqual(project.purpose, "Перевірити якість")
        self.assertEqual(reopened.list_projects(), [project])
        self.assertEqual(reopened.events(project.id)[0].operation, "project.created")

    def test_same_name_does_not_merge_projects_or_their_history(self):
        first = self.store.create("Дослідження")
        second = self.store.create("Дослідження")
        self.assertNotEqual(first.id, second.id)
        self.store.open(first.id)
        self.assertEqual(len(self.store.events(first.id)), 2)
        self.assertEqual(len(self.store.events(second.id)), 1)
        for event in self.store.events(first.id):
            self.assertEqual(event.project_id, first.id)

    def test_invalid_metadata_does_not_create_partial_project(self):
        for name, purpose in [(" ", ""), ("x" * 121, ""), ("Дані", "x" * 2001), (None, "")]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    self.store.create(name, purpose)
        self.assertEqual(self.store.list_projects(), [])

    def test_unknown_project_cannot_be_opened(self):
        project = self.store.create("Дані")
        with self.assertRaises(ProjectNotFound):
            self.store.open("missing")
        self.assertEqual(len(self.store.events(project.id)), 1)

    def test_sql_text_is_stored_as_data(self):
        name = "Research'); DROP TABLE projects; --"
        project = self.store.create(name)
        self.assertEqual(self.store.get(project.id).name, name)
        self.assertEqual(len(self.store.list_projects()), 1)

    def test_trace_failure_rolls_back_project_creation(self):
        with patch.object(self.store, "_event", side_effect=sqlite3.OperationalError("test")):
            with self.assertRaises(sqlite3.OperationalError):
                self.store.create("Не повинно зберегтися")
        self.assertEqual(self.store.list_projects(), [])

    def test_future_schema_is_rejected_without_data_changes(self):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.execute("PRAGMA user_version = 99")
        with self.assertRaises(RuntimeError):
            ProjectStore(self.database)
        with closing(sqlite3.connect(self.database)) as connection, connection:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)

    def test_unrecognized_database_is_not_overwritten(self):
        other = self.database.parent / "other.sqlite3"
        with closing(sqlite3.connect(other)) as connection, connection:
            connection.execute("CREATE TABLE author_material (text TEXT)")
            connection.execute("INSERT INTO author_material VALUES ('original')")
        with self.assertRaises(RuntimeError):
            ProjectStore(other)
        with closing(sqlite3.connect(other)) as connection, connection:
            self.assertEqual(
                connection.execute("SELECT text FROM author_material").fetchone()[0], "original"
            )


if __name__ == "__main__":
    unittest.main()


class ProjectDescriptionTests(unittest.TestCase):
    """D-64: Markdown description is saved with an event and survives reopening."""

    def test_description_saved_validated_and_migrated(self):
        import tempfile

        from data_story.contracts import Failure, Success
        from data_story.projects import ProjectStore as Store
        from data_story.service import ProjectService

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "data.sqlite3"
            service = ProjectService(Store(database))
            project = service.create({"name": "Опис"}).data
            self.assertEqual(project.description, "")
            saved = service.save_description(
                str(project.id), {"text": "# Мета\r\n\nЦіна **житла**  "}
            )
            self.assertIsInstance(saved, Success)
            self.assertEqual(saved.data.description, "# Мета\n\nЦіна **житла**")
            reopened = ProjectService(Store(database)).get(str(project.id)).data
            self.assertEqual(reopened.description, "# Мета\n\nЦіна **житла**")
            events = service.events(str(project.id)).data
            self.assertEqual(events[-1].operation, "project.description_saved")
            self.assertEqual(
                events[-1].payload, {"length": len("# Мета\n\nЦіна **житла**"), "empty": False}
            )
            for bad in ({"text": "x" * 20001}, {"text": 5}, {}):
                self.assertIsInstance(service.save_description(str(project.id), bad), Failure)
            missing = "00000000-0000-0000-0000-000000000000"
            self.assertIsInstance(service.save_description(missing, {"text": "x"}), Failure)
