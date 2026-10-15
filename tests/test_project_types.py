"""Project types, schema upgrade and editable instructions (stage 01, iteration 019)."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from data_story.configuration import InstructionStore
from data_story.icons import ICONS
from data_story.project_types import DEFAULT_TYPE, PROJECT_TYPES
from data_story.projects import ProjectStore
from data_story.storage import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]

V1_SCHEMA = """
    CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, purpose TEXT NOT NULL,
                           created_at TEXT NOT NULL);
    CREATE TABLE project_events (id TEXT PRIMARY KEY,
                                 project_id TEXT NOT NULL REFERENCES projects(id),
                                 operation TEXT NOT NULL, occurred_at TEXT NOT NULL,
                                 payload TEXT NOT NULL);
    CREATE INDEX project_events_project ON project_events(project_id, occurred_at);
    INSERT INTO projects VALUES ('11111111-1111-4111-8111-111111111111', 'Старий', '',
                                 '2026-10-10T10:00:00+00:00');
    PRAGMA user_version = 1;
"""


class ProjectTypeTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / "story.sqlite3"

    def test_registry_has_three_types_and_only_regression_is_enabled(self):
        self.assertEqual(list(PROJECT_TYPES), ["regression", "classification", "clustering"])
        self.assertEqual(DEFAULT_TYPE, "regression")
        self.assertEqual([t.key for t in PROJECT_TYPES.values() if t.enabled], ["regression"])
        for item in PROJECT_TYPES.values():
            with self.subTest(item=item.key):
                self.assertIn(item.icon, ICONS)
                self.assertGreater(len(item.default_instruction()), 200)

    def test_new_project_defaults_to_regression_and_disabled_types_are_refused(self):
        store = ProjectStore(self.database)
        project = store.create("Ціни на житло")
        self.assertEqual(project.project_type, "regression")
        self.assertEqual(store.events(project.id)[0].payload["project_type"], "regression")
        for kind in ("classification", "clustering", "unknown", None):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                store.create("Не має зберегтися", "", kind)
        self.assertEqual(len(store.list_projects()), 1)

    def test_version_1_database_is_upgraded_without_losing_projects(self):
        with closing(sqlite3.connect(self.database)) as connection, connection:
            connection.executescript(V1_SCHEMA)
        store = ProjectStore(self.database)
        [old] = store.list_projects()
        self.assertEqual((old.name, old.project_type), ("Старий", "regression"))
        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
            )
        ProjectStore(self.database)  # second start does not migrate again

    def test_instruction_default_edit_and_reset_are_logged(self):
        instructions = InstructionStore(self.database)
        default = instructions.get("regression")
        self.assertTrue(default.is_default)
        saved = instructions.save("regression", "  Спершу перевір пропуски.  ")
        self.assertEqual(saved.text, "Спершу перевір пропуски.")
        reopened = InstructionStore(self.database).get("regression")
        self.assertFalse(reopened.is_default)
        self.assertEqual(reopened.text, "Спершу перевір пропуски.")
        self.assertEqual(instructions.reset("regression").text, default.text)
        self.assertEqual(
            [e["operation"] for e in instructions.events()],
            ["instruction.saved", "instruction.reset"],
        )

    def test_disabled_type_instruction_can_be_prepared_but_empty_text_is_refused(self):
        instructions = InstructionStore(self.database)
        self.assertFalse(instructions.save("clustering", "Чернетка").is_default)
        for value in ("  ", None, "x" * 20001):
            with self.subTest(value=str(value)[:10]), self.assertRaises(ValueError):
                instructions.save("regression", value)
        with self.assertRaises(ValueError):
            instructions.get("unknown")


if __name__ == "__main__":
    unittest.main()
