"""Launcher checks: readiness identity, port collision and repository boundaries."""

import importlib.util
import json
import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from data_story.config import Settings

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("data_story_launcher", ROOT / "tools/start.py")
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_overview_shows_the_database_in_use(self):
        from data_story.screen import storage_label

        self.assertEqual(storage_label(ROOT / "runtime" / "data-story.sqlite3"),
                         "runtime/data-story.sqlite3")  # fmt: skip
        self.assertEqual(storage_label(ROOT / "tmp" / "demo" / "data.sqlite3"),
                         "tmp/demo/data.sqlite3")  # fmt: skip
        outside = Path(ROOT.anchor) / "elsewhere" / "data.sqlite3"
        self.assertEqual(storage_label(outside), str(outside.resolve()))

    def test_same_database_has_stable_identity_and_other_database_differs(self):
        first = Settings(database=ROOT / "tmp/a.sqlite3")
        self.assertEqual(first.instance, Settings(database=first.database).instance)
        self.assertNotEqual(first.instance, Settings(database=ROOT / "tmp/b.sqlite3").instance)

    def test_health_requires_matching_application_and_database(self):
        settings = Settings()
        for body, expected in [
            ({"application": "data-story", "instance": settings.instance}, True),
            ({"application": "other", "instance": settings.instance}, False),
            ({"application": "data-story", "instance": "other"}, False),
        ]:
            with self.subTest(body=body), patch.object(launcher.OPENER, "open") as mocked:
                import io

                mocked.return_value.__enter__.return_value = io.StringIO(json.dumps(body))
                self.assertEqual(launcher.running(settings), expected)

    def test_occupied_port_is_detected_without_terminating_owner(self):
        with socket.socket() as owner:
            owner.bind(("127.0.0.1", 0))
            owner.listen()
            port = owner.getsockname()[1]
            self.assertTrue(launcher.port_occupied(port))
            self.assertTrue(launcher.port_occupied(port))

    def test_outside_database_is_rejected_before_start(self):
        with patch.object(
            sys, "argv", ["start.py", "--database", str(ROOT.parent / "outside.sqlite3")]
        ):
            self.assertEqual(launcher.main(), 1)

    def test_repeat_launch_does_not_start_another_server(self):
        with (
            patch.object(sys, "argv", ["start.py", "--no-browser"]),
            patch.object(launcher, "running", return_value=True),
            patch.object(launcher, "launch_lock") as locked,
        ):
            self.assertEqual(launcher.main(), 0)
            locked.assert_not_called()
