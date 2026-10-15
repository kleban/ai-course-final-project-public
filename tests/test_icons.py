"""Icons: generated CSS matches the source dictionary; every icon used by the screen exists."""

import re
import unittest
from pathlib import Path

from data_story import icons

ROOT = Path(__file__).resolve().parents[1]


class IconTests(unittest.TestCase):
    def test_generated_css_is_up_to_date(self):
        self.assertEqual(icons.CSS_PATH.read_text(encoding="utf-8"), icons.css())

    def test_screen_uses_only_known_icons(self):
        source = (ROOT / "src/data_story/screen.py").read_text(encoding="utf-8")
        used = set(re.findall(r'icon\("([a-z_]+)"', source))
        used |= set(re.findall(r'ico="([a-z_]+)"', source))
        used |= set(re.findall(r'\("\d\d(?:–\d\d)?", "[^"]+", "([a-z_]+)"', source))
        used |= set(re.findall(r'"/[a-z]*", "[^"]+", "([a-z_]+)"\)', source))
        used |= set(re.findall(r', "([a-z_]+)", "(?:small|primary|ghost)"\)', source))
        self.assertTrue(used)
        self.assertEqual(used - set(icons.ICONS), set())


class IconGroupTests(unittest.TestCase):
    def test_every_icon_is_in_exactly_one_gallery_group(self):
        names = [name for group in icons.GROUPS.values() for name in group]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), set(icons.ICONS))
