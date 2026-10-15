"""Author feedback 097: cited sources only, A4 preview HTML, rounded conclusion facts."""

import unittest

from data_story.assistant import cited
from data_story.conclusions import rounded
from data_story.reports import preview_html


class FeedbackTests(unittest.TestCase):
    def test_only_cited_sources_are_shown(self):
        sources = [
            {"section": "Теплова карта для багатьох змінних", "title": "Правила"},
            {"section": "Вибір графіка: Динаміка", "title": "Правила"},
            {"section": "Benefits of Cross-Validation", "title": "Intro to ML", "kind": "book"},
        ]
        text = "Подивіться теплову карту. [Джерело: Теплова карта для багатьох  змінних]"
        self.assertEqual([s["section"] for s in cited(text, sources)], [sources[0]["section"]])
        self.assertEqual(cited("Див. Intro to ML, с. 268.", sources), [sources[2]])
        self.assertEqual(cited("Без посилань.", sources), [])

    def test_a4_preview_escapes_html_keeps_math_and_only_own_figures(self):
        text = (
            "## Підсумок\n\nТекст $a_0$ <script>x</script>.\n\n$$\n\\hat{y}=a_0\n$$\n\n"
            "| a | b |\n| --- | --- |\n| 1 | 2 |\n\n![Г](actual.svg)\n\n![x](../secret.svg)\n"
        )
        html = preview_html(text, lambda name: "/api/f/" + name, {"actual.svg"})
        self.assertIn("<h2>Підсумок</h2>", html)
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn('<div class="a4-math">\\[\\hat{y}=a_0\\]</div>', html)
        self.assertIn("\\(a_0\\)", html)
        self.assertIn("<table>", html)
        self.assertIn('src="/api/f/actual.svg"', html)
        self.assertNotIn("secret", html.split('alt="x"')[0].rsplit("<img", 1)[-1])

    def test_rounded_facts(self):
        self.assertEqual(
            rounded({"a": 0.37653106893338806, "b": [-0.37547714385812897, 0.0034567], "c": 5}),
            {"a": 0.38, "b": [-0.38, 0.0035], "c": 5},
        )
        self.assertIsNone(rounded(float("nan")))
        self.assertIs(rounded(True), True)


if __name__ == "__main__":
    unittest.main()
