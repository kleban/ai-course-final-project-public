"""Reports: immutable facts, explicit AI, optimistic edits, safe renderer and scoped exports."""

import json
import sqlite3
import unittest
import zipfile
from unittest.mock import patch

import test_model as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.contracts import Failure, Success
from data_story.reports import (
    MATH_HEADER,
    NARRATIVE,
    REPORT_CSS,
    ReportError,
    compose,
    embed_svg_fonts,
    number,
    offline_html,
    p_value,
    unique_notes,
    validate_markdown,
)
from data_story.storage import SCHEMA_VERSION, connect
from data_story.web import install_error_handlers, router


class ReportTests(unittest.TestCase):
    setUp = fixtures.ModelStoreTests.setUp
    _run = fixtures.ModelStoreTests._run

    def create(self):
        run = self._run(rows=40)
        model = self.service.model_train(self.project.id, {"training_run_id": run}).data
        result = self.service.report_create(
            self.project.id, {"model_id": model["id"], "title": "Тестовий звіт"}
        )
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        return result.data

    def test_snapshot_has_coefficients_and_figures_and_no_automatic_ai(self):
        with patch.object(self.service.openai, "respond", side_effect=AssertionError("No auto AI")):
            report = self.create()
        self.assertEqual(len(report["files"]), 6)
        self.assertIn("Стандартна помилка", report["markdown"])
        self.assertIn("p-value", report["markdown"])
        self.assertEqual(report["snapshot"]["model"]["metrics"]["test"]["r2"], 1)
        self.assertEqual(len(self.service.reports.list(self.project.id)), 1)
        self.assertEqual(self.store.events(self.project.id)[-1].operation, "report.created")

    def test_save_stale_and_foreign_are_refused(self):
        report = self.create()
        other = self.store.create("Other")
        self.assertIsInstance(self.service.report_get(other.id, report["id"]), Failure)
        update = self.service.report_save(
            self.project.id,
            report["id"],
            {"revision": 1, "markdown": report["markdown"] + "\nВласне пояснення."},
        )
        self.assertIsInstance(update, Success)
        self.assertEqual(update.data["revision"], 2)
        self.assertIsInstance(
            self.service.report_save(
                self.project.id, report["id"], {"revision": 1, "markdown": "stale"}
            ),
            Failure,
        )
        self.assertTrue(
            self.service.report_get(self.project.id, report["id"])
            .data["markdown"]
            .endswith("Власне пояснення.")
        )

    def test_document_cannot_execute_code_or_include_arbitrary_files(self):
        for text in [
            "---\nexecute: true\n---",
            "```{python}\nprint(1)\n```",
            "{{< include secret.txt >}}",
            "<script>x</script>",
            "![x](https://example.com/a.png)",
            "![x](../../file)",
        ]:
            with self.subTest(text=text), self.assertRaises(ReportError):
                validate_markdown(text, set())
        validate_markdown("## Текст\n$y=x$\n![Графік](actual.svg)", {"actual.svg"})

    def test_explicit_ai_uses_configured_task_and_protects_fact_tables(self):
        report = self.create()
        narrative = {k: "Перевірене пояснення." for k in NARRATIVE}

        def respond(task, instructions, text, **kwargs):
            self.assertEqual(task, "report_draft")
            self.assertEqual(kwargs["project_id"], self.project.id)
            self.assertIn("coefficients", text)
            self.assertIsNone(kwargs["accept"](json.dumps(narrative)))
            self.assertIsNotNone(kwargs["accept"]("bad"))
            return {
                "text": json.dumps(narrative),
                "model": "gpt-6-luna",
                "usage_total": 10,
                "attempts": [],
            }

        with patch.object(self.service.openai, "respond", side_effect=respond) as call:
            result = self.service.report_generate(
                self.project.id, report["id"], {"revision": 1}, ai=True
            )
        self.assertIsInstance(result, Success)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(result.data["snapshot"], report["snapshot"])
        self.assertEqual(result.data["files"], report["files"])
        self.assertEqual(result.data["markdown"], compose(report["snapshot"], narrative))
        self.assertEqual(self.store.events(self.project.id)[-1].operation, "report.ai")

    def test_missing_quarto_and_damaged_graph_do_not_create_artifact(self):
        report = self.create()
        with patch("data_story.reports.version", return_value=None):
            result = self.service.report_generate(self.project.id, report["id"], {"revision": 1})
        self.assertIsInstance(result, Failure)
        self.assertEqual(self.service.reports.get(self.project.id, report["id"])["artifacts"], [])
        image = next(iter(report["files"]))
        (self.service.reports.folder(report["id"]) / image).write_text("corrupt")
        with patch("data_story.reports.version", return_value="test"):
            with self.assertRaises(ReportError):
                self.service.reports.render(self.project.id, report["id"], 1)

    def test_atomic_save_rollback_on_audit_error(self):
        report = self.create()
        with patch(
            "data_story.reports.ProjectStore._event", side_effect=sqlite3.OperationalError("audit")
        ):
            result = self.service.report_save(
                self.project.id, report["id"], {"revision": 1, "markdown": "changed"}
            )
        self.assertIsInstance(result, Failure)
        self.assertEqual(
            self.service.reports.get(self.project.id, report["id"])["markdown"], report["markdown"]
        )

    def test_render_api_keeps_versions_and_source_bundle_without_ai_or_reanalysis(self):
        report = self.create()

        class Process:
            returncode = 0
            stdout = "ok"
            stderr = ""

        def render(command, **kwargs):
            self.assertIn("--no-execute", command)
            directory = kwargs["cwd"]
            self.assertIn("enabled: false", (directory / "report.qmd").read_text(encoding="utf-8"))
            (directory / "report.html").write_text("<html>local</html>")
            return Process()

        with (
            patch("data_story.reports.version", return_value="test"),
            patch("data_story.reports.subprocess.run", side_effect=render),
            patch.object(self.service.openai, "respond", side_effect=AssertionError("No AI")),
            patch.object(self.service, "model_train", side_effect=AssertionError("No training")),
        ):
            first = self.service.report_generate(self.project.id, report["id"], {"revision": 1})
            second = self.service.report_generate(self.project.id, report["id"], {"revision": 1})
        self.assertIsInstance(first, Success)
        self.assertEqual(len(second.data["artifacts"]), 2)
        artifact = second.data["artifacts"][-1]
        bundle = self.service.reports.download(
            self.project.id, report["id"], artifact["id"], "source"
        )
        with zipfile.ZipFile(bundle) as package:
            self.assertIn("fonts/plex/ibm-plex-sans-regular.woff2", package.namelist())
            self.assertIn("fonts/plex/OFL.txt", package.namelist())
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router(self.service))
        base = f"/api/projects/{self.project.id}/reports/{report['id']}"
        with TestClient(app) as client:
            self.assertEqual(client.get(base).status_code, 200)
            self.assertEqual(
                client.get(base + f"/files/{artifact['id']}/html").text, "<html>local</html>"
            )
            self.assertEqual(client.get(base + f"/files/{artifact['id']}/source").status_code, 200)
            view = client.get(base + f"/files/{artifact['id']}/view")
            self.assertEqual(view.text, "<html>local</html>")
            self.assertNotIn("attachment", view.headers.get("content-disposition", ""))
            self.assertIn(
                "attachment",
                client.get(base + f"/files/{artifact['id']}/html").headers["content-disposition"],
            )
            self.assertEqual(
                client.put(base, json={"revision": 0, "markdown": "x"}).status_code, 422
            )
        path = self.service.reports.download(self.project.id, report["id"], artifact["id"], "html")
        path.write_text("modified")
        with self.assertRaises(ReportError):
            self.service.reports.download(self.project.id, report["id"], artifact["id"], "html")

    def test_render_uses_app_fonts_local_math_and_themed_charts(self):
        report = self.create()

        class Process:
            returncode = 0
            stdout = "ok"
            stderr = ""

        seen = {}

        def render(command, **kwargs):
            directory = kwargs["cwd"]
            seen["qmd"] = (directory / "report.qmd").read_text(encoding="utf-8")
            seen["fonts"] = sorted(p.name for p in (directory / "fonts").rglob("*.woff2"))
            seen["svg"] = (directory / next(iter(report["files"]))).read_text(encoding="utf-8")
            (directory / "report.html").write_text(
                '<html><script src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js">'
                "</script>ok</html>"
            )
            return Process()

        with (
            patch("data_story.reports.version", return_value="test"),
            patch("data_story.reports.subprocess.run", side_effect=render),
        ):
            result = self.service.report_generate(self.project.id, report["id"], {"revision": 1})
        self.assertIsInstance(result, Success)
        self.assertIn("html-math-method: plain", seen["qmd"])
        self.assertIn("toc-title: Зміст", seen["qmd"])
        self.assertIn("ibm-plex-sans-regular.woff2", seen["fonts"])
        self.assertIn("stix-two-text-700.woff2", seen["fonts"])
        self.assertIn("@font-face", seen["svg"])
        artifact = result.data["artifacts"][-1]
        html = self.service.reports.download(self.project.id, report["id"], artifact["id"], "html")
        self.assertNotIn("polyfill", html.read_text(encoding="utf-8"))
        self.assertIn("'STIX Two Text'", REPORT_CSS)
        self.assertIn("#174690", REPORT_CSS)
        self.assertIn('src="tex-svg.js"', MATH_HEADER)

    def test_offline_html_refuses_external_resources(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as folder:
            path = Path(folder) / "r.html"
            path.write_text(
                '<link rel="stylesheet" href="https://cdn.example/x.css">', encoding="utf-8"
            )
            with self.assertRaises(ReportError):
                offline_html(path)
            path.write_text('<a href="https://example.com">посилання</a>', encoding="utf-8")
            offline_html(path)

    def test_numbers_p_values_and_notes_are_readable(self):
        self.assertEqual(number(1075684.1), "1\u202f075\u202f684,1")
        self.assertEqual(number(1075684.126), "1\u202f075\u202f684,13")
        self.assertEqual(number(0.44219), "0,44")
        self.assertEqual(number(-1.5), "-1,5")
        self.assertEqual(number(0.0034567), "0,0035")
        self.assertEqual(number(None), "не визначено")
        self.assertEqual(p_value(0.0), "< 0,001")
        self.assertEqual(p_value(0.0004), "< 0,001")
        self.assertEqual(p_value(0.0067), "0,007")
        self.assertEqual(p_value(0.0123), "0,01")
        self.assertEqual(p_value(0.7517), "0,75")
        self.assertEqual(
            unique_notes(["Метадані не підтверджено", "Метадані не підтверджено."]),
            ["Метадані не підтверджено"],
        )
        svg = '<svg style="font-family:IBM Plex Sans"><text>Ціна</text></svg>'
        self.assertIn("font/woff2;base64", embed_svg_fonts(svg))
        self.assertEqual(embed_svg_fonts("<svg><text>x</text></svg>"), "<svg><text>x</text></svg>")

    def test_schema_upgrade_preserves_projects(self):
        self.create()
        with connect(self.database) as c:
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(c.execute("SELECT count(*) FROM reports").fetchone()[0], 1)
        self.assertEqual(self.store.get(self.project.id).name, self.project.name)
