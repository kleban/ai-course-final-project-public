"""Conclusions: facts + cited fragments, explicit AI, edit/approve, report integration (v16)."""

import importlib.util
import json
import sqlite3
import unittest
from unittest.mock import patch

import test_model as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.conclusions import ConclusionError, check_text
from data_story.contracts import Failure, Success
from data_story.reports import NARRATIVE, compose
from data_story.storage import SCHEMA_VERSION, connect, ensure_schema
from data_story.web import install_error_handlers, router

HAS_DUCKDB = importlib.util.find_spec("duckdb") is not None
BOOK_TEXT = "Книжковий фрагмент про інтерпретацію R² і перенавчання. " * 30


def hit(n: int, kind: str = "rule", score: float = 0.5) -> dict:
    return {
        "id": f"{kind}-{n}",
        "kind": kind,
        "document_id": f"doc-{n}",
        "title": f"Джерело {n}",
        "section": f"Розділ {n}",
        "source": "Автор, 2020; с. 10–12 (PDF)" if kind == "book" else "Правила проєкту",
        "priority": 1,
        "text": BOOK_TEXT if kind == "book" else f"Правило {n}: значущість ≠ причинність.",
        "score": score,
    }


def fake_search(rules: int = 4, books: int = 2):
    """Deterministic retrieval instead of the local index and the user's books."""
    calls: list[str] = []

    def search(query, limit=3, books=books):
        calls.append(query)
        index = len(calls)
        found = [hit(index * 10 + i, score=0.6 - i / 10) for i in range(min(limit, rules))]
        book = [hit(index, "book", 0.55)] if books and index <= 2 else []
        return {"query": query, "method": "test", "hits": found, "book_hits": book}

    return search, calls


TEXT = (
    "- Модель пояснює частину мінливості ціни на тесті (R² test = 0,9) [1].\n"
    "- Значущість коефіцієнта не означає причинності [2].\n"
    "- Порівняйте помилку з простим прогнозом і перевірте похибки [3]."
)


class ConclusionTests(unittest.TestCase):
    setUp = fixtures.ModelStoreTests.setUp

    def model(self, project_id=None) -> dict:
        project_id = project_id or self.project.id
        lines = ["x1,x2,y"] + [
            f"{i},{(i * 7) % 5},{3 + 2 * i - 1.5 * ((i * 7) % 5) + ((i * 13) % 7 - 3) * 0.8}"
            for i in range(60)
        ]
        dataset = self.service.datasets.add(project_id, "model.csv", "\n".join(lines).encode())
        run = self.service.training_prepare(
            project_id, {"dataset_id": dataset.id, "target": "y", "features": ["x1", "x2"]}
        ).data["id"]
        return self.service.model_train(project_id, {"training_run_id": run}).data

    def prepare(self, model_id, summary_ids=(), text=TEXT, rules=4, books=2):
        search, self.queries = fake_search(rules, books)
        seen: dict = {}

        def respond(task, instructions, payload, **kwargs):
            seen.update(task=task, payload=json.loads(payload), accept=kwargs["accept"])
            self.assertEqual(kwargs["project_id"], self.project.id)
            return {
                "text": json.dumps({"text": text}, ensure_ascii=False),
                "model": "gpt-test",
                "usage_total": 321,
                "attempts": [{}],
            }

        with (
            patch.object(self.service.knowledge, "search", side_effect=search),
            patch.object(self.service.openai, "respond", side_effect=respond) as call,
        ):
            result = self.service.conclusion_prepare(
                self.project.id, {"model_id": model_id, "summary_ids": list(summary_ids)}
            )
        self.seen, self.calls = seen, call.call_count
        return result

    def test_prepare_sends_facts_and_cited_sources_and_stores_a_draft(self):
        model = self.model()
        result = self.prepare(model["id"])
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        item = result.data
        self.assertEqual((item["status"], item["revision"], self.calls), ("draft", 1, 1))
        self.assertEqual(self.seen["task"], "conclusions")
        payload = self.seen["payload"]
        self.assertEqual(payload["model"]["target"], "y")
        # Facts go to the AI with report precision (author feedback 12, history 097).
        self.assertEqual(
            payload["model"]["r2_test"], round(model["document"]["metrics"]["test"]["r2"], 2)
        )
        self.assertTrue(all(len(str(c["value"])) <= 8 for c in payload["model"]["coefficients"]))
        self.assertIn("x1", payload["model"]["significant_p_lt_0_05"])
        self.assertTrue(3 <= len(payload["sources"]) <= 5)
        self.assertEqual([s["n"] for s in payload["sources"]], list(range(1, 6)))
        self.assertIn("book", {s["kind"] for s in item["sources"]})
        # Book excerpts are kept short in the saved draft.
        self.assertTrue(all(len(s["text"]) <= 600 for s in item["sources"]))
        self.assertEqual(len(self.queries), 3)  # metrics, significance, first warning
        accept = self.seen["accept"]
        self.assertIsNone(accept(json.dumps({"text": TEXT})))
        for bad in [
            "not json",
            json.dumps({"text": "Без посилань."}),
            json.dumps({"text": "Висновок [9]."}),
            json.dumps({"text": "## Заголовок\n- пункт [1]"}),
            json.dumps({"text": "пункт [1]", "extra": 1}),
            json.dumps({"text": "<script>x</script> [1]"}),
        ]:
            with self.subTest(bad=bad):
                self.assertIsNotNone(accept(bad))
        self.assertEqual(self.store.events(self.project.id)[-1].operation, "conclusions.prepared")
        self.assertIsInstance(self.service.events(self.project.id), Success)

    def test_too_few_sources_refuses_before_any_ai_call(self):
        model = self.model()
        result = self.prepare(model["id"], rules=0)
        self.assertIsInstance(result, Failure)
        self.assertIn("щонайменше 3", result.error.message)
        self.assertEqual(self.calls, 0)
        self.assertEqual(self.service.conclusion_list(self.project.id).data, [])

    def test_edit_approve_stale_foreign_and_invalid_citations(self):
        item = self.prepare(self.model()["id"]).data
        other = self.store.create("Other")
        self.assertIsInstance(self.service.conclusion_get(other.id, item["id"]), Failure)
        bad = self.service.conclusion_save(
            self.project.id, item["id"], {"revision": 1, "markdown": "Твердження [7]."}
        )
        self.assertIsInstance(bad, Failure)
        self.assertIn("[7]", bad.error.message)
        approved = self.service.conclusion_save(
            self.project.id, item["id"], {"revision": 1}, approve=True
        )
        self.assertIsInstance(approved, Success, getattr(approved, "error", None))
        self.assertEqual((approved.data["status"], approved.data["revision"]), ("approved", 2))
        self.assertIsNotNone(approved.data["approved_at"])
        stale = self.service.conclusion_save(
            self.project.id, item["id"], {"revision": 1, "markdown": TEXT + "\n- ще [1]"}
        )
        self.assertIsInstance(stale, Failure)
        edited = self.service.conclusion_save(
            self.project.id, item["id"], {"revision": 2, "markdown": TEXT + "\n- p < 0,05 [2]"}
        ).data
        self.assertEqual((edited["status"], edited["approved_at"]), ("draft", None))
        self.assertEqual(edited["draft_markdown"], TEXT)  # the original AI wording is kept
        operations = [e.operation for e in self.store.events(self.project.id)][-2:]
        self.assertEqual(operations, ["conclusions.approved", "conclusions.saved"])

    def test_only_approved_conclusions_enter_a_new_report_without_book_text(self):
        model = self.model()
        item = self.prepare(model["id"]).data
        draft_report = self.service.report_create(self.project.id, {"model_id": model["id"]}).data
        self.assertIsNone(draft_report["snapshot"]["conclusions"])
        self.assertNotIn("Джерела висновків", draft_report["markdown"])
        self.service.conclusion_save(self.project.id, item["id"], {"revision": 1}, approve=True)
        report = self.service.report_create(self.project.id, {"model_id": model["id"]}).data
        self.assertEqual(report["snapshot"]["conclusions"]["id"], item["id"])
        self.assertIn(TEXT, report["markdown"])
        self.assertIn("**Джерела висновків**", report["markdown"])
        self.assertIn("Джерело 10 — Розділ 10", report["markdown"])
        self.assertNotIn("Книжковий фрагмент", report["markdown"])
        # A later AI narrative for the report cannot replace the approved section.
        narrative = {k: f"AI {k}" for k in NARRATIVE}
        self.assertNotIn("AI conclusions", compose(report["snapshot"], narrative))
        self.assertIn(TEXT, compose(report["snapshot"], narrative))

    @unittest.skipUnless(HAS_DUCKDB, "duckdb is not installed")
    def test_duckdb_summary_facts_and_refusals(self):
        model = self.model()
        source = self.service.datasets.add(
            self.project.id, "sales.csv", b"type,price\nh,100\nu,80\nh,200\nu,90\n"
        )
        recipe = {
            "dataset_id": str(source.id),
            "group_by": ["type"],
            "aggregates": [{"func": "avg", "field": "price"}],
        }
        token = self.service.summary_preview(self.project.id, recipe).data["preview_token"]
        summary = self.service.summary_create(
            self.project.id, {"recipe": recipe, "preview_token": token, "name": "Ціни"}
        ).data
        listed = self.service.conclusion_summaries(self.project.id).data
        self.assertEqual([s["id"] for s in listed], [str(summary.id)])
        result = self.prepare(model["id"], [str(summary.id)])
        self.assertIsInstance(result, Success, getattr(result, "error", None))
        sent = self.seen["payload"]["summaries"][0]
        self.assertEqual(sent["columns"], ["type", "avg_price"])
        self.assertEqual(sorted(sent["rows"]), [["h", 150.0], ["u", 85.0]])
        self.assertIn("AVG", sent["sql"])
        self.assertEqual(len(self.queries), 4)
        self.assertEqual(
            result.data["summaries"][0]["sha256"],
            self.service.datasets.get(str(summary.id)).sha256,
        )
        self.assertIsInstance(self.service.events(self.project.id), Success)  # journal works
        plain = self.prepare(model["id"], [str(source.id)])
        self.assertIsInstance(plain, Failure)
        self.assertEqual(self.calls, 0)
        too_many = self.service.conclusion_prepare(
            self.project.id, {"model_id": model["id"], "summary_ids": [str(source.id)] * 4}
        )
        self.assertIsInstance(too_many, Failure)

    def test_api_prepare_save_and_approve(self):
        model = self.model()
        app = FastAPI()
        install_error_handlers(app)
        app.include_router(router(self.service))
        client = TestClient(app)
        search, _ = fake_search()
        reply = {
            "text": json.dumps({"text": TEXT}),
            "model": "gpt-test",
            "usage_total": 1,
            "attempts": [],
        }
        base = f"/api/projects/{self.project.id}/conclusions"
        with (
            patch.object(self.service.knowledge, "search", side_effect=search),
            patch.object(self.service.openai, "respond", return_value=reply),
        ):
            created = client.post(base, json={"model_id": model["id"]})
        self.assertEqual(created.status_code, 200, created.text)
        item = created.json()["data"]
        self.assertEqual(client.get(base).json()["data"][0]["id"], item["id"])
        saved = client.put(f"{base}/{item['id']}", json={"revision": 1, "markdown": TEXT})
        self.assertEqual(saved.json()["data"]["revision"], 1)  # unchanged text: no new revision
        approved = client.post(f"{base}/{item['id']}/approve", json={"revision": 1})
        self.assertEqual(approved.json()["data"]["status"], "approved")
        self.assertEqual(client.put(f"{base}/{item['id']}", json={"revision": 1}).status_code, 422)

    def test_text_rules(self):
        self.assertEqual(
            check_text("  - p < 0,05 і R² > 0,4 [1]  ", 3), "- p < 0,05 і R² > 0,4 [1]"
        )
        for text in ["", "# Заголовок", "![x](a.svg)", "[4] поза межами", "x" * 8001]:
            with self.subTest(text=text[:20]), self.assertRaises(ConclusionError):
                check_text(text, 3)

    def test_schema_v15_upgrades_to_v16_and_keeps_projects(self):
        with connect(self.database) as c:
            c.execute("DROP TABLE conclusions")
            c.execute("PRAGMA user_version = 15")
        ensure_schema(self.database)
        with connect(self.database) as c:
            self.assertEqual(c.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
            self.assertEqual(SCHEMA_VERSION, 17)
            c.execute("SELECT count(*) FROM conclusions").fetchone()
            with self.assertRaises(sqlite3.IntegrityError):
                c.execute(
                    "INSERT INTO conclusions VALUES ('a', ?, 'missing', 't', 't', 1, 'other', '{}')",
                    (self.project.id,),
                )
        self.assertEqual(self.store.get(self.project.id).name, self.project.name)


if __name__ == "__main__":
    unittest.main()
