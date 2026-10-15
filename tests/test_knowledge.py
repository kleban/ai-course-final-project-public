"""Stage 07 RAG: rule documents, fragments, local search on Ukrainian control questions."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story import embeddings
from data_story.analytics import AnalyticsStore
from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.knowledge import HYBRID, METHOD, KnowledgeError, KnowledgeStore, parse
from data_story.storage import connect
from data_story.web import install_error_handlers, router

# Control questions written for the default rules: the expected section must be in the top 3.
CONTROL = [
    ("що означає R квадрат", "R² (R-квадрат) — коефіцієнт детермінації"),
    ("чому біля коефіцієнта стоять зірочки", "p-value і зірочки значущості"),
    ("як заповнити пропуски без витоку даних", "Витік даних"),
    ("графік похибок має форму лійки", "Графік похибок"),
    ("чи повинні стовпчики починатися з нуля", "Стовпці починаються з нуля"),
    ("чи можна порівнювати коефіцієнти різних ознак", "Порівняння коефіцієнтів між собою"),
    ("як закодувати категорії", "Категорії і one-hot кодування"),
    ("кореляція близька до нуля, а залежність є", "Розсіювання і кореляція"),
    ("різниця між MAE і RMSE", "RMSE — корінь середнього квадрата помилки"),
    ("R² на навчальній вибірці набагато вищий ніж на тесті", "Навчальна і тестова частини"),
    ("що таке вільний член", "Вільний член a₀"),
    ("скільки інтервалів у гістограмі", "Гістограма і ширина інтервалу"),
    ("прогноз для значень поза діапазоном", "Екстраполяція"),
    ("ознаки сильно корельовані між собою", "Мультиколінеарність"),
    ("чи заповнювати пропуски в цільовій змінній", "Ціль не заповнюємо"),
]


class KnowledgeTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def test_control_questions_find_expected_rules(self):
        misses = []
        for question, section in CONTROL:
            hits = self.service.knowledge.search(question)["hits"]
            if section not in [h["section"] for h in hits]:
                misses.append((question, [h["section"] for h in hits]))
        self.assertEqual(misses, [])

    def test_without_model_search_falls_back_to_tfidf_and_says_why(self):
        with (
            patch.object(embeddings, "embedder", return_value=None),
            patch.object(embeddings, "status", return_value="Модель не встановлена."),
        ):
            result = self.service.knowledge.search("що таке вільний член")
        self.assertEqual(result["method"], METHOD)
        self.assertEqual(result["semantic_reason"], "Модель не встановлена.")
        self.assertIsNone(result["hits"][0]["semantic"])

    def test_hybrid_ranks_by_meaning_and_caches_fragment_vectors(self):
        fragments = self.service.knowledge.fragments()
        target = next(i for i, f in enumerate(fragments) if f.section.startswith("Екстраполяція"))

        class Fake:
            version = "fake"
            calls = 0

            def encode(self, texts, kind):
                Fake.calls += kind == "passage"
                if kind == "query":
                    vector = np.zeros(len(fragments), dtype=np.float32)
                    vector[target] = 1
                    return vector[None, :]
                return np.eye(len(texts), dtype=np.float32) * 0.2 + 0.79

        fake = Fake()
        with patch.object(embeddings, "embedder", return_value=fake):
            # No shared words with the rule: only the "meaning" can find it.
            result = self.service.knowledge.search("цифри далеко за межами відомого")
            again = self.service.knowledge.search("ще раз те саме питання")
        self.assertEqual(result["method"], HYBRID)
        self.assertTrue(result["hits"][0]["section"].startswith("Екстраполяція"))
        self.assertGreater(result["hits"][0]["semantic"], result["hits"][0]["lexical"])
        self.assertEqual(Fake.calls, 1)
        self.assertEqual(again["method"], HYBRID)
        cache = list((self.service.store.database.parent / "knowledge-index").glob("fake-*.npy"))
        self.assertEqual(len(cache), 1)

    def test_unrelated_question_returns_nothing(self):
        result = self.service.knowledge.search("рецепт борщу")
        self.assertEqual(result["hits"], [])
        self.assertIn("не знайдено", result["reason"])

    def test_hits_carry_source_section_and_are_logged(self):
        result = self.service.knowledge_search({"query": "що таке вільний член", "limit": 2})
        self.assertIsInstance(result, Success)
        hit = result.data["hits"][0]
        self.assertEqual(hit["document_id"], "04-regression-coefficients")
        self.assertTrue(hit["source"])
        self.assertLessEqual(len(result.data["hits"]), 2)
        event = self.service.knowledge.events()[0]
        self.assertEqual(event["operation"], "knowledge.searched")
        self.assertEqual(event["query"], "що таке вільний член")
        self.assertEqual(event["hits"][0]["id"], hit["id"])

    def test_defaults_seeded_once_and_edits_rebuild_index(self):
        documents = self.service.knowledge.documents()
        self.assertEqual(len(documents), 5)
        self.assertTrue(all(d["is_default"] for d in documents))
        self.assertGreaterEqual(sum(d["fragments"] for d in documents), 30)
        reopened = KnowledgeStore(self.database)
        self.assertEqual(len(reopened.documents()), 5)
        before = reopened.search("правило про кавові зерна")["hits"]
        self.assertNotIn("Кавові зерна", [h["section"] for h in before])
        text = "# Кава\nДжерело: тест\nПріоритет: 1\n\n## Кавові зерна\nПравило про кавові зерна.\n"
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO knowledge_documents VALUES ('09-test', 'Кава', 'тест', 1, ?, 0, 'x')",
                (text,),
            )
        after = reopened.search("правило про кавові зерна")["hits"]
        self.assertEqual(after[0]["section"], "Кавові зерна")

    def test_parse_and_input_errors(self):
        meta, fragments = parse("x", "# T\nДжерело: s\nПріоритет: 2\n\n## A\ntext a\n\n## B\nb\n")
        self.assertEqual((meta["priority"], [f.section for f in fragments]), (2, ["A", "B"]))
        for bad in ("без заголовка", "# T\nтекст без розділів"):
            with self.assertRaises(KnowledgeError):
                parse("x", bad)
        with self.assertRaises(KnowledgeError):
            self.service.knowledge.search("   ")
        self.assertIsInstance(self.service.knowledge_search({"query": ""}), Failure)
        self.assertIsInstance(self.service.knowledge_search({"query": "x" * 501}), Failure)

    def test_http_and_assistant_tool(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            docs = client.get("/api/knowledge/documents")
            self.assertEqual(docs.status_code, 200)
            self.assertEqual(len(docs.json()["data"]), 5)
            found = client.post("/api/knowledge/search", json={"query": "форма лійки похибок"})
            self.assertEqual(found.status_code, 200)
            self.assertEqual(found.json()["data"]["hits"][0]["section"], "Графік похибок")
            self.assertEqual(client.post("/api/knowledge/search", json={}).status_code, 422)
        tool = run_tool(self.service, self.project.id, "search_knowledge", {"query": "витік даних"})
        self.assertIsInstance(tool, Success)
        json.dumps(tool.data)

    def test_rag_reads_sqlite_rules_without_analytical_store(self):
        text = "# Локальні правила\nДжерело: тест\nПріоритет: 1\n\n## Зимові велосипеди\nЗимові велосипеди потребують зимових шин.\n"
        self.service.knowledge.create(text)
        reopened = KnowledgeStore(self.database)
        with (
            patch.object(
                AnalyticsStore, "_connection", side_effect=AssertionError("RAG used DuckDB")
            ),
            patch.object(embeddings, "embedder", return_value=None),
        ):
            result = reopened.search("зимові велосипеди зимові шини")
        self.assertEqual(result["hits"][0]["section"], "Зимові велосипеди")
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT text FROM knowledge_documents WHERE id=?",
                (result["hits"][0]["document_id"],),
            ).fetchone()
        self.assertEqual(row["text"], text)


class KnowledgeEditorTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    DOC = "# Мої правила\nДжерело: власний досвід\nПріоритет: 1\n\n## Сезонність цін\nЦіни на житло навесні вищі.\n"

    def test_save_changes_search_and_logs_previous_text(self):
        store = self.service.knowledge
        original = store.document("03-regression-metrics")
        text = (
            original["text"]
            + "\n## Медіанна похибка\nМедіанна абсолютна похибка стійка до викидів.\n"
        )
        saved = self.service.knowledge_save("03-regression-metrics", {"text": text})
        self.assertIsInstance(saved, Success)
        self.assertFalse(saved.data["is_default"])
        self.assertTrue(saved.data["builtin"])
        hits = store.search("медіанна абсолютна похибка")["hits"]
        self.assertEqual(hits[0]["section"], "Медіанна похибка")
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT payload FROM config_events WHERE operation='knowledge.saved'"
            ).fetchone()
        self.assertEqual(json.loads(row[0])["previous_text"], original["text"])

    def test_reset_restores_builtin_and_unedited_defaults_follow_repository(self):
        self.service.knowledge_save("05-model-check", {"text": self.DOC})
        restored = self.service.knowledge_reset("05-model-check").data
        self.assertTrue(restored["is_default"])
        self.assertIn("Графік похибок", restored["text"])
        folder = Path(tempfile.mkdtemp(dir=self.temporary.name))
        (folder / "a.md").write_text("# A\n\n## Перше\nстарий текст\n", encoding="utf-8")
        (folder / "b.md").write_text("# B\n\n## Друге\nтекст b\n", encoding="utf-8")
        database = Path(self.temporary.name) / "kb.sqlite3"
        store = KnowledgeStore(database, defaults=folder)
        store.save("b", "# B\n\n## Друге\nмій текст\n")
        (folder / "a.md").write_text("# A\n\n## Перше\nновий текст\n", encoding="utf-8")
        (folder / "b.md").write_text("# B\n\n## Друге\nоновлений b\n", encoding="utf-8")
        (folder / "c.md").write_text("# C\n\n## Третє\nновий документ\n", encoding="utf-8")
        texts = {d["id"]: d["text"] for d in store.documents()}
        self.assertIn("новий текст", texts["a"])  # unedited default follows the repository
        self.assertIn("мій текст", texts["b"])  # the user's edit is kept
        self.assertIn("c", texts)

    def test_create_and_delete_user_document(self):
        created = self.service.knowledge_create({"text": self.DOC}).data
        self.assertTrue(created["id"].startswith("user-"))
        self.assertFalse(created["builtin"])
        self.assertEqual(self.service.knowledge.search("сезонність цін навесні")["hits"][0]["section"],
                         "Сезонність цін")  # fmt: skip
        self.assertIsInstance(self.service.knowledge_reset(created["id"]), Failure)
        self.assertIsInstance(self.service.knowledge_delete(created["id"]), Success)
        self.assertEqual(len(self.service.knowledge.documents()), 5)
        self.assertIsInstance(self.service.knowledge_delete("01-visualization"), Failure)

    def test_invalid_documents_refused(self):
        for text in (
            "без заголовка",
            "# Назва\nтекст без розділів",
            "# Назва\nПріоритет: високий\n\n## A\nb",
            "# Назва\nПріоритет: 12\n\n## A\nb",
            "# Назва\n\n## A\n" + "x" * 20001,
        ):
            self.assertIsInstance(self.service.knowledge_save("01-visualization", {"text": text}),
                                  Failure)  # fmt: skip
        self.assertTrue(self.service.knowledge.document("01-visualization")["is_default"])
        self.assertIsInstance(self.service.knowledge_save("nope", {"text": self.DOC}), Failure)

    def test_http_editor_endpoints(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            created = client.post("/api/knowledge/documents", json={"text": self.DOC})
            self.assertEqual(created.status_code, 200, created.text)
            document_id = created.json()["data"]["id"]
            route = f"/api/knowledge/documents/{document_id}"
            self.assertEqual(
                client.put(route, json={"text": self.DOC + "\n## Ще\nтекст\n"}).status_code, 200
            )
            self.assertEqual(client.delete(route).status_code, 200)
            self.assertEqual(
                client.post("/api/knowledge/documents/01-visualization/reset").status_code, 200
            )
            self.assertEqual(client.post("/api/knowledge/documents", json={}).status_code, 422)
