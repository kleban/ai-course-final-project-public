"""AI proposals cannot write until explicit apply; stale/foreign proposals are rejected."""

import sqlite3
import unittest
from unittest.mock import patch

import test_assistant as assistant_fixtures
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.assistant import Assistant
from data_story.assistant_drafts import propose_metadata, propose_project
from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.metadata import MetadataError
from data_story.web import install_error_handlers, router


class AssistantDraftTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp
    fake = assistant_fixtures.AssistantTests.fake

    def metadata_patch(self):
        return {
            "dataset_id": self.dataset.id,
            "description": "Демонстраційні вимірювання",
            "columns": [{"name": "x", "description": "Виміряний показник", "unit": "м"}],
        }

    def test_project_tool_returns_reviewable_draft_without_saving(self):
        self.fake(
            [
                assistant_fixtures.tool_reply(
                    "propose_project_description", {"text": "## Мета\nПорівняти показники."}
                )
            ]
        )
        result = self.service.assistant_ask(
            self.project.id,
            {
                "block": "data",
                "mode": "execute",
                "question": "Створи чернетку опису проєкту",
            },
        )
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data["task"], "project_description")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.store.get(self.project.id).description, "")
        draft = result.data["proposals"][0]
        self.assertEqual(self.store.events(self.project.id)[-1].payload["proposals"], [draft])
        with patch.object(self.service.openai, "run", side_effect=AssertionError("No AI on apply")):
            applied = self.service.assistant_apply_draft(self.project.id, draft)
        self.assertIsInstance(applied, Success)
        self.assertEqual(self.store.get(self.project.id).description, draft["text"])
        self.assertEqual(self.store.events(self.project.id)[-1].payload["origin"], "ai_draft")

    def test_metadata_apply_preserves_observed_and_other_fields_and_unconfirms(self):
        original = self.service.metadata(self.dataset.id).data.document.model_dump()
        self.service.save_metadata(self.dataset.id, {"document": original, "confirm": True})
        draft = propose_metadata(self.service, self.project.id, self.metadata_patch())
        self.assertIsNotNone(self.service.metadata(self.dataset.id).data.confirmed_at)
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Success)
        saved = self.service.metadata(self.dataset.id).data
        self.assertIsNone(saved.confirmed_at)
        self.assertEqual(
            saved.document.columns[0].observed.model_dump(), original["columns"][0]["observed"]
        )
        self.assertEqual(saved.document.columns[0].source, "llm")
        self.assertEqual(saved.document.columns[0].unit, "м")
        self.assertEqual(saved.document.columns[1].model_dump(), original["columns"][1])

    def test_stale_description_does_not_overwrite_manual_change(self):
        draft = propose_project(self.service, self.project.id, {"text": "AI draft"})
        self.store.save_description(self.project.id, "Author edit")
        result = self.service.assistant_apply_draft(self.project.id, draft)
        self.assertIsInstance(result, Failure)
        self.assertIn("змінився", result.error.message)
        self.assertEqual(self.store.get(self.project.id).description, "Author edit")
        with self.assertRaises(ValueError):
            self.store.save_description(self.project.id, "AI draft", expected_description="")

    def test_metadata_and_source_changes_invalidate_draft(self):
        draft = propose_metadata(self.service, self.project.id, self.metadata_patch())
        self.service.add_metadata_text(self.dataset.id, {"text": "Новий опис"})
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Failure)
        draft = propose_metadata(self.service, self.project.id, self.metadata_patch())
        original = self.service.metadata(self.dataset.id).data.document.model_dump()
        original["columns"][0]["description"] = "Author edit"
        self.service.save_metadata(self.dataset.id, {"document": original})
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Failure)
        self.assertEqual(
            self.service.metadata(self.dataset.id).data.document.columns[0].description,
            "Author edit",
        )
        with self.assertRaises(MetadataError):
            self.service.metadata_store.save(
                self.dataset.id, original, expected_version=(False, None)
            )
        with self.assertRaises(MetadataError):
            self.service.metadata_store.save(self.dataset.id, original, expected_sources=())

    def test_foreign_and_unknown_fields_are_rejected(self):
        other = self.store.create("Other")
        draft = propose_project(self.service, self.project.id, {"text": "AI draft"})
        self.assertIsInstance(self.service.assistant_apply_draft(other.id, draft), Failure)
        foreign = self.service.datasets.add(other.id, "secret.csv", b"secret\n1\n")
        args = {"dataset_id": foreign.id, "columns": [{"name": "secret", "unit": "m"}]}
        self.assertIsInstance(
            run_tool(self.service, self.project.id, "propose_metadata", args), Failure
        )
        for columns in (
            [{"name": "unknown", "description": "x"}],
            [{"name": "x", "unit": "m"}, {"name": "x", "unit": "cm"}],
            [{"name": "x", "observed": {"missing": 0}}],
        ):
            result = run_tool(
                self.service,
                self.project.id,
                "propose_metadata",
                {
                    "dataset_id": self.dataset.id,
                    "columns": columns,
                },
            )
            self.assertIsInstance(result, Failure)
        self.assertFalse(self.service.metadata(self.dataset.id).data.saved)

    def test_raw_descriptions_and_task_are_in_metadata_request(self):
        self.service.add_metadata_text(self.dataset.id, {"text": "x — площа у метрах."})
        self.fake([assistant_fixtures.tool_reply("propose_metadata", self.metadata_patch())])
        result = self.service.assistant_ask(
            self.project.id,
            {
                "block": "data",
                "mode": "execute",
                "question": "Створи чернетку метаданих",
                "page": {"dataset_id": self.dataset.id},
            },
        )
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data["task"], "metadata_draft")
        self.assertIn("x — площа у метрах.", self.requests[0]["instructions"])
        self.assertFalse(self.service.metadata(self.dataset.id).data.saved)
        self.assertEqual(len(result.data["proposals"]), 1)
        foreign = self.store.create("Other")
        context = Assistant(self.service).context(
            foreign.id, "data", {"dataset_id": self.dataset.id}
        )
        self.assertNotIn("raw_descriptions", context)

    def test_apply_api_and_repeated_apply(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        draft = propose_metadata(self.service, self.project.id, self.metadata_patch())
        with TestClient(app) as client:
            url = f"/api/projects/{self.project.id}/assistant/drafts/apply"
            self.assertEqual(client.post(url, json=draft).status_code, 200)
            self.assertEqual(client.post(url, json=draft).status_code, 422)
            self.assertEqual(client.post(url, json={**draft, "base_token": "bad"}).status_code, 422)

    def test_audit_failure_rolls_back_explicit_apply(self):
        draft = propose_project(self.service, self.project.id, {"text": "AI draft"})
        with patch(
            "data_story.projects.ProjectStore._event", side_effect=sqlite3.OperationalError("test")
        ):
            result = self.service.assistant_apply_draft(self.project.id, draft)
        self.assertIsInstance(result, Failure)
        self.assertEqual(self.store.get(self.project.id).description, "")
