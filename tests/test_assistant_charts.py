"""Text chart proposals: explicit apply, scope, validation and source integrity."""

import unittest
from unittest.mock import patch

import test_assistant as ai
import test_preparation_store as fixtures

from data_story.assistant_drafts import propose_chart
from data_story.assistant_tools import run_tool
from data_story.contracts import Failure, Success
from data_story.instructions_screen import chart_families


class AssistantChartTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp
    fake = ai.AssistantTests.fake

    def chart_query(self, **extra):
        return {
            "chart": "histogram",
            "query": {"dataset_id": self.dataset.id, "field": "x", "bins": 5, **extra},
            "explanation": "Гістограма показує розподіл числового поля.",
        }

    def test_propose_then_explicit_apply_is_local_and_audited(self):
        self.fake([ai.tool_reply("propose_eda_chart", self.chart_query())])
        result = self.service.assistant_ask(
            self.project.id,
            {
                "block": "eda",
                "mode": "execute",
                "question": "Розподіл x у 5 інтервалах",
                "page": {"dataset_id": self.dataset.id},
            },
        )
        self.assertIsInstance(result, Success)
        self.assertEqual(result.data["task"], "chart_command")
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(
            any(e.operation == "chart.built" for e in self.store.events(self.project.id))
        )
        proposal = result.data["proposals"][0]
        with patch.object(self.service.openai, "run", side_effect=AssertionError("No AI")):
            applied = self.service.assistant_apply_draft(self.project.id, proposal)
        self.assertIsInstance(applied, Success)
        self.assertEqual(sum(applied.data["result"]["counts"]), 2)
        self.assertIn("<svg", applied.data["result"]["svg"])
        event = self.store.events(self.project.id)[-1]
        self.assertEqual(event.operation, "chart.built")
        self.assertEqual(event.payload["fields"], ["x"])
        self.assertEqual(event.payload["options"]["histogram_bins"], 20)

    def test_unknown_non_numeric_repeated_and_unsupported_are_refused(self):
        for raw in [
            self.chart_query(field="unknown"),
            self.chart_query(field="s"),
            self.chart_query(bins=61),
            self.chart_query(bins=5, bin_width=1),
            {**self.chart_query(), "chart": "line"},
            {**self.chart_query(), "chart": "scatter"},
            {
                "chart": "heatmap",
                "query": {"dataset_id": self.dataset.id, "fields": ["x", "x"]},
                "explanation": "Зв’язки",
            },
        ]:
            with self.subTest(raw=raw):
                self.assertIsInstance(
                    run_tool(self.service, self.project.id, "propose_eda_chart", raw), Failure
                )
        self.assertFalse(
            any(e.operation == "chart.built" for e in self.store.events(self.project.id))
        )

    def test_stale_metadata_and_sources_refused(self):
        draft = propose_chart(self.service, self.project.id, self.chart_query())
        self.service.add_metadata_text(self.dataset.id, {"text": "Уточнення"})
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Failure)
        draft = propose_chart(self.service, self.project.id, self.chart_query())
        doc = self.service.metadata(self.dataset.id).data.document.model_dump()
        doc["columns"][0]["unit"] = "м"
        self.service.save_metadata(self.dataset.id, {"document": doc})
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Failure)

    def test_foreign_and_modified_source_refused(self):
        draft = propose_chart(self.service, self.project.id, self.chart_query())
        other = self.store.create("Other")
        self.assertIsInstance(self.service.assistant_apply_draft(other.id, draft), Failure)
        self.assertIsInstance(
            run_tool(self.service, other.id, "propose_eda_chart", self.chart_query()), Failure
        )
        path = self.service.datasets.files_root / self.dataset.stored_path
        path.write_bytes(b"x,s\n4,A\n")
        self.assertIsInstance(self.service.assistant_apply_draft(self.project.id, draft), Failure)

    def test_style_and_global_theme_preserved(self):
        options = self.service.chart_config.get().effective().model_dump()
        options.update(scatter_opacity=0.25, accent="#E07B39", grid_y=False)
        draft = propose_chart(
            self.service, self.project.id, self.chart_query(options=options, title="Розподіл")
        )
        applied = self.service.assistant_apply_draft(self.project.id, draft)
        self.assertIsInstance(applied, Success)
        self.assertEqual(applied.data["result"]["options"], options)
        self.assertNotEqual(self.service.chart_config.get().effective().accent, options["accent"])

    def test_missing_scatter_heatmap_reuse_existing_service(self):
        dataset = self.service.datasets.add(
            self.project.id, "numbers.csv", b"x,y\n1,2\n2,4\n3,6\n4,8\n5,10\n"
        )
        for chart, args in [
            ("missing", {"start": 0}),
            ("scatter", {"x": "x", "y": "y"}),
            ("heatmap", {"fields": ["x", "y"]}),
        ]:
            draft = propose_chart(
                self.service,
                self.project.id,
                {
                    "chart": chart,
                    "query": {"dataset_id": dataset.id, **args},
                    "explanation": "Перевірити дані",
                },
            )
            result = self.service.assistant_apply_draft(self.project.id, draft)
            self.assertIsInstance(result, Success)
            self.assertIn("<svg", result.data["result"]["svg"])
        self.assertEqual(self.service.datasets.get(dataset.id).sha256, dataset.sha256)

    def test_full_local_tree_and_search(self):
        families = chart_families()
        self.assertEqual(len(families), 6)
        self.assertEqual(sum(len(f[3]) for f in families), 30)
        results = self.service.knowledge.search("Вибір графіка Динаміка багато періодів", 3)
        self.assertTrue(any(h["section"] == "Вибір графіка: Динаміка" for h in results["hits"]))

    def test_only_selected_dataset_and_offered_tools_can_be_called(self):
        for mode, page, tool, args in [
            ("execute", {}, "propose_eda_chart", self.chart_query()),
            ("advise", {"dataset_id": self.dataset.id}, "propose_eda_chart", self.chart_query()),
            (
                "execute",
                {"dataset_id": self.dataset.id},
                "propose_project_description",
                {"text": "wrong block"},
            ),
        ]:
            with self.subTest(mode=mode, tool=tool):
                self.fake([ai.tool_reply(tool, args), ai.text_reply("Потрібне уточнення.")])
                result = self.service.assistant_ask(
                    self.project.id,
                    {"block": "eda", "mode": mode, "question": "Покажи графік", "page": page},
                )
                self.assertIsInstance(result, Success)
                self.assertEqual(result.data["proposals"], [])
                self.assertFalse(result.data["tools"][0]["ok"])
        self.assertEqual(self.store.get(self.project.id).description, "")

    def test_invalid_chart_proposal_is_input_error(self):
        result = run_tool(
            self.service, self.project.id, "propose_eda_chart", self.chart_query(bins=61)
        )
        self.assertIsInstance(result, Failure)
        self.assertEqual(result.error.code, "invalid_input")
