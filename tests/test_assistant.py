"""Stage 09 first piece: assistant modes, context with knowledge rules, tool calls — fake OpenAI."""

import json
import unittest

import httpx
import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient

from data_story.ai_settings import AiConfiguration
from data_story.assistant import Assistant, _context_json
from data_story.contracts import Failure, Success
from data_story.openai_client import OpenAIClient
from data_story.web import install_error_handlers, router

KEY = "sk-unit-" + "q" * 30 + "LAST"


def text_reply(text):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    }


def tool_reply(name, arguments, call="call_1"):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "encrypted_content": "x"},
            {
                "type": "function_call",
                "call_id": call,
                "name": name,
                "arguments": json.dumps(arguments),
            },
        ],  # fmt: skip
        "usage": {"input_tokens": 90, "output_tokens": 10, "total_tokens": 100},
    }


class AssistantTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def fake(self, replies):
        self.requests = []

        def handler(request):
            self.requests.append(json.loads(request.content))
            reply = replies[min(len(self.requests), len(replies)) - 1]
            return httpx.Response(200, json=reply)

        self.service.openai = OpenAIClient(
            self.database, self.service.ai_settings, key=lambda: KEY,
            transport=httpx.MockTransport(handler), sleep=lambda _: None,
        )  # fmt: skip

    def split(self):
        lines = ["x,y"] + [f"{i},{2 * i + 1 + (i % 3) * 0.2}" for i in range(20)]
        dataset = self.service.datasets.add(self.project.id, "a.csv", "\n".join(lines).encode())
        return self.service.training_prepare(
            self.project.id, {"dataset_id": dataset.id, "target": "y", "features": ["x"]}
        ).data["id"]

    def test_explain_uses_context_rules_and_logs(self):
        run = self.split()
        self.service.model_train(self.project.id, {"training_run_id": run})
        self.fake([text_reply("Модель добре пояснює ціль [Джерело: R²].")])
        result = self.service.assistant_ask(
            self.project.id,
            {"block": "model", "mode": "explain", "question": "Що означає R квадрат моделі?"},
        )
        self.assertIsInstance(result, Success)
        self.assertIn("[Джерело: R²]", result.data["text"])
        self.assertEqual(result.data["task"], "model_explain")
        self.assertTrue(result.data["sources"])
        body = self.requests[0]
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertIn("R² (R-квадрат)", body["instructions"])
        self.assertIn("latest_model", body["instructions"])
        self.assertEqual(
            [t["name"] for t in body["tools"]],
            ["ask_user", "analytics_schema", "query_analytics"],
        )
        event = self.store.events(self.project.id)[-1]
        self.assertEqual((event.operation, event.payload["status"]), ("assistant.asked", "success"))
        self.assertEqual(event.payload["usage_total"], 120)
        self.assertIsInstance(self.service.events(self.project.id), Success)

    def test_execute_runs_registered_tool_then_answers(self):
        run = self.split()
        self.fake([
            tool_reply("train_linear_regression", {"training_run_id": run}),
            text_reply("Модель навчено."),
        ])  # fmt: skip
        result = self.service.assistant_ask(
            self.project.id, {"block": "model", "mode": "execute", "question": "Навчи модель"}
        ).data
        self.assertEqual(result["tools"], [{"name": "train_linear_regression", "ok": True}])
        self.assertEqual((result["text"], result["steps"]), ("Модель навчено.", 2))
        self.assertEqual(len(self.service.model_runs(self.project.id).data), 1)
        self.assertIn("tools", self.requests[0])
        followup = self.requests[1]["input"]
        outputs = [i for i in followup if i.get("type") == "function_call_output"]
        self.assertEqual(outputs[0]["call_id"], "call_1")
        self.assertIn('"ok": true', outputs[0]["output"])
        self.assertEqual(result["usage_total"], 220)

    def test_unknown_tool_and_foreign_ids_are_harmless(self):
        other = self.store.create("Other")
        self.fake([
            tool_reply("delete_everything", {}),
            tool_reply("train_linear_regression",
                       {"training_run_id": "00000000-0000-4000-8000-000000000000"}, "call_2"),
            text_reply("Не вдалося."),
        ])  # fmt: skip
        result = self.service.assistant_ask(
            self.project.id, {"block": "model", "mode": "execute", "question": "Зроби щось"}
        ).data
        self.assertEqual([t["ok"] for t in result["tools"]], [False, False])
        self.assertEqual(self.service.model_runs(other.id).data, [])

    def test_step_limit_stops_with_explanation(self):
        config = AiConfiguration().model_dump()
        config["max_agent_steps"] = 2
        self.service.ai_settings.save(AiConfiguration.model_validate(config))
        self.fake([tool_reply("list_training_runs", {})])
        result = self.service.assistant_ask(
            self.project.id, {"block": "model", "mode": "execute", "question": "Перелічи"}
        )
        self.assertIsInstance(result, Failure)
        self.assertIn("ліміті кроків (2)", result.error.message)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "error")

    def test_invalid_requests_and_missing_key(self):
        for raw in (
            {"block": "nope", "mode": "advise", "question": "x"},
            {"block": "eda", "mode": "nope", "question": "x"},
            {"block": "eda", "mode": "advise", "question": ""},
        ):
            self.assertIsInstance(self.service.assistant_ask(self.project.id, raw), Failure)
        self.service.openai = OpenAIClient(
            self.database, self.service.ai_settings, key=lambda: None
        )
        missing = self.service.assistant_ask(
            self.project.id, {"block": "eda", "mode": "advise", "question": "Що далі?"}
        )
        self.assertIn(".env", missing.error.message)

    def test_http(self):
        self.fake([text_reply("Почніть з огляду пропусків.")])
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            response = client.post(
                f"/api/projects/{self.project.id}/assistant",
                json={"block": "eda", "mode": "advise", "question": "Що далі?"},
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["data"]["task"], "eda_advice")

    def test_asks_clarifying_question_instead_of_acting(self):
        run = self.split()
        self.fake([
            tool_reply("ask_user", {"question": "Навчити модель на вибірці a.csv?",
                                    "options": ["Так", "Ні, іншу"]}),
        ])  # fmt: skip
        result = self.service.assistant_ask(
            self.project.id, {"block": "model", "mode": "execute", "question": "Навчи модель"}
        ).data
        self.assertEqual(result["question"]["options"], ["Так", "Ні, іншу"])
        self.assertEqual(result["text"], "Навчити модель на вибірці a.csv?")
        self.assertEqual(result["tools"], [])
        self.assertEqual(self.service.model_runs(self.project.id).data, [])
        self.assertEqual(self.store.events(self.project.id)[-1].payload["status"], "clarifying")
        self.assertIn("ask_user", self.requests[0]["instructions"])
        self.fake([
            tool_reply("train_linear_regression", {"training_run_id": run}),
            text_reply("Готово."),
        ])  # fmt: skip
        history = [
            {"role": "user", "text": "Навчи модель"},
            {"role": "assistant", "text": "Навчити модель на вибірці a.csv?"},
        ]
        done = self.service.assistant_ask(
            self.project.id,
            {"block": "model", "mode": "execute", "question": "Так", "history": history},
        ).data
        self.assertEqual(done["text"], "Готово.")
        sent = self.requests[0]["input"]
        self.assertEqual([m["role"] for m in sent[:3]], ["user", "assistant", "user"])
        self.assertEqual(len(self.service.model_runs(self.project.id).data), 1)

    def test_page_context_reaches_the_model(self):
        lines = ["area,price"] + [f"{i},{3 * i}" for i in range(10)]
        dataset = self.service.datasets.add(self.project.id, "flats.csv", "\n".join(lines).encode())
        other = self.store.create("Other")
        foreign = self.service.datasets.add(other.id, "secret.csv", b"zzz,w\n1,2\n")
        self.fake([text_reply("Подивіться розподіл ціни.")])
        self.service.assistant_ask(
            self.project.id,
            {"block": "eda", "mode": "advise", "question": "Що далі?",
             "page": {"dataset_id": dataset.id, "tool": "histogram"}},
        )  # fmt: skip
        instructions = self.requests[0]["instructions"]
        self.assertIn("Поточна сторінка", instructions)
        self.assertIn("flats.csv", instructions)
        self.assertIn('"area"', instructions)
        self.assertIn("histogram", instructions)
        self.fake([text_reply("ok")])
        self.service.assistant_ask(
            self.project.id,
            {"block": "eda", "mode": "advise", "question": "Що далі?",
             "page": {"dataset_id": foreign.id}},
        )  # fmt: skip
        self.assertNotIn("zzz", self.requests[0]["instructions"])

    def test_dataset_facts_match_known_statistics_and_union(self):
        facts = Assistant(self.service).dataset_facts(
            self.project.id, {"dataset_id": self.dataset.id}
        )
        columns = {c["name"]: c for c in facts["columns"]}
        self.assertEqual(facts["rows_with_missing"], 2)
        self.assertAlmostEqual(facts["rows_with_missing_percent"], 200 / 3)
        self.assertEqual(columns["x"]["missing"], 1)
        self.assertAlmostEqual(columns["x"]["missing_percent"], 100 / 3)
        self.assertEqual(columns["x"]["statistics"]["mean"], 2)
        self.assertEqual(columns["x"]["statistics"]["median"], 2)
        self.assertEqual(columns["x"]["statistics"]["count"], 2)
        self.assertNotIn("statistics", columns["s"])
        self.assertEqual(facts["sha256"], self.dataset.sha256)

    def test_author_missing_code_and_invalid_code_are_not_zero(self):
        dataset = self.service.datasets.add(self.project.id, "codes.csv", b"x\n1\n-99\n3\n")
        metadata = self.service.metadata(dataset.id).data.document.model_dump()
        metadata["columns"][0]["missing_codes"] = ["-99"]
        self.service.save_metadata(dataset.id, {"document": metadata, "confirm": True})
        assistant = Assistant(self.service)
        facts = assistant.dataset_facts(self.project.id, {"dataset_id": dataset.id})
        self.assertEqual(facts["rows_with_missing"], 1)
        self.assertEqual(facts["columns"][0]["additional_missing"], 1)
        self.assertEqual(facts["columns"][0]["statistics"]["mean"], 2)
        self.assertEqual(facts["metadata_state"], "confirmed")
        metadata["columns"][0]["missing_codes"] = ["invalid"]
        self.service.save_metadata(dataset.id, {"document": metadata, "confirm": False})
        unknown = assistant.dataset_facts(self.project.id, {"dataset_id": dataset.id})
        self.assertIsNone(unknown["rows_with_missing"])
        self.assertIsNone(unknown["columns"][0]["missing"])
        self.assertIsNone(unknown["columns"][0]["missing_percent"])
        self.assertTrue(unknown["columns"][0]["error"])

    def test_facts_reject_foreign_or_missing_selection_without_calculation(self):
        from unittest.mock import patch

        foreign = self.store.create("Other")
        dataset = self.service.datasets.add(foreign.id, "private.csv", b"secret\n100\n")
        with patch.object(self.service.statistics_store, "calculate") as calculate:
            assistant = Assistant(self.service)
            self.assertEqual(assistant.dataset_facts(self.project.id, {})["status"], "not_selected")
            facts = assistant.dataset_facts(self.project.id, {"dataset_id": dataset.id})
            self.assertEqual(facts["status"], "unavailable")
            self.assertNotIn("secret", json.dumps(facts))
            calculate.assert_not_called()

    def test_facts_report_damaged_source_instead_of_statistics(self):
        path = self.service.datasets.files_root / self.dataset.stored_path
        path.write_bytes(b"x,s\n999,changed\n")
        facts = Assistant(self.service).dataset_facts(
            self.project.id, {"dataset_id": self.dataset.id}
        )
        self.assertEqual(facts["status"], "unavailable")
        self.assertNotIn("columns", facts)
        self.assertIn("пошкоджений", facts["message"])

    def test_eda_and_preparation_send_parseable_facts_without_table_rows(self):
        for block in ("eda", "preparation"):
            self.fake([text_reply("У x є один пропуск.")])
            self.service.assistant_ask(
                self.project.id,
                {
                    "block": block,
                    "mode": "advise",
                    "question": "Які пропуски?",
                    "page": {"dataset_id": self.dataset.id},
                },
            )
            instructions = self.requests[0]["instructions"]
            context = json.loads(instructions.split("Контекст проєкту (JSON):\n")[1])
            self.assertEqual(context["dataset_facts"]["rows_with_missing"], 2)
            self.assertNotIn('"top"', json.dumps(context["dataset_facts"]))
            self.assertNotIn(
                "dataset_facts", Assistant(self.service).context(self.project.id, "model")
            )

    def test_large_context_stays_valid_and_reports_omissions(self):
        context = {
            "project": {"name": "Test"},
            "dataset_facts": {
                "row_count": 100,
                "columns": [{"name": "field" + str(i), "mean": i} for i in range(100)],
            },
        }
        packed = _context_json(context, 600)
        self.assertLessEqual(len(packed), 600)
        decoded = json.loads(packed)
        self.assertEqual(decoded["dataset_facts"]["row_count"], 100)
        self.assertEqual(
            len(decoded["dataset_facts"]["columns"])
            + decoded["omitted_items"]["dataset_facts.columns"],
            100,
        )
        self.assertEqual(len(context["dataset_facts"]["columns"]), 100)
