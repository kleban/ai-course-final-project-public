"""Stage 08 settings: task → OpenAI model, limits, key read only from .env and never exposed."""

import json
import tempfile
import unittest
from pathlib import Path

import test_preparation_store as fixtures
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data_story.ai_settings import (
    LONG_CONTEXT,
    TASKS,
    AiConfiguration,
    AiSettingsStore,
    Price,
    cost_of,
    format_usd,
    key_status,
)
from data_story.contracts import Failure, Success
from data_story.storage import connect
from data_story.web import install_error_handlers, router

SECRET = "sk-test-" + "x" * 30 + "ABCD"


class AiSettingsTests(unittest.TestCase):
    setUp = fixtures.PreparationStoreTests.setUp

    def test_defaults_cover_every_pipeline_task(self):
        config = AiConfiguration()
        self.assertEqual(set(config.models), set(TASKS))
        self.assertIn("model_advice", config.models)
        self.assertIn("next_steps", config.models)
        self.assertEqual(config.models["knowledge_search"], "local")

    def test_save_reset_persist_and_audit(self):
        changed = AiConfiguration().model_dump()
        changed["models"]["model_explain"] = "gpt-6.1-sol"
        changed["max_output_tokens"] = 4096
        saved = self.service.save_ai_configuration(changed)
        self.assertIsInstance(saved, Success)
        reopened = AiSettingsStore(self.database)
        self.assertEqual(reopened.model_for("model_explain"), "gpt-6.1-sol")
        self.assertEqual(reopened.get().max_output_tokens, 4096)
        self.assertEqual(self.service.reset_ai_configuration().data["models"]["model_explain"],
                         "gpt-6-luna")  # fmt: skip
        with connect(self.database) as connection:
            operations = [
                r[0]
                for r in connection.execute(
                    "SELECT operation FROM config_events WHERE subject='ai' ORDER BY occurred_at"
                )
            ]
        self.assertEqual(operations, ["ai.configuration.saved", "ai.configuration.reset"])

    def test_invalid_choices_rejected(self):
        bad = [
            {"models": {"model_explain": "gpt-unknown"}},
            {"models": {"model_explain": "text-embedding-3-small"}},
            {"models": {"knowledge_search": "gpt-6-luna"}},
            {"models": {"no_such_task": "gpt-6-luna"}},
            {"max_output_tokens": 10},
            {"max_agent_steps": 0},
            {"api_key": SECRET},
        ]
        for raw in bad:
            with self.assertRaises(ValidationError):
                AiConfiguration.model_validate(raw)
            self.assertIsInstance(self.service.save_ai_configuration(raw), Failure)
        self.assertIsInstance(
            self.service.save_ai_configuration({"models": {"model_explain": "local"}}), Success
        )

    def test_prices_defaults_persist_and_validation(self):
        config = AiConfiguration()
        self.assertEqual(set(config.prices), {"gpt-6-luna", "gpt-6.1-sol"})
        self.assertEqual(config.prices["gpt-6-luna"].output, 0.50)
        legacy = AiConfiguration.model_validate({"models": {}})  # saved before prices existed
        self.assertEqual(legacy.prices["gpt-6.1-sol"].input, 2.00)
        changed = config.model_dump()
        changed["prices"]["gpt-6-luna"]["output"] = 0.6
        changed["prices_checked"] = "2026-11-01"
        self.assertIsInstance(self.service.save_ai_configuration(changed), Success)
        reopened = AiSettingsStore(self.database).get()
        self.assertEqual((reopened.prices["gpt-6-luna"].output, reopened.prices_checked),
                         (0.6, "2026-11-01"))  # fmt: skip
        for bad in ({"gpt-6-luna": {"input": -1, "cached_input": 0, "cache_write": 0,
                                     "output": 0}},
                    {"local": {"input": 1, "cached_input": 0, "cache_write": 0, "output": 0}},
                    {"gpt-6-luna": {"input": 1}}):  # fmt: skip
            raw = AiConfiguration().model_dump() | {"prices": bad}
            self.assertIsInstance(self.service.save_ai_configuration(raw), Failure)

    def test_cost_rules(self):
        price = Price(input=1.0, cached_input=0.1, cache_write=1.25, output=4.0)
        usage = {"input_tokens": 100_000, "output_tokens": 25_000}
        self.assertAlmostEqual(cost_of(usage, price)["usd"], 0.2)
        self.assertIsNone(cost_of(usage, None)["usd"])
        self.assertIsNone(cost_of({"input_tokens": 5}, price)["usd"])
        self.assertIsNone(cost_of({"input_tokens": LONG_CONTEXT + 1, "output_tokens": 1},
                                  price)["usd"])  # fmt: skip
        self.assertEqual(cost_of({"input_tokens": 0, "output_tokens": 0}, price)["usd"], 0.0)
        self.assertEqual((format_usd(None), format_usd(0.00123), format_usd(2.5)),
                         ("невідомо", "$0.0012", "$2.50"))  # fmt: skip

    def test_key_status_from_env_masked(self):
        folder = Path(tempfile.mkdtemp(dir=Path(self.temporary.name)))
        env = folder / ".env"
        self.assertEqual(key_status(env)["state"], "missing_file")
        env.write_text("OPENAI_API_KEY=\n", encoding="utf-8")
        self.assertEqual(key_status(env)["state"], "empty")
        env.write_text("OPENAI_API_KEY=hello\n", encoding="utf-8")
        self.assertEqual(key_status(env)["state"], "suspicious")
        env.write_text(f"OPENAI_API_KEY={SECRET}\n", encoding="utf-8")
        found = key_status(env)
        self.assertEqual((found["state"], found["masked"]), ("found", "sk-…ABCD"))
        self.assertNotIn(SECRET, json.dumps(found))
        self.service.ai_settings.env_file = env
        payload = json.dumps(self.service.ai_configuration().data, ensure_ascii=False)
        self.assertNotIn(SECRET, payload)
        self.assertIn("sk-…ABCD", payload)

    def test_http_endpoints(self):
        app = FastAPI()
        app.include_router(router(self.service))
        install_error_handlers(app)
        with TestClient(app) as client:
            data = client.get("/api/configuration/ai").json()["data"]
            self.assertEqual(len(data["tasks"]), len(TASKS))
            self.assertIn("key", data)
            body = data["configuration"]
            body["models"]["next_steps"] = "gpt-6-luna"
            self.assertEqual(client.put("/api/configuration/ai", json=body).status_code, 200)
            body["api_key"] = SECRET
            self.assertEqual(client.put("/api/configuration/ai", json=body).status_code, 422)
            self.assertEqual(client.post("/api/configuration/ai/reset").status_code, 200)
