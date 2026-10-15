"""Embeddings loading: memory-lean session options, visible error text, retry after failure."""

import unittest
from unittest.mock import MagicMock, patch

from data_story import embeddings


class EmbedderLoadingTests(unittest.TestCase):
    def setUp(self):
        embeddings.reset()
        self.addCleanup(embeddings.reset)

    def test_session_is_created_with_memory_lean_options(self):
        import onnxruntime

        seen = {}

        def session(path, options, providers):
            seen.update(options=options, providers=providers)
            fake = MagicMock()
            fake.get_inputs.return_value = []
            fake.get_outputs.return_value = []
            return fake

        with (
            patch.object(onnxruntime, "InferenceSession", side_effect=session),
            patch("tokenizers.Tokenizer.from_file", return_value=MagicMock()),
        ):
            embeddings.Embedder()
        options = seen["options"]
        self.assertEqual(
            options.graph_optimization_level, onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        self.assertFalse(options.enable_cpu_mem_arena)
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.get_session_config_entry("session.disable_prepacking"), "1")
        self.assertEqual(seen["providers"], ["CPUExecutionProvider"])

    def test_failure_shows_message_and_is_retried_later(self):
        error = RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation")
        ready = object()
        with (
            patch.object(embeddings, "missing", return_value=None),
            patch.object(embeddings, "Embedder", side_effect=[error, ready]) as build,
            patch.object(embeddings.time, "monotonic", side_effect=[0.0, 10.0, 200.0, 200.0]),
            self.assertLogs("data_story.embeddings", "ERROR"),
        ):
            self.assertIsNone(embeddings.embedder())
            self.assertIn("bad allocation", embeddings._state["reason"])
            self.assertIsNone(embeddings.embedder())  # within RETRY_SECONDS: cached
            self.assertIs(embeddings.embedder(), ready)  # later: loaded again
            self.assertEqual(build.call_count, 2)
            self.assertIsNone(embeddings._state["reason"])

    def test_reason_without_message(self):
        self.assertEqual(
            embeddings.failure_reason(MemoryError()),
            "Модель embeddings не запустилась: MemoryError.",
        )


if __name__ == "__main__":
    unittest.main()
