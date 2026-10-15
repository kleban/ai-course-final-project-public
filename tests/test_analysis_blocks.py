"""Analysis blocks registry: order per type, links only for implemented blocks (D-62, D-63)."""

import unittest

from data_story.analysis_blocks import BLOCKS, EDA_TOOLS, TYPE_BLOCKS, block_link, blocks_for
from data_story.project_types import PROJECT_TYPES


class AnalysisBlocksTests(unittest.TestCase):
    def test_regression_order_and_availability(self):
        blocks = blocks_for("regression")
        self.assertEqual(
            [b.title for b in blocks],
            [
                "Дані",
                "Оглядовий аналіз",
                "Підготовка даних",
                "Модель",
                "Перевірка моделі",
                "Висновки",
                "Звіт",
            ],
        )
        self.assertEqual(
            [b.key for b in blocks if b.available],
            ["data", "eda", "preparation", "model", "validation", "conclusions", "report"],
        )

    def test_links_and_registry_consistency(self):
        self.assertEqual(block_link(BLOCKS["data"], "p"), "/?project=p")
        self.assertEqual(block_link(BLOCKS["eda"], "p"), "/?project=p&view=eda")
        self.assertEqual(block_link(BLOCKS["model"], "p"), "/?project=p&view=model")
        self.assertEqual(block_link(BLOCKS["validation"], "p"), "/?project=p&view=validation")
        self.assertEqual(block_link(BLOCKS["conclusions"], "p"), "/?project=p&view=conclusions")
        self.assertEqual(set(TYPE_BLOCKS), set(PROJECT_TYPES))
        for keys in TYPE_BLOCKS.values():
            self.assertEqual(keys[:2], ["data", "eda"])
            self.assertTrue(set(keys) <= set(BLOCKS))
        self.assertEqual(
            list(EDA_TOOLS), ["statistics", "heatmap", "histogram", "scatter", "missing"]
        )
