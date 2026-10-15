"""Missing-values chart: shares from stage 03, 0-100% scale, order, pages of ten."""

import json
import tempfile
import unittest
from pathlib import Path

from data_story.chart_configuration import ChartOptions
from data_story.contracts import Failure, Success
from data_story.metadata import MetadataStore
from data_story.projects import ProjectStore
from data_story.service import ProjectService

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "statistics-melbourne-reference.json"
REFERENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


class MissingChartTests(unittest.TestCase):
    def setUp(self):
        directory = ROOT / "tmp" / "tests"
        directory.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=directory)
        self.addCleanup(self.temporary.cleanup)
        self.database = Path(self.temporary.name) / "data.sqlite3"
        self.store = ProjectStore(self.database)
        self.service = ProjectService(self.store)
        self.project = self.store.create("Missing")

    def test_melbourne_shares_order_and_pages(self):
        dataset = self.service.datasets.add(
            self.project.id, "melb.csv", (ROOT / "samples" / "melb_data.csv").read_bytes()
        )
        before = len(self.store.events(self.project.id))
        first = self.service.missing_chart(self.project.id, {"dataset_id": dataset.id})
        self.assertIsInstance(first, Success)
        data = first.data
        self.assertEqual((data.total, len(data.rows), data.start), (21, 10, 0))
        names = [row.field.name for row in data.rows]
        self.assertEqual(names[:4], ["BuildingArea", "YearBuilt", "CouncilArea", "Car"])
        for row in data.rows[:4]:
            self.assertEqual(row.missing, REFERENCE["missing"][row.field.name])
            self.assertAlmostEqual(row.percent, 100 * row.missing / 13580)
        self.assertEqual(data.rows_with_missing, REFERENCE["rows_with_missing"])
        self.assertEqual(data.svg.count("<rect"), 10)
        self.assertEqual(data.svg.count('fill="#1B4F8A"'), 1)  # one accent: the largest share
        pages = [
            self.service.missing_chart(self.project.id, {"dataset_id": dataset.id, "start": s})
            for s in (10, 20)
        ]
        self.assertEqual([len(p.data.rows) for p in pages], [10, 1])
        seen = names + [r.field.name for p in pages for r in p.data.rows]
        self.assertEqual(sorted(seen), sorted(REFERENCE["missing"]))  # 0% fields not hidden
        self.assertIsInstance(
            self.service.missing_chart(self.project.id, {"dataset_id": dataset.id, "start": 21}),
            Failure,
        )
        source = self.service.missing_chart(
            self.project.id,
            {
                "dataset_id": dataset.id,
                "options": ChartOptions(missing_order="source").model_dump(),
            },
        ).data
        self.assertEqual(source.rows[0].field.name, "Suburb")
        events = self.store.events(self.project.id)[before:]
        self.assertEqual({e.operation for e in events}, {"chart.built"})

    def test_equal_maxima_and_invalid_code(self):
        dataset = self.service.datasets.add(
            self.project.id, "t.csv", b"a,b,c\n1,,x\n,2,y\n3,4,z\n-1,5,w\n"
        )
        metadata = MetadataStore(self.database, self.service.datasets)
        document = metadata.get(dataset.id).document.model_dump()
        document["columns"][2]["missing_codes"] = []
        metadata.save(dataset.id, document)
        data = self.service.missing_chart(self.project.id, {"dataset_id": dataset.id}).data
        self.assertEqual([r.field.name for r in data.rows], ["a", "b", "c"])
        self.assertEqual(data.svg.count('fill="#1B4F8A"'), 2)  # equal maxima share the accent
        document["columns"][0]["missing_codes"] = ["abc"]
        metadata.save(dataset.id, document)
        data = self.service.missing_chart(self.project.id, {"dataset_id": dataset.id}).data
        broken = next(r for r in data.rows if r.field.name == "a")
        self.assertIsNone(broken.percent)
        self.assertIn("Код пропуску", broken.reason)
        self.assertIsNone(data.rows_with_missing)
        self.assertEqual(data.rows[-1].field.name, "a")  # undefined shares go last
