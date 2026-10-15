"""Create the demonstration project «Ціни на житло в Мельбурні» in an empty database.

Uses the same stores as the application: project (regression), the dataset from
samples/melb_data.csv and its source description as a raw metadata source. Metadata fields
stay a draft from the data — describing them is part of the demonstration (02b, later 08).
Refuses to touch an existing database: back it up first.

Run with the server stopped:
.venv\\Scripts\\python.exe tools\\make_demo_project.py [--database runtime\\data-story.sqlite3]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from data_story.datasets import DatasetStore  # noqa: E402
from data_story.metadata import MetadataStore  # noqa: E402
from data_story.projects import ProjectStore  # noqa: E402

NAME = "Ціни на житло в Мельбурні"
PURPOSE = (
    "Демонстраційний проєкт: які характеристики житла (кімнати, площа, відстань до центру, "
    "тип, регіон) пояснюють ціну продажу; лінійна регресія на даних Melbourne Housing "
    "Snapshot (Kaggle, CC BY-NC-SA 4.0)."
)
DATA = ROOT / "samples" / "melb_data.csv"
DESCRIPTION = ROOT / "samples" / "melb_data.description.md"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "runtime" / "data-story.sqlite3")
    database = parser.parse_args().database.resolve()
    if not database.is_relative_to(ROOT):
        print("База має бути всередині репозиторію.")
        return 1
    if database.exists():
        print(f"База вже існує: {database}. Спершу перенесіть її в резервну копію.")
        return 1
    projects = ProjectStore(database)
    datasets = DatasetStore(database)
    project = projects.create(NAME, PURPOSE, "regression")
    dataset = datasets.add(project.id, DATA.name, DATA.read_bytes())
    MetadataStore(database, datasets).add_file(
        dataset.id, DESCRIPTION.name, DESCRIPTION.read_bytes()
    )
    print(f"Проєкт: {project.id}\nНабір: {dataset.id} ({dataset.row_count} рядків)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
