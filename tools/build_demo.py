"""Fill the working database with the Seoul demo project through the same services as the UI.

History 137 (author's decision): the demo lives in the working runtime/ so that real token and
cost figures accumulate during rehearsals. Run with the server STOPPED (one process per base):

  .venv\\Scripts\\python.exe tools\\build_demo.py            # stage 1: data … model, AI conclusions draft
  .venv\\Scripts\\python.exe tools\\build_demo.py --report   # stage 2 (after the author approves the
                                                            # conclusions in the UI): report + AI text + HTML

Stage 1 backs the database up to tmp/runtime-before-demo-137/ first and refuses to create a
second Seoul project. AI steps make a few paid OpenAI calls (logged with tokens and cost).
"""

import argparse
import json
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from data_story.contracts import Failure  # noqa: E402
from data_story.projects import ProjectStore  # noqa: E402
from data_story.service import ProjectService  # noqa: E402

NAME = "Прокат велосипедів у Сеулі"
PURPOSE = """**Питання:** від чого залежить погодинний попит на міський прокат велосипедів — година доби, погода чи сезон?

**Дані:** UCI Seoul Bike Sharing Demand, 2017-12-01 — 2018-11-30, 8 760 годин, ліцензія CC BY 4.0 (DOI 10.24432/C5F62R).

**План:** відібрати години роботи системи, переглянути розподіли й зв’язки, побудувати лінійну регресію й чесно описати її межі: випадковий поділ 70/30 пояснює зв’язки, а не прогнозує майбутнє."""
SOURCE = ROOT / "samples" / "seoul_bike.csv"
DESCRIPTION = ROOT / "samples" / "seoul_bike.description.md"
OUT = ROOT / "tmp" / "demo-137"
FIELDS = {  # from samples/seoul_bike.description.md
    "Date": ("Дата", "Дата спостереження (дд/мм/рррр).", ""),
    "Rented Bike Count": (
        "Прокати за годину",
        "Кількість прокатів велосипедів за годину; ціль моделі.",
        "прокатів/год",
    ),
    "Hour": ("Година", "Година доби 0–23.", ""),
    "Temperature(°C)": ("Температура", "Температура повітря.", "°C"),
    "Humidity(%)": ("Вологість", "Відносна вологість.", "%"),
    "Wind speed (m/s)": ("Швидкість вітру", "Швидкість вітру.", "м/с"),
    "Visibility (10m)": ("Видимість", "Видимість в одиницях по 10 м.", "10 м"),
    "Dew point temperature(°C)": (
        "Точка роси",
        "Температура точки роси; близька до температури повітря.",
        "°C",
    ),
    "Solar Radiation (MJ/m2)": ("Сонячна радіація", "Сонячна радіація.", "МДж/м²"),
    "Rainfall(mm)": ("Опади", "Кількість опадів.", "мм"),
    "Snowfall (cm)": ("Сніг", "Висота снігу.", "см"),
    "Seasons": ("Сезон", "Winter, Spring, Summer, Autumn.", ""),
    "Holiday": ("Свято", "Holiday / No Holiday.", ""),
    "Functioning Day": ("Система працювала", "Yes / No: чи працювала система прокату.", ""),
}
DATASET = {
    "title": "Seoul Bike Sharing Demand",
    "description": "Погодинний попит на міський прокат велосипедів у Сеулі з погодними умовами й сезоном.",
    "source": "UCI Machine Learning Repository, DOI 10.24432/C5F62R",
    "unit_of_observation": "одна година роботи/стану системи прокату",
    "period": "2017-12-01 — 2018-11-30",
    "license": "CC BY 4.0",
}


def data(result):
    if isinstance(result, Failure):
        raise RuntimeError(result.error.message)
    return result.data


def say(text: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {text}", flush=True)


def describe(service: ProjectService, dataset_id: str, overrides: dict | None = None) -> None:
    store = service.metadata_store
    document = store.get(dataset_id).document
    for key, value in DATASET.items():
        setattr(document.dataset, key, value)
    for column in document.columns:
        if column.name in FIELDS:
            column.label, column.description, column.unit = FIELDS[column.name]
            column.source = "author"
        if column.name == "Rented Bike Count":
            column.role = "target"
        for attr, value in (overrides or {}).get(column.name, {}).items():
            setattr(column, attr, value)
    store.save(dataset_id, document.model_dump(), confirm=True)


def stage_one(service: ProjectService, database: Path, ai: bool = True) -> dict:
    if any(p.name == NAME for p in service.store.list_projects()):
        raise RuntimeError(f"Проєкт «{NAME}» уже є в базі; повторно не створюю.")
    project = service.store.create(NAME, project_type="regression")
    data(service.save_description(project.id, {"text": PURPOSE}))
    say(f"Проєкт: {project.id}")
    dataset = service.datasets.add(project.id, SOURCE.name, SOURCE.read_bytes())
    service.metadata_store.add_file(dataset.id, DESCRIPTION.name, DESCRIPTION.read_bytes())
    describe(service, dataset.id)
    say(f"Набір: {dataset.row_count} рядків, метадані підтверджено")
    alias = next(
        t for t in data(service.analytics_schema(project.id)) if t["object_id"] == dataset.id
    )["alias"]
    active = "\"Functioning Day\"='Yes'"
    clean = service.analytical_queries.save_summary(
        project.id,
        f'SELECT * REPLACE (CAST("Hour" AS VARCHAR) AS "Hour") FROM {alias} WHERE {active}',
        10000,
        "Відкритий прокат",
    )
    describe(service, clean.id, {"Hour": {"kind": "category"}})
    hourly = service.analytical_queries.save_summary(
        project.id,
        f'SELECT "Hour" AS "Година", avg("Rented Bike Count") AS "Середні прокати", count(*) AS "Годин" '
        f'FROM {alias} WHERE {active} GROUP BY "Hour" ORDER BY "Hour"',
        10000,
        "Попит за годинами",
    )
    seasonal = service.analytical_queries.save_summary(
        project.id,
        f'SELECT "Seasons" AS "Сезон", avg("Rented Bike Count") AS "Середні прокати", count(*) AS "Годин" '
        f'FROM {alias} WHERE {active} GROUP BY "Seasons" ORDER BY 2 DESC',
        10000,
        "Попит за сезонами",
    )
    say(f"Агрегації: відкритий прокат {clean.row_count} рядків; за годинами; за сезонами")
    for kind, body in (
        ("statistics", None),
        (
            "histogram",
            {
                "dataset_id": clean.id,
                "field": "Rented Bike Count",
                "bins": 25,
                "title": "Розподіл кількості прокатів за відкритої системи",
            },
        ),
        (
            "scatter",
            {
                "dataset_id": clean.id,
                "x": "Temperature(°C)",
                "y": "Rented Bike Count",
                "title": "Температура і прокати: зв’язок не доводить причину",
            },
        ),
        (
            "heatmap",
            {
                "dataset_id": clean.id,
                "fields": [
                    "Rented Bike Count",
                    "Temperature(°C)",
                    "Humidity(%)",
                    "Wind speed (m/s)",
                    "Visibility (10m)",
                    "Dew point temperature(°C)",
                    "Solar Radiation (MJ/m2)",
                    "Rainfall(mm)",
                    "Snowfall (cm)",
                ],
            },
        ),
    ):
        try:
            if kind == "statistics":
                data(
                    service.statistics(project.id, {"dataset_id": clean.id, "view": "descriptive"})
                )
            else:
                data(getattr(service, kind)(project.id, body))
        except Exception as error:  # a warm-up only; the page computes the chart itself
            say(f"Попередній розрахунок «{kind}» пропущено: {error}")
    say("Статистика й графіки розраховані")
    run = data(
        service.training_prepare(
            project.id,
            {
                "dataset_id": clean.id,
                "target": "Rented Bike Count",
                # History 146: the split takes every usable field; models choose their own
                # (by hand or stepwise). Date is a date; Functioning Day is constant here.
                "features": [
                    "Hour",
                    "Temperature(°C)",
                    "Humidity(%)",
                    "Wind speed (m/s)",
                    "Visibility (10m)",
                    "Dew point temperature(°C)",
                    "Solar Radiation (MJ/m2)",
                    "Rainfall(mm)",
                    "Snowfall (cm)",
                    "Seasons",
                    "Holiday",
                ],
                "encoding": "onehot",
                "test_percent": 30,
                "seed": 42,
            },
        )
    )
    model = data(service.model_train(project.id, {"training_run_id": run["id"]}))
    metrics = model["document"]["metrics"]
    say(f"Модель: {json.dumps(metrics, ensure_ascii=False)[:200]}")
    for chart in ("regression", "residuals", "actual"):
        try:
            data(service.model_check(project.id, {"model_id": model["id"], "chart": chart}))
        except Exception as error:
            say(f"Графік перевірки «{chart}» пропущено: {error}")
    ids = {
        "project_id": project.id,
        "dataset_id": dataset.id,
        "clean_id": clean.id,
        "summaries": [hourly.id, seasonal.id],
        "model_id": model["id"],
        "metrics": metrics,
    }
    if not ai:
        return ids
    say("AI: чернетка висновків (платний запит)…")
    conclusion = data(
        service.conclusion_prepare(
            project.id, {"model_id": model["id"], "summary_ids": [hourly.id, seasonal.id]}
        )
    )
    say(
        f"Висновки: чернетка {conclusion['id']}, токени {conclusion['ai'].get('usage_total')}, "
        f"вартість {conclusion['ai'].get('cost_total')}"
    )
    return ids | {"conclusion_id": conclusion["id"]}


def stage_two(service: ProjectService) -> dict:
    state = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    project_id, model_id = state["project_id"], state["model_id"]
    if service.conclusions.approved_for(project_id, model_id) is None:
        raise RuntimeError("Висновки ще не затверджені. Затвердьте їх у сервісі й запустіть знову.")
    report = data(
        service.report_create(
            project_id, {"model_id": model_id, "title": "Попит на прокат велосипедів у Сеулі"}
        )
    )
    say("AI: чернетка тексту звіту (платний запит)…")
    report = data(
        service.report_generate(project_id, report["id"], {"revision": report["revision"]}, ai=True)
    )
    say("Quarto: формування HTML…")
    report = data(
        service.report_generate(project_id, report["id"], {"revision": report["revision"]})
    )
    return state | {"report_id": report["id"], "report_revision": report["revision"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=ROOT / "runtime" / "data-story.sqlite3")
    parser.add_argument("--report", action="store_true", help="stage 2: report after approval")
    parser.add_argument("--no-ai", action="store_true", help="test run without OpenAI calls")
    args = parser.parse_args()
    database = args.database.resolve()
    if not database.is_relative_to(ROOT):
        print("База має бути всередині репозиторію.")
        return 1
    with socket.socket() as probe:
        if (
            probe.connect_ex(("127.0.0.1", 8765)) == 0
            and database == (ROOT / "runtime" / "data-story.sqlite3").resolve()
        ):
            print("Сервіс працює на 8765. Спершу Stop.cmd: одна база — один процес.")
            return 1
    OUT.mkdir(parents=True, exist_ok=True)
    # Backup BEFORE ProjectService opens (and creates) the base — fixed in history 138: the
    # first version copied the test base and then skipped the runtime backup.
    backup = ROOT / "tmp" / f"before-demo-137-{database.parent.name}"
    if not args.report and database.exists() and not backup.exists():
        shutil.copytree(
            database.parent, backup, ignore=shutil.ignore_patterns("dependencies", "*.lock")
        )
        say(f"Резервна копія: {backup.relative_to(ROOT)}")
    service = ProjectService(ProjectStore(database))
    result = stage_two(service) if args.report else stage_one(service, database, not args.no_ai)
    result["ai_totals"] = service.openai.totals()
    (OUT / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    say(f"Готово. Підсумок: {(OUT / 'result.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
