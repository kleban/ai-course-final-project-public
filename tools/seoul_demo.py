"""Reproducible Seoul demo through shared services and project SQL, in an isolated database.

Outputs verified facts and existing service charts. No OpenAI calls or runtime data changes.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story.contracts import Failure  # noqa: E402
from data_story.projects import ProjectStore  # noqa: E402
from data_story.service import ProjectService  # noqa: E402


def data(result):
    if isinstance(result, Failure):
        raise RuntimeError(result.error.message)
    return result.data


def main():
    folder = ROOT / "tmp" / "seoul-demo"
    folder.mkdir(parents=True, exist_ok=True)
    workspace = ROOT / "tmp" / "seoul-demo-115"
    workspace.mkdir(parents=True, exist_ok=True)
    service = ProjectService(ProjectStore(workspace / "data.sqlite3"))
    project = service.store.create("Seoul · демонстрація аналітики", project_type="regression")
    source = ROOT / "samples" / "seoul_bike.csv"
    dataset = service.datasets.add(project.id, source.name, source.read_bytes())
    table = next(
        t for t in data(service.analytics_schema(project.id)) if t["object_id"] == dataset.id
    )
    alias = table["alias"]
    active = "\"Functioning Day\"='Yes'"
    hourly_sql = f'SELECT "Hour" AS hour, avg("Rented Bike Count") AS mean, count(*) AS n FROM {alias} WHERE {active} GROUP BY "Hour" ORDER BY "Hour"'
    seasonal_sql = f'SELECT "Seasons" AS season, avg("Rented Bike Count") AS mean, count(*) AS n FROM {alias} WHERE {active} GROUP BY "Seasons" ORDER BY mean DESC'
    hourly = data(service.analytics_query(project.id, {"sql": hourly_sql}))
    seasonal = data(service.analytics_query(project.id, {"sql": seasonal_sql}))
    original = pd.read_csv(source)
    opened = original[original["Functioning Day"] == "Yes"]
    expected = opened.groupby("Hour")["Rented Bike Count"].agg(["mean", "count"])
    np.testing.assert_allclose([r[1] for r in hourly["rows"]], expected["mean"], rtol=1e-12)
    assert [r[2] for r in hourly["rows"]] == expected["count"].tolist()
    assert len(opened) == 8465 and len(original) == 8760
    assert max(hourly["rows"], key=lambda r: r[1])[0] == 18
    clean_sql = f'SELECT * REPLACE (CAST("Hour" AS VARCHAR) AS "Hour") FROM {alias} WHERE {active}'
    clean = service.analytical_queries.save_summary(
        project.id, clean_sql, 10000, "Відкритий прокат"
    )
    metadata = service.metadata_store.get(clean.id).document
    next(c for c in metadata.columns if c.name == "Hour").kind = "category"
    service.metadata_store.save(clean.id, metadata.model_dump(), confirm=True)
    model_run = data(
        service.training_prepare(
            project.id,
            {
                "dataset_id": clean.id,
                "target": "Rented Bike Count",
                "features": [
                    "Hour",
                    "Temperature(°C)",
                    "Humidity(%)",
                    "Wind speed (m/s)",
                    "Seasons",
                ],
                "encoding": "onehot",
                "test_percent": 30,
                "seed": 42,
            },
        )
    )
    model = data(service.model_train(project.id, {"training_run_id": model_run["id"]}))
    for name, result in (
        (
            "distribution",
            service.histogram(
                project.id,
                {
                    "dataset_id": clean.id,
                    "field": "Rented Bike Count",
                    "bins": 25,
                    "title": "Розподіл кількості прокатів за відкритої системи",
                },
            ),
        ),
        (
            "temperature",
            service.scatter(
                project.id,
                {
                    "dataset_id": clean.id,
                    "x": "Temperature(°C)",
                    "y": "Rented Bike Count",
                    "title": "Температура і прокати: зв’язок не доводить причину",
                },
            ),
        ),
    ):
        chart = data(result)
        (folder / f"{name}.svg").write_text(chart.svg, encoding="utf-8")
    model_chart = data(
        service.model_check(
            project.id,
            {
                "model_id": model["id"],
                "chart": "actual",
                "title": "Фактичні та розраховані прокати · тестова вибірка",
            },
        )
    )
    (folder / "model.svg").write_text(model_chart["svg"], encoding="utf-8")
    facts = {
        "source": "UCI Seoul Bike Sharing Demand · DOI 10.24432/C5F62R · CC BY 4.0",
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(original),
        "open_hours": len(opened),
        "closed_hours": len(original) - len(opened),
        "hourly": hourly,
        "seasonal": seasonal,
        "sql": {"hourly": hourly_sql, "seasonal": seasonal_sql, "model_data": clean_sql},
        "model": model,
        "training": model_run,
        "limitations": [
            "Спостереження за годинами мають часову залежність.",
            "Випадковий поділ 70/30 показує зв’язки; він не перевіряє прогноз майбутнього.",
            "Лінійна регресія може повертати від’ємні прогнози кількості.",
            "Вибір ознак і параметрів — демонстраційна пропозиція агента, не затверджена методика автора.",
        ],
        "workspace": str(workspace),
        "project_id": project.id,
        "dataset_id": clean.id,
    }
    (folder / "facts.json").write_text(
        json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(hourly["rows"], columns=hourly["columns"]).to_csv(
        folder / "hourly.csv", index=False
    )
    pd.DataFrame(seasonal["rows"], columns=seasonal["columns"]).to_csv(
        folder / "seasonal.csv", index=False
    )
    shutil.copyfile(ROOT / "src/data_story/assets/fonts/OpenSans.ttf", folder / "OpenSans.ttf")
    page = """<!doctype html><html lang="uk"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Story · Seoul</title><style>
@font-face{font-family:OpenSans;src:url(OpenSans.ttf)}*{box-sizing:border-box}body{margin:0;font-family:OpenSans,sans-serif;color:#202124;background:white}main{max-width:1250px;margin:auto;padding:26px}header{border-bottom:2px solid #1B4F8A;display:flex;justify-content:space-between;align-items:center;padding-bottom:16px;color:#1B4F8A}h1{font-size:24px;margin:0}nav{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}button{font:inherit;cursor:pointer;background:white;color:#1B4F8A;border:1px solid #1B4F8A;padding:9px 18px}button[aria-pressed=true]{background:#1B4F8A;color:white}figure{margin:0}img{width:100%;max-height:640px;object-fit:contain}figcaption{font-size:16px;line-height:1.6;margin:20px 0}footer{font-size:13px;color:#7A7F86;border-top:1px solid #ddd;padding-top:16px}a{color:#1B4F8A}
</style><main><header><h1>Data Story · прокат велосипедів у Сеулі</h1><span>Перевірена демонстрація</span></header><nav aria-label="Графік">
<button data-view="hourly" aria-pressed="true">За годинами</button><button data-view="seasonal" aria-pressed="false">За сезонами</button><button data-view="distribution" aria-pressed="false">Розподіл</button><button data-view="temperature" aria-pressed="false">Температура</button><button data-view="model" aria-pressed="false">Модель</button></nav>
<figure><img id="chart" src="hourly.svg" alt="Попит за годинами"><figcaption id="caption">Середній попит о 18:00 — 1 554 прокати за годину; 353 спостереження. Розглядаємо лише години відкритої системи.</figcaption></figure>
<footer>8 760 початкових рядків · 8 465 годин відкритої системи · 295 годин закриття виключено.<br>Джерело: <a href="https://doi.org/10.24432/C5F62R">UCI Seoul Bike Sharing Demand</a> · CC BY 4.0 · 2017-12-01 — 2018-11-30. <a href="facts.json">Точні числа й походження</a>.</footer></main>
<script>const captions={hourly:'Середній попит о 18:00 — 1 554 прокати за годину; 353 спостереження. Розглядаємо лише години відкритої системи.',seasonal:'Середні за сезонами: літо 1 034, осінь 924, весна 746, зима 226. Це описові зв’язки, які не доводять причинного впливу сезону.',distribution:'Розподіл кількості прокатів за годину відкритої системи. Показано всі 8 465 спостережень.',temperature:'Кожна точка — година роботи прокату. Зв’язок із температурою може залежати також від години, сезону й інших умов.',model:'Лінійна регресія: тестовий R² = 0,635; MAE = 289 проти 516 у прогнозу середнім; RMSE = 383. Випадковий поділ 70/30 демонструє зв’язки в цих даних, а не прогноз майбутнього.'};document.querySelectorAll('button').forEach(button=>button.onclick=()=>{document.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));chart.src=button.dataset.view+'.svg';chart.alt=button.textContent;caption.textContent=captions[button.dataset.view]});</script></html>"""
    (folder / "index.html").write_text(page, encoding="utf-8")
    print(
        json.dumps(
            {
                "open_hours": len(opened),
                "peak": max(hourly["rows"], key=lambda r: r[1]),
                "metrics": model["document"]["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
