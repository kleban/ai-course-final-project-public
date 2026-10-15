"""Stage 05 end-to-end: derived dataset, explicit split and train-only transformations."""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8903"
SHOTS = ROOT / "tmp" / "ui-smoke" / "060-training"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


with httpx.Client(base_url=BASE) as api:
    project = api.post("/api/projects", json={"name": "Підготовка до регресії"}).json()["data"][
        "id"
    ]
    csv = b"x,y,cat\n0,1,A\n1,3,B\n,5,A\n3,7,B\n4,9,A\n5,11,B\n6,13,A\n7,15,B\n8,17,A\n9,19,B\n"
    dataset = api.post(
        f"/api/projects/{project}/datasets", files={"file": ("training.csv", csv, "text/csv")}
    ).json()["data"]["id"]
    raw = {"dataset_id": dataset, "rules": [{"field": "x", "action": "drop"}]}
    preview = api.post(f"/api/projects/{project}/preparation/preview", json=raw).json()["data"]
    plan = api.post(
        f"/api/projects/{project}/preparation/plans",
        json={**raw, "preview_token": preview["preview_token"]},
    ).json()["data"]["plan_id"]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": 1400, "height": 1000})

    def local_only(route):
        if urlparse(route.request.url).hostname == urlparse(BASE).hostname:
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route("**/*", local_only)
    page = context.new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    page.goto(f"{BASE}/?project={project}&dataset={dataset}&view=preparation")
    page.get_by_text("План ·", exact=False).click()
    page.get_by_role("button", name="Створити підготовлений набір", exact=True).click()
    page.get_by_text("підготовлений набір · оригінал збережено", exact=False).wait_for()
    ok("apply opens independent dataset", f"dataset={dataset}" not in page.url)
    with httpx.Client(base_url=BASE) as api:
        output = api.post(f"/api/projects/{project}/preparation/plans/{plan}/apply").json()["data"]
        ok("derived nine rows", output["row_count"] == 9)
        ok(
            "original ten rows",
            api.get(f"/api/datasets/{dataset}").json()["data"]["row_count"] == 10,
        )
        ok(
            "idempotent derived output",
            len(api.get(f"/api/projects/{project}/datasets").json()["data"]) == 2,
        )
        ok(
            "metadata confirmation reset",
            api.get(f"/api/datasets/{output['id']}/metadata").json()["data"]["confirmed_at"]
            is None,
        )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "prepared-dataset.png"), full_page=True)
    page.goto(f"{BASE}/?project={project}&dataset={dataset}&view=training")
    page.get_by_role("button", name="Підготувати й зберегти вибірки", exact=True).wait_for()
    ok("training screen visible", page.get_by_text("Вибірки для регресії", exact=True).count() == 1)
    ok(
        "default counts seven three",
        page.get_by_text("Рядків: 10 · Навчальна частина: 7 · Тестова: 3", exact=True).count() == 1,
    )
    with httpx.Client(base_url=BASE) as api:
        ok(
            "opening creates no run",
            api.get(f"/api/projects/{project}/training").json()["data"] == [],
        )
    page.get_by_role("button", name="Підготувати й зберегти вибірки", exact=True).click()
    page.get_by_text("Оберіть цільову змінну та хоча б одну ознаку.", exact=True).wait_for()
    ok("explicit target and features required", True)
    page.get_by_label("Цільова змінна", exact=True).click()
    page.get_by_role("option", name="y", exact=True).click()
    page.wait_for_timeout(500)
    # History 143: split fields default to every field except the target.
    page.get_by_label("Кодування категорій", exact=True).click()
    page.get_by_role(
        "option", name="One-hot · окрема ознака для кожної категорії", exact=True
    ).click()
    page.get_by_label("Числові пропуски", exact=True).click()
    page.get_by_role("option", name="Середнім навчальної частини", exact=True).click()
    page.get_by_role("button", name="Підготувати й зберегти вибірки", exact=True).click()
    page.get_by_text("Вибірки підготовлено й збережено для моделі.", exact=True).wait_for(
        timeout=30000
    )
    ok("training saved through UI", True)
    ok(
        "train test encoded dimension shown",
        page.get_by_text(
            "Навчальних рядків: 7 · Тестових: 3 · Ознак після кодування: 2", exact=True
        ).count()
        >= 1,
    )
    ok(
        "train imputation value shown",
        page.get_by_text("x · Значення заповнення з навчальної частини:", exact=False).count() >= 1,
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "training-result.png"), full_page=True)
    with httpx.Client(base_url=BASE) as api:
        runs = api.get(f"/api/projects/{project}/training").json()["data"]
        ok("one run stored", len(runs) == 1)
        doc = runs[0]["document"]
        ok(
            "split positions complete and disjoint",
            not set(doc["train_positions"]) & set(doc["test_positions"])
            and sorted(doc["train_positions"] + doc["test_positions"]) == list(range(10)),
        )
        ok(
            "explicit choices persisted",
            doc["choices"]["encoding"] == "onehot" and doc["choices"]["numeric_missing"] == "mean",
        )
    page.reload()
    page.get_by_text("y ·", exact=False).wait_for()
    ok("saved preparation survives reload", True)
    page.get_by_label("Тестова частина, %", exact=True).fill("40")
    page.get_by_text("Рядків: 10 · Навчальна частина: 6 · Тестова: 4", exact=True).wait_for()
    ok("test share adjustable", True)
    page.set_viewport_size({"width": 390, "height": 800})
    page.reload()
    page.get_by_role("button", name="Підготувати й зберегти вибірки", exact=True).wait_for()
    page.wait_for_timeout(300)
    ok(
        "mobile page contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    ok(
        "mobile button visible",
        page.get_by_role("button", name="Підготувати й зберегти вибірки", exact=True).count() == 1,
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "training-mobile.png"), full_page=True)
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
