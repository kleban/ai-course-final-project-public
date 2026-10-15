"""Stage 06 end-to-end: explicit split choice, LinearRegression training and saved result.

Known answer: y = 3 + 2·x1 − 1.5·x2 + 4·[cat = B] without noise, so the test R² is 1 and the
coefficients are recovered exactly (the one-hot pair differs by 4). Run against an isolated
server: python tools/ui/model_ui_smoke.py http://127.0.0.1:8903
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8903"
SHOTS = ROOT / "tmp" / "ui-smoke" / "063-model"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


rows = ["x1,x2,cat,y"]
for i in range(30):
    x2, cat = (i * 7) % 5, "B" if i % 3 == 0 else "A"
    rows.append(f"{i},{x2},{cat},{3 + 2 * i - 1.5 * x2 + (4 if cat == 'B' else 0)}")

with httpx.Client(base_url=BASE, timeout=60) as api:
    project = api.post("/api/projects", json={"name": "Модель регресії"}).json()["data"]["id"]

    def upload(name: str, text: str) -> str:
        files = {"file": (name, text.encode(), "text/csv")}
        return api.post(f"/api/projects/{project}/datasets", files=files).json()["data"]["id"]

    dataset = upload("model.csv", "\n".join(rows) + "\n")
    good = api.post(
        f"/api/projects/{project}/training",
        json={
            "dataset_id": dataset,
            "target": "y",
            "features": ["x1", "x2", "cat"],
            "encoding": "onehot",
        },
    ).json()["data"]["id"]
    noisy_rows = ["signal,noise,y"]
    for i in range(80):
        signal, noise = (i * 37) % 23, (i * 11) % 7
        wobble = ((i * 53) % 9) - 4
        noisy_rows.append(f"{signal},{noise},{5 + 2 * signal + wobble}")
    noisy_set = upload("noisy.csv", "\n".join(noisy_rows) + "\n")
    noisy = api.post(
        f"/api/projects/{project}/training",
        json={"dataset_id": noisy_set, "target": "y", "features": ["signal", "noise"]},
    ).json()["data"]["id"]
    small = upload("small.csv", "a,b,y\n1,2,3\n2,1,4\n3,5,1\n4,4,2\n5,3,8\n")
    tiny = api.post(
        f"/api/projects/{project}/training",
        json={"dataset_id": small, "target": "y", "features": ["a", "b"]},
    ).json()["data"]["id"]

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

    page.goto(f"{BASE}/?project={project}")
    link = page.locator(".block-link", has_text="Модель").first
    ok("model block linked in navigation", "planned" not in (link.get_attribute("class") or ""))
    link.click()
    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).wait_for()
    ok("model screen opened from menu", "view=model" in page.url)
    ok(
        "model block marked current",
        page.locator(".block-link.current", has_text="Модель").count() == 1,
    )
    with httpx.Client(base_url=BASE) as api:
        ok(
            "opening trains nothing",
            api.get(f"/api/projects/{project}/models").json()["data"] == [],
        )
    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).click()
    page.get_by_text("Оберіть збережені вибірки й хоча б одну ознаку.", exact=True).wait_for()
    ok("explicit split selection required", True)

    page.get_by_label("Збережені вибірки", exact=True).click()
    page.get_by_role("option", name="model.csv · ціль y", exact=False).click()
    page.get_by_text("Навчальних рядків: 21 · Тестових: 9 · Стовпців після кодування: 3").wait_for()
    ok("split summary shown before training", True)
    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).click()
    page.get_by_text("Модель навчено й збережено.", exact=True).wait_for(timeout=30000)
    ok("model trained through UI", True)
    ok("test R2 equals one", page.locator(".model-kpis .kpi-value").first.inner_text() == "1")
    ok("intercept row shown", page.get_by_text("Вільний член (a₀)", exact=True).count() >= 1)
    ok("coefficient names shown", page.get_by_text("cat = B", exact=True).count() >= 1)
    ok(
        "metric explanation shown",
        page.get_by_text("MAE ≈ 0: у середньому", exact=False).count() >= 1,
    )
    ok(
        "baseline category instead of collinearity",
        page.get_by_text("Ознаки лінійно залежні", exact=False).count() == 0
        and page.get_by_text("Базові категорії", exact=False).count() >= 1,
    )
    ok(
        "limitations shown",
        page.get_by_text("не доказ причинного впливу", exact=False).count() >= 1,
    )
    ok(
        "perfect fit hides p-values with note",
        page.get_by_text("p-value ненадійні", exact=False).count() >= 1,
    )
    ok("significance note shown", page.get_by_text("як у summary() в R", exact=False).count() >= 1)
    boxes = [page.locator(".model-grid-main > .model-panel").nth(i).bounding_box() for i in (0, 1)]
    ok("desktop panels side by side", abs(boxes[0]["y"] - boxes[1]["y"]) < 2)
    ok(
        "three panels in one row",
        len({round(page.locator(".model-grid-three > .model-panel").nth(i).bounding_box()["y"])
             for i in range(3)}) == 1,
    )  # fmt: skip
    page.wait_for_selector("[data-formula][data-rendered='ok']", timeout=15000)
    ok("formulas rendered locally", page.locator("[data-rendered='ok']").count() >= 4)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "model-result.png"), full_page=True)

    with httpx.Client(base_url=BASE) as api:
        models = api.get(f"/api/projects/{project}/models").json()["data"]
        ok("one model stored", len(models) == 1)
        doc = models[0]["document"]
        values = {c["name"]: c["value"] for c in doc["coefficients"]}
        ok("x1 coefficient 2", abs(values["x1"] - 2) < 1e-8)
        ok("x2 coefficient -1.5", abs(values["x2"] + 1.5) < 1e-8)
        ok(
            "category shift vs baseline 4",
            abs(values["cat = B"] - 4) < 1e-8 and "cat = A" not in values,
        )
        ok("linked to chosen split", doc["training_run_id"] == good)
        ok("fit on train only recorded", doc["train_count"] == 21 and doc["test_count"] == 9)

    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).click()
    page.get_by_text("Показано збережену модель з цими ознаками.", exact=True).wait_for()
    with httpx.Client(base_url=BASE) as api:
        ok(
            "repeat does not duplicate",
            len(api.get(f"/api/projects/{project}/models").json()["data"]) == 1,
        )

    page.goto(f"{BASE}/?project={project}&view=model&run={noisy}")
    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).click()
    page.get_by_text("Модель навчено й збережено.", exact=True).wait_for(timeout=30000)
    row = page.locator(".coefs .trow", has_text="signal").first
    ok("p-value and stars shown", row.locator(".stars").inner_text() == "***")
    ok("p-value column present", page.get_by_text("p-value", exact=True).count() >= 1)
    with httpx.Client(base_url=BASE) as api:
        noisy_doc = next(
            m["document"]
            for m in api.get(f"/api/projects/{project}/models").json()["data"]
            if m["training_run_id"] == noisy
        )
        by_name = {c["name"]: c for c in noisy_doc["coefficients"]}
        ok("signal significant", by_name["signal"]["p_value"] < 0.001)
        ok("noise not significant", by_name["noise"]["p_value"] > 0.05)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "model-p-values.png"), full_page=True)

    page.goto(f"{BASE}/?project={project}&view=model&run={tiny}")
    page.get_by_role("button", name="Навчити лінійну регресію", exact=True).click()
    page.get_by_text("Замало навчальних рядків", exact=False).wait_for(timeout=30000)
    ok("unsuitable split refused with explanation", True)

    page.goto(f"{BASE}/?project={project}&view=model&run={good}")
    page.get_by_text("Для цих ознак є збережена модель", exact=False).wait_for()
    ok(
        "saved result survives reload",
        page.get_by_text("Вільний член (a₀)", exact=True).count() >= 1,
    )
    ok(
        "saved models listed",
        page.get_by_text("Збережені моделі проєкту", exact=True).count() == 1,
    )

    page.goto(f"{BASE}/?project={project}&dataset={dataset}&view=training")
    page.get_by_text("y ·", exact=False).first.click()
    page.get_by_role("button", name="Навчити модель на цих вибірках").first.wait_for()
    ok(
        "training screen links to model",
        page.get_by_role("button", name="Навчити модель на цих вибірках").count() >= 1,
    )

    page.goto(f"{BASE}/?project={project}")
    page.get_by_text("Навчено лінійну регресію", exact=False).first.wait_for()
    ok(
        "journal shows success and error",
        page.get_by_text("Навчено лінійну регресію: model.csv").count() >= 1
        and page.get_by_text("Навчено лінійну регресію: small.csv · помилка").count() >= 1,
    )

    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(f"{BASE}/?project={project}&view=model&run={good}")
    page.get_by_text("Для цих ознак є збережена модель", exact=False).wait_for()
    page.wait_for_timeout(300)
    mobile = [page.locator(".model-grid-main > .model-panel").nth(i).bounding_box() for i in (0, 1)]
    ok("mobile panels stacked", mobile[1]["y"] > mobile[0]["y"] + 20)
    ok(
        "mobile page contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "model-mobile.png"), full_page=True)
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
