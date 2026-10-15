"""Model check block end-to-end: regression line by feature, errors, prediction vs actual.

Run against an isolated server: python tools/ui/validation_ui_smoke.py http://127.0.0.1:8904
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8904"
SHOTS = ROOT / "tmp" / "ui-smoke" / "066-validation"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


rows = ["area,rooms,price"]
for i in range(60):
    area, rooms = 40 + (i * 37) % 90, 1 + (i * 11) % 5
    rows.append(f"{area},{rooms},{20 + 1.5 * area + 8 * rooms + ((i * 53) % 9) - 4}")

with httpx.Client(base_url=BASE, timeout=60) as api:
    project = api.post("/api/projects", json={"name": "Перевірка моделі"}).json()["data"]["id"]
    files = {"file": ("flats.csv", ("\n".join(rows) + "\n").encode(), "text/csv")}
    dataset = api.post(f"/api/projects/{project}/datasets", files=files).json()["data"]["id"]
    run = api.post(
        f"/api/projects/{project}/training",
        json={"dataset_id": dataset, "target": "price", "features": ["area", "rooms"]},
    ).json()["data"]["id"]
    model = api.post(f"/api/projects/{project}/models", json={"training_run_id": run}).json()[
        "data"
    ]["id"]

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

    page.goto(f"{BASE}/?project={project}&view=model&run={run}")
    page.get_by_role("button", name="Перевірити модель: графіки").first.click()
    page.locator("svg.diagnostic").wait_for(timeout=30000)
    ok("model page links to check", "view=validation" in page.url and model in page.url)
    ok(
        "check block current in menu",
        page.locator(".block-link.current", has_text="Перевірка моделі").count() == 1,
    )
    ok("regression line drawn", page.locator("svg .diagnostic-model").count() == 1)
    ok("first feature on x axis", page.locator("svg.diagnostic text", has_text="area").count() >= 1)
    page.wait_for_selector(".model-equation[data-rendered='ok']", timeout=15000)
    ok("equation rendered", page.locator(".model-equation svg").count() == 1)
    ok("equation text shown", page.get_by_text("ŷ(price) =", exact=False).count() >= 1)
    ok("line equation shown", page.get_by_text("Лінія на графіку", exact=False).count() == 1)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "regression.png"), full_page=True)

    page.get_by_label("Ознака для графіка регресії").click()
    page.get_by_role("option", name="rooms", exact=True).click()
    page.wait_for_function(
        "[...document.querySelectorAll('svg.diagnostic text')].some(t => t.textContent === 'rooms')"
    )
    ok("feature switch rebuilds chart", True)
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)

    page.get_by_role("button", name="Графік похибок", exact=True).click()
    page.locator("svg .diagnostic-zero").wait_for(state="attached")
    ok("errors chart with zero line", page.locator("svg .diagnostic-model").count() == 0)
    ok(
        "feature selector hidden for errors",
        not page.get_by_label("Ознака для графіка регресії").is_visible(),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "residuals.png"), full_page=True)

    page.get_by_role("button", name="Прогноз проти факту", exact=True).click()
    page.locator("svg .diagnostic-diagonal").wait_for(state="attached")
    ok("prediction vs actual with diagonal", True)

    with httpx.Client(base_url=BASE) as api:
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        checks = [e for e in events if e["operation"] == "model.checked"]
        ok("each build audited", len(checks) >= 4)
        ok("audit has no cell values", all("points" not in e["payload"] for e in checks))
        ok("models unchanged", len(api.get(f"/api/projects/{project}/models").json()["data"]) == 1)

    page.goto(f"{BASE}/?project={project}")
    ok(
        "journal shows chart builds",
        page.get_by_text("Побудовано графік перевірки моделі", exact=False).count() >= 1,
    )

    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(f"{BASE}/?project={project}&view=validation&run={model}")
    page.locator("svg.diagnostic").wait_for(timeout=30000)
    page.wait_for_timeout(300)
    ok(
        "mobile page contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
