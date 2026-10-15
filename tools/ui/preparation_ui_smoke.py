"""Explicit preparation workflow against an isolated local test server."""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8901"
SHOTS = ROOT / "tmp" / "ui-smoke" / "059-preparation"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


with httpx.Client(base_url=BASE) as api:
    project = api.post("/api/projects", json={"name": "Перевірка підготовки"}).json()["data"]["id"]
    dataset = api.post(
        f"/api/projects/{project}/datasets",
        files={"file": ("missing.csv", b"x,s\n1,A\n,B\n3,\n", "text/csv")},
    ).json()["data"]["id"]
    url = f"{BASE}/?project={project}&dataset={dataset}&view=preparation"

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
    page.goto(url)
    page.get_by_role("button", name="Додати операцію", exact=True).wait_for()
    ok(
        "preparation navigation available",
        page.get_by_role("link", name="Підготовка даних", exact=False).count() > 0,
    )
    ok(
        "source preservation explained",
        page.get_by_text("Оригінальні дані не змінюються.", exact=False).count() == 1,
    )
    ok(
        "no save without preview",
        page.get_by_role("button", name="Підтвердити й зберегти план").count() == 0,
    )
    with httpx.Client(base_url=BASE) as api:
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        ok(
            "opening does not run preparation",
            not any(e["operation"].startswith("preparation.") for e in events),
        )
    page.get_by_role("button", name="Додати операцію", exact=True).click()
    page.get_by_role("button", name="Переглянути результат", exact=True).click()
    page.get_by_text("До: 3 · Після: 2 · Вилучено рядків: 1", exact=True).wait_for()
    ok("drop correct row count", True)
    ok("ten-row preview table", page.get_by_text("Перші 10 рядків після підготовки").count() == 1)
    ok(
        "save appears after preview",
        page.get_by_role("button", name="Підтвердити й зберегти план").count() == 1,
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "drop-preview.png"), full_page=True)
    page.get_by_role("button", name="Прибрати", exact=True).click()
    page.wait_for_timeout(300)
    ok(
        "rule edit invalidates preview",
        page.get_by_role("button", name="Підтвердити й зберегти план").count() == 0,
    )
    page.get_by_label("Поле", exact=True).click()
    page.get_by_role("option", name="s · пропусків у файлі: 1", exact=True).click()
    page.get_by_label("Операція", exact=True).click()
    page.get_by_role("option", name="Заповнити заданим значенням", exact=True).click()
    page.get_by_label("Значення для заповнення", exact=True).fill("Невідомо")
    page.get_by_role("button", name="Додати операцію", exact=True).click()
    page.get_by_role("button", name="Переглянути результат", exact=True).click()
    page.get_by_text("До: 3 · Після: 3 · Вилучено рядків: 0", exact=True).wait_for()
    ok("fill keeps row count", True)
    ok("filled cell visible", page.get_by_role("cell", name="Невідомо", exact=True).count() == 1)
    page.get_by_role("button", name="Підтвердити й зберегти план", exact=True).click()
    page.get_by_text("План підготовки збережено", exact=True).wait_for()
    ok("explicit save succeeds", True)
    page.reload()
    page.get_by_text("План ·", exact=False).wait_for()
    ok("saved plan survives reload", True)
    ok(
        "reload does not restore applied rules",
        page.get_by_role("button", name="Прибрати", exact=True).count() == 0,
    )
    page.get_by_text("План ·", exact=False).click()
    page.get_by_text("SHA-256 джерела:", exact=False).wait_for()
    ok("saved provenance visible", True)
    page.get_by_role("button", name="Використати цей план", exact=True).click()
    page.wait_for_timeout(300)
    ok(
        "saved rules restored for editing",
        page.get_by_role("button", name="Прибрати", exact=True).count() == 1,
    )
    ok(
        "restoring requires fresh preview",
        page.get_by_role("button", name="Підтвердити й зберегти план").count() == 0,
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "saved-plan.png"), full_page=True)
    with httpx.Client(base_url=BASE) as api:
        plans = api.get(f"/api/projects/{project}/preparation/plans").json()["data"]
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        ok("one saved plan", len(plans) == 1)
        ok(
            "preview audit recorded",
            sum(e["operation"] == "preparation.previewed" for e in events) == 2,
        )
        ok(
            "save audit recorded once",
            sum(e["operation"] == "preparation.saved" for e in events) == 1,
        )
        ok("source unchanged", api.get(f"/api/datasets/{dataset}").json()["data"]["row_count"] == 3)
    page.goto(f"{BASE}/?project={project}")
    page.get_by_text("Збережено план підготовки:", exact=False).wait_for()
    ok("project journal shows preparation", True)
    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(url)
    page.get_by_role("button", name="Додати операцію", exact=True).wait_for()
    ok("mobile renders", True)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
