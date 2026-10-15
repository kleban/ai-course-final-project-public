"""Stage 09 piece: AI assistant panel is quiet by default and easy to open.
python tools/ui/assistant_ui_smoke.py http://127.0.0.1:8908  (server without a key in .env)
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8908"
SHOTS = ROOT / "tmp" / "ui-smoke" / "073-assistant"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


with httpx.Client(base_url=BASE) as api:
    project = api.post("/api/projects", json={"name": "Помічник"}).json()["data"]["id"]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": 1400, "height": 900})

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
    page.goto(f"{BASE}/?project={project}&view=model")
    fab = page.get_by_role("button", name="Відкрити AI-помічника")
    fab.wait_for()
    ok("corner button visible", fab.is_visible())
    ok("panel closed by default", not page.get_by_text("AI-помічник · Модель").is_visible())
    page.screenshot(path=str(SHOTS / "closed.png"))
    fab.click()
    page.get_by_text("AI-помічник · Модель").wait_for()
    ok("panel opens with block context", True)
    ok("panel states the page", page.get_by_text("Бачу сторінку: Модель", exact=False).count() == 1)
    ok(
        "explain is default on model",
        page.locator(".assistant-modes [aria-pressed=true]").inner_text() == "Поясни результат",
    )
    page.get_by_role("button", name="Поясни результат моделі").click()
    page.wait_for_function(
        "document.querySelector('.assistant-input textarea').value.length > 0", timeout=5000
    )
    ok(
        "chip fills the task",
        page.locator(".assistant-input textarea").input_value() == "Поясни результат моделі",
    )
    page.get_by_role("button", name="Надіслати").click()
    page.get_by_text("Ключ OpenAI не знайдено у файлі .env", exact=False).wait_for(timeout=20000)
    ok("missing key explained in panel", True)
    ok("question shown in thread", page.locator(".assistant-msg.user").count() == 1)
    page.get_by_role("button", name="Нова розмова").click()
    page.wait_for_timeout(300)
    ok("new conversation clears thread", page.locator(".assistant-msg").count() == 0)
    shadow = page.evaluate("getComputedStyle(document.querySelector('.q-drawer--right')).boxShadow")
    ok("open panel stands out with a shadow", shadow not in ("", "none"))
    page.screenshot(path=str(SHOTS / "open.png"))
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
    ok("escape closes panel", not page.get_by_text("AI-помічник · Модель").is_visible())
    with httpx.Client(base_url=BASE) as api:
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        ok("request logged", any(e["operation"] == "assistant.asked" for e in events))
    page.goto(f"{BASE}/?project={project}")
    page.get_by_role("button", name="З чого почати аналіз?").click()
    page.locator(".assistant-drawer").get_by_text(
        "Ключ OpenAI не знайдено у файлі .env", exact=False
    ).wait_for(timeout=20000)
    ok(
        "page shortcut opens panel and sends prepared request",
        page.locator(".assistant-msg.user").inner_text() == "З чого почати аналіз цього проєкту?",
    )
    page.screenshot(path=str(SHOTS / "shortcut.png"))
    page.goto(f"{BASE}/settings/ai")
    page.wait_for_timeout(800)
    ok(
        "no panel outside projects",
        page.get_by_role("button", name="Відкрити AI-помічника").count() == 0,
    )
    page.set_viewport_size({"width": 390, "height": 800})
    page.goto(f"{BASE}/?project={project}&view=eda")
    page.get_by_role("button", name="Відкрити AI-помічника").click()
    page.get_by_text("AI-помічник · Оглядовий аналіз").wait_for()
    page.wait_for_timeout(300)
    ok(
        "mobile contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    page.screenshot(path=str(SHOTS / "mobile.png"))
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
