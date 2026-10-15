"""Stage 08: AI models page — model per task, limits, inert key field that never leaves the browser.
python tools/ui/ai_settings_ui_smoke.py http://127.0.0.1:8906
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8906"
SHOTS = ROOT / "tmp" / "ui-smoke" / "069-ai-settings"
SHOTS.mkdir(parents=True, exist_ok=True)
FAKE = "sk-should-never-reach-server-1234567890"
results, errors, external, frames = [], [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


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
    page.on("websocket", lambda ws: ws.on("framesent", lambda f: frames.append(str(f))))
    page.on("request", lambda r: frames.append(r.post_data or ""))

    page.goto(BASE)
    page.get_by_text("Моделі AI", exact=True).first.click()
    page.get_by_role("button", name="Зберегти").wait_for()
    ok("menu opens AI page", page.url.endswith("/settings/ai"))
    state = page.locator(".ai-key-state").inner_text()
    ok("key state from .env shown", any(s in state for s in ("Ключ", "Немає файла .env")))
    ok("stage panels", page.locator(".ai-stages .model-panel").count() == 6)
    ok(
        "model advice task listed",
        page.get_by_text("Рекомендації щодо моделі", exact=False).count() == 1,
    )
    ok("next steps task listed", page.get_by_text("Наступні кроки", exact=False).count() == 1)

    field = page.get_by_label("Новий ключ OpenAI (не зберігається)")
    field.fill(FAKE)
    page.get_by_role("button", name="Замінити ключ").click()
    page.get_by_text("Ключ не надіслано й не збережено", exact=False).wait_for()
    ok("fake field cleared", field.input_value() == "")

    if "Немає файла .env" in state or "не вписано" in state:
        page.get_by_role("button", name="Пробний запит").click()
        page.get_by_text("Ключ OpenAI не знайдено у файлі .env", exact=False).first.wait_for()
        ok("missing key explained, nothing sent", True)
        page.get_by_text("eda_advice", exact=False).first.wait_for()
        ok("failed call in AI log", page.get_by_text("помилок: 1", exact=False).count() == 1)
    page.locator('[data-task="model_advice"]').click()
    page.get_by_role("option", name="gpt-6-luna", exact=False).click()
    ok("three levels per task", page.locator('[data-level="model_advice-3"]').count() == 1)
    page.locator('[data-level="model_advice-2"]').click()
    page.get_by_role("option", name="—", exact=True).click()
    page.get_by_label("Повтори на одній моделі (тайм-аут, мережа, 429, 5xx)").fill("3")

    def price(model, field):
        sel = f'[data-price="{model}-{field}"]'
        return page.locator(f"{sel} input, input{sel}").first

    ok("price panel: 4 fields x 2 models", page.locator("[data-price]").count() == 8)
    ok("luna output default 0.5", float(price("gpt-6-luna", "output").input_value()) == 0.5)
    ok("sol input default 2", float(price("gpt-6.1-sol", "input").input_value()) == 2.0)
    ok("price source link", page.get_by_text("Джерело: тарифи OpenAI").count() == 1)
    ok("cost total in AI log", page.locator(".ai-cost-total").count() == 1)
    price("gpt-6-luna", "output").fill("0.6")
    page.get_by_label("Тариф перевірено (дата)").fill("2026-10-15")
    page.get_by_role("button", name="Зберегти").click()
    page.get_by_text("Налаштування збережено", exact=False).wait_for()
    with httpx.Client(base_url=BASE) as api:
        saved = api.get("/api/configuration/ai").json()["data"]
        ok("choice persisted", saved["configuration"]["models"]["model_advice"] == "gpt-6-luna")
        ok("level 2 cleared", saved["configuration"]["fallbacks"]["model_advice"] == [])
        ok("retries saved", saved["configuration"]["retries"] == 3)
        ok("api has no key value", FAKE not in str(saved))
        prices = saved["configuration"]["prices"]
        ok("price saved", prices["gpt-6-luna"]["output"] == 0.6)
        ok("price date saved", saved["configuration"]["prices_checked"] == "2026-10-15")
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "ai-settings.png"), full_page=True)
    page.locator(".model-top").screenshot(path=str(SHOTS / "ai-prices.png"))
    page.get_by_role("button", name="Повернути типові").click()
    page.get_by_text("Повернуто типові", exact=False).wait_for()
    with httpx.Client(base_url=BASE) as api:
        ok(
            "reset to defaults",
            api.get("/api/configuration/ai").json()["data"]["configuration"]["models"][
                "model_advice"
            ]
            == "gpt-6.1-sol",
        )
    ok("price reset in form", float(price("gpt-6-luna", "output").input_value()) == 0.5)
    page.set_viewport_size({"width": 390, "height": 800})
    page.reload()
    page.get_by_role("button", name="Зберегти").wait_for()
    page.wait_for_timeout(300)
    ok(
        "mobile contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
    browser.close()
ok("typed key never sent to server", not any(FAKE in f for f in frames))
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
