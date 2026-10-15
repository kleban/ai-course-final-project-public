"""Stage 08b: monitoring page lists every part with its state.
python tools/ui/monitoring_ui_smoke.py http://127.0.0.1:8910
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8910"
SHOTS = ROOT / "tmp" / "ui-smoke" / "077-monitoring"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append(bool(condition))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


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
    page.goto(BASE)
    page.get_by_text("Моніторинг", exact=True).first.click()
    page.locator(".monitor-grid .model-panel").first.wait_for(timeout=20000)
    ok("menu opens monitoring", page.url.endswith("/settings/monitoring"))
    ok("all parts listed", page.locator(".monitor-grid").first.locator(".model-panel").count() == 6)
    for title in (
        "База SQLite",
        "DuckDB",
        "Доступ до OpenAI",
        "Сховище даних",
        "Вибірки й моделі",
    ):
        ok(f"part {title}", page.get_by_text(title, exact=True).count() == 1)
    ok("database works", page.locator(".monitor-ok", has_text="База SQLite").count() == 1)
    page.screenshot(path=str(SHOTS / "monitoring.png"), full_page=True)
    before = page.locator(".footnote").last.inner_text()
    page.wait_for_timeout(1100)
    page.get_by_role("button", name="Перевірити").click()
    page.wait_for_function(
        f"[...document.querySelectorAll('.footnote')].pop().innerText !== {before!r}"
    )
    ok("re-check updates time", True)
    page.set_viewport_size({"width": 390, "height": 800})
    page.reload()
    page.locator(".monitor-grid .model-panel").first.wait_for(timeout=20000)
    page.wait_for_timeout(300)
    ok(
        "mobile contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
