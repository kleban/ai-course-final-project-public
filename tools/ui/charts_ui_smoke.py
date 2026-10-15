"""Stage 04 browser acceptance (heatmap, histogram, scatter, settings) on a test server."""

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8895"
SHOTS = ROOT / "tmp" / "ui-smoke" / "057-missing"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


def ok(name, condition):
    results.append((name, bool(condition)))
    print(("PASS " if condition else "FAIL ") + name, flush=True)


with httpx.Client(base_url=B) as api:
    project = api.post("/api/projects", json={"name": "Перевірка графіків Melbourne"}).json()[
        "data"
    ]["id"]
    uploaded = api.post(
        f"/api/projects/{project}/datasets",
        files={"file": ("melb.csv", (ROOT / "samples" / "melb_data.csv").read_bytes(), "text/csv")},
    )
    uploaded.raise_for_status()
    dataset = uploaded.json()["data"]["id"]
    api.post("/api/configuration/charts/reset").raise_for_status()

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    context = browser.new_context(viewport={"width": 1400, "height": 1000})

    def local_only(route):
        if urlparse(route.request.url).hostname == urlparse(B).hostname:
            route.continue_()
        else:
            external.append(route.request.url)
            route.abort()

    context.route("**/*", local_only)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: message.type == "error" and errors.append(message.text))
    page.goto(B + "/settings/charts")
    page.locator(".heatmap-cell").first.wait_for()
    ok("storytelling default six lower cells", page.locator(".heatmap-cell").count() == 6)
    page.get_by_label("Тема", exact=True).click()
    page.get_by_role("option", name="Академічна", exact=True).click()
    page.wait_for_timeout(400)
    ok("academic preview nine cells", page.locator(".heatmap-cell").count() == 9)
    with httpx.Client(base_url=B) as api:
        ok(
            "preview not saved",
            api.get("/api/configuration/charts").json()["data"]["theme"] == "storytelling",
        )
    page.get_by_role("button", name="Зберегти налаштування", exact=True).click()
    page.wait_for_timeout(800)
    page.reload()
    page.locator(".heatmap-cell").first.wait_for()
    ok("academic survives reload", page.locator(".heatmap-cell").count() == 9)
    page.get_by_label("Додатні значення · HEX", exact=True).fill("#264653")
    page.get_by_role("checkbox", name="Числа в клітинках", exact=True).click()
    page.wait_for_timeout(400)
    ok("numbers can be hidden undefined remains", page.locator(".heatmap-cell text").count() == 2)
    page.get_by_role("button", name="Зберегти налаштування", exact=True).click()
    page.wait_for_timeout(800)
    with httpx.Client(base_url=B) as api:
        saved = api.get("/api/configuration/charts").json()["data"]
        ok(
            "custom persisted",
            saved["theme"] == "custom" and saved["custom"]["positive"] == "#264653",
        )
    page.get_by_label("Текст · HEX", exact=True).fill("#FFFFFF")
    page.get_by_text("Низький контраст тексту й фону:", exact=False).wait_for()
    ok("contrast warning in preview", True)
    page.get_by_label("Додатні значення · HEX", exact=True).fill("invalid")
    page.get_by_role("button", name="Зберегти налаштування", exact=True).click()
    page.get_by_text("Перевірте кольори, розміри та поріг перед збереженням.").wait_for()
    ok("invalid configuration rejected", True)
    page.get_by_role("button", name="Відновити початкові налаштування", exact=True).click()
    page.wait_for_timeout(800)
    page.get_by_role("button", name="Гістограма", exact=True).click()
    page.locator(".histogram .chart-item").first.wait_for()
    ok(
        "settings histogram preview with median",
        page.locator(".histogram .chart-item").count() == 20
        and page.locator(".histogram .median-line").count() == 1,
    )
    page.get_by_role("checkbox", name="Лінія медіани", exact=True).click()
    page.get_by_label("Інтервалів за замовчуванням (5–60)", exact=True).fill("10")
    page.wait_for_timeout(500)
    ok(
        "histogram preview follows bins and median switch",
        page.locator(".histogram .chart-item").count() == 10
        and page.locator(".histogram .median-line").count() == 0,
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "settings-histogram.png"), full_page=True)
    page.get_by_role("button", name="Розсіювання", exact=True).click()
    page.locator(".scatter circle").first.wait_for()
    ok("settings scatter preview", page.locator(".scatter circle").count() == 120)
    page.get_by_role("button", name="Пропуски за полями", exact=True).click()
    page.locator(".missing-chart .chart-item").first.wait_for()
    ok("settings missing preview", page.locator(".missing-chart .chart-item").count() == 5)
    page.get_by_role("button", name="Відновити початкові налаштування", exact=True).click()
    page.wait_for_timeout(800)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "settings.png"), full_page=True)
    ok(
        "reset restores defaults",
        page.get_by_role("checkbox", name="Числа в клітинках", exact=True).is_checked(),
    )
    page.goto(B + f"/?project={project}&dataset={dataset}&view=eda&tool=heatmap")
    selector = page.get_by_label("Числові поля · оберіть 2–12", exact=True)
    selector.click()
    page.get_by_role("option", name="Price", exact=True).click()

    page.get_by_role("option", name="Rooms", exact=True).click()
    selector.press("Escape")
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.locator(".heatmap-cell").first.wait_for()
    ok("heatmap three cells", page.locator(".heatmap-cell").count() == 3)
    pair = page.locator(".heatmap-cell").nth(1)
    pair.focus()
    page.wait_for_timeout(100)
    detail = page.locator(".heatmap-detail").inner_text()
    ok(
        "keyboard pair shows r n exclusions",
        "0.50" in detail and "13580" in detail and "виключено пар: 0" in detail,
    )
    page.get_by_text("Як розраховано · формули", exact=True).click()
    page.wait_for_function("document.querySelectorAll('.math-formula mjx-container').length === 4")
    ok(
        "four local MathJax formulas rendered",
        page.locator(".math-formula[data-rendered=ok]").count() == 4,
    )
    ok(
        "Pearson formula keeps TeX sums",
        "sum" in page.locator(".math-formula").first.get_attribute("data-formula"),
    )
    page.evaluate("scrollTo(0,0)")
    page.wait_for_timeout(200)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "heatmap.png"), full_page=True)
    # History 139: the per-chart formatting panel was removed; charts use the common theme.
    ok(
        "no per-chart formatting panel",
        page.get_by_text("Тимчасове оформлення графіка").count() == 0,
    )
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.wait_for_timeout(1500)
    with httpx.Client(base_url=B) as api:
        ok(
            "temporary changes not global",
            api.get("/api/configuration/charts").json()["data"]["theme"] == "storytelling",
        )
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        ok(
            "one chart event per build no statistics duplicates",
            sum(e["operation"] == "chart.built" for e in events) == 2
            and not any(e["operation"] == "statistics.calculated" for e in events),
        )
    page.goto(B + f"/?project={project}&dataset={dataset}&view=eda&tool=histogram")
    page.get_by_label("Числове поле", exact=True).click()
    page.get_by_role("option", name="Price", exact=True).click()
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.locator(".histogram .chart-item").first.wait_for()
    bars = page.locator(".histogram .chart-item")
    labels = [bars.nth(i).get_attribute("aria-label") for i in range(bars.count())]
    total = sum(int(label.split(": ")[1].split(" ")[0]) for label in labels)
    ok("histogram twenty bins sum equals n", len(labels) == 20 and total == 13580)
    ok(
        "histogram last bin closed includes max",
        labels[-1].startswith("[") and "9 000 000]" in labels[-1],
    )
    ok(
        "histogram facts and median from statistics",
        page.get_by_text("сума частот 13 580 = n", exact=False).count() == 1
        and page.get_by_text("Медіана (статистика 03): 903 000", exact=True).count() == 1
        and page.locator(".histogram .median-line").count() == 1,
    )
    bars.nth(0).focus()
    page.wait_for_timeout(100)
    ok(
        "keyboard bar shows bounds and count",
        "рядків" in page.locator(".heatmap-detail").inner_text(),
    )
    page.get_by_text("Як розраховано · формули", exact=True).click()
    page.wait_for_function("document.querySelectorAll('.math-formula mjx-container').length === 4")
    ok(
        "four histogram formulas rendered",
        page.locator(".math-formula[data-rendered=ok]").count() == 4,
    )
    page.evaluate("scrollTo(0,0)")
    page.wait_for_timeout(200)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "histogram.png"), full_page=True)
    page.get_by_role("button", name="Ширина інтервалу", exact=True).click()
    page.get_by_label("Ширина інтервалу (> 0)", exact=True).fill("500000")
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.get_by_text("Інтервалів: 18 (задана ширина)", exact=False).wait_for()
    ok("width mode builds eighteen bins", page.locator(".histogram .chart-item").count() == 18)
    page.get_by_label("Ширина інтервалу (> 0)", exact=True).fill("1000")
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.get_by_text("допустимо до 200", exact=False).wait_for()
    ok("too narrow width explained", True)
    with httpx.Client(base_url=B) as api:
        events = api.get(f"/api/projects/{project}/events").json()["data"]
        histogram_events = [
            e
            for e in events
            if e["operation"] == "chart.built" and e["payload"].get("chart") == "histogram"
        ]
        ok(
            "histogram events success and error logged",
            [e["payload"]["status"] for e in histogram_events].count("success") == 2
            and [e["payload"]["status"] for e in histogram_events].count("error") == 1,
        )
    page.goto(B + f"/?project={project}&dataset={dataset}&view=eda&tool=scatter")
    page.get_by_role("button", name="Побудувати", exact=True).click()
    started = time.monotonic()
    page.locator(".scatter circle").first.wait_for(timeout=60000)
    elapsed = time.monotonic() - started
    ok(
        "scatter all Melbourne pairs drawn",
        page.locator(".scatter circle").count() == 13580
        and page.get_by_text("Точок n = 13 580", exact=False).count() == 1,
    )
    ok("scatter r from statistics", "Pearson r = 0.50" in page.locator(".chart-facts").inner_text())
    ok(f"scatter built in {elapsed:.1f}s (< 15s)", elapsed < 15)
    page.evaluate("scrollTo(0,0)")
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "scatter.png"), full_page=True)
    page.goto(B + f"/?project={project}&dataset={dataset}&view=eda&tool=missing")
    page.get_by_role("button", name="Побудувати", exact=True).click()
    page.locator(".missing-chart .chart-item").first.wait_for()
    first = page.locator(".missing-chart .chart-item").first.get_attribute("aria-label")
    ok(
        "missing chart ten fields largest first",
        page.locator(".missing-chart .chart-item").count() == 10
        and first.startswith("BuildingArea: 6450"),
    )
    page.get_by_role("button", name="Наступні 10", exact=True).click()
    page.get_by_text("Поля 11–20 з 21", exact=True).wait_for()
    ok("missing chart pages of ten", True)
    page.get_by_text("Як розраховано · формули", exact=True).click()
    page.wait_for_function("document.querySelectorAll('.math-formula mjx-container').length === 1")
    ok("missing formula rendered", page.locator(".math-formula[data-rendered=ok]").count() == 1)
    page.evaluate("scrollTo(0,0)")
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "missing.png"), full_page=True)
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(500)
    ok(
        "mobile page has no horizontal overflow",
        page.evaluate("document.documentElement.scrollWidth <= innerWidth"),
    )
    ok(
        "mobile chart scrolls internally",
        page.locator(".heatmap-scroll").evaluate("el=>el.scrollWidth>el.clientWidth"),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
    ok("no external requests", not external)
    ok("no browser errors", not errors)
    browser.close()

(ROOT / "tmp" / "charts057-results.json").write_text(
    json.dumps(
        {"results": results, "errors": errors, "external": external}, ensure_ascii=False, indent=2
    ),
    encoding="utf-8",
)
raise SystemExit(0 if all(condition for _, condition in results) else 1)
