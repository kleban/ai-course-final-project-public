"""Stage 07: knowledge base page, local search with sources. Isolated server required:
python tools/ui/knowledge_ui_smoke.py http://127.0.0.1:8905
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8905"
SHOTS = ROOT / "tmp" / "ui-smoke" / "067-knowledge"
SHOTS.mkdir(parents=True, exist_ok=True)
results, errors, external = [], [], []


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

    page.goto(BASE)
    page.get_by_text("База знань", exact=True).first.click()
    page.get_by_role("button", name="Знайти").wait_for()
    ok("menu opens knowledge page", page.url.endswith("/settings/knowledge"))
    ok("documents listed", page.get_by_text("Документів: 5", exact=False).count() == 1)
    page.get_by_label("Запитання").fill("що означають зірочки біля коефіцієнта")
    page.get_by_label("Запитання").press("Enter")
    page.locator(".knowledge-hit").first.wait_for(timeout=120000)
    first = page.locator(".knowledge-hit").first.inner_text()
    ok("best hit is significance rule", "p-value і зірочки значущості" in first)
    ok("hit shows source", "Джерело:" in first)
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "search.png"), full_page=True)
    page.get_by_label("Запитання").fill("рецепт борщу")
    page.get_by_role("button", name="Знайти").click()
    page.get_by_text("не знайдено достатньо схожих", exact=False).wait_for()
    ok("unrelated question explained", page.locator(".knowledge-hit").count() == 0)
    page.get_by_text("Показники якості регресії", exact=False).first.click()
    page.get_by_text("Порівняння з простим прогнозом", exact=False).first.wait_for()
    ok("document text readable", True)
    page.get_by_role("button", name="Новий документ").click()
    page.locator(".knowledge-editor .toastui-editor").first.wait_for()
    page.locator(".knowledge-editor").get_by_role("button", name="Зберегти").click()
    page.get_by_text("Назва документа · 1 розд. · ваш", exact=False).wait_for()
    ok("new document from template", True)
    page.get_by_text("Назва документа · 1 розд. · ваш", exact=False).click()
    page.locator(".knowledge-doc", has_text="Назва документа").get_by_role(
        "button", name="Видалити"
    ).click()
    page.locator(".knowledge-dialog").get_by_role("button", name="Видалити").click()
    page.wait_for_timeout(800)
    ok(
        "user document deleted",
        page.get_by_text("Назва документа · 1 розд.", exact=False).count() == 0,
    )
    doc = page.locator(".knowledge-doc", has_text="Перевірка моделі графіками")
    doc.locator(".q-expansion-item__container > .q-item").first.click()
    doc.get_by_role("button", name="Редагувати").click()
    page.locator(".knowledge-editor .ProseMirror").first.wait_for()
    page.locator(".knowledge-editor .ProseMirror").first.click()
    page.keyboard.press("Control+End")
    page.keyboard.press("Enter")
    page.keyboard.type("Додаткове речення про залишки.")
    page.locator(".knowledge-editor").get_by_role("button", name="Зберегти").click()
    page.get_by_text("Перевірка моделі графіками · 5 розд. · змінений", exact=False).wait_for()
    ok("builtin document edited", True)
    page.screenshot(path=str(SHOTS / "edited.png"), full_page=True)
    page.get_by_text("Перевірка моделі графіками · 5 розд. · змінений", exact=False).click()
    page.locator(".knowledge-doc", has_text="Перевірка моделі графіками").get_by_role(
        "button", name="Повернути типовий"
    ).click()
    page.get_by_text("Перевірка моделі графіками · 5 розд. · типовий", exact=False).wait_for()
    ok("builtin document reset", True)
    with httpx.Client(base_url=BASE) as api:
        found = api.post("/api/knowledge/search", json={"query": "форма лійки"}).json()["data"]
        ok("api search same index", found["hits"][0]["section"] == "Графік похибок")
    page.set_viewport_size({"width": 390, "height": 800})
    page.reload()
    page.get_by_role("button", name="Знайти").wait_for()
    page.wait_for_timeout(300)
    ok(
        "mobile contained",
        page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"),
    )
    page.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    page.screenshot(path=str(SHOTS / "mobile.png"), full_page=True)
    browser.close()
ok("no external requests", not external)
ok("no browser errors", not errors)
print(f"{sum(results)}/{len(results)} passed; errors: {errors}")
sys.exit(0 if all(results) else 1)
