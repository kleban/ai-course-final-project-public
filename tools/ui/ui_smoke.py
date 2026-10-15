"""End-to-end UI smoke test of every screen through the real browser UI (no API shortcuts).

Needs Playwright with Chromium and a running Data Story with an EMPTY database, e.g.:
    python tools/start.py --no-browser --port 8899 --database tmp/smoke/data.sqlite3
    python tools/ui/ui_smoke.py http://127.0.0.1:8899
Not part of Check.cmd: Playwright is not in requirements.lock.
"""

import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8899"
SAMPLE = str(Path(__file__).resolve().parents[2] / "samples" / "demo_apartments.csv")
MELBOURNE = str(Path(__file__).resolve().parents[2] / "samples" / "melb_data.csv")
SHOTS = Path(__file__).resolve().parents[2] / "tmp" / "ui-smoke"
SHOTS.mkdir(parents=True, exist_ok=True)
errors, results = [], []
external_requests = []
row_requests: list[int] = []  # "limit" of every page request from the data grid


def ok(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


def markdown(page):
    # Read the component's source text; inner_text adds visual paragraph breaks.
    return page.locator(".markdown-editor").evaluate(
        "el => runMethod(el.id.slice(1), 'getMarkdown', [])"
    )


with sync_playwright() as pw:
    b = pw.chromium.launch()
    context = b.new_context(viewport={"width": 1400, "height": 900})

    def local_only(route):
        if urlparse(route.request.url).hostname == urlparse(B).hostname:
            route.continue_()
        else:
            external_requests.append(route.request.url)
            route.abort()

    context.route("**/*", local_only)
    pg = context.new_page()
    pg.on("console", lambda m: m.type == "error" and errors.append(m.text))
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on(
        "request",
        lambda r: (
            r.method == "POST"
            and r.url.endswith("/rows")
            and row_requests.append(int((r.post_data_json or {}).get("limit", 10**6)))
        ),
    )
    pg.goto(B)
    pg.wait_for_timeout(1500)
    ok("empty overview", pg.get_by_text("Тут з’являться ваші проєкти").count() == 1)
    ok("panel logo", pg.locator("img.brand-logo").count() == 1)
    ok(
        "menu count starts at 0",
        pg.locator(".nav-link.current .nav-count").inner_text().strip() == "0",
    )
    pg.get_by_text("Створити проєкт").first.click()
    pg.wait_for_timeout(700)
    ok("type default regression", "Регресія" in pg.locator(".type-current").inner_text())
    pg.get_by_text("Зберегти проєкт").click()
    pg.wait_for_timeout(600)
    ok("empty name rejected", pg.get_by_text("Назва: введіть значення.").count() == 1)
    pg.get_by_label("Назва проєкту").fill("Ціни на квартири")
    pg.get_by_label("Мета дослідження (необов’язково)").fill("Що впливає на ціну")
    pg.get_by_text("Зберегти проєкт").click()
    pg.wait_for_timeout(1500)
    ok(
        "menu shows project count",
        pg.locator(".nav-link .nav-count").inner_text().strip() == "1",
    )
    ok(
        "project description empty state",
        pg.get_by_text("Опису ще немає.", exact=False).count() == 1,
    )
    pg.get_by_role("button", name="Редагувати опис", exact=True).click()
    pg.locator(".markdown-editor .toastui-editor").first.wait_for()
    pg.locator(".markdown-editor .ProseMirror").first.click()
    pg.keyboard.type("Мета: **ціна** житла")
    pg.get_by_role("button", name="Зберегти опис", exact=True).click()
    pg.locator(".project-description strong").first.wait_for()
    ok(
        "project description saved and rendered",
        pg.locator(".project-description strong").inner_text() == "ціна",
    )
    ok(
        "opened project highlights block data",
        "Дані" in pg.locator(".block-link.current").inner_text(),
    )
    ok("project opened", "project=" in pg.url and pg.get_by_text("Ціни на квартири").count() >= 1)
    project_url = pg.url
    # Updated in history 132: EDA is a menu block link; all seven blocks are implemented.
    pg.locator(".block-link", has_text="Оглядовий аналіз").click()
    pg.get_by_text("Для статистики потрібен набір даних.", exact=False).wait_for()
    blocks = pg.locator(".block-link")
    ok(
        "analysis blocks: seven, none planned, exploratory current",
        blocks.count() == 7
        and pg.locator(".block-link.planned").count() == 0
        and "Оглядовий аналіз" in pg.locator(".block-link.current").inner_text()
        and "eda" in pg.url,
    )
    ok("statistics empty state has upload", pg.locator(".uploader input[type=file]").count() == 1)
    pg.goto(project_url)
    pg.wait_for_timeout(1500)
    pg.locator(".uploader input[type=file]").first.set_input_files(SAMPLE)
    pg.wait_for_timeout(3000)
    ok(
        "csv uploaded -> dataset view",
        "dataset=" in pg.url and pg.get_by_text("Таблиця даних").count() == 1,
    )
    ok("150 rows", pg.locator(".kpi-value").first.inner_text().strip() == "150")
    dataset_url = pg.url
    pg.get_by_text("Додати опис").click()
    pg.wait_for_timeout(600)
    pg.get_by_label("Текст опису").fill("ціна_тис_грн — ціна продажу, тис. грн")
    pg.get_by_text("Зберегти текст").click()
    pg.wait_for_timeout(1500)
    ok("raw description saved", pg.get_by_text("Сирі описи даних (1)").count() == 1)
    pg.get_by_text("Редагувати метадані").click()
    pg.wait_for_timeout(1500)
    card = pg.locator(".meta-card").nth(6)
    card.get_by_label("Назва для людей").fill("Ціна")
    card.get_by_label("Роль").click()
    pg.wait_for_timeout(300)
    pg.get_by_role("option", name="цільова").click()
    pg.get_by_text("Зберегти й підтвердити").click()
    pg.wait_for_timeout(1500)
    ok(
        "metadata confirmed",
        pg.get_by_text("Підтверджено").count() >= 1 and pg.locator(".role.target").count() == 1,
    )
    pg.goto(dataset_url.split("&dataset=")[0])
    pg.wait_for_timeout(1500)
    pg.locator(".uploader input[type=file]").first.set_input_files(MELBOURNE)
    pg.wait_for_timeout(6000)
    rows_text = pg.locator(".kpi-value").first.inner_text()
    ok("Melbourne CSV uploaded", "".join(c for c in rows_text if c.isdigit()) == "13580")
    date_row = pg.locator(".trow").filter(has=pg.get_by_text("Date", exact=True))
    ok(
        "CSV date column recognised",
        date_row.get_by_text("дата", exact=True).count() == 1
        and date_row.get_by_text("28.01.2016").count() == 1
        and date_row.get_by_text("23.09.2017").count() == 1,
    )
    date_row.get_by_text("дата", exact=True).hover()
    pg.wait_for_timeout(600)
    ok("date format tooltip", pg.get_by_text("Дата у файлі: ДД/ММ/РРРР").count() == 1)
    grid = pg.locator(".data-grid")
    ok(
        "AG Grid shows all columns and Ukrainian lead",
        grid.locator(".ag-header-cell-text").get_by_text("Suburb", exact=True).count() == 1
        and grid.locator(".ag-header-cell-text").get_by_text("Price", exact=True).count() == 1
        and pg.get_by_text("Усі 13").count() == 1
        and grid.locator(".ag-paging-panel").get_by_text("Сторінка").count() >= 1
        and pg.get_by_text("Дати показано як ДД.ММ.РРРР").count() == 1,
    )
    ok("grid shows dates as ДД.ММ.РРРР", grid.get_by_text("03.12.2016").count() >= 1)
    grid.locator(".ag-header-cell[col-id='c4'] .ag-header-cell-label").click()
    pg.wait_for_timeout(800)
    first_price = grid.locator(".ag-row[row-index='0'] .ag-cell[col-id='c4']").inner_text()
    ok("grid sorts numbers as numbers", first_price.strip() == "85000")
    grid.locator(".ag-floating-filter[aria-colindex='4'] input").fill("u")
    pg.wait_for_timeout(1500)
    ok("grid filter updates count", pg.get_by_text("За фільтрами знайдено 3").count() == 1)
    grid.locator(".ag-paging-button[aria-label='Наступна сторінка']").click()
    pg.wait_for_timeout(1200)
    ok("next page loads", grid.locator(".ag-row[row-index='50']").count() == 1)
    ok(
        "server pages only",
        bool(row_requests) and max(row_requests) <= 100 and len(row_requests) <= 10,
    )
    grid.scroll_into_view_if_needed()
    pg.screenshot(path=str(SHOTS / "melbourne-grid.png"))
    melbourne_url = pg.url
    pg.get_by_role("button", name="Оглядовий аналіз", exact=True).click()
    pg.locator(".stat-table tbody tr").first.wait_for()
    ok(
        "statistics context entire Melbourne",
        "13\u202f580" in pg.locator(".stat-context").inner_text(),
    )
    ok("statistics descriptive page is 10 fields", pg.locator(".stat-table tbody tr").count() == 10)
    price = pg.locator(".stat-table tr[data-field='Price']")
    ok("statistics price matches control", "1\u202f075\u202f684,079" in price.inner_text())
    ok(
        "statistics draft visible",
        pg.get_by_text("Метадані не підтверджено", exact=False).count() >= 1,
    )
    pg.screenshot(path=str(SHOTS / "statistics-descriptive.png"))
    pg.get_by_role("button", name="Наступні поля", exact=True).click()
    pg.locator(".stat-table tr[data-field='BuildingArea']").wait_for()
    ok(
        "statistics server pagination",
        pg.locator(".stat-table tr[data-field='Price']").count() == 0,
    )
    pg.get_by_role("button", name="Пропуски", exact=True).click()
    pg.locator(".stat-summary").wait_for()
    ok("statistics missing row union", "7\u202f384" in pg.locator(".stat-summary").inner_text())
    pg.screenshot(path=str(SHOTS / "statistics-missing.png"))
    pg.get_by_role("button", name="Кореляції", exact=True).click()
    pg.get_by_label("Числові поля для кореляції").wait_for()
    x = pg.locator(".stat-table tr[data-field='Price']")
    # Numeric fields begin Rooms, Price, Distance...; first cell is Price x Rooms.
    x.locator(".stat-correlation").first.click()
    pg.wait_for_timeout(400)
    ok(
        "statistics Pearson pair n",
        "0,497" in pg.locator(".stat-pair").inner_text()
        and "13580" in pg.locator(".stat-pair").inner_text(),
    )
    pg.screenshot(path=str(SHOTS / "statistics-correlations.png"))
    for field in (
        "Distance",
        "Postcode",
        "Bedroom2",
        "Bathroom",
        "Car",
        "Landsize",
        "BuildingArea",
        "YearBuilt",
        "Lattitude",
        "Longtitude",
        "Propertycount",
    ):
        pg.locator(".stat-fields .q-chip").filter(has_text=field).locator(
            ".q-chip__icon--remove"
        ).click()
        pg.wait_for_timeout(300)
    pg.get_by_role("button", name="Застосувати поля", exact=True).click()
    pg.wait_for_timeout(700)
    ok(
        "statistics selected fields only",
        pg.locator(".stat-table tbody tr").count() == 2
        and pg.locator(".stat-table thead th").count() == 3,
    )
    pg.locator(".stat-fields .q-chip").filter(has_text="Price").locator(
        ".q-chip__icon--remove"
    ).click()
    pg.get_by_role("button", name="Застосувати поля", exact=True).click()
    pg.get_by_text("Оберіть щонайменше два придатні числові поля.", exact=True).wait_for()
    ok("statistics explains fewer than two fields", pg.locator(".stat-table").count() == 0)
    pg.reload()
    pg.locator(".stat-table tbody tr").first.wait_for()
    pg.get_by_role("button", name="Кореляції", exact=True).click()
    pg.get_by_label("Числові поля для кореляції").wait_for()
    pg.get_by_text("Виключені поля", exact=False).click()
    ok("statistics excludes date", "Date: До Pearson" in pg.locator(".howto").last.inner_text())
    pg.set_viewport_size({"width": 390, "height": 800})
    pg.wait_for_timeout(500)
    ok(
        "statistics mobile contained scroll",
        pg.evaluate("document.documentElement.scrollWidth <= innerWidth"),
    )
    pg.screenshot(path=str(SHOTS / "statistics-mobile.png"))
    pg.set_viewport_size({"width": 1400, "height": 900})
    pg.get_by_role("button", name="Огляд і метадані", exact=True).click()
    pg.get_by_text("Редагувати метадані", exact=True).click()
    pg.locator(".meta-card").first.wait_for()
    postcode_card = pg.locator(".meta-card").nth(9)
    postcode_card.get_by_label("Тип").click()
    pg.get_by_role("option", name="категорія", exact=True).click()
    pg.get_by_text("Зберегти й підтвердити", exact=True).click()
    pg.get_by_role("button", name="Оглядовий аналіз", exact=True).wait_for()
    pg.get_by_role("button", name="Оглядовий аналіз", exact=True).click()
    pg.get_by_role("button", name="Кореляції", exact=True).click()
    pg.get_by_label("Числові поля для кореляції").wait_for()
    pg.get_by_text("Виключені поля", exact=False).click()
    ok(
        "statistics reflects saved metadata",
        "Postcode: До Pearson" in pg.locator(".howto").last.inner_text(),
    )
    ok(
        "statistics confirmed state",
        pg.get_by_text("Метадані підтверджено", exact=True).count() == 1,
    )
    pg.goto(melbourne_url)
    pg.wait_for_timeout(1500)
    # stage 02c: a second project takes an independent copy of the Melbourne dataset
    pg.goto(B)
    pg.wait_for_timeout(1200)
    pg.get_by_text("Створити проєкт").first.click()
    pg.wait_for_timeout(700)
    pg.get_by_label("Назва проєкту").fill("Друга аналітика")
    pg.get_by_text("Зберегти проєкт").click()
    pg.wait_for_timeout(1500)
    ok("menu count follows projects", pg.locator(".nav-count").first.inner_text().strip() == "2")
    pg.get_by_text("Додати наявний набір").click()
    pg.wait_for_timeout(1000)
    dialog = pg.locator(".copy-dialog")
    ok(
        "copy dialog lists datasets of other projects",
        dialog.locator(".trow").count() == 2
        and dialog.get_by_text("проєкт «Ціни на квартири»").count() == 2
        and dialog.get_by_text("метадані підтверджено").count() == 2,
    )
    pg.screenshot(path=str(SHOTS / "copy-dialog.png"))
    dialog.locator(".trow").filter(has=pg.get_by_text("melb_data.csv", exact=True)).get_by_text(
        "Додати", exact=True
    ).click()
    pg.wait_for_timeout(3000)
    rows_text = pg.locator(".kpi-value").first.inner_text()
    ok(
        "copied dataset opens with provenance",
        "dataset=" in pg.url
        and "".join(c for c in rows_text if c.isdigit()) == "13580"
        and pg.get_by_text("копія з проєкту «Ціни на квартири»").count() == 1,
    )
    pg.goto(pg.url.split("&dataset=")[0])
    pg.wait_for_timeout(1500)
    ok(
        "copy marked in project datasets and journal",
        pg.locator(".datasets .tag").get_by_text("копія").count() == 1
        and pg.get_by_text("Додано копію наявного набору").count() >= 1,
    )
    pg.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    pg.screenshot(path=str(SHOTS / "melbourne-dataset.png"), full_page=True)
    pg.goto(B + "/settings")
    pg.wait_for_timeout(1200)
    ok("settings page", pg.get_by_text("Типи проєктів та інструкції").count() == 1)
    editor = pg.get_by_role("textbox", name="Текст інструкції Markdown")
    editor.wait_for()
    original = markdown(pg).strip()
    ok("markdown editor mounted", len(original) > 100)
    ok("icon toolbar", pg.locator(".markdown-tools button > .ico").count() == 9)
    pg.get_by_role("button", name="Жирний", exact=True).hover()
    ok(
        "visible Ukrainian tooltip",
        pg.get_by_role("tooltip").filter(has_text="Ctrl+B").is_visible(),
    )
    pg.screenshot(path=str(SHOTS / "markdown-tooltip.png"))
    editor.hover()
    ok("preview hidden by default", not pg.locator(".toastui-editor-md-preview").is_visible())
    pg.get_by_text("Підказки Markdown", exact=True).click()
    ok("markdown help", pg.get_by_text("Виділіть текст", exact=False).is_visible())
    ok(
        "ukrainian button hint",
        "Ctrl+B" in pg.get_by_role("button", name="Жирний", exact=True).get_attribute("title"),
    )
    pg.get_by_role("button", name="Перегляд", exact=True).click()
    ok("preview toggles", pg.locator(".toastui-editor-md-preview").is_visible())
    pg.screenshot(path=str(SHOTS / "markdown-preview.png"))
    pg.get_by_role("button", name="Перегляд", exact=True).click()
    pg.get_by_text("Підказки Markdown", exact=True).click()
    draft = "## Перевірка Markdown\n\nУкраїнський **текст** і *курсив*.\n\n- Перший пункт\n- Другий пункт\n\n| Поле | Опис |\n| --- | --- |\n| Ціна | грн |\n\n```python\nx = 1\n```"
    editor.fill(draft)
    ok(
        "uniform Markdown source size",
        editor.evaluate("""el => [...el.querySelectorAll('[class*=toastui-editor-md-]')].every(n =>
            getComputedStyle(n).fontSize === '14px')"""),
    )
    ok(
        "Markdown bold italic and colour",
        editor.evaluate("""el => getComputedStyle(el.querySelector('.toastui-editor-md-strong')).fontWeight === '700'
            && getComputedStyle(el.querySelector('.toastui-editor-md-emph')).fontStyle === 'italic'
            && getComputedStyle(el.querySelector('.toastui-editor-md-heading')).color === 'rgb(23, 70, 144)'"""),
    )
    ok(
        "no empty toolbar gap",
        pg.locator(".toastui-editor-defaultUI").evaluate("""el =>
            el.querySelector('.toastui-editor-main').getBoundingClientRect().top - el.getBoundingClientRect().top <= 2"""),
    )
    # Save immediately after input: verifies that the last edit is not lost in transit.
    pg.get_by_text("Зберегти інструкцію", exact=True).click()
    pg.wait_for_timeout(1000)
    ok("markdown save and reload", markdown(pg).strip() == draft)
    pg.reload()
    editor.wait_for()
    ok("markdown survives page reload", markdown(pg).strip() == draft)
    # ProseMirror groups edits within 500 ms; make this a distinct undo operation.
    pg.wait_for_timeout(700)
    pg.get_by_role("button", name="Таблиця", exact=True).click()
    ok("table template inserted", "Пояснення" in editor.inner_text())
    pg.get_by_role("button", name="Скасувати", exact=True).click()
    ok("format undo", markdown(pg).strip() == draft)
    editor.hover()
    pg.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    pg.screenshot(path=str(SHOTS / "markdown-editor.png"), full_page=True)
    editor.fill("")
    pg.get_by_text("Зберегти інструкцію", exact=True).click()
    pg.wait_for_timeout(500)
    ok("empty markdown rejected", pg.get_by_text("Інструкція: введіть значення.").count() == 1)
    editor.fill("а" * 20001)
    pg.get_by_text("Зберегти інструкцію", exact=True).click()
    pg.wait_for_timeout(500)
    ok("oversized markdown rejected", "20000" in pg.locator(".error-message").last.inner_text())
    pg.get_by_text("Відновити типову", exact=True).click()
    pg.wait_for_timeout(1000)
    ok("default instruction restored", markdown(pg).strip() == original)
    pg.set_viewport_size({"width": 390, "height": 800})
    pg.reload()
    editor.wait_for()
    pg.evaluate("window.scrollTo(0, 0)")  # fixed menu at top
    pg.screenshot(path=str(SHOTS / "markdown-mobile.png"), full_page=True)
    ok("mobile editor fits", pg.evaluate("document.documentElement.scrollWidth <= innerWidth"))
    pg.set_viewport_size({"width": 1400, "height": 900})
    pg.goto(B + "/icons")
    pg.wait_for_timeout(1200)
    ok("icons page", pg.locator(".icon-tile").count() >= 30)
    ok("Markdown icons in gallery", pg.get_by_text("Редактор Markdown", exact=True).count() == 1)
    pg.goto(B)
    pg.wait_for_timeout(1200)
    ok("project listed with type", pg.get_by_text("Регресія").count() >= 1)
    pg.locator(".nav-link.nav-toggle").click()
    pg.wait_for_timeout(700)
    pg.goto(B + "/settings")
    pg.wait_for_timeout(1200)
    ok(
        "collapsed menu remembered",
        "q-drawer--mini" in pg.evaluate("document.querySelector('.q-drawer').className"),
    )
    pg.locator(".nav-link.nav-toggle").click()
    pg.wait_for_timeout(500)
    pg.goto(dataset_url)
    pg.wait_for_timeout(1500)
    pg.screenshot(path=str(SHOTS / "final-dataset.png"))
    pg.goto(B)
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(SHOTS / "final-home.png"))
    pg.set_viewport_size({"width": 390, "height": 800})
    pg.goto(B)
    pg.wait_for_timeout(1200)
    ok("mobile renders", pg.get_by_text("Ваші дослідження").count() == 1)
    pg.screenshot(path=str(SHOTS / "final-mobile.png"))
    b.close()
ok("no external resource requests", not external_requests)
print("console errors:", errors)
print(f"{sum(r for _, r in results)}/{len(results)} passed")
sys.exit(0 if all(r for _, r in results) and not errors else 1)
