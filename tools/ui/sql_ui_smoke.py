"""Reviewed SQL summary: explicit AI action, preview, apply, persisted normal dataset.

Run against the isolated mock-AI server tmp/sql-ui-server-115.py, never production.
"""

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8915"
FOLDER = ROOT / "tmp" / "sql-ui-115"
state = json.loads((FOLDER / "context.json").read_text(encoding="utf-8"))
project = state["project"]
errors, external = [], []
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
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(f"{BASE}/?project={project}&view=combine")
    page.get_by_role("button", name="Підготуй агрегацію", exact=True).wait_for()
    with httpx.Client(base_url=BASE) as api:
        before = len(api.get(f"/api/projects/{project}/datasets").json()["data"])
    page.get_by_role("button", name="Підготуй агрегацію", exact=True).click()
    page.get_by_text("Пропозиція агрегації", exact=True).wait_for(timeout=60000)
    draft = page.locator(".assistant-draft")
    assert draft.locator("table").count() == 1
    assert "3" in draft.inner_text()
    with httpx.Client(base_url=BASE) as api:
        assert len(api.get(f"/api/projects/{project}/datasets").json()["data"]) == before
    page.screenshot(path=str(FOLDER / "before-apply.png"))
    draft.get_by_role("button", name="Застосувати", exact=True).click()
    page.wait_for_timeout(1500)
    with httpx.Client(base_url=BASE) as api:
        datasets = api.get(f"/api/projects/{project}/datasets").json()["data"]
        assert len(datasets) == before + 1
        created = next(d for d in datasets if d["name"] == "Зведення SQL.csv")
        assert created["row_count"] == 3
    page.screenshot(path=str(FOLDER / "after-apply.png"))
    browser.close()
assert not errors, errors
assert not external, external
(FOLDER / "evidence.json").write_text(
    json.dumps(
        {
            "preview": True,
            "explicit_apply": True,
            "rows": 3,
            "browser_errors": errors,
            "external_requests": external,
        },
        ensure_ascii=False,
    ),
    encoding="utf-8",
)
print("PASS: SQL preview, explicit apply, persisted dataset, local resources, no browser errors")
