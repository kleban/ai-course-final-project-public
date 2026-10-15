"""Reports in the project: select saved model, edit, explicitly ask AI, render, preview, download."""

from functools import partial
from typing import Any

from nicegui import run, ui

from .ai_settings import format_usd
from .contracts import Failure
from .markdown_editor import MarkdownEditor
from .model_screen import panel
from .reports import preview_html
from .screen import button, error_line, page_header, progress_line


def _when(value: str) -> str:
    return value[:16].replace("T", " ") + " UTC"


def report_screen(service, active, report_id=None):
    project_id = str(active.id)
    page_header(
        "Етап 7 · " + active.name,
        "doc",
        "Аналітичний звіт",
        "Збережені результати → редагована чернетка → локальний HTML через Quarto. "
        "AI — лише за вашою дією.",
    )
    models = service.model_runs(project_id)
    if isinstance(models, Failure):
        error_line(models)
        return
    if not models.data:
        ui.label("Спочатку підготуйте вибірки й навчіть модель у цьому проєкті.").classes(
            "section-lead"
        )
        return
    listing = service.report_list(project_id)
    if isinstance(listing, Failure):
        error_line(listing)
        return
    reports = {r["id"]: r for r in listing.data}
    selected = report_id or next(iter(reports), None)
    if selected is not None and selected not in reports:
        ui.label("Звіт не знайдено в цьому проєкті.").classes("error-message")
        selected = next(iter(reports), None)

    state: dict[str, Any] = {"report": reports.get(selected), "busy": False}
    ref: dict[str, Any] = {}
    controls: list[Any] = []

    def busy(value: bool) -> None:
        state["busy"] = value
        for item in controls:
            if hasattr(item, "set_enabled"):
                item.set_enabled(not value)
            elif value:
                item.props("disabled")
            else:
                item.props(remove="disabled")

    with ui.element("div").classes("model-grid model-top report-top"):
        with panel("Звіт проєкту"):
            if state["report"] is None:
                ui.label(
                    "Звітів ще немає. Створіть чернетку зі збереженої моделі праворуч."
                ).classes("model-note")
            else:
                ui.select(
                    {k: f"{r['title']} · {_when(r['created_at'])}" for k, r in reports.items()},
                    value=selected,
                    label="Звіти проєкту",
                    on_change=lambda e: ui.navigate.to(
                        f"/?project={project_id}&view=report&run={e.value}"
                    ),
                ).props("outlined square").classes("w-full")
                summary = ui.column().classes("w-full gap-0")
                with ui.row().classes("report-actions"):
                    controls.append(button("Зберегти чернетку", lambda: action("save"), "check"))
                    controls.append(
                        button("Підготувати пояснення AI", lambda: action("ai"), "spark")
                    )
                    controls.append(button("Переглянути як документ", lambda: sheet(), "preview"))
                    controls.append(
                        button("Сформувати HTML", lambda: action("render"), "doc", "primary")
                    )
                ref["summary"] = summary
        with panel("Нова чернетка"):
            choices = {
                m["id"]: (
                    f"{m['document']['dataset_name']} · {m['document']['choices']['target']} · "
                    f"{_when(m['created_at'])}"
                )
                for m in models.data
            }
            model = (
                ui.select(choices, value=next(iter(choices)), label="Збережена модель")
                .props("outlined square")
                .classes("w-full")
            )
            title = (
                ui.input("Назва звіту", value="Аналітичний звіт · " + active.name)
                .props("outlined square maxlength=180")
                .classes("w-full")
            )
            ui.label(
                "Чернетка бере числа, таблиці й графіки зі знімка моделі; навчання не повторюється."
            ).classes("model-note")
            controls.extend([model, title])
            controls.append(
                button("Створити чернетку зі знімка результатів", lambda: create(), "doc")
            )
    status = ui.column().classes("w-full report-status")

    async def create():
        if state["busy"]:
            return
        busy(True)
        status.clear()
        try:
            result = await run.io_bound(
                service.report_create, project_id, {"model_id": model.value, "title": title.value}
            )
            if isinstance(result, Failure):
                with status:
                    error_line(result)
                return
            ui.navigate.to(f"/?project={project_id}&view=report&run={result.data['id']}")
        finally:
            busy(False)

    if state["report"] is None:
        return

    def show_summary() -> None:
        report = state["report"]
        box = ref["summary"]
        box.clear()
        with box:
            ui.label(
                f"Версія чернетки {report['revision']} · графіків: "
                f"{len(report['snapshot']['figures'])} · знімок {_when(report['created_at'])}"
            ).classes("stat-summary")
            if report.get("ai"):
                ui.label(
                    f"Пояснення AI: {report['ai']['model']} · токени: "
                    f"{report['ai']['usage_total']} · {format_usd(report['ai'].get('cost_total'))}. "
                    "Перевірте текст перед поширенням."
                ).classes("model-note")
            ui.label(
                "AI замінює лише тексти розділів поясненнями зі збережених фактів; числа, таблиці "
                "й графіки залишаються. Формування HTML не викликає AI й не повторює аналіз."
            ).classes("model-note")

    def figure_link(report: dict, name: str) -> str:
        return f"/api/projects/{project_id}/reports/{report['id']}/figures/{name}"

    with panel("Чернетка Markdown", "report-editor"):

        @ui.refreshable
        def editor():
            report = state["report"]
            preview_text = report["markdown"]
            for name in report["files"]:
                preview_text = preview_text.replace(f"]({name})", f"]({figure_link(report, name)})")
            ref["editor"] = MarkdownEditor(preview_text, max_length=100000)

        editor()

    async def sheet() -> None:
        """The current draft as one continuous document page in a dialog (feedback 13, history 140).

        Rendered in the app from the editor text — no Quarto run, nothing is saved.
        """
        report = state["report"]
        text = await ref["editor"].markdown()
        for name in report["files"]:
            text = text.replace(f"]({figure_link(report, name)})", f"]({name})")
        html = preview_html(text, lambda name: figure_link(report, name), set(report["files"]))
        with ui.dialog().props("maximized=false") as dialog, ui.element("div").classes("a4-wrap"):
            with ui.element("div").classes("a4-bar"):
                ui.label(f"{report['title']} · попередній перегляд чернетки").classes("a4-title")
                button("Закрити", dialog.close, "prev", "small")
            with ui.element("div").classes("a4-sheet"):
                ui.label(report["title"]).classes("a4-heading")
                ui.html(html, sanitize=False).classes("a4-body")
        dialog.on("hide", dialog.delete)
        dialog.open()
        await ui.run_javascript(
            "window.MathJax && MathJax.typesetPromise && "
            "MathJax.typesetPromise([document.querySelector('.a4-sheet')]);",
            timeout=10,
        )

    async def action(kind):
        if state["busy"]:
            return
        busy(True)
        status.clear()
        try:
            report = state["report"]
            text = await ref["editor"].markdown()
            for name in report["files"]:
                text = text.replace(f"]({figure_link(report, name)})", f"]({name})")
            if text != report["markdown"]:
                saved = await run.io_bound(
                    service.report_save,
                    project_id,
                    report["id"],
                    {"revision": report["revision"], "markdown": text},
                )
                if isinstance(saved, Failure):
                    with status:
                        error_line(saved)
                    return
                state["report"] = report = saved.data
            if kind != "save":
                with status:
                    progress_line(
                        "Готуємо пояснення AI…" if kind == "ai" else "Quarto формує локальний HTML…"
                    )
                result = await run.io_bound(
                    service.report_generate,
                    project_id,
                    report["id"],
                    {"revision": report["revision"]},
                    ai=kind == "ai",
                )
                status.clear()
                if isinstance(result, Failure):
                    with status:
                        error_line(result)
                    return
                state["report"] = result.data
            if kind == "ai":
                editor.refresh()
            show_summary()
            outputs.refresh()
            with status:
                ui.label(
                    "Чернетку збережено."
                    if kind == "save"
                    else "Пояснення AI готові для перевірки."
                    if kind == "ai"
                    else "HTML-звіт готовий."
                ).classes("model-note")
        finally:
            busy(False)

    show_summary()

    @ui.refreshable
    def outputs():
        report = state["report"]
        artifacts = list(reversed(report["artifacts"]))
        base = f"/api/projects/{project_id}/reports/{report['id']}/files"
        with ui.element("div").classes("model-grid model-grid-main"):
            with panel("Перегляд HTML"):
                if not artifacts:
                    ui.label(
                        "HTML ще не сформовано. Натисніть «Сформувати HTML» — перегляд з’явиться тут."
                    ).classes("model-note")
                else:
                    latest = artifacts[0]
                    if latest["revision"] != report["revision"]:
                        ui.label(
                            f"Показано версію {latest['revision']}; чернетка змінена після неї. "
                            "Сформуйте HTML ще раз, щоб побачити зміни."
                        ).classes("model-warning")
                    ui.element("iframe").props(
                        f'src="{base}/{latest["id"]}/view" title="Перегляд звіту" loading="lazy"'
                    ).classes("report-frame")
            with panel("Версії та файли"):
                if not artifacts:
                    ui.label("Сформовані версії й файли для завантаження з’являться тут.").classes(
                        "model-note"
                    )
                for artifact in artifacts:
                    prefix = f"{base}/{artifact['id']}"
                    with ui.element("div").classes("report-version"):
                        ui.label(
                            f"Версія {artifact['revision']} · {_when(artifact['created_at'])}"
                        ).classes("report-version-title")
                        ui.label(f"Quarto {artifact['quarto_version']}").classes("model-note")
                        with ui.row().classes("report-files"):
                            ui.link("Відкрити", prefix + "/view", new_tab=True).classes("btn")
                            button("HTML", partial(ui.download, prefix + "/html"), "doc")
                            button("Markdown (.qmd)", partial(ui.download, prefix + "/qmd"), "doc")
                            button("Джерело (ZIP)", partial(ui.download, prefix + "/source"), "doc")
                with ui.expansion("Графіки знімка").classes("w-full"):
                    for figure in report["snapshot"]["figures"]:
                        ui.label(figure["caption"]).classes("report-version-title")
                        ui.image(figure_link(report, figure["file"])).classes("report-figure")

    outputs()
