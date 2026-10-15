"""Conclusions block: saved model (+ DuckDB summaries) → explicit AI draft with cited sources →
author's edit and approval → the approved text enters new reports."""

from typing import Any

from nicegui import run, ui

from .ai_settings import format_usd
from .contracts import Failure
from .markdown_editor import MarkdownEditor
from .model_screen import panel
from .reports import number
from .screen import button, error_line, page_header, progress_line


def _when(value: str) -> str:
    return value[:16].replace("T", " ") + " UTC"


def conclusions_screen(service, active, conclusion_id=None):
    project_id = str(active.id)
    page_header(
        "Етап 6 · " + active.name,
        "spark",
        "Висновки",
        "Факти збереженої моделі та агрегацій DuckDB → чернетка AI з посиланнями на правила й "
        "книжки → ваше редагування й затвердження → висновки у звіті. AI — лише за вашою дією.",
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
    listing = service.conclusion_list(project_id)
    summaries = service.conclusion_summaries(project_id)
    for result in (listing, summaries):
        if isinstance(result, Failure):
            error_line(result)
            return
    items = {c["id"]: c for c in listing.data}
    selected = conclusion_id or next(iter(items), None)
    if selected is not None and selected not in items:
        ui.label("Висновки не знайдено в цьому проєкті.").classes("error-message")
        selected = next(iter(items), None)
    model_names = {
        m["id"]: (
            f"{m['document']['dataset_name']} · {m['document']['choices']['target']} · "
            f"{_when(m['created_at'])}"
        )
        for m in models.data
    }

    targets = {
        m["id"]: f"{m['document']['dataset_name']} → {m['document']['choices']['target']}"
        for m in models.data
    }
    state: dict[str, Any] = {"item": items.get(selected), "busy": False}
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

    def link(item_id: str) -> str:
        return f"/?project={project_id}&view=conclusions&run={item_id}"

    with ui.element("div").classes("model-grid model-top report-top"):
        with panel("Висновки проєкту"):
            if state["item"] is None:
                ui.label(
                    "Висновків ще немає. Оберіть модель праворуч і підготуйте чернетку AI."
                ).classes("model-note")
            else:
                ui.select(
                    {
                        k: ("✓ " if c["status"] == "approved" else "")
                        + f"{_when(c['created_at'])} · модель: {targets.get(c['model_id'], '?')}"
                        for k, c in items.items()
                    },
                    value=selected,
                    label="Версії висновків",
                    on_change=lambda e: ui.navigate.to(link(e.value)),
                ).props("outlined square").classes("w-full")
                ref["summary"] = ui.column().classes("w-full gap-0")
                with ui.row().classes("report-actions"):
                    controls.append(button("Зберегти", lambda: action("save"), "check"))
                    controls.append(
                        button("Затвердити висновки", lambda: action("approve"), "check", "primary")
                    )
                    controls.append(button("Створити звіт із висновками", lambda: report(), "doc"))
        with panel("Нові висновки"):
            model = (
                ui.select(model_names, value=next(iter(model_names)), label="Збережена модель")
                .props("outlined square")
                .classes("w-full")
            )
            options = {s["id"]: f"{s['name']} · {s['rows']} рядків" for s in summaries.data}
            chosen = (
                ui.select(options, value=[], multiple=True, label="Агрегації DuckDB (до 3)")
                .props("outlined square use-chips")
                .classes("w-full")
            )
            if not options:
                chosen.set_enabled(False)
                ui.link(
                    "Агрегацій ще немає — створити в «Підготовка даних → Агрегація та з'єднання»",
                    f"/?project={project_id}&view=combine",
                ).classes("model-note")
            ui.label(
                "Один виклик AI: числа моделі й агрегацій (до 20 рядків кожного) та 3–5 фрагментів "
                "правил і книжок, знайдених за цими фактами. Аналіз не повторюється."
            ).classes("model-note")
            controls.extend([model, chosen])
            controls.append(
                button("Підготувати висновки AI", lambda: prepare(), "spark", "primary")
            )
    status = ui.column().classes("w-full report-status")

    async def prepare():
        if state["busy"]:
            return
        if len(chosen.value or []) > 3:
            with status:
                ui.label("Оберіть не більше трьох агрегацій.").classes("error-message")
            return
        busy(True)
        status.clear()
        with status:
            progress_line(
                "Шукаємо джерела й готуємо висновки AI…",
                "Зазвичай до хвилини: пошук у базі знань і один запит до OpenAI.",
            )
        try:
            result = await run.io_bound(
                service.conclusion_prepare,
                project_id,
                {"model_id": model.value, "summary_ids": list(chosen.value or [])},
            )
            status.clear()
            if isinstance(result, Failure):
                with status:
                    error_line(result)
                return
            ui.navigate.to(link(result.data["id"]))
        finally:
            busy(False)

    if state["item"] is None:
        return

    def show_summary() -> None:
        item = state["item"]
        box = ref["summary"]
        box.clear()
        with box:
            if item["status"] == "approved":
                ui.label(
                    f"Затверджено автором · {_when(item['approved_at'])} · версія {item['revision']}"
                ).classes("stat-summary")
                ui.label(
                    "Нові звіти цієї моделі містять ці висновки й джерела. Будь-яка зміна тексту "
                    "повертає висновки в чернетку до повторного затвердження."
                ).classes("model-note")
            else:
                ui.label(f"Чернетка · версія {item['revision']} · не затверджено").classes(
                    "stat-summary"
                )
                ui.label(
                    "Перевірте числа й посилання [n] за фактами та джерелами нижче, відредагуйте "
                    "й затвердіть. До звіту потрапляють лише затверджені висновки."
                ).classes("model-warning")
            ai = item["ai"]
            ui.label(
                f"Чернетка AI: {ai['model']} · токени: {ai['usage_total']} · "
                f"{format_usd(ai.get('cost_total'))} · "
                f"джерел: {len(item['sources'])}"
            ).classes("model-note")

    with panel("Текст висновків (Markdown, посилання [1]…[n])", "report-editor"):

        @ui.refreshable
        def editor():
            ref["editor"] = MarkdownEditor(state["item"]["markdown"], max_length=8000)

        editor()

    async def action(kind: str) -> None:
        if state["busy"]:
            return
        busy(True)
        status.clear()
        try:
            item = state["item"]
            text = (await ref["editor"].markdown()).strip()
            result = await run.io_bound(
                service.conclusion_save,
                project_id,
                item["id"],
                {"revision": item["revision"], "markdown": text},
                approve=kind == "approve",
            )
            if isinstance(result, Failure):
                with status:
                    error_line(result)
                return
            state["item"] = result.data
            editor.refresh()
            show_summary()
            with status:
                ui.label(
                    "Висновки затверджено." if kind == "approve" else "Чернетку збережено."
                ).classes("model-note")
        finally:
            busy(False)

    async def report() -> None:
        if state["busy"]:
            return
        item = state["item"]
        status.clear()
        if item["status"] != "approved":
            with status:
                ui.label(
                    "Спершу затвердьте висновки — у звіт потрапляють лише затверджені."
                ).classes("model-warning")
            return
        busy(True)
        try:
            result = await run.io_bound(
                service.report_create,
                project_id,
                {"model_id": item["model_id"], "title": "Аналітичний звіт · " + active.name},
            )
            if isinstance(result, Failure):
                with status:
                    error_line(result)
                return
            ui.navigate.to(f"/?project={project_id}&view=report&run={result.data['id']}")
        finally:
            busy(False)

    show_summary()
    item = state["item"]
    with ui.element("div").classes("model-grid model-grid-main"):
        with panel("Джерела"):
            for source in item["sources"]:
                kind = "Книжка" if source["kind"] == "book" else "Правило"
                with ui.element("div").classes("report-version"):
                    ui.label(f"[{source['n']}] {source['title']}").classes("report-version-title")
                    ui.label(f"{kind} · {source['section']} · {source['source']}").classes(
                        "model-note"
                    )
                    with ui.expansion("Фрагмент").classes("w-full"):
                        ui.label(source["text"]).classes("model-note")
        with panel("Факти, передані AI"):
            facts = item["facts"]
            rows = [
                ("Ціль", facts["target"]),
                ("Ознаки", ", ".join(facts["features"])),
                ("R² train / test", f"{number(facts['r2_train'])} / {number(facts['r2_test'])}"),
                ("MAE test", number(facts["mae_test"])),
                ("MAE простого прогнозу", number(facts["baseline_mae"])),
                ("Значущі (p < 0,05)", ", ".join(facts["significant_p_lt_0_05"]) or "немає"),
            ]
            for label, value in rows:
                with ui.row().classes("w-full justify-between"):
                    ui.label(label).classes("model-note")
                    ui.label(str(value))
            for summary in item["summaries"]:
                with ui.expansion(f"Агрегація: {summary['name']}").classes("w-full"):
                    ui.label(summary["sql"]).classes("model-note mono")
                    ui.table(
                        columns=[
                            {"name": str(i), "label": c, "field": str(i)}
                            for i, c in enumerate(summary["columns"])
                        ],
                        rows=[
                            {str(i): number(v) for i, v in enumerate(row)}
                            for row in summary["rows"]
                        ],
                    ).props("dense flat square").classes("w-full")
