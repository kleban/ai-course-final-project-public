"""AI assistant panel: a quiet button in the corner, a side panel only when the user opens it.

Closed by default and never opens by itself; Escape or the close button hides it. The panel
knows the current page (block, selected dataset, chart, split or model), keeps a short
conversation and shows the assistant's clarifying questions with one-click answers.
"""

from typing import Any
from uuid import UUID

from nicegui import run, ui

from .ai_settings import format_usd
from .assistant import BLOCKS, MAX_HISTORY, MODES, SUGGESTIONS
from .contracts import Failure
from .metadata import KIND_TITLES, ROLE_TITLES
from .screen import button, error_line, icon, progress_line
from .service import ProjectService

HINTS = {
    "advise": "Порада без змін у даних.",
    "execute": "Помічник може запустити дозволені дії сервісу (навчити модель, побудувати "
    "графік перевірки). Перед змінами він уточнить.",
    "explain": "Пояснення збережених результатів простими словами.",
}


def _clean_page(page: dict) -> dict:
    clean: dict = {}
    for key in ("dataset_id", "run_id"):
        try:
            if page.get(key):
                clean[key] = str(UUID(str(page[key])))
        except ValueError:
            pass
    if page.get("tool"):
        clean["tool"] = str(page["tool"])[:40]
    return clean


def assistant_panel(
    service: ProjectService, project_id: str, block: str, page: dict | None = None
) -> None:
    page = _clean_page(page or {})
    hints = dict(HINTS)
    if block in ("data", "eda"):
        hints["execute"] = "Створить чернетку для перегляду. Запис — лише після «Застосувати»."
    history: list[dict] = []
    ref: dict[str, Any] = {}
    drawer = (
        ui.right_drawer(value=False, fixed=True, elevated=False)
        .props("overlay bordered width=420 behavior=desktop")
        .classes("assistant-drawer")
    )
    with drawer:
        with ui.element("div").classes("assistant-head"):
            icon("spark")
            ui.label(f"AI-помічник · {BLOCKS[block]}").classes("panel-title")
            with (
                ui.element("button")
                .classes("assistant-close")
                .props('type="button" aria-label="Закрити помічника"')
                .on("click", drawer.hide)
            ):
                ui.label("×")
        where = [BLOCKS[block]]
        if page.get("dataset_id"):
            found = service.dataset(page["dataset_id"])
            if not isinstance(found, Failure):
                where.append(f"набір «{found.data.name}»")
        if page.get("tool"):
            where.append(f"інструмент «{page['tool']}»")
        ui.label("Бачу сторінку: " + " · ".join(where)).classes("model-note assistant-where")
        mode = ui.toggle(MODES, value="explain" if block in ("model", "validation") else "advise")
        mode.props("square no-caps unelevated dense").classes("assistant-modes")
        hint = ui.label(hints[mode.value]).classes("model-note")
        mode.on_value_change(lambda e: hint.set_text(hints[e.value]))
        thread = ui.column().classes("w-full assistant-thread")
        with ui.element("div").classes("assistant-chips") as chips:
            for text in SUGGESTIONS[block]:
                with (
                    ui.element("button")
                    .classes("assistant-chip")
                    .props('type="button"')
                    .on("click", lambda _, t=text: ref["question"].set_value(t))
                ):
                    ui.label(text)
        ref["question"] = (
            ui.textarea(placeholder="Напишіть завдання або оберіть підказку")
            .props("outlined square autogrow maxlength=2000")
            .classes("w-full assistant-input")
        )
        status = ui.column().classes("w-full")

        def bubble(role: str, text: str) -> None:
            with thread:
                with ui.element("div").classes(f"assistant-msg {role}"):
                    if role == "assistant":
                        ui.markdown(text).classes("md-text")
                    else:
                        ui.label(text)

        async def send(text: str | None = None) -> None:
            if ref.get("busy"):
                return
            question = (text if text is not None else ref["question"].value or "").strip()
            status.clear()
            if not question:
                with status:
                    ui.label("Напишіть завдання або оберіть підказку.").classes("error-message")
                return
            request_page = dict(page)
            reader = getattr(ui.context.client, "assistant_chart_context", None)
            if reader:
                from pydantic import ValidationError

                try:
                    current = reader()
                except ValidationError:
                    with status:
                        ui.label("Перевірте поточні параметри оформлення графіка.").classes(
                            "error-message"
                        )
                    return
                request_page.update(
                    chart=current, tool=current["chart"], dataset_id=current["query"]["dataset_id"]
                )
            ref["busy"] = True
            ref["send"].props("disabled")
            bubble("user", question)
            ref["question"].set_value("")
            chips.set_visibility(False)
            with status:
                progress_line("Помічник працює…", "Зазвичай до хвилини.")
            try:
                response = await run.io_bound(
                    service.assistant_ask,
                    project_id,
                    {
                        "block": block,
                        "mode": mode.value,
                        "question": question,
                        "page": request_page,
                        "history": history[-MAX_HISTORY:],
                    },
                )
            finally:
                ref["busy"] = False
                ref["send"].props(remove="disabled")
            status.clear()
            if isinstance(response, Failure):
                with status:
                    error_line(response)
                return
            data = response.data
            history.extend(
                [{"role": "user", "text": question}, {"role": "assistant", "text": data["text"]}]
            )
            bubble("assistant", data["text"])
            with thread:
                for proposal in data.get("proposals", []):
                    draft_preview(service, project_id, proposal)
                if data.get("question") and data["question"]["options"]:
                    with ui.element("div").classes("assistant-chips"):
                        for option in data["question"]["options"]:
                            with (
                                ui.element("button")
                                .classes("assistant-chip answer")
                                .props('type="button"')
                                .on("click", lambda _, o=option: send(o))
                            ):
                                ui.label(option)
                details = []
                if data["tools"]:
                    details.append(
                        "виконано: "
                        + ", ".join(
                            f"{t['name']} ({'успіх' if t['ok'] else 'помилка'})"
                            for t in data["tools"]
                        )
                    )
                tokens = data["usage_total"]
                details.append(
                    f"{data['model']} · токени: {tokens if tokens is not None else 'невідомо'}"
                    f" · {format_usd(data.get('cost_total'))}"
                )
                # Only the sources the answer refers to; the full retrieval is in the journal.
                if data.get("cited"):
                    ui.label(
                        "Джерела: "
                        + "; ".join(
                            s["title"] + " — " + s["section"]
                            if s.get("kind") == "book"
                            else s["section"]
                            for s in data["cited"]
                        )
                    ).classes("model-note assistant-sources")
                ui.label(" · ".join(details)).classes("assistant-meta")

        def restart() -> None:
            history.clear()
            thread.clear()
            status.clear()
            chips.set_visibility(True)

        async def send_typed() -> None:
            await send()

        async def open_with(mode_value: str, text: str) -> None:
            """Open the panel from a shortcut in the page and run the prepared request."""
            drawer.show()
            mode.set_value(mode_value)
            await send(text)

        ui.context.client.assistant_open = open_with

        with ui.row().classes("gap-2"):
            ref["send"] = button("Надіслати", send_typed, "spark", "primary")
            button("Нова розмова", restart, "undo")
        ui.label("AI викликається лише за вашою дією; усі запити — у журналі проєкту.").classes(
            "footnote"
        )
    with (
        ui.element("button")
        .classes("assistant-fab")
        .props('type="button" aria-label="Відкрити AI-помічника" title="AI-помічник"')
        .on("click", drawer.toggle)
    ):
        icon("spark")
        ui.label("AI")
    ui.keyboard(
        on_key=lambda e: drawer.hide() if e.key == "Escape" and e.action.keydown else None,
        ignore=[],
    )


def ai_shortcut(label: str, mode: str, question: str) -> None:
    """A small «AI» link next to a result; does nothing visible without the panel."""
    opener = getattr(ui.context.client, "assistant_open", None)
    if opener is None:
        return
    with (
        ui.element("button")
        .classes("btn ghost small ai-shortcut")
        .props(f'type="button" title="AI-помічник: {question}"')
        .on("click", lambda: opener(mode, question))
    ):
        icon("spark")
        ui.label(label)


def draft_preview(service: ProjectService, project_id: str, proposal: dict) -> None:
    """Show editable project text or every proposed metadata change before explicit apply."""
    with ui.column().classes("w-full assistant-draft").style("gap: 4px"):
        is_project = proposal["kind"] == "project_description"
        is_chart = proposal["kind"] == "chart"
        is_sql = proposal["kind"] == "sql_summary"
        ui.label(
            "Пропозиція агрегації"
            if is_sql
            else "Пропозиція графіка"
            if is_chart
            else "Чернетка опису проєкту"
            if is_project
            else "Чернетка метаданих"
        ).classes("section-title")
        if is_project:
            editor = (
                ui.textarea(value=proposal["text"])
                .props("outlined square autogrow maxlength=20000")
                .classes("w-full")
            )
        elif is_chart:
            from .analysis_blocks import EDA_TOOLS
            from .chart_screen import FIELDS, GROUPS

            chart = proposal["chart"]
            ui.label(EDA_TOOLS[chart["chart"]]).classes("font-bold")
            ui.label(chart["explanation"]).classes("model-note")
            names = {
                "field": "Поле",
                "fields": "Поля",
                "x": "Вісь X",
                "y": "Вісь Y",
                "bins": "Інтервалів",
                "bin_width": "Ширина інтервалу",
                "start": "Початкове поле",
                "title": "Заголовок",
            }
            for key, label in names.items():
                value = chart["query"].get(key)
                if value is not None:
                    ui.label(f"{label}: {value}").classes("model-note")
            with ui.expansion("Оформлення"):
                options = chart["query"]["options"]
                for group in ("common", chart["chart"]):
                    ui.label(GROUPS[group]).classes("font-bold")
                    for key, _, label, _ in FIELDS[group]:
                        ui.label(f"{label}: {options[key]}").classes("model-note")
        elif is_sql:
            summary = proposal["sql_summary"]
            ui.label(summary["name"]).classes("font-bold")
            ui.label(summary["explanation"]).classes("model-note")
            preview = service.analytics_query(project_id, {k: summary[k] for k in ("sql", "limit")})
            if not isinstance(preview, Failure):
                from .combine_screen import _table

                ui.html(_table(preview.data["columns"], preview.data["rows"][:10]), sanitize=False)
                ui.label(
                    f"Рядків: {preview.data['row_count']}. "
                    "Після застосування набір буде доступний для графіків."
                ).classes("model-note")
            with ui.expansion("Запит до даних"):
                ui.code(summary["sql"], language="sql").classes("w-full")
        else:
            metadata = proposal["metadata"]
            if metadata.get("description") is not None:
                ui.label(metadata["description"]).classes("model-note")
            for column in metadata["columns"]:
                ui.label(column["name"]).classes("font-bold")
                for key, title in (
                    ("label", "Назва"),
                    ("description", "Опис"),
                    ("unit", "Одиниця"),
                    ("kind", "Тип"),
                    ("role", "Роль"),
                ):
                    if column.get(key) is not None:
                        value = column[key]
                        if key == "kind":
                            value = KIND_TITLES[value]
                        elif key == "role":
                            value = ROLE_TITLES[value]
                        ui.label(f"{title}: {value or 'порожньо'}").classes("model-note")
            ui.label(
                "Після застосування метадані залишаться чернеткою. Підтвердіть їх окремо після перевірки."
            ).classes("model-note")
        result_line = ui.label().classes("model-note")

        async def apply_draft() -> None:
            apply_button.props("disabled")
            try:
                payload = {**proposal, "text": editor.value} if is_project else proposal
                result = await run.io_bound(service.assistant_apply_draft, project_id, payload)
                if isinstance(result, Failure):
                    result_line.set_text(result.error.message)
                    apply_button.props(remove="disabled")
                    return
                if is_chart:
                    callback = getattr(ui.context.client, "assistant_chart_apply", None)
                    if callback:
                        callback(result.data)
                    result_line.set_text("Графік перебудовано в робочій області.")
                else:
                    result_line.set_text(
                        "Застосовано. Оновіть сторінку, щоб побачити збережені зміни."
                    )
                ui.notify("Чернетку застосовано", type="positive")
            except Exception:
                apply_button.props(remove="disabled")
                raise

        apply_button = button("Застосувати", apply_draft, "check", "primary")
