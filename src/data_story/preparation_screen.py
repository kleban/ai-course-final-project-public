"""Small explicit workflow: choose rules, inspect changes, confirm a saved recipe."""

from functools import partial

from nicegui import ui

from .assistant_panel import ai_shortcut
from .contracts import Failure, ProjectView, Success
from .preparation import MissingRule
from .screen import button, error_line, page_header
from .service import ProjectService

ACTION_TITLES = {"drop": "Вилучити рядки з пропуском", "constant": "Заповнити заданим значенням"}


def preparation_screen(
    service: ProjectService, active: ProjectView, dataset_id: str | None
) -> None:
    project_id = str(active.id)
    page_header(
        "Підготовка даних",
        "methods",
        "Пропуски",
        "Оберіть операції, перегляньте результат і підтвердьте збереження плану.",
    )
    ai_shortcut(
        "Порадь, як обробити пропуски", "advise", "Як краще обробити пропуски в цьому наборі?"
    )
    ui.label(
        "Оригінальні дані не змінюються. Зберігаємо план операцій, а не новий набір даних."
    ).classes("section-lead")
    listing = service.list_datasets(project_id)
    if isinstance(listing, Failure):
        error_line(listing)
        return
    choices = {str(dataset.id): dataset.name for dataset in listing.data}
    if not choices:
        ui.label("Завантажте CSV у цей проєкт, щоб підготувати дані.").classes("section-lead")
        return
    if dataset_id and dataset_id not in choices:
        ui.label("Набір не належить цьому проєкту або не знайдений.").classes("error-message")
        return
    selected = dataset_id or next(iter(choices))
    ui.select(
        choices,
        value=selected,
        label="Набір даних",
        on_change=lambda e: ui.navigate.to(
            f"/?project={project_id}&dataset={e.value}&view=preparation"
        ),
    ).props("outlined square")
    button(
        "Вибірки для регресії",
        partial(ui.navigate.to, f"/?project={project_id}&dataset={selected}&view=training"),
        "next",
    )
    button(
        "Агрегація та з'єднання (DuckDB)",
        partial(ui.navigate.to, f"/?project={project_id}&dataset={selected}&view=combine"),
        "table",
    )
    metadata = service.metadata(selected)
    if isinstance(metadata, Failure):
        error_line(metadata)
        return
    columns = {
        c.name: c
        for c in metadata.data.document.columns
        if c.role not in ("ignore", "id") and c.kind != "id"
    }
    if not columns:
        ui.label("Немає полів для підготовки. Перевірте ролі в метаданих.").classes("section-lead")
        return
    if not metadata.data.confirmed_at:
        ui.label(
            "Метадані не підтверджено. Перевірте типи, ролі й коди пропусків перед підготовкою."
        ).classes("section-lead")
    ui.label(
        "Середнє, медіана, кодування категорій і поділ 70/30 — наступні компоненти. Поради AI з’являться після налаштування OpenAI."
    ).classes("section-lead")
    state: dict = {"rules": [], "preview": None}
    with ui.row().classes("stat-controls"):
        field = ui.select(
            {
                name: f"{name} · пропусків у файлі: {c.observed.missing}"
                for name, c in columns.items()
            },
            value=next(iter(columns)),
            label="Поле",
        ).props("outlined square")
        action = ui.select(ACTION_TITLES, value="drop", label="Операція").props("outlined square")
        value = ui.input(label="Значення для заповнення").props("outlined square")
        value.set_visibility(False)
        action.on_value_change(lambda e: value.set_visibility(e.value == "constant"))
    message = ui.label("").classes("error-message")

    def invalidate() -> None:
        state["preview"] = None
        result.refresh()
        plan.refresh()

    def add_rule() -> None:
        message.set_text("")
        name = field.value
        raw: dict = {"field": name, "action": action.value}
        if action.value == "constant":
            column = columns[name]
            try:
                if column.kind == "bool":
                    text = str(value.value or "").strip().lower()
                    if text not in ("так", "ні"):
                        raise ValueError("Для логічного поля введіть так або ні.")
                    raw["value"] = text == "так"
                elif column.kind == "number" or column.observed.kind == "number":
                    raw["value"] = float(str(value.value or "").replace(",", "."))
                else:
                    raw["value"] = value.value
                MissingRule.model_validate(raw)
            except ValueError:
                message.set_text(
                    "Перевірте значення: число, непорожній текст або так/ні відповідно до типу поля."
                )
                return
        state["rules"] = [r for r in state["rules"] if r["field"] != name] + [raw]
        invalidate()

    button("Додати операцію", add_rule, "plus")

    def remove(name: str) -> None:
        state["rules"] = [r for r in state["rules"] if r["field"] != name]
        invalidate()

    def restore(rules: list[dict]) -> None:
        if any(rule["field"] not in columns for rule in rules):
            message.set_text("Поля плану не відповідають поточним ролям. Перевірте метадані.")
            return
        state["rules"] = [dict(rule) for rule in rules]
        message.set_text("")
        invalidate()
        ui.notify("Операції відновлено. Перегляньте результат перед збереженням.")

    @ui.refreshable
    def plan() -> None:
        ui.label("Вибрані операції").classes("section-title")
        if not state["rules"]:
            ui.label("Операцій ще немає — дані залишаться без змін.").classes("section-lead")
        for rule in state["rules"]:
            with ui.row().classes("stat-controls"):
                suffix = f": {rule['value']}" if rule["action"] == "constant" else ""
                ui.label(f"{rule['field']} · {ACTION_TITLES[rule['action']]}{suffix}")
                button("Прибрати", partial(remove, rule["field"]), "reset", "small")

    def preview() -> None:
        response = service.preparation_preview(
            project_id, {"dataset_id": selected, "rules": state["rules"]}
        )
        state["preview"] = response
        result.refresh()

    def save() -> None:
        previous = state["preview"]
        if not isinstance(previous, Success):
            return
        response = service.preparation_save(
            project_id,
            {
                "dataset_id": selected,
                "rules": state["rules"],
                "preview_token": previous.data["preview_token"],
            },
        )
        if isinstance(response, Failure):
            state["preview"] = response
            result.refresh()
        else:
            state["preview"] = None
            result.refresh()
            ui.notify("План підготовки збережено", type="positive")
            saved.refresh()

    @ui.refreshable
    def result() -> None:
        response = state["preview"]
        if isinstance(response, Failure):
            error_line(response)
        elif isinstance(response, Success):
            data = response.data
            report = data["report"]
            ui.label("Попередній перегляд").classes("section-title")
            ui.label(
                f"До: {report['rows_before']} · Після: {report['rows_after']} · Вилучено рядків: {report['rows_removed']}"
            ).classes("stat-context")
            for change in report["changes"]:
                ui.label(
                    f"{change['field']}: пропусків до — {change['missing_before']}; заповнено — {change['filled']}; вилучено рядків із пропуском — {change['removed_with_missing']}"
                )
            ui.label("Перші 10 рядків після підготовки").classes("section-lead")
            ui.table(
                columns=[
                    {"name": f"c{i}", "label": name, "field": f"c{i}", "align": "left"}
                    for i, name in enumerate(data["columns"])
                ],
                rows=[{f"c{i}": cell for i, cell in enumerate(row)} for row in data["rows"]],
            ).classes("w-full")
            if state["rules"]:
                button("Підтвердити й зберегти план", save, "check")

    def apply(plan_id: str) -> None:
        response = service.preparation_apply(project_id, plan_id)
        if isinstance(response, Failure):
            message.set_text(response.error.message)
        else:
            ui.navigate.to(f"/?project={project_id}&dataset={response.data.id}")

    @ui.refreshable
    def saved() -> None:
        ui.label("Збережені плани підготовки").classes("section-title")
        response = service.preparation_plans(project_id)
        if isinstance(response, Failure):
            error_line(response)
            return
        plans = [p for p in response.data if p["dataset_id"] == selected]
        if not plans:
            ui.label("Для цього набору ще немає збережених планів.").classes("section-lead")
        for item in plans:
            doc = item["document"]
            with ui.expansion(
                f"План · {item['created_at'][:19].replace('T', ' ')} UTC · {doc['report']['rows_after']} рядків"
            ).classes("w-full"):
                ui.label(f"SHA-256 джерела: {doc['sha256']}")
                for rule in doc["rules"]:
                    suffix = f": {rule['value']}" if rule["action"] == "constant" else ""
                    ui.label(f"{rule['field']} · {ACTION_TITLES[rule['action']]}{suffix}")
                ui.label("Збережений план описує вибрані операції; оригінал не змінено.").classes(
                    "section-lead"
                )

                button("Використати цей план", partial(restore, doc["rules"]), "copy")
                button("Створити підготовлений набір", partial(apply, item["id"]), "data")

    plan()
    button("Переглянути результат", preview, "compare")
    result()
    saved()
