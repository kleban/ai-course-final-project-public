"""Joins and summaries (DuckDB): the user assembles steps, the service writes and runs the SQL."""

from functools import partial
from html import escape
from typing import Any

from nicegui import run, ui

from .assistant_panel import ai_shortcut
from .combine import FUNC_TITLES
from .contracts import Failure, ProjectView
from .model_screen import panel
from .screen import button, error_line, page_header
from .service import ProjectService

OPS = {
    "eq": "дорівнює",
    "ne": "не дорівнює",
    "gt": "більше",
    "ge": "не менше",
    "lt": "менше",
    "le": "не більше",
    "in": "один із (через кому)",
    "is_null": "порожнє",
    "not_null": "не порожнє",
}
KIND_TITLES = {"number": "число", "category": "категорія", "text": "текст", "date": "дата",
               "bool": "так/ні", "id": "ідентифікатор"}  # fmt: skip


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return escape(str(value))
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{value:,.0f}".replace(",", "\u202f")
    text = f"{value:,.2f}" if abs(value) >= 100 else f"{value:.4g}"
    return text.replace(",", "\u202f").replace(".", ",")


def _table(columns: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{escape(str(c))}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(
            f"<td>{_cell(v)}</td>" for v in row
        ) + "</tr>"
        for row in rows
    )  # fmt: skip
    return f'<div class="combine-table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def combine_screen(service: ProjectService, active: ProjectView, dataset_id: str | None) -> None:
    project_id = str(active.id)
    page_header(
        "Підготовка даних",
        "methods",
        "Агрегація та з'єднання",
        "Оберіть кроки — сервіс сам складе SQL і виконає його в DuckDB. Оригінали не змінюються; "
        "результат можна зберегти як новий набір.",
    )
    ai_shortcut(
        "Порадь агрегацію",
        "advise",
        "Яка агрегація чи з'єднання наборів допоможе відповісти на питання проєкту?",
    )
    ai_shortcut(
        "Підготуй агрегацію",
        "execute",
        "Підготуй пропозицію агрегації за питанням і метою проєкту. "
        "Покажи таблицю результатів для перегляду перед збереженням. "
        "Якщо мети недостатньо — постав одне коротке уточнення.",
    )
    listing = service.list_datasets(project_id)
    if isinstance(listing, Failure):
        error_line(listing)
        return
    datasets = {str(d.id): d.name for d in listing.data}
    if not datasets:
        ui.label("Завантажте CSV у цей проєкт, щоб робити агрегації.").classes("section-lead")
        return
    base = dataset_id if dataset_id in datasets else next(iter(datasets))
    fields_cache: dict[str, list[dict]] = {}

    def fields(ds: str) -> list[dict]:
        if ds not in fields_cache:
            result = service.summary_fields(project_id, ds)
            fields_cache[ds] = [] if isinstance(result, Failure) else result.data
        return fields_cache[ds]

    state: dict[str, Any] = {
        "join": False,
        "join_dataset": next((d for d in datasets if d != base), base),
        "left_key": None,
        "right_key": None,
        "how": "inner",
        "join_columns": [],
        "filters": [],
        "group_by": [],
        "aggregates": [],
        "order_by": None,
        "descending": True,
        "limit": None,
        "preview": None,
    }

    def output_fields() -> dict[str, str]:
        """Names available after the join (the same rule as the service: clashes get _2)."""
        names = {f["name"]: f["kind"] for f in fields(base)}
        if state["join"]:
            for f in fields(state["join_dataset"]):
                if state["join_columns"] and f["name"] not in state["join_columns"]:
                    continue
                if f["name"] == state["right_key"] and state["right_key"] == state["left_key"]:
                    continue
                name = f["name"] if f["name"] not in names else f"{f['name']}_2"
                while name in names:
                    name += "_2"
                names[name] = f["kind"]
        return names

    def labelled(names: dict[str, str]) -> dict[str, str]:
        return {n: f"{n} · {KIND_TITLES.get(k, k)}" for n, k in names.items()}

    def recipe() -> dict:
        data: dict[str, Any] = {"dataset_id": base}
        if state["join"]:
            data["join"] = {
                "dataset_id": state["join_dataset"],
                "left_key": state["left_key"] or "",
                "right_key": state["right_key"] or "",
                "how": state["how"],
                "columns": state["join_columns"] or None,
            }
        filters = []
        for item in state["filters"]:
            value: Any = item.get("value")
            if item["op"] == "in":
                value = [v.strip() for v in str(value or "").split(",") if v.strip()]
            filters.append({"field": item["field"], "op": item["op"], "value": value})
        data["filters"] = filters
        data["group_by"] = state["group_by"]
        data["aggregates"] = [
            {"func": a["func"], "field": a.get("field") or None, "name": a.get("name") or None}
            for a in state["aggregates"]
        ]
        data["order_by"] = state["order_by"] or None
        data["descending"] = state["descending"]
        data["limit"] = int(state["limit"]) if state["limit"] else None
        return data

    ui.select(
        datasets,
        value=base,
        label="Основний набір",
        on_change=lambda e: ui.navigate.to(
            f"/?project={project_id}&dataset={e.value}&view=combine"
        ),
    ).props("outlined square").classes("w-full")
    button(
        "До пропусків",
        partial(ui.navigate.to, f"/?project={project_id}&dataset={base}&view=preparation"),
        "prev",
        "small back",
    )

    def remove(key: str, index: int) -> None:
        state[key].pop(index)
        builder.refresh()

    def add(key: str, item: dict) -> None:
        state[key].append(dict(item))
        builder.refresh()

    @ui.refreshable
    def builder() -> None:
        available = output_fields()
        with ui.element("div").classes("model-grid model-grid-main"):
            with panel("1. З'єднання з іншим набором"):
                ui.switch("Додати поля з іншого набору", value=state["join"]).on_value_change(
                    lambda e: (state.update(join=e.value), builder.refresh())
                )
                if state["join"]:
                    others = {k: v for k, v in datasets.items() if k != base} or datasets
                    if state["join_dataset"] not in others:
                        state["join_dataset"] = next(iter(others))
                    ui.select(others, value=state["join_dataset"], label="Набір для з'єднання").props(
                        "outlined square"
                    ).classes("w-full").on_value_change(
                        lambda e: (state.update(join_dataset=e.value, right_key=None, join_columns=[]),
                                   builder.refresh())
                    )  # fmt: skip
                    left = labelled({f["name"]: f["kind"] for f in fields(base)})
                    right = labelled({f["name"]: f["kind"] for f in fields(state["join_dataset"])})
                    with ui.row().classes("stat-controls"):
                        ui.select(left, value=state["left_key"], label="Ключ в основному").props(
                            "outlined square"
                        ).classes("combine-field").on_value_change(
                            lambda e: (state.update(left_key=e.value), builder.refresh())
                        )
                        ui.select(right, value=state["right_key"], label="Ключ у другому").props(
                            "outlined square"
                        ).classes("combine-field").on_value_change(
                            lambda e: (state.update(right_key=e.value), builder.refresh())
                        )
                        ui.select(
                            {"inner": "Лише збіги (INNER)", "left": "Усі рядки основного (LEFT)"},
                            value=state["how"],
                            label="Тип з'єднання",
                        ).props("outlined square").classes("combine-field").on_value_change(
                            lambda e: state.update(how=e.value)
                        )
                    ui.select(
                        list(right_names := [f["name"] for f in fields(state["join_dataset"])]),
                        value=[c for c in state["join_columns"] if c in right_names],
                        multiple=True,
                        label="Які поля додати (порожньо — усі)",
                    ).props("outlined square use-chips").classes("w-full").on_value_change(
                        lambda e: (state.update(join_columns=e.value or []), builder.refresh())
                    )
                else:
                    ui.label("Без з'єднання агрегація рахується лише з основного набору.").classes(
                        "model-note"
                    )
            with panel("2. Умови відбору рядків"):
                for index, item in enumerate(state["filters"]):
                    with ui.row().classes("stat-controls"):
                        ui.select(labelled(available), value=item["field"], label="Поле").props(
                            "outlined square"
                        ).classes("combine-field").on_value_change(
                            lambda e, i=item: i.update(field=e.value)
                        )
                        ui.select(OPS, value=item["op"], label="Умова").props(
                            "outlined square"
                        ).classes("combine-op").on_value_change(
                            lambda e, i=item: (i.update(op=e.value), builder.refresh())
                        )
                        if item["op"] not in ("is_null", "not_null"):
                            ui.input("Значення", value=item.get("value") or "").props(
                                "outlined square"
                            ).classes("combine-field").on_value_change(
                                lambda e, i=item: i.update(value=e.value)
                            )
                        button(
                            "Прибрати",
                            partial(remove, "filters", index),
                            "reset",
                            "ghost",
                        )
                if not state["filters"]:
                    ui.label("Без умов — беремо всі рядки.").classes("model-note")
                button(
                    "Додати умову",
                    partial(
                        add, "filters", {"field": next(iter(available)), "op": "eq", "value": ""}
                    ),
                    "plus",
                )
        with ui.element("div").classes("model-grid model-grid-main"):
            with panel("3. Групування й показники"):
                ui.select(
                    labelled(available),
                    value=[g for g in state["group_by"] if g in available],
                    multiple=True,
                    label="Групувати за (до 5 полів; порожньо — без групування)",
                ).props("outlined square use-chips").classes("w-full").on_value_change(
                    lambda e: state.update(group_by=e.value or [])
                )
                funcs = {k: v for k, v in FUNC_TITLES.items()}
                for index, item in enumerate(state["aggregates"]):
                    with ui.row().classes("stat-controls"):
                        ui.select(funcs, value=item["func"], label="Показник").props(
                            "outlined square"
                        ).classes("combine-op").on_value_change(
                            lambda e, i=item: i.update(func=e.value)
                        )
                        ui.select(
                            {"": "— усі рядки —", **labelled(available)},
                            value=item.get("field") or "",
                            label="Поле",
                        ).props("outlined square").classes("combine-field").on_value_change(
                            lambda e, i=item: i.update(field=e.value or None)
                        )
                        ui.input(
                            "Назва стовпця (необов'язково)", value=item.get("name") or ""
                        ).props("outlined square").classes("combine-field").on_value_change(
                            lambda e, i=item: i.update(name=e.value)
                        )
                        button(
                            "Прибрати",
                            partial(remove, "aggregates", index),
                            "reset",
                            "ghost",
                        )
                if not state["aggregates"]:
                    ui.label(
                        "Без показників результатом буде таблиця рядків (після з'єднання й умов)."
                    ).classes("model-note")
                button(
                    "Додати показник",
                    partial(add, "aggregates", {"func": "count", "field": None, "name": ""}),
                    "plus",
                )
            with panel("4. Порядок і обсяг"):
                ui.input(
                    "Сортувати за (стовпець результату, напр. median_Price)",
                    value=state["order_by"] or "",
                ).props("outlined square").classes("w-full").on_value_change(
                    lambda e: state.update(order_by=e.value.strip() or None)
                )
                ui.switch("За спаданням", value=state["descending"]).on_value_change(
                    lambda e: state.update(descending=e.value)
                )
                ui.number("Не більше рядків (необов'язково)", value=state["limit"], min=1,
                          format="%.0f").props("outlined square").classes("w-full").on_value_change(
                    lambda e: state.update(limit=e.value)
                )  # fmt: skip

    builder()
    with ui.row().classes("report-actions"):
        button("Переглянути результат", lambda: preview(), "search", "primary")
    status = ui.column().classes("w-full")
    result_box = ui.column().classes("w-full")

    async def preview() -> None:
        status.clear()
        result_box.clear()
        state["preview"] = None
        data = recipe()
        response = await run.io_bound(service.summary_preview, project_id, data)
        if isinstance(response, Failure):
            with status:
                error_line(response)
            return
        state["preview"] = (data, response.data)
        show(response.data)

    def show(data: dict) -> None:
        with result_box:
            with panel("Результат", "combine-result"):
                ui.label(
                    f"Рядків у результаті: {data['row_count']:,}".replace(",", " ")
                    + f" · показано перші {len(data['rows'])}"
                ).classes("stat-summary")
                for warning in data["warnings"]:
                    ui.label(warning).classes("model-warning")
                ui.html(_table(data["columns"], data["rows"]), sanitize=False).classes("w-full")
                with ui.expansion("SQL, який склав сервіс (лише для читання)").classes("w-full"):
                    ui.code(data["sql"], language="sql").classes("w-full")
                with ui.row().classes("stat-controls"):
                    name = (
                        ui.input("Назва нового набору (необов'язково)")
                        .props("outlined square maxlength=160")
                        .classes("combine-field")
                    )
                    button("Зберегти як новий набір", lambda: save(name.value), "check", "primary")

    async def save(name: str) -> None:
        if not state["preview"]:
            return
        data, shown = state["preview"]
        response = await run.io_bound(
            service.summary_create,
            project_id,
            {"recipe": data, "preview_token": shown["preview_token"], "name": name or None},
        )
        status.clear()
        if isinstance(response, Failure):
            with status:
                error_line(response)
            return
        ui.navigate.to(f"/?project={project_id}&dataset={response.data.id}")
