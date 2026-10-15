"""Model check block: one chart at a time on the test part, switchable chart and feature."""

from functools import partial

from nicegui import run, ui

from .assistant_panel import ai_shortcut
from .contracts import Failure, ProjectView
from .model_check import CHARTS
from .model_screen import panel, precise
from .screen import button, error_line, page_header, progress_line
from .service import ProjectService


def model_title(item: dict) -> str:
    doc = item["document"]
    r2 = doc["metrics"]["test"]["r2"]
    return (
        f"{doc['dataset_name']} · ціль {doc['choices']['target']} · R² {precise(r2)} · "
        f"{item['created_at'][:16].replace('T', ' ')} UTC"
    )


def validation_screen(service: ProjectService, active: ProjectView, model_id: str | None) -> None:
    project_id = str(active.id)
    page_header(
        "Перевірка моделі",
        "check",
        "Графіки перевірки",
        "Усі графіки — на тестовій частині, якої модель не бачила під час навчання.",
    )
    models = service.model_runs(project_id)
    if isinstance(models, Failure):
        error_line(models)
        return
    if not models.data:
        ui.label("Спочатку навчіть модель у блоці «Модель».").classes("section-lead")
        button("Модель", partial(ui.navigate.to, f"/?project={project_id}&view=model"), "next")
        return
    options = {item["id"]: model_title(item) for item in models.data}
    if model_id not in options:
        if model_id:
            ui.label("Ця модель не належить проєкту або не знайдена.").classes("error-message")
        model_id = models.data[0]["id"]
    by_id = {item["id"]: item for item in models.data}
    updating = False
    with ui.element("div").classes("model-grid model-top"):
        left = panel("Модель")
        right = panel("Графік")
    with left:
        selector = (
            ui.select(options, value=model_id, label="Збережена модель")
            .props("outlined square")
            .classes("model-select")
        )
    with right:
        chart = ui.toggle(CHARTS, value="regression").props("square no-caps unelevated")
        feature = (
            ui.select({}, label="Ознака для графіка регресії")
            .props("outlined square")
            .classes("model-select")
        )
    output = ui.column().classes("w-full")

    def feature_options() -> None:
        doc = by_id[selector.value]["document"]
        chosen = set(doc["choices"]["features"])
        names = [n for n in doc["feature_names"] if n in chosen]
        feature.set_options(names, value=names[0] if names else None)

    latest = 0

    async def build() -> None:
        nonlocal latest
        latest += 1
        token = latest
        feature.set_visibility(chart.value == "regression")
        output.clear()
        with output:
            progress_line("Будуємо графік…")
        raw = {"model_id": selector.value, "chart": chart.value}
        if chart.value == "regression" and feature.value:
            raw["feature"] = feature.value
        response = await run.io_bound(service.model_check, project_id, raw)
        if token != latest:  # a newer choice was made while this chart was being built
            return
        output.clear()
        with output:
            if isinstance(response, Failure):
                error_line(response)
                return
            show(response.data)

    def show(view: dict) -> None:
        with ui.element("div").classes("model-grid model-grid-main"):
            with ui.element("section").classes("model-panel"):
                with ui.element("div").classes("heatmap-wrap"):
                    ui.html(view["svg"], sanitize=False).classes("diagnostic-wrap")
                ui.label("Рівняння моделі").classes("panel-title equation-title")
                formula = ui.element("div").classes("math-formula model-equation")
                formula._props["data-formula"] = view["equation_tex"]
                ui.label(view["equation"]).classes("model-note mono-wrap equation-text")
                if view.get("partial"):
                    ui.label(
                        f"Лінія на графіку (інші ознаки — на середніх): {view['partial']}"
                    ).classes("model-note mono-wrap")
            with ui.element("section").classes("model-panel"):
                ui.label("Як читати").classes("panel-title")
                ai_shortcut(
                    "Поясни цей графік",
                    "explain",
                    f"Поясни графік «{view['title']}» цієї моделі: що на ньому видно "
                    "й чи є ознаки проблем?",
                )
                ui.label(view["hint"]).classes("model-explain")
                for note in view["notes"] + view["warnings"]:
                    with ui.element("div").classes("model-warning"):
                        ui.label(note)
                ui.label(f"Тестових рядків: {view['n']}").classes("model-note")

    async def model_changed(_event) -> None:
        nonlocal updating
        updating = True
        try:
            feature_options()
        finally:
            updating = False
        await build()

    async def chart_changed(_event) -> None:
        await build()

    async def feature_changed(_event) -> None:
        if not updating and chart.value == "regression":
            await build()

    selector.on_value_change(model_changed)
    chart.on_value_change(chart_changed)
    feature.on_value_change(feature_changed)
    feature_options()
    ui.timer(0.05, build, once=True)
