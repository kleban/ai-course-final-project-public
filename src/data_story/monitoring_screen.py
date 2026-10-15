"""Settings → Monitoring: state of every part of the service, checked locally on demand."""

from nicegui import run, ui

from .contracts import Failure
from .monitoring import STATE_TITLES
from .screen import Shell, button, error_line, kpi, page_header
from .service import ProjectService

ICONS = {"ok": "check", "warning": "warning", "error": "warning", "not_configured": "info",
         "not_implemented": "when"}  # fmt: skip


def part(row: dict) -> None:
    with ui.element("section").classes(f"model-panel monitor-{row['state']}"):
        ui.label(row["title"]).classes("panel-title")
        with ui.element("div").classes("monitor-state"):
            ui.element("span").classes(f"ico ico-{ICONS[row['state']]}").props('aria-hidden="true"')
            ui.label(row["state_title"])
        ui.label(row["detail"]).classes("model-note mono-wrap")


def monitoring_screen(service: ProjectService) -> None:
    ui.page_title("Моніторинг · Data Story")
    Shell(service, "monitoring", trail=[("Налаштування", None), ("Моніторинг", None)])
    with ui.column().classes("page"):
        page_header(
            "Налаштування",
            "shield",
            "Моніторинг сервісу",
            "Чи працюють складові сервісу: вебсервер і API, бази даних, пошук у базі знань, "
            "звіти та доступ до OpenAI. Перевірки локальні; OpenAI тут не викликається.",
        )

        async def check() -> None:
            response = await run.io_bound(service.monitoring)
            output.clear()
            with output:
                if isinstance(response, Failure):
                    error_line(response)
                    return
                data = response.data
                summary = data["summary"]
                with ui.element("div").classes("grid grid-4"):
                    kpi("check", STATE_TITLES["ok"], str(summary["ok"]), accent=True)
                    kpi("warning", "Із зауваженнями", str(summary["warning"]))
                    kpi("warning", STATE_TITLES["error"], str(summary["error"]))
                    kpi("info", "Не налаштовані / заплановані",
                        str(summary["not_configured"] + summary["not_implemented"]))  # fmt: skip
                services = [r for r in data["rows"] if r["group"] == "service"]
                details = [r for r in data["rows"] if r["group"] == "detail"]
                with ui.element("div").classes("model-grid model-grid-three monitor-grid"):
                    for row in services:
                        part(row)
                problems = data.get("detail_problems", 0)
                title = "Технічні деталі проєкту" + (
                    f" · зауважень: {problems}" if problems else " · усе гаразд"
                )
                with ui.expansion(title).classes("w-full monitor-details"):
                    with ui.element("div").classes("model-grid model-grid-three monitor-grid"):
                        for row in details:
                            part(row)
                ui.label(
                    f"Перевірено {data['checked_at'][:19].replace('T', ' ')} UTC · "
                    f"{data['duration_ms']:.0f} мс"
                ).classes("footnote")

        button("Перевірити", check, "check", "primary")
        output = ui.column().classes("w-full")
        ui.timer(0.05, check, once=True)
