"""Settings → AI models: OpenAI model per pipeline task, limits, key state from .env."""

from nicegui import run, ui

from .ai_settings import PRICES_SOURCE, format_usd
from .contracts import Failure
from .model_screen import panel
from .screen import Shell, button, error_line, error_text, page_header, progress_line
from .service import ProjectService

# The key field is deliberately inert: the value stays in the browser, is never sent to the
# server and is cleared on click (D-76: the key comes only from .env).
FAKE_KEY = """
<div class="fake-key">
  <input type="password" autocomplete="off" spellcheck="false" placeholder="sk-…"
         aria-label="Новий ключ OpenAI (не зберігається)">
  <button type="button" class="btn small" onclick="
    this.previousElementSibling.value='';
    this.nextElementSibling.textContent='Ключ не надіслано й не збережено. Щоб змінити ключ, ' +
      'відредагуйте OPENAI_API_KEY у файлі .env і перезапустіть сервіс.';">Замінити ключ</button>
  <div class="model-note" aria-live="polite"></div>
</div>
"""
PRICE_FIELDS = {
    "input": "Вхід",
    "cached_input": "Вхід із кешу",
    "cache_write": "Запис у кеш",
    "output": "Вихід (з міркуванням)",
}
KEY_STATES = {
    "found": "Ключ знайдено",
    "empty": "Ключ не вписано",
    "missing_file": "Немає файла .env",
    "suspicious": "Перевірте ключ",
}


def ai_settings_screen(service: ProjectService) -> None:
    ui.page_title("Моделі AI · Data Story")
    Shell(service, "ai_settings", trail=[("Налаштування", None), ("Моделі AI", None)])
    with ui.column().classes("page"):
        page_header(
            "Налаштування",
            "spark",
            "Моделі AI",
            "Яка модель OpenAI виконує яку задачу аналізу. Налаштування спільні для сервісу; "
            "AI викликається лише явною дією користувача.",
        )
        response = service.ai_configuration()
        if isinstance(response, Failure):
            error_line(response)
            return
        data = response.data
        config, key = data["configuration"], data["key"]
        models = {m["id"]: m for m in data["models"]}
        selects: dict[str, ui.select] = {}
        fallback_selects: dict[str, list[ui.select]] = {}
        price_inputs: dict[str, dict[str, ui.number]] = {}
        with ui.element("div").classes("model-grid model-grid-three model-top"):
            with panel("Ключ OpenAI"):
                ui.label(
                    KEY_STATES[key["state"]] + (f" · {key['masked']}" if key["masked"] else "")
                ).classes("stat-context ai-key-state")
                ui.label(key["message"]).classes("model-note")
                ui.label(
                    "Джерело ключа — лише файл .env у корені проєкту (OPENAI_API_KEY). "
                    "Поле нижче не передає ключ на сервер."
                ).classes("model-note")
                ui.html(FAKE_KEY, sanitize=False)
            with panel("Ліміти"):
                tokens = ui.number(
                    "Максимум вихідних токенів на відповідь",
                    value=config["max_output_tokens"],
                    min=256,
                    max=32000,
                    step=256,
                ).props("outlined square")
                retries = ui.number(
                    "Повтори на одній моделі (тайм-аут, мережа, 429, 5xx)",
                    value=config["retries"],
                    min=0,
                    max=5,
                    step=1,
                ).props("outlined square")
                steps = ui.number(
                    "Максимум кроків агента", value=config["max_agent_steps"], min=1, max=20, step=1
                ).props("outlined square")
                ui.label("Ліміти зупиняють відповідь або агента зі зрозумілим станом.").classes(
                    "model-note"
                )
            with panel("Ціни моделей · $ за 1 млн токенів"):
                for model_id, price in config["prices"].items():
                    ui.label(model_id).classes("knowledge-section")
                    with ui.element("div").classes("price-grid"):
                        price_inputs[model_id] = {
                            field: ui.number(
                                title, value=price[field], min=0, max=1000, step=0.005,
                                format="%.3f",
                            ).props(f'outlined square dense data-price="{model_id}-{field}"')
                            for field, title in PRICE_FIELDS.items()
                        }  # fmt: skip
                checked = ui.input("Тариф перевірено (дата)", value=config["prices_checked"]).props(
                    "outlined square dense"
                )
                ui.label(
                    "Кожен вхідний токен оплачується за одним тарифом: звичайним, із кешу або "
                    "запису в кеш. Вартість рахується з фактичних токенів відповіді OpenAI; "
                    "без токенів або тарифу — «невідомо». Стандартний рівень, контекст до 272 тис."
                ).classes("model-note")
                ui.link("Джерело: тарифи OpenAI", PRICES_SOURCE, new_tab=True).classes("model-note")
        with ui.element("div").classes("model-grid model-grid-main"):
            with panel("Перевірка з'єднання"):
                ui.label(
                    "«Перевірити доступ» лише питає OpenAI, чи доступні обрані моделі, без "
                    "генерації тексту. «Пробний запит» — один короткий платний запит "
                    "(до 256 вихідних токенів)."
                ).classes("model-note")
                check_out = ui.column().classes("w-full")

                async def check() -> None:
                    check_out.clear()
                    with check_out:
                        progress_line("Перевіряємо доступ до OpenAI…")
                    result = await run.io_bound(service.ai_check_access)
                    check_out.clear()
                    with check_out:
                        if isinstance(result, Failure):
                            error_line(result)
                        else:
                            for row in result.data:
                                ui.label(f"{row['model']}: {row['message']}").classes(
                                    "stat-context" if row["available"] else "error-message"
                                )
                    calls.refresh()

                async def test() -> None:
                    check_out.clear()
                    with check_out:
                        progress_line("Надсилаємо пробний запит…")
                    result = await run.io_bound(service.ai_test_request)
                    check_out.clear()
                    with check_out:
                        if isinstance(result, Failure):
                            error_line(result)
                        else:
                            ui.label(result.data["text"] or "(порожня відповідь)").classes(
                                "model-explain"
                            )
                            total = result.data["usage_total"]
                            ui.label(
                                f"{result.data['model']} (рівень {result.data['level']}) · "
                                f"спроб: {len(result.data['attempts'])} · токени всіх спроб: "
                                f"{total if total is not None else 'невідомо'} · вартість: "
                                f"{format_usd(result.data.get('cost_total'))}"
                            ).classes("model-note")
                    calls.refresh()

                with ui.row().classes("gap-2"):
                    button("Перевірити доступ", check, "check")
                    button("Пробний запит", test, "spark")
            with panel("Журнал викликів AI"):

                @ui.refreshable
                def calls() -> None:
                    response = service.ai_calls()
                    if isinstance(response, Failure):
                        error_line(response)
                        return
                    totals = response.data["totals"]
                    ui.label(
                        f"Викликів: {totals['calls']} · помилок: {totals['errors']} · "
                        f"відомих токенів: {totals['total_tokens']} "
                        f"(вхід {totals['input_tokens']}, вихід {totals['output_tokens']}) · "
                        f"без даних про токени: {totals['unknown_usage']}"
                    ).classes("model-note")
                    ui.label(
                        f"Вартість відомих викликів: {format_usd(totals['cost_usd'])}"
                        + (
                            f" (з них {totals['cost_estimated']} оцінено за поточним тарифом)"
                            if totals["cost_estimated"]
                            else ""
                        )
                        + (f" · без вартості: {totals['cost_unknown']}" if totals["cost_unknown"]
                           else "")
                    ).classes("stat-context ai-cost-total")  # fmt: skip
                    daily_costs(response.data["daily"])
                    if not response.data["calls"]:
                        ui.label("Викликів ще не було.").classes("model-note")
                    for call in response.data["calls"][:8]:
                        usage = call.get("usage") or {}
                        tokens = usage.get("total_tokens")
                        usd = (call.get("cost") or {}).get("usd")
                        line = (
                            f"{call['started_at'][:19].replace('T', ' ')} UTC · {call['task']} · "
                            f"{call['model']} · "
                            + ("успіх" if call["status"] == "success" else "помилка")
                            + (f" · {tokens} ток." if tokens is not None else "")
                            + (f" · {format_usd(usd)}" if usd is not None else "")
                        )
                        ui.label(line).classes("model-note mono-wrap")
                        if call.get("error"):
                            ui.label(call["error"]).classes("error-message")

                calls()
        with ui.element("div").classes("model-grid model-grid-three ai-stages"):
            for stage in data["stages"]:
                with panel(stage):
                    for task in [t for t in data["tasks"] if t["stage"] == stage]:
                        allowed = {
                            m: "Без OpenAI (локально)" if m == "local" else m
                            for m in models
                            if models[m]["kind"] in (task["kind"], "local")
                        }
                        with ui.element("div").classes("ai-task"):
                            ui.label(f"{task['title']} · {task['mode']}").classes(
                                "knowledge-section"
                            )
                            ui.label(task["purpose"]).classes("model-note")
                            selects[task["key"]] = (
                                ui.select(
                                    allowed,
                                    value=config["models"][task["key"]],
                                    label="Рівень 1 · основна модель",
                                )
                                .props(f'outlined square dense data-task="{task["key"]}"')
                                .classes("model-select")
                            )
                            if task["kind"] == "text":
                                spare = {
                                    "": "—",
                                    **{m: t for m, t in allowed.items() if m != "local"},
                                }
                                chain = config["fallbacks"].get(task["key"], [])
                                fallback_selects[task["key"]] = [
                                    ui.select(
                                        spare,
                                        value=chain[i] if i < len(chain) else "",
                                        label=f"Рівень {i + 2} · якщо попередній не впорався",
                                    )
                                    .props(
                                        f'outlined square dense data-level="{task["key"]}-{i + 2}"'
                                    )
                                    .classes("model-select")
                                    for i in range(2)
                                ]
        message = ui.label("").classes("model-note")

        def save() -> None:
            raw = {
                "models": {key: select.value for key, select in selects.items()},
                "fallbacks": {
                    key: [select.value for select in levels if select.value]
                    for key, levels in fallback_selects.items()
                },
                "retries": int(retries.value if retries.value is not None else -1),
                "max_output_tokens": int(tokens.value or 0),
                "max_agent_steps": int(steps.value or 0),
                "prices": {
                    model_id: {
                        field: float(box.value) if box.value is not None else -1.0
                        for field, box in boxes.items()
                    }
                    for model_id, boxes in price_inputs.items()
                },
                "prices_checked": str(checked.value or ""),
            }
            result = service.save_ai_configuration(raw)
            if isinstance(result, Failure):
                message.set_text(error_text(result))
                message.classes(replace="error-message")
            else:
                message.set_text("Налаштування збережено. Ключ не змінювався.")
                message.classes(replace="stat-context")
                calls.refresh()

        def reset() -> None:
            result = service.reset_ai_configuration()
            if isinstance(result, Failure):
                message.set_text(error_text(result))
                return
            for key, select in selects.items():
                select.set_value(result.data["models"][key])
            for key, levels in fallback_selects.items():
                chain = result.data["fallbacks"].get(key, [])
                for i, select in enumerate(levels):
                    select.set_value(chain[i] if i < len(chain) else "")
            retries.set_value(result.data["retries"])
            tokens.set_value(result.data["max_output_tokens"])
            steps.set_value(result.data["max_agent_steps"])
            for model_id, boxes in price_inputs.items():
                for field, box in boxes.items():
                    box.set_value(result.data["prices"][model_id][field])
            checked.set_value(result.data["prices_checked"])
            calls.refresh()
            message.set_text("Повернуто типові моделі й ліміти.")

        with ui.row().classes("gap-2"):
            button("Зберегти", save, "check", "primary")
            button("Повернути типові", reset, "undo")
        for model in data["models"]:
            ui.label(model["title"]).classes("model-note")
        ui.label(
            "Моделі — кандидати з офіційного каталогу OpenAI (перевірено 2026-10-10); доступ "
            "у вашому обліковому записі ще не перевірявся. Розрахунки, статистика й регресія "
            "AI не використовують."
        ).classes("footnote")


def daily_costs(days: list[dict]) -> None:
    """Spending per day in USD (author request, history 141): date, calls, tokens, cost."""
    if not days:
        return
    with ui.element("div").classes("stat-table-wrap ai-daily-costs"):
        with ui.element("table").classes("stat-table"):
            with ui.element("caption"):
                ui.label("Витрати за днями")
            with ui.element("thead"), ui.element("tr"):
                for title in ("Дата", "Викликів", "Токени", "Вартість"):
                    with ui.element("th").props('scope="col"'):
                        ui.label(title)
            with ui.element("tbody"):
                for day in days:
                    with ui.element("tr") as row:
                        row.props["data-day"] = day["day"]
                        with ui.element("th").props('scope="row"'):
                            ui.label(day["day"])
                        with ui.element("td"):
                            ui.label(str(day["calls"]))
                        with ui.element("td"):
                            ui.label(f"{day['total_tokens']:,}".replace(",", " "))
                        with ui.element("td"):
                            ui.label(
                                format_usd(day["cost_usd"])
                                + (f" + {day['cost_unknown']} невідомо" if day["cost_unknown"]
                                   else "")
                            )  # fmt: skip
            total = sum(d["cost_usd"] for d in days)
            with ui.element("tfoot"), ui.element("tr"):
                with ui.element("th").props('scope="row"'):
                    ui.label("Разом")
                with ui.element("td"):
                    ui.label(str(sum(d["calls"] for d in days)))
                with ui.element("td"):
                    ui.label(f"{sum(d['total_tokens'] for d in days):,}".replace(",", " "))
                with ui.element("td"):
                    ui.label(format_usd(total))
