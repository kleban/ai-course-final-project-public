"""Portable, local reader instructions; no source repository or external resources needed."""

import json
from pathlib import Path

from nicegui import ui

from .knowledge import DEFAULTS
from .screen import Shell, page_header

GUIDES = Path(__file__).parent / "guides"
SUPPORTED = {"hist": "Гістограма", "scatter": "Розсіювання", "corr": "Теплова карта"}


def chart_families() -> list:
    return json.loads((GUIDES / "chart-selection.json").read_text(encoding="utf-8"))


def instructions_screen(service) -> None:
    ui.page_title("Інструкції · Data Story")
    Shell(service, "guides", trail=[("Інструкції", None)])
    with ui.column().classes("page"):
        page_header(
            "Довідник",
            "doc",
            "Інструкції",
            "Оберіть питання, підготуйте дані й перевірте результат. Усі матеріали доступні локально.",
        )
        with ui.tabs().classes("w-full") as tabs:
            charts = ui.tab("Вибір графіка")
            regression = ui.tab("Лінійна регресія")
        with ui.tab_panels(tabs, value=charts).classes("w-full"):
            with ui.tab_panel(charts):
                ui.label("Що ви хочете показати?").classes("section-title")
                ui.markdown(
                    "Спершу сформулюйте питання. Потім визначте типи полів, кількість груп "
                    "і наявність часу або географії. Схема нижче веде від питання до графіка."
                )
                ui.label(
                    "Позначка «Є в сервісі» означає, що цей графік уже можна побудувати. "
                    "Інші картки пояснюють загальну методику."
                ).classes("model-note")
                icons = json.loads((GUIDES / "chart-icons.json").read_text(encoding="utf-8"))
                for family, question, _, items in chart_families():
                    with ui.expansion(
                        f"{family} · {question}", value=family in ("Розподіл", "Зв'язок")
                    ).classes("w-full"):
                        with ui.element("div").classes("guide-tree"):
                            for condition, name, key, tip in items:
                                with ui.column().classes("guide-card"):
                                    ui.label(condition).classes("model-note")
                                    ui.html(icons[key], sanitize=False)
                                    ui.label("↓ " + name).classes("font-bold")
                                    ui.label(tip)
                                    if key in SUPPORTED:
                                        ui.label("Є в сервісі").classes("tag")
                ui.markdown(
                    "### Як працює текстова команда\n"
                    "1. Відкрийте проєкт → Оглядовий аналіз та оберіть набір.\n"
                    "2. Натисніть AI → Виконай і напишіть, що потрібно показати.\n"
                    "3. Перевірте запропоновані поля, тип, інтервали й оформлення.\n"
                    "4. Натисніть Застосувати: сервіс обчислить і покаже графік.\n\n"
                    "Приклади: «Покажи розподіл Price у 25 інтервалах», "
                    "«Покажи зв’язок Distance і Price», «Зміни колір точок на помаранчевий». "
                    "Для зміни поточного графіка відкрийте відповідний інструмент; "
                    "помічник бачить його поля й оформлення. AI не запускається автоматично."
                )
                ui.markdown(
                    "### Пропуски та межі MVP\n"
                    "Частки пропусків показуємо горизонтальними стовпцями від нуля, "
                    "відсортованими за часткою. Зараз також доступні гістограма числового поля, "
                    "розсіювання двох числових полів і кореляційна карта 2–12 числових полів. "
                    "Лінії часу, boxplot, кругові, звичайні категоріальні стовпці й географічні "
                    "карти ще не реалізовано. Помічник пояснить обмеження або поставить уточнення."
                )
            with ui.tab_panel(regression):
                ui.markdown(
                    "## Як виконати лінійну регресію\n"
                    "1. **Дані:** оберіть числову ціль і ознаки; перевірте одиниці та підтвердьте метадані.\n"
                    "2. **Огляд:** перевірте пропуски, розподіл цілі й зв’язки ознак.\n"
                    "3. **Підготовка:** оберіть обробку пропусків і поділ train/test. "
                    "Перетворення навчаються лише на train; категорії кодуються з базовою категорією.\n"
                    "4. **Модель:** навчіть регресію на збережених вибірках.\n"
                    "5. **Перевірка:** прочитайте R², MAE, RMSE на тесті, коефіцієнти та графіки похибок.\n"
                    "6. **Висновок:** вкажіть якість прогнозу, обмеження й те, що потрібно перевірити. "
                    "Окремі поради AI запускайте кнопками Порадь / Поясни результат."
                )
                formula = ui.element("div").classes("math-formula")
                formula._props["data-formula"] = r"\hat{y}=a_0+a_1x_1+\cdots+a_px_p"
                ui.label("ŷ — прогноз цілі, a₀ — вільний член, aⱼ — коефіцієнт ознаки xⱼ.").classes(
                    "model-note"
                )
                for name in (
                    "03-regression-metrics",
                    "04-regression-coefficients",
                    "05-model-check",
                ):
                    text = (DEFAULTS / f"{name}.md").read_text(encoding="utf-8")
                    title = text.splitlines()[0][2:]
                    with ui.expansion(title, value=name == "03-regression-metrics").classes(
                        "w-full"
                    ):
                        ui.markdown("## " + text.split("\n## ", 1)[1]).classes("md-text")
                ui.markdown(
                    "### Приклад читання результату\n"
                    "Якщо ціль — ціна, а MAE = 100 000, середня абсолютна помилка — "
                    "100 000 в одиницях ціни. Це умовний приклад, не результат вашого набору. "
                    "Коефіцієнт площі описує зміну прогнозу за +1 одиницю площі при незмінних "
                    "інших ознаках. Він не доводить, що збільшення площі спричиняє таку зміну ціни."
                )
