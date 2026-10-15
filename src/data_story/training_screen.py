"""Human-selected target, features, transformations and split for the model block."""

import math
from functools import partial

from nicegui import run, ui

from .assistant_panel import ai_shortcut
from .contracts import Failure, ProjectView
from .screen import button, error_line, page_header, progress_line
from .service import ProjectService


def training_screen(service: ProjectService, active: ProjectView, dataset_id: str | None) -> None:
    project_id = str(active.id)
    page_header(
        "Підготовка даних",
        "methods",
        "Вибірки для регресії",
        "Оберіть ціль, ознаки й перетворення. Підготовка запускається лише за кнопкою.",
    )
    ai_shortcut(
        "Порадь ціль і ознаки",
        "advise",
        "Які ціль і ознаки обрати для регресії в цьому наборі й чому?",
    )
    listing = service.list_datasets(project_id)
    if isinstance(listing, Failure):
        error_line(listing)
        return
    choices = {str(d.id): d.name for d in listing.data}
    if not choices:
        ui.label("Спочатку завантажте набір даних у проєкт.").classes("section-lead")
        return
    if dataset_id and dataset_id not in choices:
        ui.label("Набір не належить цьому проєкту або не знайдений.").classes("error-message")
        return
    selected = dataset_id or next(iter(choices))
    dataset = next(d for d in listing.data if str(d.id) == selected)
    ui.select(
        choices,
        value=selected,
        label="Набір даних",
        on_change=lambda e: ui.navigate.to(
            f"/?project={project_id}&dataset={e.value}&view=training"
        ),
    ).props("outlined square")
    button(
        "Обробка пропусків",
        partial(ui.navigate.to, f"/?project={project_id}&dataset={selected}&view=preparation"),
        "methods",
    )
    metadata = service.metadata(selected)
    if isinstance(metadata, Failure):
        error_line(metadata)
        return
    columns = [
        c
        for c in metadata.data.document.columns
        if c.role not in ("id", "ignore") and c.kind != "id"
    ]
    targets = {c.name: c.label or c.name for c in columns if c.kind == "number"}
    available = {
        c.name: c.label or c.name
        for c in columns
        if c.kind in ("number", "category", "bool") and c.role != "target"
    }
    if not targets:
        ui.label("Немає числової цілі. Перевірте типи й ролі у метаданих.").classes("section-lead")
        return
    ui.label(
        "Випадковий поділ призначений для незалежних рядків. Часові ряди та повторні спостереження однієї групи потребують іншого способу поділу."
    ).classes("section-lead")
    with ui.row().classes("stat-controls training-controls"):
        target = (
            ui.select(targets, label="Цільова змінна")
            .props("outlined square")
            .classes("stat-dataset")
        )
        features = (
            ui.select(available, multiple=True, label="Поля вибірок", value=list(available))
            .props("outlined square use-chips")
            .classes("stat-dataset")
        )

    ui.label(
        "Вибірки беруть усі поля, крім цілі; ознаки конкретної моделі обираються в блоці "
        "«Модель» — вручну або методом stepwise."
    ).classes("section-lead")

    def target_changed(e) -> None:
        features.set_options({name: title for name, title in available.items() if name != e.value})
        features.set_value([name for name in features.value if name != e.value])

    target.on_value_change(target_changed)
    with ui.row().classes("stat-controls training-controls"):
        percent = ui.number("Тестова частина, %", value=30, min=10, max=50, step=1).props(
            "outlined square"
        )
        seed = ui.number(
            "Число для відтворення поділу", value=42, min=0, max=2**32 - 1, step=1
        ).props("outlined square")
    count = ui.label("").classes("stat-context")

    def counts() -> None:
        value = float(percent.value or 30)
        test = math.ceil(dataset.row_count * value / 100)
        count.set_text(
            f"Рядків: {dataset.row_count} · Навчальна частина: {dataset.row_count - test} · Тестова: {test}"
        )

    percent.on_value_change(lambda _: counts())
    counts()
    ui.label(
        "Словник категорій і значення для заповнення визначаються тільки на навчальній частині. Невідомі тестові категорії кодуються нулями з попередженням. Ціль не заповнюємо."
    ).classes("section-lead")
    with ui.row().classes("stat-controls training-controls"):
        encoding = ui.select(
            {"none": "Без кодування", "onehot": "One-hot · окрема ознака для кожної категорії"},
            value="none",
            label="Кодування категорій",
        ).props("outlined square")
        numeric = ui.select(
            {
                "reject": "Не заповнювати",
                "mean": "Середнім навчальної частини",
                "median": "Медіаною навчальної частини",
            },
            value="reject",
            label="Числові пропуски",
        ).props("outlined square")
        category = ui.select(
            {"reject": "Не заповнювати", "most_frequent": "Найчастішою навчальною категорією"},
            value="reject",
            label="Категорійні пропуски",
        ).props("outlined square")
    ui.label(
        "Обмеження MVP: до 30 вихідних ознак, до 200 після кодування. Більші набори категорій потрібно підготувати окремо."
    ).classes("section-lead")
    feedback = ui.column().classes("w-full")

    busy = False

    async def prepare() -> None:
        nonlocal busy
        if busy:
            return
        feedback.clear()
        if not target.value or not features.value:
            with feedback:
                ui.label("Оберіть цільову змінну та хоча б одну ознаку.").classes("error-message")
            return
        if (
            percent.value is None
            or seed.value is None
            or percent.value != int(percent.value)
            or seed.value != int(seed.value)
        ):
            with feedback:
                ui.label("Частка й число відтворення мають бути цілими числами.").classes(
                    "error-message"
                )
            return
        busy = True
        prepare_button.props("disabled")
        with feedback:
            progress_line("Готуємо вибірки…")
        try:
            response = await run.io_bound(
                service.training_prepare,
                project_id,
                {
                    "dataset_id": selected,
                    "target": target.value,
                    "features": features.value,
                    "test_percent": int(percent.value),
                    "seed": int(seed.value),
                    "encoding": encoding.value,
                    "numeric_missing": numeric.value,
                    "category_missing": category.value,
                },
            )
            feedback.clear()
            with feedback:
                if isinstance(response, Failure):
                    error_line(response)
                else:
                    ui.label("Вибірки підготовлено й збережено для моделі.").classes("stat-context")
                    report(response.data["document"])
                    model_link(response.data["id"])
            saved.refresh()
        finally:
            busy = False
            prepare_button.props(remove="disabled")

    prepare_button = button("Підготувати й зберегти вибірки", prepare, "check")

    def report(document: dict) -> None:
        details = document["report"]
        ui.label(
            f"Навчальних рядків: {details['train_count']} · Тестових: {details['test_count']} · Ознак після кодування: {details['feature_count']}"
        ).classes("stat-context")
        for warning in details["warnings"]:
            ui.label(warning).classes("section-lead")
        for name, transformation in details["transforms"].items():
            if "fill" in transformation:
                ui.label(
                    f"{name} · Значення заповнення з навчальної частини: {transformation['fill']}"
                )
        ui.label("Ознаки: " + ", ".join(details["feature_names"]))
        for name, transformation in details["transforms"].items():
            if transformation.get("baseline") is not None:
                ui.label(
                    f"{name} · базова категорія: {transformation['baseline']} "
                    "(окремого стовпця немає, інші — відносно неї)"
                )

    def model_link(run_id: str) -> None:
        button(
            "Навчити модель на цих вибірках",
            partial(ui.navigate.to, f"/?project={project_id}&view=model&run={run_id}"),
            "regression",
            "small",
        )

    @ui.refreshable
    def saved() -> None:
        ui.label("Збережена підготовка для моделі").classes("section-title")
        response = service.training_runs(project_id)
        if isinstance(response, Failure):
            error_line(response)
            return
        runs = [r for r in response.data if r["dataset_id"] == selected]
        if not runs:
            ui.label("Для цього набору вибірки ще не підготовлено.").classes("section-lead")
        for item in runs:
            doc = item["document"]
            with ui.expansion(
                f"{doc['choices']['target']} · {item['created_at'][:19].replace('T', ' ')} UTC"
            ).classes("w-full"):
                report(doc)
                ui.label(
                    f"Навчання/тест: {100 - doc['choices']['test_percent']}/{doc['choices']['test_percent']} · Число відтворення: {doc['choices']['seed']}"
                )
                ui.label(f"SHA-256 джерела: {doc['sha256']}")
                ui.label(f"Ідентифікатор підготовки: {item['id']}")
                model_link(item["id"])

    saved()
