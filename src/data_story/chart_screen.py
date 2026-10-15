"""One chart at a time (heatmap, histogram) and an independent configuration page."""

from typing import Any

from nicegui import ui
from pydantic import BaseModel, ValidationError

from .analysis_blocks import eda_link
from .chart_configuration import FONTS, THEMES, ChartConfiguration, ChartOptions, preset
from .charts import (
    bin_text,
    contrast_warnings,
    heatmap_svg,
    histogram_bins,
    histogram_svg,
    missing_order,
    missing_svg,
    number_text,
    scatter_svg,
)
from .contracts import (
    Failure,
    HeatmapView,
    HistogramView,
    MissingView,
    ProjectView,
    ScatterView,
    Success,
)
from .service import ProjectService

GROUPS = {
    "common": "Спільні: шрифт, розміри, кольори, осі",
    "heatmap": "Теплова карта",
    "histogram": "Гістограма",
    "scatter": "Розсіювання",
    "missing": "Пропуски",
}
FIELDS: dict[str, list[tuple[str, str, str, Any]]] = {
    # (name, control, title, bounds) - only parameters of implemented charts are shown.
    "common": [
        ("font", "select", "Шрифт", FONTS),
        ("title_size", "int", "Розмір заголовка", (16, 32)),
        ("label_size", "int", "Розмір підписів", (12, 22)),
        ("decimals", "int", "Знаків після коми", (1, 4)),
        ("text", "color", "Текст", None),
        ("background", "color", "Фон", None),
        ("accent", "color", "Акцент (стовпці)", None),
        ("axis_labels", "bool", "Підписи осей", None),
        ("grid_x", "bool", "Сітка X", None),
        ("grid_y", "bool", "Сітка Y", None),
        ("frame", "bool", "Рамка області графіка", None),
    ],
    "heatmap": [
        ("positive", "color", "Додатні значення", None),
        ("negative", "color", "Від’ємні значення · лінія медіани", None),
        ("neutral", "color", "Нуль / слабкий зв’язок", None),
        ("numbers", "bool", "Числа в клітинках", None),
        ("lower_triangle", "bool", "Лише нижній трикутник", None),
        ("threshold_enabled", "bool", "Акцент за порогом |r|", None),
        ("threshold", "float", "Поріг акценту |r|", (0, 1, 0.05)),
    ],
    "histogram": [
        ("histogram_bins", "int", "Інтервалів за замовчуванням (5–60)", (5, 60)),
        ("histogram_median", "bool", "Лінія медіани", None),
        ("histogram_opacity", "float", "Непрозорість стовпців (0,2–1)", (0.2, 1, 0.05)),
    ],
    "scatter": [
        ("scatter_size", "int", "Радіус точки (1–10)", (1, 10)),
        ("scatter_opacity", "float", "Непрозорість точок (0,05–1)", (0.05, 1, 0.05)),
        ("scatter_outline", "bool", "Контур точок", None),
    ],
    "missing": [
        ("context", "color", "Сірий контекст", None),
        ("missing_labels", "bool", "Прямі підписи часток", None),
        (
            "missing_order",
            "select",
            "Порядок полів",
            {"desc": "За спаданням частки", "source": "Як у наборі"},
        ),
    ],
}
INVALID = "Перевірте HEX-кольори, розміри тексту, поріг 0…1 і непрозорість 0,2…1."


def options_editor(options: ChartOptions, changed, groups: tuple[str, ...]) -> Any:
    """Controls for the given groups; hidden groups keep the values of ``options``."""
    controls: dict[str, Any] = {}

    def notify() -> None:
        changed()

    for group in groups:
        if len(groups) > 1:
            ui.label(GROUPS[group]).classes("chart-options-title")
        with ui.element("div").classes("chart-options"):
            for name, control, title, bounds in FIELDS[group]:
                value = getattr(options, name)
                if control == "select":
                    controls[name] = ui.select(bounds, value=value, label=title, on_change=notify)
                elif control in ("int", "float"):
                    low, high, *step = bounds
                    controls[name] = ui.number(
                        title,
                        value=value,
                        min=low,
                        max=high,
                        step=step[0] if step else 1,
                        on_change=notify,
                    ).props("outlined square")
                elif control == "color":
                    # Colour picker with a preview of the colour itself (author feedback 3);
                    # the HEX code can still be typed. The picker is local (Quasar), no CDN.
                    controls[name] = (
                        ui.color_input(
                            title + " · HEX", value=value, on_change=notify, preview=True
                        )
                        .props("outlined square")
                        .classes("color-field")
                    )
                else:
                    controls[name] = ui.checkbox(title, value=value, on_change=notify)

    def read() -> ChartOptions:
        values = options.model_dump()
        values.update({k: c.value for k, c in controls.items()})
        for key in ("title_size", "label_size", "decimals", "histogram_bins", "scatter_size"):
            value = values[key]
            if value is not None and float(value).is_integer():
                values[key] = int(value)
        for key in ("threshold", "histogram_opacity", "scatter_opacity"):
            if isinstance(values[key], int) and not isinstance(values[key], bool):
                values[key] = float(values[key])
        return ChartOptions.model_validate(values)

    return read


def show_chart(svg: str, hint: str) -> None:
    with ui.element("div").classes("heatmap-wrap"):
        ui.html(svg, sanitize=False).classes("heatmap-scroll")
        ui.label(hint).classes("heatmap-detail").props('aria-live="polite"')


HEATMAP_HINT = "Оберіть клітинку мишею або клавішею Tab: тут з’являться r, n і виключення."
MISSING_HINT = "Оберіть стовпець мишею або клавішею Tab: тут з’являться кількість і частка."
SCATTER_HINT = "Кожна точка — рядок з обома значеннями; накладання видно за прозорістю."
HISTOGRAM_HINT = "Оберіть стовпець мишею або клавішею Tab: тут з’являться межі й частота."

FORMULAS = {
    "heatmap": (
        "Pearson показує лінійний зв’язок, а не причинність. Для кожної пари беремо лише "
        "спільні скінченні значення: n ≥ 3, обидва поля мають варіацію.",
        [
            (
                r"r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2\sum_{i=1}^{n}(y_i-\bar{y})^2}}",
                "xᵢ, yᵢ — значення спільної пари; середні x̄, ȳ обчислено саме на цих n парах.",
            ),
            (
                r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i",
                "Середнє: сума придатних значень, поділена на n.",
            ),
            (
                r"s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2}",
                "Вибіркове стандартне відхилення: знаменник n−1; потрібні хоча б два значення.",
            ),
            (
                r"p_{\mathrm{missing}}=100\frac{m}{N}",
                "m — кількість пропусків поля за метаданими; N — усі рядки. "
                "Нескінченності не є пропусками.",
            ),
        ],
    ),
    "missing": (
        "Пропуски — порожні клітинки й явні коди пропуску з метаданих. Нескінченності не є "
        "пропусками й показані окремо. Шкала завжди 0–100%, стовпці від нуля.",
        [
            (
                r"p=100\,\frac{m}{N}",
                "m — пропуски поля, N — усі рядки набору. При N = 0 або помилковому коді "
                "частку не визначено.",
            ),
        ],
    ),
    "scatter": (
        "Розсіювання показує всі рядки, де є обидва значення (попарне вилучення, як у Pearson). "
        "Без вибірки, видалення викидів чи обрізання осей; r — з ядра статистики 03.",
        [
            (
                r"r=\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2\sum_{i=1}^{n}(y_i-\bar{y})^2}}",
                "n — кількість точок; r визначено при n ≥ 3 і варіації обох полів. "
                "Кореляція не доводить причинності й не описує нелінійні зв’язки.",
            ),
        ],
    ),
    "histogram": (
        "Гістограма показує, скільки придатних значень потрапило в кожен рівний інтервал. "
        "Пропуски (за метаданими) і нескінченності не входять до n; діапазон не обрізається.",
        [
            (
                r"h=\frac{x_{\max}-x_{\min}}{k}",
                "Ширина h для k інтервалів. Якщо задано ширину, k — найменша кількість "
                "інтервалів від мінімуму, що покриває максимум.",
            ),
            (
                r"x\in[a_j,\,a_{j+1}),\quad a_j=x_{\min}+j\,h",
                "Ліва межа включається, права — ні; останній інтервал включає максимум.",
            ),
            (
                r"\sum_{j=1}^{k}f_j=n",
                "fⱼ — частота інтервалу; сума частот дорівнює кількості придатних значень n.",
            ),
            (
                r"Q(0{,}5)=x_{(\lfloor t\rfloor)}+(t-\lfloor t\rfloor)\,"
                r"\bigl(x_{(\lfloor t\rfloor+1)}-x_{(\lfloor t\rfloor)}\bigr),\quad t=1+0{,}5\,(n-1)",
                "Медіана — лінійний квантиль 0,5 упорядкованих значень, той самий, що в "
                "статистиці 03.",
            ),
        ],
    ),
}


def formulas(kind: str) -> None:
    lead, items = FORMULAS[kind]
    with ui.element("details").classes("howto"):
        with ui.element("summary"):
            ui.label("Як розраховано · формули")
        ui.label(lead)
        for tex, explanation in items:
            formula = ui.element("div").classes("math-formula")
            formula._props["data-formula"] = tex
            ui.label(explanation)
        ui.label(
            "Якщо формула не завантажилась, пояснення методу вище залишається доступним."
        ).classes("muted")


CHARTS = {
    "heatmap": "Теплова карта кореляцій",
    "histogram": "Гістограма",
    "scatter": "Розсіювання",
    "missing": "Пропуски за полями",
}


def charts_screen(
    service: ProjectService, active: ProjectView, dataset_id: str | None, chart: str | None
) -> None:
    """One chart at a time: choose the chart type, then its own controls."""
    from .screen import datasets_section, error_line

    kind = chart if chart in CHARTS else "heatmap"
    project_id = str(active.id)
    ui.label(
        CHARTS[kind] + " · один графік, увесь набір даних. Числа з перевіреного ядра статистики."
    ).classes("section-lead")
    listing = service.list_datasets(project_id)
    configuration = service.chart_configuration()
    if isinstance(listing, Failure):
        error_line(listing)
        return
    if isinstance(configuration, Failure):
        error_line(configuration)
        return
    if not listing.data:
        ui.label("Для графіка потрібен набір даних. Завантажте CSV у цей проєкт.")
        datasets_section(service, active, listing)
        return
    choices = {str(d.id): d.name for d in listing.data}
    if dataset_id and dataset_id not in choices:
        ui.label("Набір не належить цьому проєкту або не знайдений.").classes("error-message")
        return
    selected = dataset_id or next(iter(choices))

    def go(dataset: str, chart_kind: str) -> None:
        ui.navigate.to(eda_link(project_id, chart_kind, dataset))

    with ui.element("div").classes("chart-choice"):
        ui.select(
            choices,
            value=selected,
            label="Набір даних",
            on_change=lambda e: go(e.value, kind),
        ).props("outlined square")
    seed = getattr(ui.context.client, "assistant_chart_seed", None)
    if seed and seed["query"]["dataset_id"] == selected:
        configuration.data = ChartConfiguration(
            theme="custom", custom=ChartOptions.model_validate(seed["query"]["options"])
        )
    meta = service.metadata(selected)
    if isinstance(meta, Failure):
        error_line(meta)
        return
    eligible = {
        c.name: (c.label or c.name) + (f" · {c.name}" if c.label and c.label != c.name else "")
        for c in meta.data.document.columns
        if c.kind == "number" and c.role not in ("id", "ignore")
    }
    if kind == "missing":
        missing_body(service, project_id, selected, configuration.data)
    elif kind == "scatter":
        scatter_body(service, project_id, selected, eligible, configuration.data)
    elif kind == "histogram":
        histogram_body(service, project_id, selected, eligible, configuration.data)
    else:
        heatmap_body(service, project_id, selected, eligible, configuration.data)


def connect_assistant_chart(kind, selected, state, title, read_options, controls):
    """Hydrate an explicitly applied result and expose live controls for text commands."""
    seed = getattr(ui.context.client, "assistant_chart_seed", None)
    if seed and seed["chart"] == kind and seed["query"]["dataset_id"] == selected:
        query = seed["query"]
        title.set_value(query.get("title", ""))
        for key, control in controls.items():
            if key in query and query[key] is not None:
                control.set_value(query[key])
        if "mode" in controls:
            controls["mode"].set_value("width" if query.get("bin_width") else "count")
        views: dict[str, type[BaseModel]] = {
            "heatmap": HeatmapView,
            "histogram": HistogramView,
            "scatter": ScatterView,
            "missing": MissingView,
        }
        state["result"] = views[kind].model_validate(seed["result"])
        ui.context.client.assistant_chart_seed = None

    def context():
        query = {
            "dataset_id": selected,
            "title": title.value or "",
            "options": read_options().model_dump(),
        }
        if kind == "histogram":
            if controls["mode"].value == "width":
                query["bin_width"] = controls["bin_width"].value
            else:
                value = controls["bins"].value
                query["bins"] = (
                    int(value) if value is not None and float(value).is_integer() else value
                )
            query["field"] = controls["field"].value
        else:
            query.update({key: control.value for key, control in controls.items()})
        if kind == "missing":
            query["start"] = state["result"].start if state["result"] else 0
        return {"chart": kind, "query": query}

    ui.context.client.assistant_chart_context = context


def theme_line(configuration: ChartConfiguration) -> None:
    ui.label(
        "Оформлення — загальна тема графіків: "
        + THEMES[configuration.theme]
        + ". Змінюється в «Налаштування → Теми»."
    )


def theme_options(options: ChartOptions):
    """History 139 (author): no per-chart formatting panel — every chart uses the common theme."""
    return lambda: options


def heatmap_body(
    service: ProjectService,
    project_id: str,
    selected: str,
    eligible: dict[str, str],
    configuration: ChartConfiguration,
) -> None:
    from .screen import button, error_text

    # Explicit selection: never silently take a subset when >12 fields are available.
    fields = (
        ui.select(
            eligible,
            value=list(eligible) if len(eligible) <= 12 else [],
            multiple=True,
            label="Числові поля · оберіть 2–12",
        )
        .props("outlined square use-chips")
        .classes("w-full")
    )
    theme_line(configuration)
    button("Теми графіків", lambda: ui.navigate.to("/settings/charts"), "palette")
    title = (
        ui.input("Заголовок (необов’язково)")
        .props("outlined square maxlength=180")
        .classes("w-full")
    )
    read_options = theme_options(configuration.effective())
    message = ui.label("").classes("error-message").props('role="alert"')
    state: dict[str, Any] = {"result": None}

    @ui.refreshable
    def render() -> None:
        result = state["result"]
        if result is None:
            ui.label("Оберіть поля й натисніть «Побудувати».").classes("section-lead")
            return
        metadata_title = {"none": "не збережено", "draft": "чернетка", "confirmed": "підтверджено"}[
            result.metadata_state
        ]
        ui.label(f"Увесь набір · {result.row_count:,} рядків · метадані: {metadata_title}").classes(
            "stat-summary"
        )
        ui.label("Тема при побудові: " + THEMES[result.theme])
        for warning in result.warnings:
            ui.label(warning).classes("metadata-warning")
        if result.options.threshold_enabled:
            ui.label(
                f"Акцент: |r| ≥ {result.options.threshold:g}. Інші значення приглушені, доступні в підказках."
            )
        show_chart(result.svg, HEATMAP_HINT)
        with ui.element("details").classes("howto"):
            with ui.element("summary"):
                ui.label("Повні назви полів, кількість значень і виключення")
            for field in result.fields:
                ui.label(f"{field.label} · {field.name} · одиниці: {field.unit or 'не задані'}")
            for name, reason in result.excluded.items():
                ui.label(name + ": " + reason)
            for i, field in enumerate(result.fields):
                for j in range(i + 1):
                    value = result.r[i][j]
                    ui.label(
                        f"{field.name} ↔ {result.fields[j].name}: r={value if value is not None else '—'}; "
                        f"n={result.n[i][j]}; виключено пар={result.row_count - result.n[i][j]}"
                        + ("; " + result.reasons[i][j] if result.reasons[i][j] else "")
                    )
        formulas("heatmap")

    def build() -> None:
        try:
            options = read_options()
        except ValidationError:
            message.set_text(INVALID)
            return
        result = service.heatmap(
            project_id,
            {
                "dataset_id": selected,
                "fields": fields.value,
                "options": options.model_dump(),
                "title": title.value or "",
            },
        )
        if isinstance(result, Failure):
            message.set_text(error_text(result))
            return
        state["result"] = result.data
        message.set_text("")
        render.refresh()

    button("Побудувати", build, "formula", "primary")
    connect_assistant_chart("heatmap", selected, state, title, read_options, {"fields": fields})
    render()


def histogram_body(
    service: ProjectService,
    project_id: str,
    selected: str,
    eligible: dict[str, str],
    configuration: ChartConfiguration,
) -> None:
    from .screen import button, error_text

    if not eligible:
        ui.label("У наборі немає придатних числових полів для гістограми.")
        return
    defaults = configuration.effective()
    field = (
        ui.select(eligible, value=next(iter(eligible)), label="Числове поле")
        .props("outlined square")
        .classes("chart-field")
    )
    with ui.element("div").classes("chart-options"):
        mode = ui.toggle(
            {"count": "Кількість інтервалів", "width": "Ширина інтервалу"}, value="count"
        ).props('no-caps unelevated aria-label="Спосіб задати інтервали"')
        bins = ui.number(
            "Кількість інтервалів (5–60)", value=defaults.histogram_bins, min=5, max=60, step=1
        ).props("outlined square")
        width = ui.number("Ширина інтервалу (> 0)", value=None, min=0).props("outlined square")
    bins.bind_visibility_from(mode, "value", value="count")
    width.bind_visibility_from(mode, "value", value="width")
    theme_line(configuration)
    title = (
        ui.input("Заголовок (необов’язково)")
        .props("outlined square maxlength=180")
        .classes("w-full")
    )
    read_options = theme_options(defaults)
    message = ui.label("").classes("error-message").props('role="alert"')
    state: dict[str, Any] = {"result": None}

    @ui.refreshable
    def render() -> None:
        result = state["result"]
        if result is None:
            ui.label("Оберіть поле й натисніть «Побудувати».").classes("section-lead")
            return
        decimals = result.options.decimals
        metadata_title = {"none": "не збережено", "draft": "чернетка", "confirmed": "підтверджено"}[
            result.metadata_state
        ]
        ui.label(f"Увесь набір · {result.row_count:,} рядків · метадані: {metadata_title}").classes(
            "stat-summary"
        )
        ui.label("Тема при побудові: " + THEMES[result.theme])
        for warning in result.warnings:
            ui.label(warning).classes("metadata-warning")
        excluded = result.row_count - result.n
        parts = [f"n = {result.n:,} придатних значень", f"виключено {excluded:,}"]
        if result.missing is not None:
            parts.append(f"пропуски {result.missing:,}")
        parts.append(f"нескінченності {result.invalid_infinite:,}")
        ui.label(" · ".join(parts).replace(",", " ")).classes("chart-facts")
        if result.reason:
            ui.label(result.reason).classes("metadata-warning")
        if not result.svg:
            return
        if not result.constant:
            mode_text = "задана ширина" if result.mode == "width" else "задана кількість"
            ui.label(
                f"Інтервалів: {len(result.counts)} ({mode_text}) · ширина "
                f"{number_text(result.width or 0, decimals)} · діапазон "
                f"{number_text(result.edges[0], decimals)}…{number_text(result.edges[-1], decimals)}"
                f" · сума частот {sum(result.counts):,} = n".replace(",", " ")
            ).classes("chart-facts")
        if result.median is not None:
            ui.label("Медіана (статистика 03): " + number_text(result.median, decimals)).classes(
                "chart-facts"
            )
        show_chart(result.svg, HISTOGRAM_HINT)
        with ui.element("details").classes("howto"):
            with ui.element("summary"):
                ui.label("Таблиця інтервалів і частот")
            field_info = result.field
            ui.label(
                f"{field_info.label or field_info.name} · {field_info.name} · одиниці: "
                f"{field_info.unit or 'не задані'}"
            )
            for index, count in enumerate(result.counts):
                label = (
                    "усі значення = " + number_text(result.edges[0], decimals)
                    if result.constant
                    else bin_text(result.edges, index, decimals)
                )
                ui.label(f"{label}: {count}")
        formulas("histogram")

    def build() -> None:
        try:
            options = read_options()
        except ValidationError:
            message.set_text(INVALID)
            return
        request: dict[str, Any] = {
            "dataset_id": selected,
            "field": field.value,
            "options": options.model_dump(),
            "title": title.value or "",
        }
        if mode.value == "width":
            if not width.value or width.value <= 0:
                message.set_text("Вкажіть ширину інтервалу більшу за нуль.")
                return
            request["bin_width"] = float(width.value)
        else:
            value = bins.value
            if value is None or not float(value).is_integer() or not 5 <= value <= 60:
                message.set_text("Кількість інтервалів — ціле число від 5 до 60.")
                return
            request["bins"] = int(value)
        result = service.histogram(project_id, request)
        if isinstance(result, Failure):
            message.set_text(error_text(result))
            return
        state["result"] = result.data
        message.set_text("")
        render.refresh()

    button("Побудувати", build, "formula", "primary")
    connect_assistant_chart(
        "histogram",
        selected,
        state,
        title,
        read_options,
        {"field": field, "bins": bins, "bin_width": width, "mode": mode},
    )
    render()


def missing_body(
    service: ProjectService,
    project_id: str,
    selected: str,
    configuration: ChartConfiguration,
) -> None:
    from .screen import button, error_text

    theme_line(configuration)
    title = (
        ui.input("Заголовок (необов’язково)")
        .props("outlined square maxlength=180")
        .classes("w-full")
    )
    read_options = theme_options(configuration.effective())
    message = ui.label("").classes("error-message").props('role="alert"')
    state: dict[str, Any] = {"result": None, "start": 0}

    @ui.refreshable
    def render() -> None:
        result = state["result"]
        if result is None:
            ui.label("Натисніть «Побудувати»: усі поля набору, по 10 на сторінці.").classes(
                "section-lead"
            )
            return
        ui.label(f"Увесь набір · N = {result.row_count:,} рядків".replace(",", " ")).classes(
            "stat-summary"
        )
        ui.label("Тема при побудові: " + THEMES[result.theme])
        for warning in result.warnings:
            ui.label(warning).classes("metadata-warning")
        if result.rows_with_missing is not None:
            share = result.rows_with_missing_percent or 0
            ui.label(
                f"Рядків хоча б з одним пропуском: {result.rows_with_missing:,} "
                f"({share:.{result.options.decimals}f}%)".replace(",", " ")
            ).classes("chart-facts")
        end = min(result.start + len(result.rows), result.total)
        ui.label(f"Поля {result.start + 1}–{end} з {result.total}").classes("chart-facts")
        show_chart(result.svg, MISSING_HINT)
        with ui.element("div").classes("dialog-actions left"):
            if result.start > 0:
                button("Попередні 10", lambda: build(result.start - 10), "prev", "small")
            if end < result.total:
                button("Наступні 10", lambda: build(result.start + 10), "next", "small")
        formulas("missing")

    def build(start: int = 0) -> None:
        try:
            options = read_options()
        except ValidationError:
            message.set_text(INVALID)
            return
        result = service.missing_chart(
            project_id,
            {
                "dataset_id": selected,
                "start": start,
                "options": options.model_dump(),
                "title": title.value or "",
            },
        )
        if isinstance(result, Failure):
            message.set_text(error_text(result))
            return
        state["result"] = result.data
        message.set_text("")
        render.refresh()

    button("Побудувати", lambda: build(0), "formula", "primary")
    connect_assistant_chart("missing", selected, state, title, read_options, {})
    render()


def scatter_body(
    service: ProjectService,
    project_id: str,
    selected: str,
    eligible: dict[str, str],
    configuration: ChartConfiguration,
) -> None:
    from .screen import button, error_text

    if len(eligible) < 2:
        ui.label("Для розсіювання потрібні два придатні числові поля.")
        return
    names = list(eligible)
    with ui.element("div").classes("chart-options"):
        x = ui.select(eligible, value=names[0], label="Вісь X").props("outlined square")
        y = ui.select(eligible, value=names[1], label="Вісь Y").props("outlined square")
    theme_line(configuration)
    title = (
        ui.input("Заголовок (необов’язково)")
        .props("outlined square maxlength=180")
        .classes("w-full")
    )
    read_options = theme_options(configuration.effective())
    message = ui.label("").classes("error-message").props('role="alert"')
    state: dict[str, Any] = {"result": None}

    @ui.refreshable
    def render() -> None:
        result = state["result"]
        if result is None:
            ui.label("Оберіть поля X і Y й натисніть «Побудувати».").classes("section-lead")
            return
        decimals = result.options.decimals
        ui.label(f"Увесь набір · {result.row_count:,} рядків".replace(",", " ")).classes(
            "stat-summary"
        )
        ui.label("Тема при побудові: " + THEMES[result.theme])
        for warning in result.warnings:
            ui.label(warning).classes("metadata-warning")
        r_text = "—" if result.r is None else number_text(result.r, decimals)
        ui.label(
            f"Точок n = {result.n:,} · виключено рядків {result.row_count - result.n:,} · "
            f"Pearson r = {r_text}".replace(",", " ")
        ).classes("chart-facts")
        if result.reason:
            ui.label(result.reason).classes("metadata-warning")
        if result.svg:
            show_chart(result.svg, SCATTER_HINT)
        formulas("scatter")

    def build() -> None:
        if x.value == y.value:
            message.set_text("Оберіть два різні поля.")
            return
        try:
            options = read_options()
        except ValidationError:
            message.set_text(INVALID)
            return
        result = service.scatter(
            project_id,
            {
                "dataset_id": selected,
                "x": x.value,
                "y": y.value,
                "options": options.model_dump(),
                "title": title.value or "",
            },
        )
        if isinstance(result, Failure):
            message.set_text(error_text(result))
            return
        state["result"] = result.data
        message.set_text("")
        render.refresh()

    button("Побудувати", build, "formula", "primary")
    connect_assistant_chart("scatter", selected, state, title, read_options, {"x": x, "y": y})
    render()


def chart_settings_screen(service: ProjectService) -> None:
    from .screen import Shell, button, error_line, error_text, page_header

    ui.page_title("Теми графіків · Data Story")
    Shell(service, "chart_settings", trail=[("Налаштування", None), ("Теми", None)])
    with ui.column().classes("page"):
        page_header(
            "Налаштування",
            "palette",
            "Теми графіків",
            "Спільні для сервісу. Попередній перегляд не змінює збережені налаштування. "
            "Доступні параметри всіх графіків оглядового аналізу; "
            "інші додамо разом із графіками.",
        )
        response = service.chart_configuration()
        if isinstance(response, Failure):
            error_line(response)
            return
        original = response.data
        options_state = {"value": original.effective()}
        read_state: dict[str, Any] = {}
        sample = {"kind": "heatmap"}
        message = ui.label("").classes("error-message")

        @ui.refreshable
        def preview() -> None:
            try:
                options = read_state["read"]() if "read" in read_state else options_state["value"]
            except ValidationError:
                ui.label(INVALID).classes("error-message")
                return
            for warning in contrast_warnings(options):
                ui.label(warning).classes("metadata-warning")
            ui.label("Попередній перегляд · демонстраційні значення, не ваші дані").classes(
                "section-lead"
            )
            if sample["kind"] == "missing":
                show_chart(demo_missing(options), MISSING_HINT)
                return
            if sample["kind"] == "scatter":
                show_chart(demo_scatter(options), SCATTER_HINT)
                return
            if sample["kind"] == "histogram":
                show_chart(demo_histogram(options), HISTOGRAM_HINT)
                return
            columns = [{"name": n, "label": n, "unit": ""} for n in ("Поле A", "Поле B", "Поле C")]
            demo = {
                "columns": columns,
                "fields": [c["name"] for c in columns],
                "row_count": 100,
                "r": [[1, 0.5, -0.7], [0.5, 1, None], [-0.7, None, 1]],
                "n": [[100, 90, 80], [90, 100, 2], [80, 2, 100]],
                "reasons": [
                    [None] * 3,
                    [None, None, "Потрібно 3 пари."],
                    [None, "Потрібно 3 пари.", None],
                ],
            }
            show_chart(heatmap_svg(demo, options), HEATMAP_HINT)

        @ui.refreshable
        def editor() -> None:
            read_state["read"] = options_editor(
                options_state["value"],
                preview.refresh,
                ("common", "heatmap", "histogram", "scatter", "missing"),
            )

        def choose(e) -> None:
            options_state["value"] = (
                (original.custom or preset("storytelling"))
                if e.value == "custom"
                else preset(e.value)
            )
            read_state.clear()
            editor.refresh()
            preview.refresh()

        theme = ui.select(THEMES, value=original.theme, label="Тема", on_change=choose).props(
            "outlined square"
        )
        ui.label(
            "Змінені параметри збережуться як одна власна тема. Готові теми залишаються незмінними."
        )
        editor()

        def switch(e) -> None:
            sample["kind"] = e.value
            preview.refresh()

        ui.toggle(CHARTS, value="heatmap", on_change=switch).props(
            'no-caps unelevated aria-label="Приклад попереднього перегляду"'
        ).classes("chart-type")
        preview()

        def save() -> None:
            try:
                options = read_state["read"]()
                active = theme.value
                custom = original.custom
                if active == "custom" or options != preset(active):
                    active, custom = "custom", options
                request = ChartConfiguration(theme=active, custom=custom)
            except ValidationError:
                message.set_text("Перевірте кольори, розміри та поріг перед збереженням.")
                return
            result = service.save_chart_configuration(request.model_dump())
            if isinstance(result, Failure):
                message.set_text(error_text(result))
                return
            ui.notify("Тему графіків збережено", type="positive")
            ui.navigate.to("/settings/charts")

        def reset() -> None:
            result = service.reset_chart_configuration()
            if isinstance(result, Failure):
                message.set_text(error_text(result))
                return
            ui.navigate.to("/settings/charts")

        with ui.element("div").classes("dialog-actions left"):
            button("Зберегти налаштування", save, "check", "primary")
            button("Відновити початкові налаштування", reset, "reset")
        events = service.chart_configuration_events()
        if isinstance(events, Success):
            with ui.element("details").classes("howto"):
                with ui.element("summary"):
                    ui.label("Журнал налаштувань · останні 10 дій")
                for event in events.data:
                    ui.label(
                        str(event["occurred_at"])
                        + " · "
                        + ("Збережено" if event["operation"].endswith("saved") else "Відновлено")
                    )


def demo_histogram(options: ChartOptions) -> str:
    """Deterministic demonstration values for the settings preview, not user data."""
    values = [float(v) for v in range(10, 90, 3)] + [float(v) for v in range(30, 60, 2)] * 2
    binned = histogram_bins(values, options.histogram_bins)
    ordered = sorted(values)
    middle = len(ordered) // 2
    binned["median"] = (
        ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    )
    return histogram_svg(binned, {"name": "Поле A", "label": "", "unit": ""}, options)


def demo_scatter(options: ChartOptions) -> str:
    """Deterministic demonstration points for the settings preview, not user data."""
    pairs = [[float(i % 37), float((i * 7) % 23 + i % 37 * 0.6)] for i in range(120)]
    column = {"label": "", "unit": ""}
    return scatter_svg(
        pairs, {**column, "name": "Поле A"}, {**column, "name": "Поле B"}, None, options
    )


def demo_missing(options: ChartOptions) -> str:
    """Demonstration shares for the settings preview, not user data."""
    rows = [
        {
            "field": {"name": f"Поле {letter}", "label": "", "unit": ""},
            "missing": missing,
            "percent": missing,
            "invalid_infinite": 0,
            "reason": None,
        }
        for letter, missing in (("A", 0), ("B", 47), ("C", 12), ("D", 3), ("E", 47))
    ]
    return missing_svg(missing_order(rows, options.missing_order), 47, options)
