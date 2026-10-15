"""Heatmap SVG adapted from SRC-004 slides_4_types.js divCol; no browser calculations."""

import math
from html import escape
from typing import Any

from .chart_configuration import ChartOptions


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def color(value: float, options: ChartOptions) -> str:
    # Lecture divCol: diverging interpolation, fixed center zero and endpoints +/-1.
    start = rgb(options.neutral)
    end = rgb(options.positive if value >= 0 else options.negative)
    t = min(1.0, abs(value))
    return "#{:02x}{:02x}{:02x}".format(*(round(a + (b - a) * t) for a, b in zip(start, end)))


def luminance(color_value: str) -> float:
    values = [v / 255 for v in rgb(color_value)]
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in values]
    return sum(v * w for v, w in zip(linear, (0.2126, 0.7152, 0.0722)))


def contrast(a: str, b: str) -> float:
    x, y = sorted((luminance(a), luminance(b)))
    return (y + 0.05) / (x + 0.05)


def contrast_warnings(options: ChartOptions) -> list[str]:
    warnings = []
    if contrast(options.text, options.background) < 4.5:
        warnings.append("Низький контраст тексту й фону: оберіть контрастніші кольори.")
    if contrast(options.negative, options.positive) < 1.2:
        warnings.append("Кольори додатних і від’ємних значень схожі; перевірте розрізнення.")
    return warnings


def field_label(column: dict[str, Any]) -> str:
    label = column["label"] or column["name"]
    if label != column["name"]:
        label += " · " + column["name"]
    if column["unit"]:
        label += " (" + column["unit"] + ")"
    return label


def heatmap_svg(result: dict[str, Any], options: ChartOptions, title: str = "") -> str:
    names = result["fields"]
    columns = {c["name"]: c for c in result["columns"]}
    size, left, top = 64, 240, 170
    width, height = left + len(names) * size + 140, top + max(len(names) * size, 252) + 55
    parts = [
        f'<svg class="heatmap" xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="Теплова карта кореляцій Pearson" style="font-family:{options.font};'
        f'background:{options.background}">',
        f'<text x="20" y="35" font-size="{options.title_size}" fill="{options.text}">'
        f"{escape(title or 'Зв’язки числових полів · Pearson')}</text>",
    ]

    def label(x: float, y: float, value: str, anchor: str = "start", rotate: bool = False) -> str:
        transform = f' transform="rotate(-35 {x} {y})"' if rotate else ""
        return (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{options.label_size}" '
            f'fill="{options.text}"{transform}>{escape(value)}</text>'
        )

    for i, name in enumerate(names):
        full = field_label(columns[name])
        short = full if len(full) <= 25 else full[:22] + "…"
        parts.append(label(left - 12, top + i * size + size / 2 + 5, short, "end"))
        parts.append(label(left + i * size + size / 2, top - 15, short, rotate=True))
        for j, other in enumerate(names):
            if options.lower_triangle and j > i:
                continue
            value, n, reason = result["r"][i][j], result["n"][i][j], result["reasons"][i][j]
            value_text = "—" if value is None else f"{value:.{options.decimals}f}"
            detail = f"{full} ↔ {field_label(columns[other])}: r={value_text}; n={n}"
            detail += f"; виключено пар: {result['row_count'] - n}"
            if reason:
                detail += "; " + reason
            fill = "#E5E7EB" if value is None else color(value, options)
            muted = (
                value is not None and options.threshold_enabled and abs(value) < options.threshold
            )
            if muted:
                fill = options.neutral
                detail += f"; нижче порога акценту {options.threshold:g}"
            ink = "#000000" if contrast("#000000", fill) >= contrast("#FFFFFF", fill) else "#FFFFFF"
            x, y = left + j * size, top + i * size
            parts.append(
                f'<g class="heatmap-cell" tabindex="0" role="img" aria-label="{escape(detail, quote=True)}">'
                f'<title>{escape(detail)}</title><rect x="{x}" y="{y}" width="{size - 2}" '
                f'height="{size - 2}" fill="{fill}" />'
            )
            if options.numbers or value is None:
                parts.append(
                    f'<text x="{x + size / 2}" y="{y + size / 2 + 5}" text-anchor="middle" '
                    f'font-size="{options.label_size}" fill="{ink}">{value_text}</text>'
                )
            parts.append("</g>")
    if options.frame:
        parts.append(
            f'<rect x="{left - 1}" y="{top - 1}" width="{len(names) * size}" '
            f'height="{len(names) * size}" fill="none" stroke="{options.text}" />'
        )
    # Discrete legend has all three exact anchors; no configurable truncation.
    legend_x = left + len(names) * size + 22
    for i in range(21):
        value = 1 - i / 10
        parts.append(
            f'<rect x="{legend_x}" y="{top + i * 12}" width="20" height="13" '
            f'fill="{color(value, options)}" />'
        )
    for i, value in ((0, "1"), (10, "0"), (20, "−1")):
        parts.append(label(legend_x + 30, top + i * 12 + 12, value))
    parts.append(label(20, height - 15, "Невизначено: — · Шкала кольору: −1…1"))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- histogram (stage 04, 052)
# Adapted idea from SRC-004 slides_4_types.js:93 (bin width, frequencies, median line);
# no fixed demonstration range: the full range of eligible values is always shown.

MAX_WIDTH_BINS = 200
SMALL_SAMPLE = 30


class BinningError(ValueError):
    """Safe Ukrainian message for impossible bin settings."""


def number_text(value: float, decimals: int) -> str:
    if float(value).is_integer() and abs(value) < 1e15:
        return f"{int(value):,}".replace(",", " ")
    text = f"{value:,.{decimals}f}".replace(",", " ")
    return text if text.strip("-0. ") else f"{value:.{decimals}g}"


def nice_ticks(low: float, high: float, target: int = 6, integer: bool = False) -> list[float]:
    """Readable ticks (1/2/5 x 10^k) inside [low, high]; both ends stay data-driven."""
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return [low]
    raw = (high - low) / max(1, target)
    if integer:
        raw = max(raw, 1.0)
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.ceil(low / step) * step
    ticks, i = [], 0
    while first + i * step <= high + step * 1e-9 and i < 100:
        ticks.append(round(first + i * step, 12))
        i += 1
    return ticks


def histogram_bins(
    values: list[float], bins: int | None = None, width: float | None = None
) -> dict[str, Any]:
    """Equal bins [a, b) over the full range; the last bin includes the maximum."""
    n = len(values)
    if n == 0:
        return {
            "mode": "width" if width else "count",
            "edges": [],
            "counts": [],
            "width": None,
            "constant": False,
            "reason": "Немає придатних значень: графік не будується.",
        }
    low, high = min(values), max(values)
    if low == high:
        return {
            "mode": "width" if width else "count",
            "edges": [low, high],
            "counts": [n],
            "width": 0.0,
            "constant": True,
            "reason": "Усі значення однакові: один стовпець без ширини інтервалу.",
        }
    if width is not None:
        if not math.isfinite(width) or width <= 0:
            raise BinningError("Ширина інтервалу має бути додатним числом.")
        count = math.ceil((high - low) / width)
        if low + count * width < high:
            count += 1
        if count > MAX_WIDTH_BINS:
            raise BinningError(
                f"Ця ширина дає {count} інтервалів; допустимо до {MAX_WIDTH_BINS}. Збільште ширину."
            )
        edges = [low + i * width for i in range(count + 1)]
        mode, step = "width", width
    else:
        count = bins or 20
        if not 5 <= count <= 60:
            raise BinningError("Кількість інтервалів має бути від 5 до 60.")
        step = (high - low) / count
        edges = [low + i * step for i in range(count)] + [high]
        mode = "count"
    counts = [0] * count
    for value in values:
        index = min(int((value - low) // step), count - 1)
        # Guard floating edges: the value must satisfy edges[index] <= value < edges[index+1].
        while index > 0 and value < edges[index]:
            index -= 1
        while index < count - 1 and value >= edges[index + 1]:
            index += 1
        counts[index] += 1
    if sum(counts) != n:  # pragma: no cover - invariant
        raise BinningError("Сума частот не збігається з кількістю значень.")
    return {
        "mode": mode,
        "edges": edges,
        "counts": counts,
        "width": step,
        "constant": False,
        "reason": None,
    }


def bin_text(edges: list[float], index: int, decimals: int) -> str:
    last = index == len(edges) - 2
    left, right = number_text(edges[index], decimals), number_text(edges[index + 1], decimals)
    return f"[{left}; {right}{']' if last else ')'}"


def histogram_svg(
    data: dict[str, Any], column: dict[str, Any], options: ChartOptions, title: str = ""
) -> str:
    edges, counts, n = data["edges"], data["counts"], sum(data["counts"])
    width, height = 820, 470
    left, right, top, bottom = 90, 30, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    name = field_label(column)
    heading = title or f"Розподіл: {name}"
    description = (
        f"Гістограма поля {name}. Вісь X: значення поля; вісь Y: кількість рядків. "
        f"n={n}; інтервалів: {len(counts)}."
    )
    parts = [
        f'<svg class="histogram" xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="{escape(description, quote=True)}" style="font-family:{options.font};'
        f'background:{options.background}">',
        f"<desc>{escape(description)}</desc>",
        f'<text x="20" y="35" font-size="{options.title_size}" fill="{options.text}">'
        f"{escape(heading)}</text>",
    ]
    tick_size = max(11, options.label_size - 2)

    def text(x: float, y: float, value: str, anchor: str = "middle", size: int = 0) -> str:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
            f'font-size="{size or tick_size}" fill="{options.text}">{escape(value)}</text>'
        )

    grid = "#E3E5E8"
    if data["constant"]:
        low = edges[0] - 0.5 if edges[0] == 0 else edges[0] - abs(edges[0]) * 0.05
        high = edges[0] + 0.5 if edges[0] == 0 else edges[0] + abs(edges[0]) * 0.05
    else:
        low, high = edges[0], edges[-1]
    top_count = max(counts) if counts else 1
    # Axis ends at the first tick at or above the tallest bar, so it is never clipped.
    y_ticks = nice_ticks(0, top_count, 5, integer=True)
    step = y_ticks[1] - y_ticks[0] if len(y_ticks) > 1 else 1
    while y_ticks[-1] < top_count:
        y_ticks.append(y_ticks[-1] + step)
    y_max = y_ticks[-1] or 1

    def sx(value: float) -> float:
        return left + (value - low) / (high - low) * plot_w

    def sy(count: float) -> float:
        return top + plot_h - count / y_max * plot_h

    for tick in y_ticks:
        if options.grid_y and tick > 0:
            parts.append(
                f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(tick):.2f}" '
                f'y2="{sy(tick):.2f}" stroke="{grid}" stroke-width="1" />'
            )
        parts.append(text(left - 10, sy(tick) + 4, number_text(tick, 0), "end"))
    x_ticks = nice_ticks(low, high, 6)
    for tick in x_ticks:
        if options.grid_x:
            parts.append(
                f'<line x1="{sx(tick):.2f}" x2="{sx(tick):.2f}" y1="{top}" '
                f'y2="{top + plot_h}" stroke="{grid}" stroke-width="1" />'
            )
        parts.append(
            f'<line x1="{sx(tick):.2f}" x2="{sx(tick):.2f}" y1="{top + plot_h}" '
            f'y2="{top + plot_h + 5}" stroke="{options.text}" />'
        )
        parts.append(text(sx(tick), top + plot_h + 22, number_text(tick, options.decimals)))
    for index, count in enumerate(counts):
        if data["constant"]:
            x0, x1 = sx(edges[0]) - plot_w * 0.04, sx(edges[0]) + plot_w * 0.04
            label = f"Усі значення = {number_text(edges[0], options.decimals)}"
        else:
            x0, x1 = sx(edges[index]), sx(edges[index + 1])
            label = bin_text(edges, index, options.decimals)
        share = 100 * count / n if n else 0
        detail = f"{label}: {count} рядків ({share:.1f}% від n={n})"
        bar_h = plot_h - (sy(count) - top)
        parts.append(
            f'<g class="chart-item" tabindex="0" role="img" aria-label="{escape(detail, quote=True)}">'
            f"<title>{escape(detail)}</title>"
            f'<rect x="{x0:.2f}" y="{top:.2f}" width="{max(x1 - x0, 1):.2f}" height="{plot_h}" '
            f'fill="transparent" />'
            f'<rect x="{x0 + 0.5:.2f}" y="{sy(count):.2f}" width="{max(x1 - x0 - 1, 1):.2f}" '
            f'height="{max(bar_h, 0):.2f}" fill="{options.accent}" '
            f'fill-opacity="{options.histogram_opacity:g}" /></g>'
        )
    # Bars start at zero: the baseline is always drawn.
    parts.append(
        f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" '
        f'stroke="{options.text}" stroke-width="1" />'
    )
    median = data.get("median")
    if options.histogram_median and median is not None:
        mx = sx(median)
        parts.append(
            f'<line class="median-line" x1="{mx:.2f}" x2="{mx:.2f}" y1="{top - 8}" '
            f'y2="{top + plot_h}" stroke="{options.negative}" stroke-width="2" '
            f'stroke-dasharray="6 4" />'
        )
        anchor = "start" if mx < left + plot_w * 0.75 else "end"
        offset = 6 if anchor == "start" else -6
        parts.append(
            text(
                mx + offset,
                top - 12,
                "Медіана " + number_text(median, options.decimals),
                anchor,
                options.label_size,
            )
        )
    if options.frame:
        parts.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
            f'stroke="{options.text}" />'
        )
    if options.axis_labels:
        parts.append(text(left + plot_w / 2, height - 30, name, size=options.label_size))
        parts.append(
            f'<text x="22" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
            f'font-size="{options.label_size}" fill="{options.text}" '
            f'transform="rotate(-90 22 {top + plot_h / 2:.2f})">Кількість рядків</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- scatter (stage 04, 056)
# Idea from SRC-004 slides_4_types.js:137 (points + Pearson); every pairwise complete row
# is a point: no sampling, outlier removal, log scale or trimmed range.


def padded_range(values: list[float]) -> tuple[float, float]:
    low, high = min(values), max(values)
    if low == high:  # constant axis: explicit padding for readability
        pad = abs(low) * 0.05 or 0.5
        return low - pad, high + pad
    return low, high


def scatter_svg(
    pairs: list[list[float]],
    x_column: dict[str, Any],
    y_column: dict[str, Any],
    r: float | None,
    options: ChartOptions,
    title: str = "",
) -> str:
    width, height = 820, 520
    left, right, top, bottom = 100, 30, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    x_name, y_name = field_label(x_column), field_label(y_column)
    r_text = "—" if r is None else f"{r:.{options.decimals}f}"
    heading = title or f"{y_name} і {x_name}"
    description = (
        f"Діаграма розсіювання: X — {x_name}, Y — {y_name}; точок n={len(pairs)}; "
        f"Pearson r={r_text}."
    )
    x_low, x_high = padded_range([p[0] for p in pairs])
    y_low, y_high = padded_range([p[1] for p in pairs])

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_low) / (y_high - y_low) * plot_h

    tick_size = max(11, options.label_size - 2)
    grid = "#E3E5E8"
    parts = [
        f'<svg class="scatter" xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(description, quote=True)}" style="font-family:{options.font};'
        f'background:{options.background}">',
        f"<desc>{escape(description)}</desc>",
        f'<text x="20" y="35" font-size="{options.title_size}" fill="{options.text}">'
        f"{escape(heading)}</text>",
    ]

    def text(x: float, y: float, value: str, anchor: str = "middle", size: int = 0) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size or tick_size}" fill="{options.text}">{escape(value)}</text>'
        )

    for tick in nice_ticks(x_low, x_high, 6):
        if options.grid_x:
            parts.append(
                f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top}" '
                f'y2="{top + plot_h}" stroke="{grid}" />'
            )
        parts.append(text(sx(tick), top + plot_h + 22, number_text(tick, options.decimals)))
    for tick in nice_ticks(y_low, y_high, 5):
        if options.grid_y:
            parts.append(
                f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(tick):.1f}" '
                f'y2="{sy(tick):.1f}" stroke="{grid}" />'
            )
        parts.append(text(left - 10, sy(tick) + 4, number_text(tick, options.decimals), "end"))
    stroke = f' stroke="{options.text}" stroke-width="0.5"' if options.scatter_outline else ""
    parts.append(
        f'<g class="scatter-points" fill="{options.accent}" '
        f'fill-opacity="{options.scatter_opacity:g}"{stroke}>'
    )
    radius = options.scatter_size
    parts.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}"/>' for x, y in pairs)
    parts.append("</g>")
    parts.append(
        f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" '
        f'stroke="{options.text}" /><line x1="{left}" x2="{left}" y1="{top}" '
        f'y2="{top + plot_h}" stroke="{options.text}" />'
    )
    if options.frame:
        parts.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
            f'stroke="{options.text}" />'
        )
    if options.axis_labels:
        parts.append(text(left + plot_w / 2, height - 30, x_name, size=options.label_size))
        parts.append(
            f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            f'font-size="{options.label_size}" fill="{options.text}" '
            f'transform="rotate(-90 22 {top + plot_h / 2:.1f})">{escape(y_name)}</text>'
        )
    parts.append(text(left + plot_w, 35, f"n = {len(pairs)} · r = {r_text}", "end"))
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- missing values (stage 04, 057)
# Bars start at zero on a fixed 0-100% axis (SRC-004 slides_4_types.js:21); grey context and one
# accent for the largest share (slides_2_design.js:79). Equal maxima share the same accent.

MISSING_PAGE = 10


def missing_order(rows: list[dict[str, Any]], order: str) -> list[dict[str, Any]]:
    if order == "source":
        return list(rows)
    # Undefined shares go last; equal shares keep the source order (stable sort).
    return sorted(rows, key=lambda row: (row["percent"] is None, -(row["percent"] or 0.0)))


def missing_svg(
    rows: list[dict[str, Any]], top_share: float | None, options: ChartOptions, title: str = ""
) -> str:
    width, row_h = 820, 34
    left, right, top = 250, 150, 80
    plot_w = width - left - right
    height = top + max(1, len(rows)) * row_h + 60
    heading = title or "Пропуски за полями, % рядків"
    description = f"Горизонтальні стовпці: частка пропусків для {len(rows)} полів, шкала 0–100%."
    parts = [
        f'<svg class="missing-chart" xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="group" '
        f'aria-label="{escape(description, quote=True)}" style="font-family:{options.font};'
        f'background:{options.background}">',
        f'<text x="20" y="35" font-size="{options.title_size}" fill="{options.text}">'
        f"{escape(heading)}</text>",
    ]
    tick_size = max(11, options.label_size - 2)
    bottom = top + len(rows) * row_h
    for tick in (0, 25, 50, 75, 100):
        x = left + tick / 100 * plot_w
        if options.grid_x:
            parts.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top - 6}" y2="{bottom}" stroke="#E3E5E8" />'
            )
        parts.append(
            f'<text x="{x:.1f}" y="{bottom + 20}" text-anchor="middle" font-size="{tick_size}" '
            f'fill="{options.text}">{tick}%</text>'
        )
    for index, row in enumerate(rows):
        y = top + index * row_h
        name = field_label(row["field"])
        short = name if len(name) <= 30 else name[:27] + "…"
        percent = row["percent"]
        if percent is None:
            label = "— · " + (row["reason"] or "частку не визначено")
            detail = f"{name}: частку пропусків не визначено. {row['reason'] or ''}".strip()
        else:
            label = f"{percent:.{options.decimals}f}% ({row['missing']})"
            detail = f"{name}: {row['missing']} пропусків, {percent:.{options.decimals}f}% рядків"
        if row["invalid_infinite"]:
            detail += f"; нескінченностей окремо: {row['invalid_infinite']}"
        parts.append(
            f'<g class="chart-item" tabindex="0" role="img" aria-label="{escape(detail, quote=True)}">'
            f"<title>{escape(detail)}</title>"
            f'<text x="{left - 10}" y="{y + row_h / 2 + 5:.1f}" text-anchor="end" '
            f'font-size="{options.label_size}" fill="{options.text}">{escape(short)}</text>'
        )
        if percent is not None:
            fill = options.accent if top_share and percent == top_share else options.context
            bar = percent / 100 * plot_w
            parts.append(
                f'<rect x="{left}" y="{y + 6}" width="{max(bar, 0):.2f}" height="{row_h - 12}" '
                f'fill="{fill}" />'
            )
            if options.missing_labels:
                parts.append(
                    f'<text x="{left + bar + 6:.1f}" y="{y + row_h / 2 + 5:.1f}" '
                    f'font-size="{tick_size}" fill="{options.text}">{escape(label)}</text>'
                )
        else:
            parts.append(
                f'<text x="{left + 6}" y="{y + row_h / 2 + 5:.1f}" font-size="{tick_size}" '
                f'fill="{options.text}">{escape(label[:60])}</text>'
            )
        parts.append("</g>")
    parts.append(
        f'<line x1="{left}" x2="{left}" y1="{top - 6}" y2="{bottom}" stroke="{options.text}" />'
    )
    if options.frame:
        parts.append(
            f'<rect x="{left}" y="{top - 6}" width="{plot_w}" height="{bottom - top + 6}" '
            f'fill="none" stroke="{options.text}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------- model check (stage 06, 066)
# Same scatter grammar as stage 04 (points in the accent colour, light grid); the model line is
# the single highlight in the contrasting colour, reference lines are thin dashed text colour.


def diagnostic_svg(view: dict[str, Any], options: ChartOptions, title: str = "") -> str:
    """Points plus model/reference lines for the model check block; computed on the server."""
    width, height = 820, 520
    left, right, top, bottom = 100, 30, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    points = view["points"]
    xs = [p[0] for p in points] + [q[0] for line in view["lines"] for q in line["points"]]
    ys = [p[1] for p in points] + [q[1] for line in view["lines"] for q in line["points"]]
    x_low, x_high = padded_range(xs)
    y_low, y_high = padded_range(ys)
    x_pad, y_pad = (x_high - x_low) * 0.04, (y_high - y_low) * 0.04  # keep points off the axes
    x_low, x_high, y_low, y_high = x_low - x_pad, x_high + x_pad, y_low - y_pad, y_high + y_pad
    heading = title or view["title"]
    description = (
        f"{view['title']}: X — {view['x_label']}, Y — {view['y_label']}; точок n={len(points)}."
    )

    def sx(value: float) -> float:
        return left + (value - x_low) / (x_high - x_low) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - (value - y_low) / (y_high - y_low) * plot_h

    tick_size = max(11, options.label_size - 2)
    grid = "#E3E5E8"
    parts = [
        f'<svg class="diagnostic" xmlns="http://www.w3.org/2000/svg" width="100%" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(description, quote=True)}" style="max-width:{width}px;'
        f'height:auto;font-family:{options.font};background:{options.background}">',
        f"<desc>{escape(description)}</desc>",
        f'<text x="20" y="35" font-size="{options.title_size}" fill="{options.text}">'
        f"{escape(heading)}</text>",
    ]

    def text(x: float, y: float, value: str, anchor: str = "middle", size: int = 0) -> str:
        return (
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
            f'font-size="{size or tick_size}" fill="{options.text}">{escape(value)}</text>'
        )

    for tick in nice_ticks(x_low, x_high, 6):
        if options.grid_x:
            parts.append(
                f'<line x1="{sx(tick):.1f}" x2="{sx(tick):.1f}" y1="{top}" '
                f'y2="{top + plot_h}" stroke="{grid}" />'
            )
        parts.append(text(sx(tick), top + plot_h + 22, number_text(tick, options.decimals)))
    for tick in nice_ticks(y_low, y_high, 5):
        if options.grid_y:
            parts.append(
                f'<line x1="{left}" x2="{left + plot_w}" y1="{sy(tick):.1f}" '
                f'y2="{sy(tick):.1f}" stroke="{grid}" />'
            )
        parts.append(text(left - 10, sy(tick) + 4, number_text(tick, options.decimals), "end"))
    stroke = f' stroke="{options.text}" stroke-width="0.5"' if options.scatter_outline else ""
    parts.append(
        f'<g class="diagnostic-points" fill="{options.accent}" '
        f'fill-opacity="{options.scatter_opacity:g}"{stroke}>'
    )
    radius = options.scatter_size
    parts.extend(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}"/>' for x, y in points)
    parts.append("</g>")
    for line in view["lines"]:
        (x1, y1), (x2, y2) = line["points"]
        style = (
            f'stroke="{options.negative}" stroke-width="2.5"'
            if line["kind"] == "model"
            else f'stroke="{options.text}" stroke-width="1" stroke-dasharray="5 4" opacity="0.7"'
        )
        parts.append(
            f'<line class="diagnostic-{line["kind"]}" x1="{sx(x1):.1f}" y1="{sy(y1):.1f}" '
            f'x2="{sx(x2):.1f}" y2="{sy(y2):.1f}" {style} />'
        )
    parts.append(
        f'<line x1="{left}" x2="{left + plot_w}" y1="{top + plot_h}" y2="{top + plot_h}" '
        f'stroke="{options.text}" /><line x1="{left}" x2="{left}" y1="{top}" '
        f'y2="{top + plot_h}" stroke="{options.text}" />'
    )
    if options.frame:
        parts.append(
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" '
            f'stroke="{options.text}" />'
        )
    if options.axis_labels:
        parts.append(text(left + plot_w / 2, height - 30, view["x_label"], size=options.label_size))
        parts.append(
            f'<text x="22" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
            f'font-size="{options.label_size}" fill="{options.text}" '
            f'transform="rotate(-90 22 {top + plot_h / 2:.1f})">{escape(view["y_label"])}</text>'
        )
    parts.append(text(left + plot_w, 35, f"n = {len(points)}", "end"))
    parts.append("</svg>")
    return "".join(parts)
