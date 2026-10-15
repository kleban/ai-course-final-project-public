"""Model check views on the saved test part: regression line, errors, prediction vs actual.

Predictions are reproduced from the saved coefficients (no pickle) and must agree with the
stored MAE; otherwise the saved model and matrices do not belong together.
"""

import math
from typing import Literal

import numpy as np

from .model import ModelError, check_matrices

Chart = Literal["regression", "residuals", "actual"]
CHARTS: dict[str, str] = {
    "regression": "Графік регресії",
    "residuals": "Графік похибок",
    "actual": "Прогноз проти факту",
}
HINTS = {
    "regression": (
        "Точки — тестові рядки: обрана ознака і фактична ціль. Лінія — прогноз моделі, коли "
        "змінюється лише ця ознака, а інші зафіксовані на середніх навчальної частини."
    ),
    "residuals": (
        "Похибка = факт − прогноз. Добре, коли точки рівномірно розкидані навколо нуля без "
        "вигину чи «лійки»; візерунок означає, що лінійна модель щось пропускає."
    ),
    "actual": (
        "Кожна точка — тестовий рядок: факт по горизонталі, прогноз по вертикалі. Що ближче "
        "точки до діагоналі, то точніший прогноз."
    ),
}


def _num(value: float) -> str:
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return text.replace(",", " ").replace(".", ",")


def _signed(value: float) -> str:
    return ("− " if value < 0 else "+ ") + _num(abs(value))


def numeric_features(document: dict) -> list[str]:
    """Columns that are original numeric features (one-hot columns are 'name = value')."""
    chosen = set(document["choices"]["features"])
    return [name for name in document["feature_names"] if name in chosen]


def _tex_num(value: float) -> str:
    return _num(value).replace("\u202f", "\\,").replace(",", "{,}").replace("\\{,}", "\\,")


def _tex(name: str) -> str:
    return "".join("\\" + ch if ch in "#$%&_{}" else ch for ch in name).replace("~", " ")


def equation(document: dict) -> tuple[str, str]:
    """Full model equation as plain text and as TeX; a₀ is the intercept."""
    terms = [(c["value"], c["name"]) for c in document["coefficients"]]
    target = document["choices"]["target"]
    text = f"ŷ({target}) = {_num(document['intercept'])} " + " ".join(
        f"{_signed(value)}·{name}" for value, name in terms
    )
    tex = f"\\hat y = {_tex_num(document['intercept'])}" + "".join(
        f" {'-' if value < 0 else '+'} {_tex_num(abs(value))}\\cdot\\text{{{_tex(name)}}}"
        for value, name in terms
    )
    return text, tex


def diagnostics(
    document: dict, arrays: dict[str, np.ndarray], chart: str, feature: str | None = None
) -> dict:
    check_matrices(
        arrays,
        {
            "train_count": document["train_count"],
            "test_count": document["test_count"],
            "feature_count": document["feature_count"],
            "feature_names": document["feature_names"],
        },
    )
    if chart not in CHARTS:
        raise ModelError("Такого графіка перевірки немає.")
    coefficients = np.array([c["value"] for c in document["coefficients"]], dtype=float)
    intercept = float(document["intercept"])
    x_train, x_test = arrays["x_train"].astype(float), arrays["x_test"].astype(float)
    y_test = arrays["y_test"].astype(float)
    predicted = intercept + x_test @ coefficients
    stored = document["metrics"]["test"]["mae"]
    mae = float(np.abs(y_test - predicted).mean())
    if not math.isclose(mae, stored, rel_tol=1e-6, abs_tol=1e-9 * (1 + abs(stored))):
        raise ModelError("Збережена модель не відповідає файлу вибірок. Навчіть модель заново.")
    residuals = y_test - predicted
    target = document["choices"]["target"]
    text, tex = equation(document)
    view: dict = {
        "chart": chart,
        "title": CHARTS[chart],
        "hint": HINTS[chart],
        "n": len(y_test),
        "equation": text,
        "equation_tex": tex,
        "features": numeric_features(document),
        "feature": None,
        "notes": [],
    }
    if chart == "regression":
        options = view["features"]
        if not options:
            raise ModelError(
                "Немає числових ознак для графіка регресії: категорії показуються лише "
                "в коефіцієнтах."
            )
        name = feature or options[0]
        if name not in options:
            raise ModelError("Оберіть числову ознаку цієї моделі.")
        j = document["feature_names"].index(name)
        means = x_train.mean(axis=0)
        base = intercept + float(coefficients @ means - coefficients[j] * means[j])
        low, high = float(x_test[:, j].min()), float(x_test[:, j].max())
        if low == high:
            low, high = float(x_train[:, j].min()), float(x_train[:, j].max())
        slope = float(coefficients[j])
        view.update(
            feature=name,
            x_label=name,
            y_label=target,
            points=[[float(a), float(b)] for a, b in zip(x_test[:, j], y_test)],
            lines=[
                {
                    "kind": "model",
                    "points": [[low, base + slope * low], [high, base + slope * high]],
                }
            ],
            partial=f"ŷ({target}) = {_num(base)} {_signed(slope)}·{name}",
        )
        if len(options) < len(document["feature_names"]) or len(document["feature_names"]) > 1:
            view["notes"].append(
                "Інші ознаки зафіксовані на середніх навчальної частини; розкид точок "
                "навколо лінії включає і їхній вплив."
            )
    elif chart == "residuals":
        low, high = float(predicted.min()), float(predicted.max())
        view.update(
            x_label=f"Прогноз «{target}»",
            y_label="Похибка: факт − прогноз",
            points=[[float(a), float(b)] for a, b in zip(predicted, residuals)],
            lines=[{"kind": "zero", "points": [[low, 0.0], [high, 0.0]]}],
        )
        view["notes"].append(
            f"Середня похибка {_num(float(residuals.mean()))}; найбільша за модулем "
            f"{_num(float(np.abs(residuals).max()))}."
        )
    else:
        low = float(min(y_test.min(), predicted.min()))
        high = float(max(y_test.max(), predicted.max()))
        view.update(
            x_label=f"Факт «{target}»",
            y_label=f"Прогноз «{target}»",
            points=[[float(a), float(b)] for a, b in zip(y_test, predicted)],
            lines=[{"kind": "diagonal", "points": [[low, low], [high, high]]}],
        )
    if len(y_test) < 30:
        view["notes"].append(f"Мала тестова частина (n={len(y_test)}): візерунки нестійкі.")
    return view
