"""Stepwise choice of fields for linear regression on the train part only (history 143).

Bidirectional search like R ``step()``: start from the intercept-only model; on every step try
adding each unused field and dropping each used one, take the move that improves the chosen
criterion most, stop when nothing improves it. A category field moves as one whole group of
one-hot columns. The test part is never looked at, so the final test metrics stay honest.
"""

import math
from typing import Literal

import numpy as np

from .model import ModelError

Criterion = Literal["adj_r2", "aic", "bic"]
CRITERIA: dict[str, dict] = {
    "adj_r2": {"title": "Скоригований R²", "better": "max"},
    "aic": {"title": "AIC", "better": "min"},
    "bic": {"title": "BIC", "better": "min"},
}
MAX_STEPS = 100


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS with intercept on the train part: R², adjusted R², AIC and BIC.

    AIC/BIC as in R ``extractAIC`` for lm: n·ln(RSS/n) + k·(p + 1), k = 2 or ln(n);
    p counts matrix columns (a category adds one column per non-baseline category).
    """
    n, p = x.shape
    design = np.column_stack([np.ones(n), x]) if p else np.ones((n, 1))
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    rss = float(np.sum((y - design @ coefficients) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    if n <= p + 1 or tss <= 0:
        raise ModelError("Замало навчальних рядків або стала ціль: stepwise неможливий.")
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    log_term = n * math.log(max(rss, 1e-300) / n)
    return {
        "r2": r2,
        "adj_r2": adj,
        "aic": log_term + 2 * (p + 1),
        "bic": log_term + math.log(n) * (p + 1),
        "columns": p,
    }


def _better(criterion: str, new: float, old: float) -> bool:
    gain = new - old if CRITERIA[criterion]["better"] == "max" else old - new
    return gain > 1e-9 * max(1.0, abs(old))


def stepwise(
    x_train: np.ndarray,
    y_train: np.ndarray,
    columns: dict[str, list[int]],
    criterion: str = "adj_r2",
) -> dict:
    """Return the chosen fields (in split order) and every step taken."""
    if criterion not in CRITERIA:
        raise ModelError("Невідомий критерій stepwise.")
    order = list(columns)
    x = x_train.astype(float)
    y = y_train.astype(float)

    def stats(fields: list[str]) -> dict:
        index = [i for name in order if name in fields for i in columns[name]]
        return fit_stats(x[:, index], y)

    chosen: list[str] = []
    current = stats(chosen)
    steps = [{"step": 0, "action": "start", "field": None, **current, "fields": []}]
    for number in range(1, MAX_STEPS + 1):
        best = None
        # Like the R step() trace: every move tried on this step plus "<none>" (stay as is).
        tried = [{"action": "none", "field": None, "value": current[criterion],
                  "adj_r2": current["adj_r2"]}]  # fmt: skip
        candidates = [("add", f) for f in order if f not in chosen] + [("drop", f) for f in chosen]
        for action, field in candidates:
            fields = chosen + [field] if action == "add" else [f for f in chosen if f != field]
            try:
                result = stats(fields)
            except ModelError:
                continue
            tried.append(
                {
                    "action": action,
                    "field": field,
                    "value": result[criterion],
                    "adj_r2": result["adj_r2"],
                }  # fmt: skip
            )
            if best is None or _better(criterion, result[criterion], best[2][criterion]):
                best = (action, field, result, fields)
        tried.sort(
            key=lambda t: -t["value"] if CRITERIA[criterion]["better"] == "max" else t["value"]
        )
        if best is None or not _better(criterion, best[2][criterion], current[criterion]):
            steps.append(
                {
                    "step": number,
                    "action": "stop",
                    "field": None,
                    **current,
                    "fields": list(chosen),
                    "candidates": tried,
                }  # fmt: skip
            )
            break
        action, field, current, chosen = best
        chosen = [f for f in order if f in chosen]
        steps.append(
            {
                "step": number,
                "action": action,
                "field": field,
                **current,
                "fields": list(chosen),
                "candidates": tried,
            }  # fmt: skip
        )
    if not chosen:
        raise ModelError(
            "Жодна ознака не покращує критерій порівняно з моделлю лише з вільним членом."
        )
    return {
        "method": "stepwise",
        "direction": "both",
        "criterion": criterion,
        "criterion_title": CRITERIA[criterion]["title"],
        "better": CRITERIA[criterion]["better"],
        "part": "train",
        "features": chosen,
        "steps": steps,
    }
