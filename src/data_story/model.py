"""Linear regression on saved train/test matrices: fit on train only, score on test."""

import math
import warnings as py_warnings

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ESTIMATOR = "sklearn.linear_model.LinearRegression"
INFERENCE_VERSION = 2
MATRIX_KEYS = ("x_train", "x_test", "y_train", "y_test")


class ModelError(ValueError):
    """The saved matrices cannot produce a trustworthy linear regression."""


class ModelNotFound(LookupError):
    """No saved model with this id in the project."""


SIGNIFICANCE = ((0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, "."))
SIGNIFICANCE_NOTE = (
    "Позначки значущості, як у summary() в R: *** p < 0,001; ** p < 0,01; * p < 0,05; "
    ". p < 0,1; без позначки — p ≥ 0,1."
)
INFERENCE_NOTES = [
    "p-value перевіряє гіпотезу, що коефіцієнт дорівнює нулю за незмінних інших ознак. "
    "Мале p-value означає, що за гіпотези нульового коефіцієнта та припущень моделі "
    "такий або більш крайній результат малоймовірний; "
    "воно не вимірює силу чи важливість ознаки і не доводить причинність.",
    "Стандартні помилки, t і p-value розраховано класичним методом найменших квадратів "
    "на навчальній частині (як lm у R). Вони коректні, якщо помилки незалежні, мають однакову "
    "мінливість і приблизно нормальний розподіл; інакше p-value можуть бути занижені.",
    "Коли перевіряється багато коефіцієнтів, деякі виявляться «значущими» випадково; "
    "сильна кореляція між ознаками збільшує стандартні помилки.",
]


def stars(p_value: float | None) -> str:
    if p_value is None:
        return ""
    return next((mark for limit, mark in SIGNIFICANCE if p_value < limit), "")


def format_p_value(value: float | None) -> str:
    """Short display whose rounding never crosses a significance threshold."""
    if value is None:
        return "не визначено"
    if value < 0.001:
        return "< 0,001"
    digits = 3 if value < 0.01 else 2
    while digits < 16 and stars(float(f"{value:.{digits}f}")) != stars(value):
        digits += 1
    return f"{value:.{digits}f}".replace(".", ",")


def inference(
    x_train: np.ndarray, y_train: np.ndarray, coefficients: np.ndarray, intercept: float, rank: int
) -> dict:
    """OLS standard errors, t and two-sided p-values on the training part (R ``summary.lm``).

    Uses the pseudo-inverse: with linearly dependent features only estimable coefficients get
    statistics; the others are reported as undefined instead of misleading numbers.
    """
    design = np.column_stack([np.ones(len(y_train)), x_train])
    estimate = np.concatenate([[intercept], coefficients])
    residual = y_train - design @ estimate
    df = len(y_train) - rank
    rss = float(residual @ residual)
    tss = float(((y_train - y_train.mean()) ** 2).sum())
    empty: dict[str, object] = {"std_error": None, "t": None, "p_value": None, "stars": ""}
    if df < 1:
        reason = "Немає залишкових ступенів свободи."
        return {"df": df, "sigma": None, "reason": reason, "rows": [empty] * len(estimate)}
    if rss <= 1e-24 * max(tss, 1.0) or rss == 0:
        reason = (
            "Модель майже ідеально відтворює навчальні дані (залишки ≈ 0), тому p-value "
            "ненадійні й не показуються."
        )
        return {"df": df, "sigma": 0.0, "reason": reason, "rows": [empty] * len(estimate)}
    sigma2 = rss / df
    with np.errstate(all="ignore"):
        # Forming X.T @ X squares the condition number and can erase intercept variance.
        # Use the same retained SVD rank for covariance and the estimable subspace.
        _, singular, right = np.linalg.svd(design, full_matrices=False)
        scaled = right[:rank].T / singular[:rank]
        covariance = scaled @ scaled.T
        estimable = (
            np.ones(design.shape[1], dtype=bool)
            if rank == design.shape[1]
            else (1 - np.sum(right[:rank] ** 2, axis=0) < 1e-10)
        )
    rows: list[dict[str, object]] = []
    for j, value in enumerate(estimate):
        variance = sigma2 * covariance[j, j]
        if not estimable[j] or not math.isfinite(variance) or variance <= 0:
            rows.append({**empty, "reason": "Не визначено через лінійну залежність ознак."})
            continue
        error = math.sqrt(variance)
        t_value = float(value) / error
        p_value = float(2 * stats.t.sf(abs(t_value), df))
        rows.append({"std_error": error, "t": t_value, "p_value": p_value, "stars": stars(p_value)})
    return {"df": df, "sigma": math.sqrt(sigma2), "reason": None, "rows": rows}


LIMITATIONS = [
    "Модель припускає лінійний зв'язок ознак із ціллю; нелінійні залежності й взаємодії не враховано.",
    "Коефіцієнт показує зміну прогнозу при зміні ознаки на одиницю за незмінних інших ознак; "
    "це асоціація в даних, а не доказ причинного впливу.",
    "Ознаки не масштабувались: величини коефіцієнтів залежать від одиниць вимірювання й "
    "напряму не порівнюються між собою.",
    "Для категорії коефіцієнт — зсув прогнозу відносно базової категорії цієї ознаки "
    "(першої за алфавітом); у базової категорії окремого коефіцієнта немає.",
    "Показники отримано на одному випадковому поділі; інший поділ може дати інші значення.",
    "Прогнози поза діапазоном навчальних значень (екстраполяція) ненадійні.",
    "Викиди й неоднакова мінливість помилок можуть сильно впливати на коефіцієнти; "
    "залишки перевіряються в блоці «Перевірка моделі».",
]


def check_matrices(arrays: dict[str, np.ndarray], expected: dict) -> None:
    """Reject damaged or mismatching matrices before any fitting."""
    if set(arrays) != set(MATRIX_KEYS):
        raise ModelError("Файл вибірок не містить потрібних матриць.")
    x_train, x_test = arrays["x_train"], arrays["x_test"]
    y_train, y_test = arrays["y_train"], arrays["y_test"]
    if x_train.ndim != 2 or x_test.ndim != 2 or y_train.ndim != 1 or y_test.ndim != 1:
        raise ModelError("Розмірність матриць вибірок пошкоджена.")
    if any(not np.issubdtype(a.dtype, np.number) or a.dtype == bool for a in arrays.values()):
        raise ModelError("Матриці вибірок мають містити лише числа.")
    if (
        x_train.shape != (expected["train_count"], expected["feature_count"])
        or x_test.shape != (expected["test_count"], expected["feature_count"])
        or len(y_train) != expected["train_count"]
        or len(y_test) != expected["test_count"]
    ):
        raise ModelError("Розміри матриць не збігаються з описом збереженої підготовки.")
    if len(expected["feature_names"]) != expected["feature_count"]:
        raise ModelError("Назви ознак не відповідають матриці.")
    if not all(np.isfinite(a.astype(float)).all() for a in arrays.values()):
        raise ModelError("У матрицях є пропуски або нескінченні значення.")


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def train_linear_regression(arrays: dict[str, np.ndarray], feature_names: list[str]) -> dict:
    x_train = arrays["x_train"].astype(float)
    x_test = arrays["x_test"].astype(float)
    y_train = arrays["y_train"].astype(float)
    y_test = arrays["y_test"].astype(float)
    rows, width = x_train.shape
    if rows <= width + 1:
        raise ModelError(
            f"Замало навчальних рядків ({rows}) для {width} ознак: потрібно більше, ніж ознак + 1. "
            "Зменшіть кількість ознак або збільшіть навчальну частину."
        )
    estimator = LinearRegression()
    try:
        return _fit_and_score(estimator, x_train, y_train, x_test, y_test, feature_names)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        if isinstance(error, ModelError):
            raise
        raise ModelError(
            "Результат числово нестабільний (нескінченні значення). Перевірте масштаб ознак і цілі."
        ) from error


def _fit_and_score(
    estimator: LinearRegression,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
) -> dict:
    rows, width = x_train.shape
    with np.errstate(all="ignore"), py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        estimator.fit(x_train, y_train)
        train_prediction = estimator.predict(x_train)
        test_prediction = estimator.predict(x_test)
        r2_test = float(r2_score(y_test, test_prediction, force_finite=False))
        r2_train = float(r2_score(y_train, train_prediction, force_finite=False))
        mae = float(mean_absolute_error(y_test, test_prediction))
        rmse = float(root_mean_squared_error(y_test, test_prediction))
        baseline = float(mean_absolute_error(y_test, np.full(len(y_test), y_train.mean())))
    coefficients = np.asarray(estimator.coef_, dtype=float)
    intercept = float(estimator.intercept_)
    if (
        not np.isfinite(coefficients).all()
        or not math.isfinite(intercept)
        or not np.isfinite(test_prediction).all()
        or not math.isfinite(mae)
        or not math.isfinite(rmse)
    ):
        raise ModelError(
            "Результат числово нестабільний (нескінченні значення). Перевірте масштаб ознак і цілі."
        )
    rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(rows), x_train])))
    with np.errstate(all="ignore"), py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore")
        tests = inference(x_train, y_train, coefficients, intercept, rank)
    warnings: list[str] = []
    r2_reason = None
    if not math.isfinite(r2_test):
        r2_reason = (
            "R² не визначений: усі значення цілі в тестовій частині однакові, "
            "тому немає мінливості, яку можна пояснити."
        )
    if rank < width + 1:
        warnings.append(
            "Ознаки лінійно залежні (наприклад, усі категорії one-hot разом із вільним членом "
            "або стала ознака). Прогноз коректний, але окремі коефіцієнти не визначені однозначно; "
            "scikit-learn повертає розв'язок із найменшою нормою."
        )
    if len(y_test) < 30:
        warnings.append(
            f"Тестова частина мала (n={len(y_test)}): показники нестійкі й можуть суттєво "
            "змінюватися з іншим поділом."
        )
    if math.isfinite(r2_test) and math.isfinite(r2_train) and r2_train - r2_test > 0.2:
        warnings.append(
            "R² на навчальній частині помітно вищий за тестовий: можливе перенавчання "
            "або відмінність тестових даних."
        )
    if math.isfinite(r2_test) and r2_test < 0:
        warnings.append("Тестовий R² від'ємний: модель прогнозує гірше, ніж середнє значення цілі.")
    return {
        "estimator": ESTIMATOR,
        "params": {
            k: v
            for k, v in estimator.get_params().items()
            if v is None or isinstance(v, (bool, int, float, str))
        },
        "feature_names": list(feature_names),
        "coefficients": [
            {"name": name, "value": float(value), **row}
            for name, value, row in zip(feature_names, coefficients, tests["rows"][1:])
        ],
        "intercept": intercept,
        "intercept_inference": tests["rows"][0],
        "inference": {
            "version": INFERENCE_VERSION,
            "method": "OLS на навчальній частині; t-розподіл; двосторонні p-value",
            "df_residual": tests["df"],
            "sigma": tests["sigma"],
            "reason": tests["reason"],
            "significance_note": SIGNIFICANCE_NOTE,
            "notes": INFERENCE_NOTES,
        },
        "metrics": {
            "test": {
                "r2": _finite(r2_test),
                "r2_reason": r2_reason,
                "mae": mae,
                "rmse": rmse,
            },
            "train": {"r2": _finite(r2_train)},
            "baseline_mae": baseline,
        },
        "train_count": rows,
        "test_count": len(y_test),
        "feature_count": width,
        "rank": rank,
        "warnings": warnings,
        "limitations": LIMITATIONS,
    }


def _num(value: float) -> str:
    """Ukrainian number text for explanations: thin-space thousands, decimal comma."""
    if abs(value) < 1e-10:
        return "≈ 0"
    text = f"{value:,.0f}" if abs(value) >= 1000 else f"{value:.4g}"
    return text.replace(",", " ").replace(".", ",")


def _eq(value: float) -> str:
    text = _num(value)
    return text if text.startswith("≈") else f"= {text}"


def explain(document: dict) -> list[str]:
    """Deterministic plain-language reading of saved numbers (no AI, no new calculation)."""
    test = document["metrics"]["test"]
    target = document["choices"]["target"]
    lines = []
    if test["r2"] is None:
        lines.append(test["r2_reason"] or "R² не визначений.")
    else:
        lines.append(
            f"R² {_eq(test['r2'])}: на тестовій частині модель пояснює приблизно "
            f"{_num(max(test['r2'], 0) * 100)}% мінливості «{target}» відносно середнього."
            if test["r2"] >= 0
            else f"R² {_eq(test['r2'])}: прогноз гірший за просте середнє значення цілі."
        )
    lines.append(
        f"MAE {_eq(test['mae'])}: у середньому прогноз відхиляється від факту на стільки "
        f"одиниць «{target}»."
    )
    lines.append(
        f"RMSE {_eq(test['rmse'])}: сильніше враховує великі помилки; що більший він "
        "за MAE, то більше окремих великих відхилень."
    )
    tested = [c for c in document["coefficients"] if c.get("p_value") is not None]
    if tested:
        significant = [c["name"] for c in tested if c["p_value"] < 0.05]
        lines.append(
            "На рівні 0,05 статистично значущі: " + ", ".join(significant) + "."
            if significant
            else "Жоден коефіцієнт не є статистично значущим на рівні 0,05."
        )
    baseline = document["metrics"]["baseline_mae"]
    if baseline > 0:
        change = (baseline - test["mae"]) / baseline * 100
        lines.append(
            f"Прогноз середнім навчальної цілі дав би MAE {_eq(baseline)}; модель "
            + (
                f"зменшує середню помилку на {_num(change)}%."
                if change >= 0
                else f"збільшує її на {_num(-change)}%."
            )
        )
    return lines
