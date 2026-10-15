"""Explicit train/test preparation. Fit imputers and encoders on training rows only."""

import warnings as py_warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api import types as ptypes
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from .metadata import Metadata
from .missing_values import missing_masks
from .preparation import PreparationError

MAX_CELLS = 2_000_000
MAX_FEATURES = 200


class TrainingChoices(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target: str = Field(min_length=1)
    features: list[str] = Field(min_length=1, max_length=30)
    test_percent: int = Field(default=30, ge=10, le=50)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    encoding: Literal["none", "onehot"] = "none"
    numeric_missing: Literal["reject", "mean", "median"] = "reject"
    category_missing: Literal["reject", "most_frequent"] = "reject"

    @model_validator(mode="after")
    def distinct(self) -> "TrainingChoices":
        if self.target in self.features or len(set(self.features)) != len(self.features):
            raise ValueError("Ціль і ознаки мають бути різними, без повторів.")
        return self


@dataclass
class TrainingMatrices:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_positions: list[int]
    test_positions: list[int]
    report: dict


def prepare_training(
    frame: pd.DataFrame, metadata: Metadata, choices: TrainingChoices
) -> TrainingMatrices:
    columns = {c.name: c for c in metadata.document.columns}
    if list(frame.columns) != list(columns):
        raise PreparationError("Стовпці файла не збігаються зі збереженими метаданими.")
    if choices.target not in columns or any(f not in columns for f in choices.features):
        raise PreparationError("Оберіть ціль і ознаки серед полів набору.")
    target = columns[choices.target]
    if target.kind != "number" or target.role in ("id", "ignore"):
        raise PreparationError(
            "Для регресії оберіть числову ціль без ролі ідентифікатора або ігнорування."
        )
    if len(frame) < 5:
        raise PreparationError("Для навчальної й тестової вибірок потрібно щонайменше 5 рядків.")
    source = frame.reset_index(drop=True)
    cleaned: dict[str, pd.Series] = {}
    kinds: dict[str, str] = {}
    for name in [choices.target, *choices.features]:
        column = columns[name]
        if column.role in ("id", "ignore") or column.kind == "id":
            raise PreparationError("Ідентифікатори та ігноровані поля не входять у модель.")
        if name != choices.target and column.role == "target":
            raise PreparationError("Інші цільові поля не можна використовувати як ознаки.")
        series = source[name]
        base, extra, error = missing_masks(series, column.missing_codes)
        if error:
            raise PreparationError(error)
        missing = base | extra
        if column.kind == "number":
            if not ptypes.is_numeric_dtype(series) or ptypes.is_bool_dtype(series):
                raise PreparationError("Тип числового поля не відповідає фактичним даним.")
            numeric = series.astype(float).where(~missing, np.nan)
            if np.isinf(numeric.to_numpy()).any():
                raise PreparationError(
                    "В обраних полях є нескінченні значення. Виправте їх перед моделлю."
                )
            cleaned[name] = numeric
            kinds[name] = "number"
        elif column.kind in ("category", "bool") and name != choices.target:
            cleaned[name] = series.astype("string").astype(object).where(~missing, np.nan)
            kinds[name] = "category"
        else:
            raise PreparationError(
                "Ознаки мають бути числами, категоріями або так/ні. Дати й текст спочатку підготуйте."
            )
    y = cleaned[choices.target].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise PreparationError("Ціль має пропуски. Явно вилучіть ці рядки у підготовці даних.")
    train, test = train_test_split(
        np.arange(len(frame)),
        test_size=choices.test_percent / 100,
        random_state=choices.seed,
        shuffle=True,
    )
    if len(train) < 2 or len(test) < 2:
        raise PreparationError("У навчальній і тестовій вибірках має бути щонайменше по 2 рядки.")
    train_parts, test_parts, labels, transforms, warnings = [], [], [], {}, []
    width = 0
    for name in choices.features:
        values = cleaned[name].to_numpy().reshape(-1, 1)
        left, right = values[train], values[test]
        is_numeric = kinds[name] == "number"
        strategy = choices.numeric_missing if is_numeric else choices.category_missing
        if pd.isna(left).all():
            raise PreparationError("Обрана ознака повністю порожня у навчальній вибірці.")
        if strategy == "reject":
            if pd.isna(values).any():
                raise PreparationError(
                    "В ознаках є пропуски. Оберіть спосіб заповнення або підготуйте набір."
                )
        else:
            imputer = SimpleImputer(strategy=strategy)
            left = imputer.fit_transform(left)
            right = imputer.transform(right)
            statistic = imputer.statistics_[0]
            transforms[name] = {
                "missing_strategy": strategy,
                "fill": float(statistic) if is_numeric else str(statistic),
            }
        if not is_numeric:
            if choices.encoding != "onehot":
                raise PreparationError("Для категорій явно оберіть кодування One-hot.")
            category_count = len(pd.unique(left.ravel()))
            if category_count < 2:
                raise PreparationError(
                    f"Ознака «{name}» у навчальній частині має лише одну категорію: "
                    "вона не несе інформації для моделі."
                )
            if (
                width + category_count > MAX_FEATURES
                or len(frame) * (width + category_count) > MAX_CELLS
            ):
                raise PreparationError(
                    "Забагато категорій для MVP. Оберіть менше ознак або об'єднайте категорії."
                )
            # Treatment coding like R: the first category (alphabetical) is the baseline, so the
            # columns are not collinear with the intercept and each category gets its own
            # p-value as a shift relative to the baseline (D-85).
            encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
            with py_warnings.catch_warnings():
                py_warnings.simplefilter("ignore")  # unknown test categories are reported below
                encoder.fit(left)
                categories = encoder.categories_[0].tolist()
                unknown = int(sum(value not in categories for value in right.ravel()))
                left, right = encoder.transform(left), encoder.transform(right)
            baseline = categories[0]
            if unknown:
                warnings.append(
                    f"{name}: {unknown} тестових рядків мають категорії, яких немає в навчальній "
                    f"частині; для них прогноз рахується як для базової категорії «{baseline}»."
                )
            transforms.setdefault(name, {}).update(
                categories=categories, baseline=str(baseline), unknown_test_rows=unknown
            )
            labels.extend([f"{name} = {value}" for value in categories[1:]])
        else:
            labels.append(name)
        width += left.shape[1]
        if width > MAX_FEATURES or len(frame) * width > MAX_CELLS:
            raise PreparationError("Підготовлена матриця завелика для MVP.")
        train_parts.append(left.astype(float))
        test_parts.append(right.astype(float))
    x_train, x_test = np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise PreparationError("Після підготовки залишилися непридатні значення.")
    if not metadata.confirmed_at:
        warnings.append("Метадані не підтверджено.")
    warnings.append(
        "Випадковий поділ придатний для незалежних рядків. Для часових або групових даних потрібен окремий спосіб поділу."
    )
    return TrainingMatrices(
        x_train,
        x_test,
        y[train],
        y[test],
        train.tolist(),
        test.tolist(),
        {
            "row_count": len(frame),
            "train_count": len(train),
            "test_count": len(test),
            "feature_count": width,
            "feature_names": labels,
            "transforms": transforms,
            "warnings": warnings,
            "method": "train_test_split; fit on train only; onehot drop=first "
            "(baseline category), handle_unknown=ignore",
        },
    )
