"""First preparation component: explicit missing-value preview on an independent copy.

No storage, AI, inferred choices, learned imputation or modification of source data.
Row positions refer to the original frame, independently of pandas index labels.
"""

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from pandas.api import types as ptypes
from pydantic import BaseModel, ConfigDict, model_validator

from .metadata import Metadata
from .missing_values import missing_masks


class PreparationError(ValueError):
    """Safe Ukrainian error for the future UI/API adapter."""


class MissingRule(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    field: str
    action: Literal["drop", "constant"]
    value: float | str | bool | None = None

    @model_validator(mode="after")
    def check_value(self) -> "MissingRule":
        if self.action == "constant" and self.value is None:
            raise ValueError("Укажіть значення для заповнення пропусків.")
        if self.action == "drop" and self.value is not None:
            raise ValueError("Для вилучення рядків значення не потрібне.")
        return self


@dataclass(frozen=True)
class MissingPreview:
    frame: pd.DataFrame
    retained_positions: list[int]
    report: dict


def preview_missing(
    frame: pd.DataFrame, metadata: Metadata, rules: list[MissingRule]
) -> MissingPreview:
    """Validate the complete plan before dropping rows and filling surviving cells.

    Unselected fields remain untouched. No filling of the target or unsupported types.
    All drop masks are unioned before filling, so order cannot affect the result.
    """
    columns = {column.name: column for column in metadata.document.columns}
    if list(frame.columns) != list(columns):
        raise PreparationError("Стовпці файла не збігаються зі збереженими метаданими.")
    names = [rule.field for rule in rules]
    if len(set(names)) != len(names):
        raise PreparationError("Для кожного поля можна вибрати лише одну операцію.")
    # Internal positional index handles even repeated labels without row duplication.
    source = frame.reset_index(drop=True)
    drop = pd.Series(False, index=source.index)
    masks: dict[str, pd.Series] = {}
    for rule in rules:
        if rule.field not in columns:
            raise PreparationError("Поле не знайдено в наборі.")
        column = columns[rule.field]
        if column.role in ("id", "ignore") or column.kind == "id":
            raise PreparationError("Поле виключено за типом або роллю.")
        series = source[rule.field]
        base, extra, error = missing_masks(series, column.missing_codes)
        if error:
            raise PreparationError(error)
        mask = base | extra
        masks[rule.field] = mask
        if rule.action == "drop":
            drop |= mask
            continue
        if column.role == "target":
            raise PreparationError("Цільову змінну не заповнюємо: виберіть вилучення рядків.")
        value = rule.value
        if column.kind == "number":
            if (
                not ptypes.is_numeric_dtype(series)
                or ptypes.is_bool_dtype(series)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise PreparationError("Для числового поля вкажіть скінченне число.")
        elif column.kind == "bool":
            if not isinstance(value, bool) or not ptypes.is_bool_dtype(series):
                raise PreparationError("Для логічного поля вкажіть так або ні.")
        elif column.kind in ("category", "text"):
            # Numeric categories keep their existing representation; never coerce codes silently.
            if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise PreparationError("Категорії цього поля задано числами: вкажіть число.")
            elif not isinstance(value, str) or not value.strip():
                raise PreparationError("Для текстового поля вкажіть непорожній текст.")
        else:
            raise PreparationError("Заповнення дат у цьому компоненті ще не підтримується.")
        if isinstance(value, str) and value.strip() in [c.strip() for c in column.missing_codes]:
            raise PreparationError("Значення заповнення саме позначене як код пропуску.")
        if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
            if value in [float(c.strip()) for c in column.missing_codes]:
                raise PreparationError("Значення заповнення саме позначене як код пропуску.")
    positions = source.index[~drop].tolist()
    if rules and not positions:
        raise PreparationError("Після підготовки не залишиться жодного рядка.")
    prepared = source.loc[~drop].copy(deep=True)
    changes = []
    for rule in rules:
        selected = masks[rule.field].loc[~drop]
        affected = int(selected.sum())
        if rule.action == "constant" and affected:
            series = prepared[rule.field]
            if (
                ptypes.is_integer_dtype(series)
                and not ptypes.is_bool_dtype(series)
                and isinstance(rule.value, float)
                and not rule.value.is_integer()
            ):
                # Promote only fractional fills; preserve exact large integer categories.
                prepared[rule.field] = series.astype(float)
            try:
                prepared.loc[selected, rule.field] = rule.value
            except (TypeError, ValueError) as error:
                raise PreparationError("Значення несумісне з типом поля.") from error
        changes.append(
            {
                "field": rule.field,
                "action": rule.action,
                "missing_before": int(masks[rule.field].sum()),
                "filled": affected if rule.action == "constant" else 0,
                "removed_with_missing": int((masks[rule.field] & drop).sum()),
            }
        )
    prepared.index = frame.index.take(positions)
    return MissingPreview(
        frame=prepared,
        retained_positions=positions,
        report={
            "rows_before": len(frame),
            "rows_after": len(prepared),
            "rows_removed": int(drop.sum()),
            "changes": changes,
            "warnings": [] if metadata.confirmed_at else ["Метадані не підтверджено"],
        },
    )
