"""Shared missing-value semantics for exploratory analysis and data preparation."""

import math

import pandas as pd
from pandas.api import types as ptypes


def missing_masks(series: pd.Series, codes: list[str]) -> tuple[pd.Series, pd.Series, str | None]:
    """Return parser NA, additional explicit codes and a safe validation error."""
    base = series.isna()
    extra = pd.Series(False, index=series.index)
    if ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series):
        try:
            numeric_codes = [float(code.strip()) for code in codes]
            if not all(math.isfinite(code) for code in numeric_codes):
                raise ValueError
            extra = series.isin(numeric_codes) & ~base
        except ValueError:
            return base, extra, "Код пропуску має бути скінченним числом."
    else:
        extra = series.astype("string").str.strip().isin([code.strip() for code in codes]) & ~base
    return base, extra, None
