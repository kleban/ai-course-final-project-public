"""Stage 03 calculation core. No AI, UI, rounding or changes to source data."""

import hashlib
import json
import math
from collections.abc import Callable
from time import perf_counter
from typing import Any, Literal

import pandas as pd
from pandas.api import types as ptypes

from .datasets import DatasetError, DatasetStore, _date_text
from .metadata import Metadata, MetadataStore, issues
from .missing_values import missing_masks
from .projects import ProjectStore
from .storage import connect

View = Literal["descriptive", "missing", "correlations"]
METHODS = {
    "descriptive": "sample std (n-1); linear quantiles (n-1)*p",
    "missing": "parser NA + explicit codes; union of rows",
    "correlations": "Pearson; pairwise finite observations; n>=3",
}

CACHE_VERSION = 1


def result_tables(result: dict) -> list[tuple[str, pd.DataFrame]]:
    """Plain numeric tables for SQL; the JSON cache also preserves explanations and labels."""
    rows = []
    for column in result["columns"]:
        row = {
            k: column[k]
            for k in (
                "name",
                "label",
                "kind",
                "role",
                "base_missing",
                "additional_missing",
                "missing",
                "missing_percent",
                "invalid_infinite",
            )
        }
        row.update(
            {
                k: v
                for k, v in (column.get("statistics") or {}).items()
                if v is None or isinstance(v, int | float | bool)
            }
        )
        rows.append(row)
    tables = [("fields", pd.DataFrame(rows))]
    if result.get("r"):
        pairs = [
            {
                "left": left,
                "right": right,
                "r": result["r"][i][j],
                "n": result["n"][i][j],
                "reason": result["reasons"][i][j],
            }
            for i, left in enumerate(result["fields"])
            for j, right in enumerate(result["fields"])
        ]
        tables.append(("correlations", pd.DataFrame(pairs)))
    if "pairs" in result:
        tables.append(("pairs", pd.DataFrame(result["pairs"], columns=["x", "y"])))
    if "values" in result:
        tables.append(("values", pd.DataFrame({"value": result["values"]})))
    if "histogram" in result:
        histogram = result["histogram"]
        tables.append(
            (
                "bins",
                pd.DataFrame(
                    {
                        "left": histogram["edges"][:-1],
                        "right": histogram["edges"][1:],
                        "count": histogram["counts"],
                    }
                ),
            )
        )
    return tables


class StatisticsError(ValueError):
    """Safe Ukrainian calculation error."""


def _finite(value: Any) -> float | None:
    return float(value) if pd.notna(value) and math.isfinite(float(value)) else None


def calculate(
    frame: pd.DataFrame,
    profile: dict,
    metadata: Metadata,
    view: View,
    fields: list[str] | None = None,
    values_for: str | None = None,
    with_pairs: bool = False,
) -> dict[str, Any]:
    """Calculate one view from a snapshot; invalid field rules do not hide other fields.

    ``values_for`` returns the cleaned finite values of one numeric field (stage 04
    histogram) using exactly these missing/infinity rules; nothing is recalculated elsewhere.

    A malformed numeric missing code invalidates that field's additional count and the
    dataset-wide missing union. Known base counts stay visible, never presented as totals.
    """
    if view not in METHODS:
        raise StatisticsError("Невідомий вид розрахунку.")
    if list(frame.columns) != [c.name for c in metadata.document.columns]:
        raise StatisticsError("Стовпці файла не збігаються зі збереженими метаданими.")
    described = {c["name"]: c for c in profile["columns"]}
    rows = len(frame)
    state = "confirmed" if metadata.confirmed_at else "draft" if metadata.saved else "none"
    result: dict[str, Any] = {
        "view": view,
        "method": METHODS[view],
        "row_count": rows,
        "metadata_state": state,
        "metadata_updated_at": metadata.updated_at,
        "warnings": [] if metadata.confirmed_at else ["Метадані не підтверджено"],
        "issues": [vars(issue) for issue in issues(metadata.document)],
        "columns": [],
    }
    numeric: dict[str, pd.Series] = {}
    excluded: dict[str, str] = {}
    union = pd.Series(False, index=frame.index)
    union_valid = True
    for column in metadata.document.columns:
        name = column.name
        series = frame[name]
        actual = described.get(name, {})
        is_number = ptypes.is_numeric_dtype(series) and not ptypes.is_bool_dtype(series)
        base, extra, error = missing_masks(series, column.missing_codes)
        if error:
            union_valid = False
        missing = base | extra
        union |= missing
        invalid = series.isin([math.inf, -math.inf]) & ~missing if is_number else extra & False
        values = series[~missing & ~invalid]
        item: dict[str, Any] = {
            "name": name,
            "label": column.label,
            "unit": column.unit,
            "kind": column.kind,
            "role": column.role,
            "base_missing": int(base.sum()),
            "additional_missing": int(extra.sum()) if error is None else None,
            "missing": int(missing.sum()) if error is None else None,
            "missing_percent": 100 * int(missing.sum()) / rows if rows and error is None else None,
            "invalid_infinite": int(invalid.sum()),
            "error": error,
            "statistics": None,
            "reason": None,
        }
        reason = error
        if reason is None and (column.role in ("id", "ignore") or column.kind == "id"):
            reason = "Поле виключено за типом або роллю."
        if reason is None and column.kind == "number" and not is_number:
            reason = "Описано як число, але фактичні значення не є числовими."
        if reason is None and column.kind == "date":
            fmt = actual.get("date_format")
            if actual.get("kind") != "date" or not fmt:
                reason = "У профілі немає розпізнаного формату дати."
            else:
                dates = pd.to_datetime(
                    values.astype("string").str.strip(), format=fmt, errors="coerce"
                )
                if dates.isna().any():
                    reason = "Значення не відповідають розпізнаному формату дати."
                else:
                    item["date_format"] = fmt
                    item["date_ambiguous"] = bool(actual.get("date_ambiguous"))
                    if item["date_ambiguous"]:
                        item["warning"] = "Неоднозначні дати: використано формат " + fmt
                    item["statistics"] = {
                        "count": len(dates),
                        "unique": int(dates.nunique()),
                        "first": _date_text(dates.min(), fmt) if len(dates) else None,
                        "last": _date_text(dates.max(), fmt) if len(dates) else None,
                    }
        if reason is None and column.kind == "number":
            numeric[name] = series.where(~missing & ~invalid)
            item["statistics"] = {
                "count": len(values),
                "mean": _finite(values.mean()),
                "std": _finite(values.std(ddof=1)) if len(values) >= 2 else None,
                "min": _finite(values.min()),
                "q1": _finite(values.quantile(0.25)),
                "median": _finite(values.quantile(0.5)),
                "q3": _finite(values.quantile(0.75)),
                "max": _finite(values.max()),
                "std_reason": "Потрібно щонайменше 2 значення." if len(values) < 2 else None,
                "empty_reason": "Немає придатних значень." if not len(values) else None,
            }
            item["statistics"]["reasons"] = {
                key: (
                    "Немає придатних значень."
                    if not len(values)
                    else "Потрібно щонайменше 2 значення."
                    if key == "std" and len(values) < 2
                    else "Результат виходить за межі числової точності."
                )
                for key in ("mean", "std", "min", "q1", "median", "q3", "max")
                if item["statistics"][key] is None
            }
        elif reason is None and column.kind in ("category", "text", "bool"):
            counts = values.astype(str).value_counts()
            top = (
                sorted(str(v) for v in counts[counts == counts.max()].index) if len(counts) else []
            )
            item["statistics"] = {
                "count": len(values),
                "unique": len(counts),
                "top": top[0] if top else None,
                "frequency": int(counts.max()) if len(counts) else None,
                "tied": len(top) > 1,
                "empty_reason": "Немає придатних значень." if not len(values) else None,
            }
        item["reason"] = reason
        if name not in numeric:
            excluded[name] = reason or "До Pearson входять лише числові поля."
        if view == "missing":
            item.pop("statistics")
        result["columns"].append(item)
    if values_for is not None:
        if values_for not in numeric:
            known = {c.name for c in metadata.document.columns}
            raise StatisticsError(
                excluded.get(values_for, "Поле не знайдено в наборі.")
                if values_for in known
                else "Поле не знайдено в наборі."
            )
        result["values"] = numeric[values_for].dropna().astype(float).tolist()
    if view == "missing":
        result["rows_with_missing"] = int(union.sum()) if union_valid else None
        result["rows_with_missing_percent"] = (
            100 * int(union.sum()) / rows if rows and union_valid else None
        )
        result["reason"] = (
            None if union_valid else "Виправте помилкові коди пропуску для підсумку набору."
        )
    if view == "correlations":
        selected = list(numeric) if fields is None else fields
        if len(set(selected)) != len(selected) or any(name not in numeric for name in selected):
            raise StatisticsError("Оберіть різні поля серед придатних числових.")
        result.update(fields=selected, excluded=excluded, r=[], n=[], reasons=[])
        result["reason"] = (
            "Оберіть щонайменше два придатні числові поля." if len(selected) < 2 else None
        )
        if len(selected) >= 2:
            for left in selected:
                r_row, n_row, reasons = [], [], []
                for right in selected:
                    pairs = pd.concat([numeric[left], numeric[right]], axis=1).dropna()
                    count = len(pairs)
                    reason = "Потрібно щонайменше 3 спільні значення." if count < 3 else None
                    if reason is None and any(pairs.iloc[:, i].nunique() < 2 for i in (0, 1)):
                        reason = "Одне з полів має нульову варіацію."
                    value = None if reason else _finite(pairs.iloc[:, 0].corr(pairs.iloc[:, 1]))
                    if value is None and reason is None:
                        reason = "Числова точність не дозволяє обчислити кореляцію."
                    r_row.append(value)
                    n_row.append(count)
                    reasons.append(reason)
                result["r"].append(r_row)
                result["n"].append(n_row)
                result["reasons"].append(reasons)
        if with_pairs and len(selected) == 2:
            # Scatter (stage 04): exactly the pairs Pearson used, no sampling or trimming.
            pairs = pd.concat([numeric[selected[0]], numeric[selected[1]]], axis=1).dropna()
            result["pairs"] = pairs.astype(float).values.tolist()
    return result


class StatisticsStore:
    """Load a fresh file/metadata snapshot and record exactly one event per request."""

    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets
        self.metadata = MetadataStore(datasets.database, datasets)

    def calculate(
        self,
        project_id: str,
        dataset_id: str,
        view: View,
        fields: list[str] | None = None,
        *,
        values_for: str | None = None,
        with_pairs: bool = False,
        operation: str = "statistics.calculated",
        audit_extra: dict[str, Any] | None = None,
        finish: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """``finish`` derives chart data inside the audited block: its errors log as errors."""
        started = perf_counter()
        ProjectStore(self.datasets.database).get(project_id)
        audit: dict[str, Any] = {
            "dataset_id": dataset_id,
            "view": view,
            "fields": fields,
            "method": METHODS.get(view),
            "sha256": None,
            "metadata_state": None,
            "metadata_updated_at": None,
            "row_count": None,
        }
        if audit_extra:
            audit.update(audit_extra)
        try:
            dataset = self.datasets.get(dataset_id)
            if dataset.project_id != project_id:
                raise StatisticsError("Набір не належить поточному проєкту.")
            audit["dataset_name"] = dataset.name
            path = (self.datasets.files_root / dataset.stored_path).resolve()
            if not path.is_relative_to(self.datasets.files_root) or not path.is_file():
                raise StatisticsError("Файл набору даних відсутній у сховищі.")
            content = path.read_bytes()
            audit["sha256"] = hashlib.sha256(content).hexdigest()
            if audit["sha256"] != dataset.sha256:
                raise StatisticsError(
                    "Файл набору змінився або пошкоджений. Завантажте його заново."
                )
            frame = self.datasets.frame(dataset_id)
            metadata = self.metadata.get(dataset_id)
            audit.update(
                row_count=len(frame),
                metadata_state="confirmed"
                if metadata.confirmed_at
                else "draft"
                if metadata.saved
                else "none",
                metadata_updated_at=metadata.updated_at,
            )
            identity = {
                "version": CACHE_VERSION,
                "pandas": pd.__version__,
                "method": METHODS.get(view),
                "sha256": dataset.sha256,
                "profile": dataset.profile,
                "metadata": metadata.document.model_dump(mode="json"),
                "saved": metadata.saved,
                "updated_at": metadata.updated_at,
                "confirmed_at": metadata.confirmed_at,
                "view": view,
                "fields": fields,
                "values_for": values_for,
                "with_pairs": with_pairs,
            }
            source = json.dumps(identity, sort_keys=True, ensure_ascii=False, allow_nan=False)
            object_id = dataset_id + "." + hashlib.sha256(source.encode()).hexdigest()
            analytics = self.datasets.analytics
            result = analytics.result(project_id, "statistics", object_id, source)
            audit["reused"] = result is not None
            if result is None:
                result = calculate(
                    frame, dataset.profile, metadata, view, fields, values_for, with_pairs
                )
                result.update(project_id=project_id, dataset_id=dataset_id, sha256=audit["sha256"])
                analytics.save_result(
                    project_id, "statistics", object_id, source, result, result_tables(result)
                )
            audit.update(
                {key: result[key] for key in ("metadata_state", "metadata_updated_at", "row_count")}
            )
            audit["fields"] = (
                [values_for]
                if values_for is not None
                else result.get("fields", [c["name"] for c in result["columns"]])
            )
            if operation == "chart.built" and "n" in result:
                audit["pair_counts"] = result["n"]
            if "values" in result:
                audit["value_count"] = len(result["values"])
            if finish is not None:
                # Style/title never change the numeric cache. Only bin parameters do.
                chart_source = source + json.dumps(
                    {k: audit.get(k) for k in ("chart", "bins", "bin_width")}, sort_keys=True
                )
                chart_id = dataset_id + "." + hashlib.sha256(chart_source.encode()).hexdigest()
                cached_chart = analytics.result(project_id, "chart_data", chart_id, chart_source)
                if cached_chart is None:
                    details = finish(result)
                    analytics.save_result(
                        project_id,
                        "chart_data",
                        chart_id,
                        chart_source,
                        {"result": result, "audit": details},
                        result_tables(result),
                    )
                else:
                    result = cached_chart["result"]
                    details = cached_chart["audit"]
                audit.update(details)
            audit["status"] = "success"
            return result
        except (StatisticsError, DatasetError, LookupError, OSError, ValueError) as error:
            message = (
                str(error)
                if isinstance(error, (StatisticsError, DatasetError, LookupError))
                else "Не вдалося прочитати файл або метадані набору."
            )
            audit.update(status="error", error=message)
            raise StatisticsError(message) from error
        finally:
            audit["duration_ms"] = (perf_counter() - started) * 1000
            with connect(self.datasets.database) as connection:
                ProjectStore._event(connection, project_id, operation, audit)
