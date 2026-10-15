"""Uploaded datasets: validation, safe storage of the original file, column profile and preview.

No UI or AI here. pandas reads the file once at upload time to reject broken files early and to
compute a profile; the original bytes are kept unchanged so later stages can derive new versions.
"""

import csv
import hashlib
import io
import json
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any
from uuid import uuid4

import pandas as pd
from pandas.api import types as ptypes

from .analytics import AnalyticsStore
from .projects import ProjectStore
from .storage import connect, ensure_schema

MAX_BYTES = 5 * 1024 * 1024
FORMATS = {".csv": "csv"}
PREVIEW_ROWS = 20
NAME_LIMIT = 200
ENCODINGS = ("utf-8-sig", "cp1251")
DELIMITERS = ",;\t|"
KIND_TITLES = {"number": "число", "text": "текст", "date": "дата", "bool": "так/ні"}
# Explicit date formats, tried in order; a text column becomes a date only when every filled
# value matches one format. Day-first comes before month-first (Ukrainian convention).
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%d.%m.%Y",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%Y.%m.%d",
)
DATE_FORMAT_TITLES = {"%Y": "РРРР", "%m": "ММ", "%d": "ДД", "%H": "гг", "%M": "хх", "%S": "сс"}


class DatasetError(ValueError):
    """The uploaded file cannot become a dataset; the message is safe to show to the user."""


class DatasetNotFound(LookupError):
    """A requested dataset does not exist."""


@dataclass(frozen=True)
class Dataset:
    id: str
    project_id: str
    name: str
    file_format: str
    stored_path: str
    size_bytes: int
    sha256: str
    row_count: int
    column_count: int
    profile: dict
    created_at: str
    copied_from: str | None = None


@dataclass(frozen=True)
class Preview:
    columns: list[str]
    rows: list[list[str | None]]
    total_rows: int


@dataclass(frozen=True)
class Page:
    """One page of rows for the grid after server-side filtering and sorting.
    Keys are c0, c1, … by column position; numbers stay numbers, dates are ISO text."""

    rows: list[dict[str, Any]]
    total: int
    all_rows: int


PAGE_LIMIT = 500
TYPED_CACHE_SIZE = 4
# dataset id -> (typed frame, kinds, date formats); a few recent sets stay parsed in memory
_typed_cache: OrderedDict[str, tuple[pd.DataFrame, list[str], list[str | None]]] = OrderedDict()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_name(filename: object) -> str:
    """Display name only — never used as a path."""
    if not isinstance(filename, str):
        raise DatasetError("Файл: не вдалося визначити назву файла.")
    name = PurePath(filename.replace("\\", "/")).name.strip()
    if not name:
        raise DatasetError("Файл: не вдалося визначити назву файла.")
    return name[:NAME_LIMIT]


def file_format(name: str) -> str:
    suffix = PurePath(name).suffix.lower()
    if suffix not in FORMATS:
        raise DatasetError("Формат: підтримуються лише файли CSV.")
    return FORMATS[suffix]


# ------------------------------------------------------------------ reading


def _decode(content: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise DatasetError("CSV: не вдалося прочитати текст (очікується UTF-8 або Windows-1251).")


def _delimiter(text: str) -> str:
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=DELIMITERS).delimiter
    except csv.Error:
        first = sample.splitlines()[0] if sample else ""
        counts = {d: first.count(d) for d in DELIMITERS}
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] else ","


def read_table(
    content: bytes, fmt: str, *, meta: dict | None = None, nrows: int | None = None
) -> tuple[pd.DataFrame, dict]:
    """Parse bytes into a DataFrame. `meta` (encoding, delimiter) repeats an earlier detection."""
    meta = dict(meta or {})
    if fmt != "csv":
        raise DatasetError("Формат: підтримуються лише файли CSV.")
    try:
        if "encoding" in meta:
            text = content.decode(meta["encoding"])
        else:
            text, meta["encoding"] = _decode(content)
        if not text.strip():
            raise DatasetError("Файл порожній.")
        meta.setdefault("delimiter", _delimiter(text))
        options = {}
        if "prepared_csv" in meta:
            options = {
                "dtype": meta["prepared_csv"]["dtypes"],
                "keep_default_na": False,
                "na_values": [meta["prepared_csv"]["na_token"]],
                "float_precision": "round_trip",
            }
        frame = pd.read_csv(io.StringIO(text), sep=meta["delimiter"], nrows=nrows, **options)
    except DatasetError:
        raise
    except pd.errors.EmptyDataError as error:
        raise DatasetError("Файл порожній: немає заголовків і даних.") from error
    except pd.errors.ParserError as error:
        raise DatasetError(f"CSV: неправильна структура рядків ({error}).") from error
    except Exception as error:  # unexpected bytes or structure
        raise DatasetError("Файл пошкоджений або має непідтримувану структуру.") from error
    frame.columns = [str(c) for c in frame.columns]
    if frame.shape[1] == 0:
        raise DatasetError("Файл не містить стовпців.")
    return frame, meta


# ------------------------------------------------------------------ profile


def _kind(series: pd.Series) -> str:
    if ptypes.is_bool_dtype(series):
        return "bool"
    if ptypes.is_numeric_dtype(series):
        return "number"
    if ptypes.is_datetime64_any_dtype(series):
        return "date"
    return "text"


def date_format_title(fmt: str) -> str:
    """'%d/%m/%Y' -> 'ДД/ММ/РРРР' for the interface."""
    for code, title in DATE_FORMAT_TITLES.items():
        fmt = fmt.replace(code, title)
    return fmt


def detect_dates(series: pd.Series) -> tuple[str, pd.Series, bool] | None:
    """Recognise a text column as dates without AI: one explicit format must fit all values.

    Returns the format, the parsed filled values and whether another format also fits
    (for example 03/04/2020 is valid both day-first and month-first).
    """
    if not ptypes.is_string_dtype(series) and not ptypes.is_object_dtype(series):
        return None
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return None
    unique = pd.Series(values.unique())
    matches = [
        fmt
        for fmt in DATE_FORMATS
        if pd.to_datetime(unique, format=fmt, errors="coerce").notna().all()
    ]
    if not matches:
        return None
    parsed = pd.to_datetime(values, format=matches[0])
    return matches[0], parsed, len(matches) > 1


def _date_text(value: pd.Timestamp, fmt: str) -> str:
    """ISO text with the same precision as the source format."""
    if "%S" in fmt:
        return value.isoformat(timespec="seconds")
    if "%H" in fmt:
        return value.isoformat(timespec="minutes")
    return value.date().isoformat()


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 6)


def profile(frame: pd.DataFrame) -> list[dict]:
    columns = []
    for name in frame.columns:
        series = frame[name]
        kind = _kind(series)
        item: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
            "min": None,
            "max": None,
            "mean": None,
        }
        if kind == "number" and series.notna().any():
            item.update(
                min=_number(series.min()), max=_number(series.max()), mean=_number(series.mean())
            )
        dates = detect_dates(series) if kind == "text" else None
        if dates is not None:
            fmt, parsed, ambiguous = dates
            item.update(
                kind="date",
                date_format=fmt,
                first=_date_text(parsed.min(), fmt),
                last=_date_text(parsed.max(), fmt),
            )
            if ambiguous:
                item["date_ambiguous"] = True
        columns.append(item)
    return columns


def _typed_number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return int(number) if number.is_integer() and abs(number) < 2**53 else number


def _cell(value: Any) -> str | None:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


# ------------------------------------------------------------------ store


class DatasetStore:
    def __init__(self, database: Path, files_root: Path | None = None):
        self.database = ensure_schema(database)
        self.files_root = Path(files_root or self.database.parent / "projects").resolve()
        self.analytics = AnalyticsStore(self.database.parent / "analytics.duckdb")

    @staticmethod
    def source(dataset: Dataset) -> str:
        return json.dumps(
            {"parser_version": 1, "sha256": dataset.sha256, "profile": dataset.profile},
            sort_keys=True,
            ensure_ascii=False,
        )

    def persist(self, dataset: Dataset, frame: pd.DataFrame) -> dict:
        return self.analytics.put(
            dataset.project_id, "dataset", dataset.id, frame, source=self.source(dataset)
        )

    def verified_content(self, dataset: Dataset) -> bytes:
        path = (self.files_root / dataset.stored_path).resolve()
        if not path.is_relative_to(self.files_root) or not path.is_file():
            raise DatasetNotFound("Файл набору даних відсутній у сховищі.")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != dataset.sha256:
            raise DatasetError("Файл набору змінився або пошкоджений. Завантажте його заново.")
        return content

    def frame(self, dataset_id: str) -> pd.DataFrame:
        """Read the persistent raw table, checking the original even on a cache hit."""
        dataset = self.get(dataset_id)
        content = self.verified_content(dataset)
        fingerprint = self.analytics.find(
            dataset.project_id, "dataset", dataset.id, self.source(dataset)
        )
        if fingerprint is None:
            frame, _ = read_table(content, dataset.file_format, meta=dataset.profile)
            fingerprint = self.persist(dataset, frame)["fingerprint"]
        return self.analytics.read(dataset.project_id, "dataset", dataset.id, fingerprint)

    def add(
        self, project_id: str, filename: object, content: bytes, *, parser: dict | None = None
    ) -> Dataset:
        name = clean_name(filename)
        fmt = file_format(name)
        if not content:
            raise DatasetError("Файл порожній.")
        if len(content) > MAX_BYTES:
            raise DatasetError(f"Розмір: максимум {MAX_BYTES // (1024 * 1024)} МБ.")
        with connect(self.database) as connection:  # fail fast for an unknown project
            ProjectStore._get(connection, project_id)
        frame, meta = read_table(content, fmt, meta=parser)
        if parser is not None:
            meta = {**parser, **meta}

        dataset_id = str(uuid4())
        relative = Path(project_id) / "datasets" / dataset_id / f"original.{fmt}"
        target = self.files_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        partial.write_bytes(content)
        os.replace(partial, target)

        dataset = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=name,
            file_format=fmt,
            stored_path=relative.as_posix(),
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            row_count=int(frame.shape[0]),
            column_count=int(frame.shape[1]),
            profile={**meta, "columns": profile(frame)},
            created_at=_now(),
        )
        reference = None
        try:
            reference = self.persist(dataset, frame)
            with connect(self.database) as connection:
                ProjectStore._get(connection, project_id)
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dataset.id,
                        dataset.project_id,
                        dataset.name,
                        dataset.file_format,
                        dataset.stored_path,
                        dataset.size_bytes,
                        dataset.sha256,
                        dataset.row_count,
                        dataset.column_count,
                        json.dumps(dataset.profile, ensure_ascii=False),
                        dataset.created_at,
                        None,
                    ),
                )
                ProjectStore._event(
                    connection,
                    project_id,
                    "dataset.uploaded",
                    {
                        "dataset_id": dataset.id,
                        "name": name,
                        "rows": str(dataset.row_count),
                        "columns": str(dataset.column_count),
                    },
                )
        except BaseException:
            self.analytics.discard(reference)
            shutil.rmtree(target.parent, ignore_errors=True)
            raise
        return dataset

    @staticmethod
    def _row(row) -> Dataset:
        data = dict(row)
        data["profile"] = json.loads(data["profile"])
        return Dataset(**data)

    def list_all(self) -> list[Dataset]:
        """Datasets of every project, newest first (choice of a dataset to copy)."""
        with connect(self.database) as connection:
            rows = connection.execute(
                "SELECT * FROM datasets ORDER BY created_at DESC, id"
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_datasets(self, project_id: str) -> list[Dataset]:
        with connect(self.database) as connection:
            ProjectStore._get(connection, project_id)
            rows = connection.execute(
                "SELECT * FROM datasets WHERE project_id = ? ORDER BY created_at, id", (project_id,)
            ).fetchall()
        return [self._row(row) for row in rows]

    def get(self, dataset_id: str) -> Dataset:
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            raise DatasetNotFound("Набір даних не знайдено. Оновіть список і виберіть інший.")
        return self._row(row)

    def preview(self, dataset_id: str, rows: int = PREVIEW_ROWS) -> Preview:
        dataset = self.get(dataset_id)
        frame = self.frame(dataset_id).head(rows)
        values = [[_cell(v) for v in record] for record in frame.itertuples(index=False)]
        return Preview(list(frame.columns), values, dataset.row_count)

    def _typed(self, dataset_id: str) -> tuple[pd.DataFrame, list[str], list[str | None]]:
        """The whole file parsed once with types from the profile; reused for every page."""
        dataset = self.get(dataset_id)
        self.verified_content(dataset)
        key = f"{self.database.resolve()}:{dataset.id}:{dataset.sha256}"
        if key in _typed_cache:
            _typed_cache.move_to_end(key)
            return _typed_cache[key]
        frame = self.frame(dataset_id)
        described = {c["name"]: c for c in dataset.profile.get("columns", [])}
        typed: dict[str, pd.Series] = {}
        kinds: list[str] = []
        formats: list[str | None] = []
        for index, name in enumerate(frame.columns):
            item = described.get(name, {})
            kind, fmt = item.get("kind", "text"), item.get("date_format")
            series = frame[name]
            if kind == "date" and fmt:
                series = pd.to_datetime(
                    series.astype("string").str.strip(), format=fmt, errors="coerce"
                )
            elif kind == "number":
                series = pd.to_numeric(series, errors="coerce")
            else:
                kind, fmt = "text", None
                series = series.astype("string")
            typed[f"c{index}"] = series
            kinds.append(kind)
            formats.append(fmt)
        result = (pd.DataFrame(typed), kinds, formats)
        _typed_cache[key] = result
        while len(_typed_cache) > TYPED_CACHE_SIZE:
            _typed_cache.popitem(last=False)
        return result

    def page(
        self,
        dataset_id: str,
        start: int = 0,
        limit: int = 100,
        sort: list[dict[str, str]] | None = None,
        filters: dict[str, dict[str, Any]] | None = None,
    ) -> Page:
        """Only the requested rows leave the server; filtering and sorting happen here."""
        if start < 0 or not 1 <= limit <= PAGE_LIMIT:
            raise DatasetError(f"Сторінка: від 1 до {PAGE_LIMIT} рядків, початок не менше 0.")
        frame, kinds, formats = self._typed(dataset_id)
        mask = pd.Series(True, index=frame.index)
        for column, model in (filters or {}).items():
            index = _column_index(column, len(kinds))
            mask &= _filter_mask(frame[column], kinds[index], model)
        selected = frame[mask]
        if sort:
            columns, ascending = [], []
            for item in sort:
                _column_index(item.get("colId", ""), len(kinds))
                if item.get("sort") not in ("asc", "desc"):
                    raise DatasetError("Сортування: очікується asc або desc.")
                columns.append(item["colId"])
                ascending.append(item["sort"] == "asc")
            selected = selected.sort_values(
                columns, ascending=ascending, na_position="last", kind="stable"
            )
        chunk = selected.iloc[start : start + limit]
        rows = []
        for record in chunk.itertuples(index=False, name=None):
            row: dict[str, Any] = {}
            for index, value in enumerate(record):
                if kinds[index] == "date":
                    fmt = formats[index] or "%Y-%m-%d"
                    row[f"c{index}"] = None if pd.isna(value) else _date_text(value, fmt)
                elif kinds[index] == "number":
                    row[f"c{index}"] = _typed_number(value)
                else:
                    row[f"c{index}"] = None if pd.isna(value) else str(value)
            rows.append(row)
        return Page(rows, int(mask.sum()), int(len(frame)))


def _column_index(column: str, count: int) -> int:
    if not column.startswith("c") or not column[1:].isdigit() or int(column[1:]) >= count:
        raise DatasetError(f"Невідомий стовпець: {column}.")
    return int(column[1:])


def _filter_mask(series: pd.Series, kind: str, model: dict[str, Any]) -> pd.Series:
    """AG Grid filter model -> boolean mask (text, number and date filters, AND/OR)."""
    if not isinstance(model, dict):
        raise DatasetError("Фільтр: неправильний формат.")
    if "conditions" in model:
        parts = [_filter_mask(series, kind, c) for c in model["conditions"]]
        if not parts:
            return pd.Series(True, index=series.index)
        combined = parts[0]
        for part in parts[1:]:
            combined = combined | part if model.get("operator") == "OR" else combined & part
        return combined
    kind_of_filter = {"text": "text", "number": "number", "date": "date"}
    if kind_of_filter.get(model.get("filterType", "")) != kind:
        raise DatasetError("Фільтр не відповідає типу стовпця.")
    how = model.get("type")
    if how == "blank":
        return series.isna()
    if how == "notBlank":
        return series.notna()
    try:
        if kind == "text":
            text = series.str.lower()
            value = str(model.get("filter", "")).lower()
            tests = {
                "contains": lambda: text.str.contains(value, regex=False),
                "notContains": lambda: ~text.str.contains(value, regex=False),
                "equals": lambda: text == value,
                "notEqual": lambda: text != value,
                "startsWith": lambda: text.str.startswith(value),
                "endsWith": lambda: text.str.endswith(value),
            }
        else:
            if kind == "number":
                values = series
                low = float(model["filter"])
                high = float(model.get("filterTo", low))
            else:
                values = series.dt.normalize()
                low = pd.Timestamp(str(model["dateFrom"])[:10])
                high = pd.Timestamp(str(model.get("dateTo") or model["dateFrom"])[:10])
            tests = {
                "equals": lambda: values == low,
                "notEqual": lambda: values != low,
                "lessThan": lambda: values < low,
                "lessThanOrEqual": lambda: values <= low,
                "greaterThan": lambda: values > low,
                "greaterThanOrEqual": lambda: values >= low,
                "inRange": lambda: (values >= low) & (values <= high),
            }
    except (KeyError, TypeError, ValueError) as error:
        raise DatasetError("Фільтр: неправильне значення.") from error
    if how not in tests:
        raise DatasetError(f"Фільтр: невідома умова {how}.")
    return tests[how]().fillna(how in ("notContains", "notEqual")).astype(bool)
