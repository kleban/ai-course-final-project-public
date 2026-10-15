"""Data dictionary of a dataset (stage 02b): structured field descriptions plus raw author notes.

Stored as one versioned JSON document per dataset in SQLite. Facts measured in the data
(`observed`) are always recomputed from the file and never taken from user input or an LLM;
descriptive fields come from the author now and from an LLM draft later (stage 08).
"""

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .datasets import DatasetStore, clean_name
from .projects import ProjectStore
from .storage import connect, ensure_schema

CATEGORY_LIMIT = 20
SOURCE_BYTES = 2 * 1024 * 1024
SOURCE_TEXT_LIMIT = 50_000
SOURCE_FORMATS = {".txt", ".md", ".csv", ".json", ".xlsx"}

Kind = Literal["number", "category", "text", "date", "bool", "id"]
Role = Literal["feature", "target", "id", "ignore"]
KIND_TITLES = {
    "number": "число",
    "category": "категорія",
    "text": "текст",
    "date": "дата",
    "bool": "так/ні",
    "id": "ідентифікатор",
}
ROLE_TITLES = {
    "feature": "ознака",
    "target": "цільова",
    "id": "ідентифікатор",
    "ignore": "не використовувати",
}


class MetadataError(ValueError):
    """Invalid metadata or description source; the message is safe to show."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CategoryValue(_Model):
    value: str = Field(max_length=200)
    meaning: str = Field(default="", max_length=500)


class Observed(_Model):
    """Facts measured in the file at upload time; read-only."""

    kind: Kind
    missing: int
    unique: int
    min: float | None = None
    max: float | None = None
    values: list[str] = Field(default_factory=list)


class ColumnMeta(_Model):
    name: str
    label: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=2000)
    kind: Kind
    unit: str = Field(default="", max_length=100)
    computation: str = Field(default="", max_length=1000)
    categories: list[CategoryValue] = Field(default_factory=list, max_length=200)
    range_min: float | None = None
    range_max: float | None = None
    missing_codes: list[str] = Field(default_factory=list, max_length=20)
    role: Role = "feature"
    source: Literal["data", "author", "llm"] = "data"
    observed: Observed


class DatasetMeta(_Model):
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=4000)
    source: str = Field(default="", max_length=1000)
    unit_of_observation: str = Field(default="", max_length=300)
    period: str = Field(default="", max_length=300)
    license: str = Field(default="", max_length=500)


class MetadataDocument(_Model):
    version: Literal[1] = 1
    dataset: DatasetMeta = Field(default_factory=DatasetMeta)
    columns: list[ColumnMeta]


@dataclass(frozen=True)
class Issue:
    column: str
    level: Literal["warning", "info"]
    message: str


@dataclass(frozen=True)
class Metadata:
    dataset_id: str
    document: MetadataDocument
    saved: bool
    updated_at: str | None
    confirmed_at: str | None


@dataclass(frozen=True)
class MetadataSource:
    id: str
    dataset_id: str
    kind: str
    name: str
    stored_path: str | None
    text: str
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ skeleton from data


def _observed(frame: pd.DataFrame, column: dict) -> Observed:
    series = frame[column["name"]]
    kinds: dict[str, Kind] = {"number": "number", "date": "date", "bool": "bool"}
    kind: Kind = kinds.get(column["kind"], "text")
    unique = int(column["unique"])
    rows = int(series.notna().sum())
    if kind == "text" and 0 < unique <= CATEGORY_LIMIT and unique < max(rows, 1):
        kind = "category"
    elif kind == "text" and rows > CATEGORY_LIMIT and unique == rows:
        kind = "id"
    values: list[str] = []
    if kind in ("category", "bool"):
        counts = series.dropna().astype(str).value_counts()
        values = [str(v) for v in counts.index[:CATEGORY_LIMIT]]
    return Observed(
        kind=kind,
        missing=int(column["missing"]),
        unique=unique,
        min=column.get("min"),
        max=column.get("max"),
        values=values,
    )


def skeleton(frame: pd.DataFrame, profile: dict) -> MetadataDocument:
    columns = []
    for column in profile["columns"]:
        observed = _observed(frame, column)
        columns.append(
            ColumnMeta(
                name=column["name"],
                kind=observed.kind,
                categories=[CategoryValue(value=v) for v in observed.values]
                if observed.kind == "category"
                else [],
                role="id" if observed.kind == "id" else "feature",
                observed=observed,
            )
        )
    return MetadataDocument(columns=columns)


# ------------------------------------------------------------------ checks


def issues(document: MetadataDocument) -> list[Issue]:
    found: list[Issue] = []
    targets = [c.name for c in document.columns if c.role == "target"]
    if len(targets) > 1:
        found.append(Issue("", "warning", f"Кілька цільових змінних: {', '.join(targets)}."))
    for column in document.columns:
        seen = column.observed
        if column.kind == "number" and seen.kind != "number":
            found.append(
                Issue(column.name, "warning", "Описано як число, але в даних не лише числа.")
            )
        if column.role == "target" and seen.kind != "number":
            found.append(
                Issue(column.name, "warning", "Для регресії цільова змінна має бути числовою.")
            )
        if column.range_min is not None and column.range_max is not None:
            if column.range_min > column.range_max:
                found.append(
                    Issue(column.name, "warning", "Мінімум діапазону більший за максимум.")
                )
        if column.range_min is not None and seen.min is not None and seen.min < column.range_min:
            found.append(
                Issue(
                    column.name,
                    "warning",
                    f"У даних є значення {seen.min:g} < {column.range_min:g}.",
                )
            )
        if column.range_max is not None and seen.max is not None and seen.max > column.range_max:
            found.append(
                Issue(
                    column.name,
                    "warning",
                    f"У даних є значення {seen.max:g} > {column.range_max:g}.",
                )
            )
        if column.kind == "category" and column.categories and seen.values:
            described = {c.value for c in column.categories}
            unknown = [v for v in seen.values if v not in described]
            if unknown:
                found.append(
                    Issue(column.name, "warning", "Немає в описі: " + ", ".join(unknown[:10]) + ".")
                )
            absent = [c.value for c in column.categories if c.value not in seen.values]
            if absent and seen.unique <= CATEGORY_LIMIT:
                found.append(
                    Issue(
                        column.name,
                        "info",
                        "Описано, але немає в даних: " + ", ".join(absent[:10]) + ".",
                    )
                )
        codes = [c for c in column.missing_codes if c in seen.values]
        if codes:
            found.append(
                Issue(
                    column.name,
                    "info",
                    "Коди пропуску трапляються в даних: " + ", ".join(codes) + ".",
                )
            )
    return found


# ------------------------------------------------------------------ store


class MetadataStore:
    def __init__(self, database: Path, datasets: DatasetStore | None = None):
        self.database = ensure_schema(database)
        self.datasets = datasets or DatasetStore(self.database)

    def _frame(self, dataset_id: str):
        dataset = self.datasets.get(dataset_id)
        frame = self.datasets.frame(dataset_id)
        return dataset, frame

    def _fresh(self, dataset_id: str) -> MetadataDocument:
        dataset, frame = self._frame(dataset_id)
        return skeleton(frame, dataset.profile)

    def get(self, dataset_id: str) -> Metadata:
        self.datasets.get(dataset_id)
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT * FROM dataset_metadata WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
        if row is None:
            return Metadata(dataset_id, self._fresh(dataset_id), False, None, None)
        document = MetadataDocument.model_validate_json(row["document"])
        return Metadata(dataset_id, document, True, row["updated_at"], row["confirmed_at"])

    def save(
        self,
        dataset_id: str,
        raw: object,
        *,
        confirm: bool = False,
        expected_version: tuple[bool, str | None] | None = None,
        expected_sources: tuple[str, ...] | None = None,
    ) -> Metadata:
        fresh = self._fresh(dataset_id)
        try:
            document = MetadataDocument.model_validate(raw)
        except ValidationError as error:
            first = error.errors()[0]
            place = ".".join(str(part) for part in first["loc"])
            raise MetadataError(f"Метадані: {place} — {first['msg']}.") from error
        expected = [c.name for c in fresh.columns]
        if [c.name for c in document.columns] != expected:
            raise MetadataError(
                "Метадані: стовпці мають збігатися з набором даних і його порядком."
            )
        observed = {c.name: c.observed for c in fresh.columns}
        columns = []
        for column in document.columns:
            source = column.source
            original = next(c for c in fresh.columns if c.name == column.name)
            descriptive = column.model_dump(exclude={"observed", "source", "name"})
            if descriptive != original.model_dump(exclude={"observed", "source", "name"}):
                source = "author" if source == "data" else source
            columns.append(
                column.model_copy(update={"observed": observed[column.name], "source": source})
            )
        document = document.model_copy(update={"columns": columns})
        dataset = self.datasets.get(dataset_id)
        now = _now()
        with connect(self.database) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if expected_sources is not None:
                sources = connection.execute(
                    "SELECT id FROM metadata_sources WHERE dataset_id=? ORDER BY created_at, id",
                    (dataset_id,),
                ).fetchall()
                if tuple(row["id"] for row in sources) != expected_sources:
                    raise MetadataError(
                        "Джерела змінилися після створення чернетки. Створіть нову чернетку."
                    )
            previous = connection.execute(
                "SELECT confirmed_at, updated_at FROM dataset_metadata WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if expected_version is not None and expected_version != (
                previous is not None,
                previous["updated_at"] if previous else None,
            ):
                raise MetadataError(
                    "Метадані змінилися після створення чернетки. Створіть нову чернетку."
                )
            confirmed_at = now if confirm else None
            connection.execute(
                "INSERT INTO dataset_metadata VALUES (?, ?, ?, ?) ON CONFLICT(dataset_id) DO UPDATE "
                "SET document = excluded.document, updated_at = excluded.updated_at, "
                "confirmed_at = excluded.confirmed_at",
                (dataset_id, document.model_dump_json(), now, confirmed_at),
            )
            ProjectStore._event(
                connection,
                dataset.project_id,
                "metadata.confirmed" if confirm else "metadata.saved",
                {
                    "dataset_id": dataset_id,
                    "name": dataset.name,
                    "was_confirmed": "так" if previous and previous["confirmed_at"] else "ні",
                    **({"origin": "ai_draft"} if expected_version is not None else {}),
                },
            )
        return Metadata(dataset_id, document, True, now, confirmed_at)

    # --- raw descriptions attached by the author

    def add_text(self, dataset_id: str, text: object) -> MetadataSource:
        if not isinstance(text, str) or not text.strip():
            raise MetadataError("Опис: вставте текст.")
        if len(text) > SOURCE_TEXT_LIMIT:
            raise MetadataError(f"Опис: максимум {SOURCE_TEXT_LIMIT} символів.")
        return self._add_source(dataset_id, "text", "Вставлений текст", None, None, text.strip())

    def add_file(self, dataset_id: str, filename: object, content: bytes) -> MetadataSource:
        name = clean_name(filename)
        suffix = PurePath(name).suffix.lower()
        if suffix not in SOURCE_FORMATS:
            raise MetadataError("Файл опису: підтримуються TXT, MD, CSV, JSON і XLSX.")
        if not content:
            raise MetadataError("Файл опису порожній.")
        if len(content) > SOURCE_BYTES:
            raise MetadataError("Файл опису: максимум 2 МБ.")
        text = extract_text(suffix, content)
        return self._add_source(dataset_id, "file", name, suffix, content, text)

    def _add_source(
        self,
        dataset_id: str,
        kind: str,
        name: str,
        suffix: str | None,
        content: bytes | None,
        text: str,
    ) -> MetadataSource:
        dataset = self.datasets.get(dataset_id)
        source_id = str(uuid4())
        stored_path = None
        target = None
        if content is not None and suffix:
            relative = Path(dataset.stored_path).parent / "metadata" / f"{source_id}{suffix}"
            target = self.datasets.files_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            stored_path = relative.as_posix()
        source = MetadataSource(
            source_id, dataset_id, kind, name, stored_path, text[:SOURCE_TEXT_LIMIT], _now()
        )
        try:
            with connect(self.database) as connection:
                connection.execute(
                    "INSERT INTO metadata_sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        source.id,
                        dataset_id,
                        kind,
                        name,
                        stored_path,
                        source.text,
                        source.created_at,
                    ),
                )
                ProjectStore._event(
                    connection,
                    dataset.project_id,
                    "metadata.source_added",
                    {"dataset_id": dataset_id, "name": dataset.name, "source": name},
                )
        except BaseException:
            if target is not None:
                target.unlink(missing_ok=True)
            raise
        return source

    def sources(self, dataset_id: str) -> list[MetadataSource]:
        self.datasets.get(dataset_id)
        with connect(self.database) as connection:
            rows = connection.execute(
                "SELECT * FROM metadata_sources WHERE dataset_id = ? ORDER BY created_at, id",
                (dataset_id,),
            ).fetchall()
        return [MetadataSource(**dict(row)) for row in rows]


def extract_text(suffix: str, content: bytes) -> str:
    """Plain text of a description file, for showing it and later for the LLM."""
    if suffix == ".xlsx":
        try:
            book = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
            parts = []
            for sheet in book.sheet_names:
                frame = book.parse(sheet, header=None, dtype=str).fillna("")
                parts.append(f"# {sheet}\n" + frame.to_csv(sep="\t", index=False, header=False))
            return "\n".join(parts).strip()
        except Exception as error:
            raise MetadataError("Файл опису XLSX пошкоджений.") from error
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise MetadataError("Файл опису: очікується текст UTF-8 або Windows-1251.")
    if suffix == ".json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except ValueError as error:
            raise MetadataError("Файл опису JSON має помилку синтаксису.") from error
    if not text.strip():
        raise MetadataError("Файл опису порожній.")
    return text.strip()
