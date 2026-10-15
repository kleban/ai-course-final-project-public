"""Joins and summaries over project datasets with DuckDB, built from a recipe — never from SQL.

The user (or the AI assistant) chooses steps: one optional join, filters, grouping with
aggregates, order and limit. The service validates every field against the dataset metadata,
generates the SQL itself (quoted identifiers, values as parameters), runs it in an in-memory
DuckDB over verified snapshots of the files and can save the result as a new, independent
dataset with its recipe, SQL and source hashes. Free-form SQL is never accepted (D-92).
"""

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ROOT
from .datasets import MAX_BYTES, Dataset, DatasetError, DatasetStore, profile
from .metadata import skeleton
from .missing_values import missing_masks
from .preparation import PreparationError
from .prepared_datasets import csv_snapshot, snapshot
from .projects import ProjectStore
from .storage import connect

PREVIEW_ROWS = 50
MAX_RESULT_ROWS = 1_000_000
Name = Annotated[str, Field(min_length=1, max_length=200)]
Value = str | int | float | bool


class SummaryError(ValueError):
    """The recipe cannot be built or executed safely."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Join(_Model):
    dataset_id: UUID
    left_key: Name
    right_key: Name
    how: Literal["inner", "left"] = "inner"
    columns: list[Name] | None = Field(default=None, max_length=200)


class Filter(_Model):
    field: Name
    op: Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "is_null", "not_null"]
    value: Value | list[Value] | None = None

    @model_validator(mode="after")
    def value_shape(self) -> "Filter":
        if self.op in ("is_null", "not_null"):
            self.value = None
        elif self.op == "in":
            if not isinstance(self.value, list) or not 1 <= len(self.value) <= 100:
                raise ValueError("Для «один із» вкажіть від 1 до 100 значень.")
        elif self.value is None or isinstance(self.value, list):
            raise ValueError("Вкажіть одне значення для умови.")
        return self


class Aggregate(_Model):
    func: Literal["count", "count_distinct", "sum", "avg", "median", "min", "max"]
    field: Name | None = None
    name: Annotated[str, Field(max_length=120)] | None = None

    @model_validator(mode="after")
    def needs_field(self) -> "Aggregate":
        if self.field is None and self.func != "count":
            raise ValueError("Оберіть поле для обчислення.")
        return self


class Recipe(_Model):
    dataset_id: UUID
    join: Join | None = None
    filters: list[Filter] = Field(default_factory=list, max_length=20)
    group_by: list[Name] = Field(default_factory=list, max_length=5)
    aggregates: list[Aggregate] = Field(default_factory=list, max_length=12)
    order_by: Name | None = None
    descending: bool = False
    limit: int | None = Field(default=None, ge=1, le=MAX_RESULT_ROWS)

    @model_validator(mode="after")
    def grouping(self) -> "Recipe":
        if self.group_by and not self.aggregates:
            raise ValueError("Для групування додайте хоча б один показник (наприклад, кількість).")
        return self


FUNC_SQL = {
    "count": "COUNT({})",
    "count_distinct": "COUNT(DISTINCT {})",
    "sum": "SUM({})",
    "avg": "AVG({})",
    "median": "MEDIAN({})",
    "min": "MIN({})",
    "max": "MAX({})",
}
FUNC_TITLES = {
    "count": "Кількість",
    "count_distinct": "Кількість різних",
    "sum": "Сума",
    "avg": "Середнє",
    "median": "Медіана",
    "min": "Мінімум",
    "max": "Максимум",
}
NUMERIC_ONLY = {"sum", "avg", "median"}
ORDERED = {"gt", "ge", "lt", "le"}
OP_SQL = {"eq": "=", "ne": "<>", "gt": ">", "ge": ">=", "lt": "<", "le": "<="}


def quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def default_name(aggregate: Aggregate) -> str:
    return aggregate.name or (
        "count_rows" if aggregate.field is None else f"{aggregate.func}_{aggregate.field}"
    )


def _clean(frame: pd.DataFrame, metadata) -> pd.DataFrame:
    """Explicit missing codes from the metadata become missing values, as in statistics."""
    frame = frame.copy()
    for column in metadata.document.columns:
        if column.missing_codes and column.name in frame.columns:
            base, extra, error = missing_masks(frame[column.name], column.missing_codes)
            if error is None and extra.any():
                frame[column.name] = frame[column.name].mask(extra)
    return frame


class Plan:
    """A validated recipe: SQL text, parameters and the meaning of every output column."""

    def __init__(self, recipe: Recipe, left, right=None) -> None:
        # left/right: (dataset, frame, metadata)
        self.recipe = recipe
        self.sources = [left] + ([right] if right else [])
        kinds = {"a": {c.name: c for c in left[2].document.columns}}
        columns: dict[str, tuple[str, str]] = {name: ("a", name) for name in left[1].columns}
        if recipe.join:
            if right is None:
                raise SummaryError("Набір для з'єднання не знайдено.")
            kinds["b"] = {c.name: c for c in right[2].document.columns}
            join = recipe.join
            if join.left_key not in left[1].columns:
                raise SummaryError(f"Ключа «{join.left_key}» немає в основному наборі.")
            if join.right_key not in right[1].columns:
                raise SummaryError(f"Ключа «{join.right_key}» немає в наборі для з'єднання.")
            left_kind = kinds["a"][join.left_key].kind
            right_kind = kinds["b"][join.right_key].kind
            if (left_kind == "number") != (right_kind == "number"):
                raise SummaryError(
                    "Ключі з'єднання мають бути одного типу: обидва числа або обидва тексти."
                )
            wanted = join.columns if join.columns is not None else list(right[1].columns)
            for name in wanted:
                if name not in right[1].columns:
                    raise SummaryError(f"Поля «{name}» немає в наборі для з'єднання.")
                if name == join.right_key and join.right_key == join.left_key:
                    continue
                output = name if name not in columns else f"{name}_2"
                while output in columns:
                    output += "_2"
                columns[output] = ("b", name)
        self.columns = columns
        self.kinds = kinds
        expressions = {
            name: f"{alias}.{quote(source)}" for name, (alias, source) in columns.items()
        }

        def column(name: str, purpose: str) -> str:
            if name not in expressions:
                raise SummaryError(f"Поля «{name}» немає серед доступних ({purpose}).")
            return expressions[name]

        def kind(name: str) -> str:
            alias, source = columns[name]
            meta = kinds[alias].get(source)
            return meta.kind if meta else "text"

        params: list[Any] = []
        where = []
        for item in recipe.filters:
            expr = column(item.field, "умова")
            if item.op == "is_null":
                where.append(f"{expr} IS NULL")
            elif item.op == "not_null":
                where.append(f"{expr} IS NOT NULL")
            elif item.op == "in":
                values = item.value if isinstance(item.value, list) else [item.value]
                where.append(f"{expr} IN ({', '.join('?' for _ in values)})")
                params.extend(self._typed(v, kind(item.field), item.field) for v in values)
            else:
                if item.op in ORDERED and kind(item.field) not in ("number", "date"):
                    raise SummaryError(
                        f"«Більше/менше» доступне лише для чисел і дат («{item.field}»)."
                    )
                where.append(f"{expr} {OP_SQL[item.op]} ?")
                params.append(self._typed(item.value, kind(item.field), item.field))
        output_names: list[str] = []
        select = []
        self.meaning: dict[str, dict] = {}
        if recipe.aggregates:
            groups = []
            for name in recipe.group_by:
                groups.append(column(name, "групування"))
                select.append(f"{column(name, 'групування')} AS {quote(name)}")
                output_names.append(name)
                self.meaning[name] = {"from": columns[name], "func": None}
            for aggregate in recipe.aggregates:
                target = "*" if aggregate.field is None else column(aggregate.field, "показник")
                if aggregate.func in NUMERIC_ONLY and kind(str(aggregate.field)) != "number":
                    raise SummaryError(
                        f"{FUNC_TITLES[aggregate.func]} рахується лише для числових полів («{aggregate.field}»)."
                    )
                name = default_name(aggregate)
                if not re.fullmatch(r"[\w .,%()\-]{1,120}", name):
                    raise SummaryError("Назва показника: літери, цифри, пробіл і _ . , % ( ) -.")
                if name in output_names:
                    raise SummaryError(f"Назва показника «{name}» повторюється.")
                select.append(f"{FUNC_SQL[aggregate.func].format(target)} AS {quote(name)}")
                output_names.append(name)
                self.meaning[name] = {
                    "from": columns[aggregate.field] if aggregate.field else None,
                    "func": aggregate.func,
                }
        else:
            for name, expr in expressions.items():
                select.append(f"{expr} AS {quote(name)}")
                output_names.append(name)
                self.meaning[name] = {"from": columns[name], "func": None}
        sql = "SELECT " + ",\n       ".join(select) + "\nFROM a"
        if recipe.join:
            how = "INNER JOIN" if recipe.join.how == "inner" else "LEFT JOIN"
            sql += (
                f"\n{how} b ON a.{quote(recipe.join.left_key)} = b.{quote(recipe.join.right_key)}"
            )
        if where:
            sql += "\nWHERE " + "\n  AND ".join(where)
        if recipe.aggregates and recipe.group_by:
            sql += "\nGROUP BY " + ", ".join(groups)
        if recipe.order_by:
            if recipe.order_by not in output_names:
                raise SummaryError("Сортувати можна лише за полем результату.")
            sql += f"\nORDER BY {quote(recipe.order_by)} {'DESC' if recipe.descending else 'ASC'}"
            sql += " NULLS LAST"
        if recipe.limit:
            sql += f"\nLIMIT {int(recipe.limit)}"
        self.sql = sql
        self.params = params
        self.output_names = output_names

    @staticmethod
    def _typed(value: Any, kind: str, field: str) -> Any:
        if kind == "number":
            if isinstance(value, bool):
                raise SummaryError(f"Для числового поля «{field}» вкажіть число.")
            try:
                return float(str(value).replace(",", "."))
            except ValueError as error:
                raise SummaryError(f"Для числового поля «{field}» вкажіть число.") from error
        if kind == "bool":
            return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "так")
        if kind == "date":
            try:
                return pd.Timestamp(str(value)).to_pydatetime()
            except ValueError as error:
                raise SummaryError(
                    f"Для дати «{field}» вкажіть дату, наприклад 2017-03-25."
                ) from error
        return str(value)

    def literal_sql(self) -> str:
        """SQL with values shown inline — for the reader; execution always uses parameters."""
        parts = self.sql.split("?")
        out = parts[0]
        for value, rest in zip(self.params, parts[1:]):
            shown = (
                "'" + str(value).replace("'", "''") + "'"
                if isinstance(value, str | datetime)
                else str(value)
            )
            out += shown + rest
        return out


def run(plan: Plan, limit: int | None = None) -> tuple[pd.DataFrame, int]:
    try:
        import duckdb
    except ImportError as error:
        raise SummaryError("DuckDB не встановлено. Запустіть Setup.cmd.") from error
    temp = ROOT / "tmp" / "duckdb"
    temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(
        ":memory:",
        config={"threads": 2, "memory_limit": "1GB", "temp_directory": str(temp)},
    )
    try:
        connection.register("a", plan.sources[0][1])
        if len(plan.sources) > 1:
            connection.register("b", plan.sources[1][1])
        total = connection.execute(
            f"SELECT COUNT(*) FROM ({plan.sql}) AS q", plan.params
        ).fetchone()
        count = int(total[0]) if total else 0
        if count > MAX_RESULT_ROWS:
            raise SummaryError(
                f"Результат завеликий: {count} рядків (до {MAX_RESULT_ROWS}). Додайте умови або групування."
            )
        query = plan.sql if limit is None else f"SELECT * FROM ({plan.sql}) AS q LIMIT {int(limit)}"
        frame = connection.execute(query, plan.params).df()
    except duckdb.Error as error:
        raise SummaryError(
            "DuckDB не виконав агрегацію: " + str(error).splitlines()[0][:300]
        ) from error
    finally:
        connection.close()
    return frame, count


def _json_rows(frame: pd.DataFrame) -> list[list]:
    rows = []
    for record in frame.itertuples(index=False):
        row: list[Any] = []
        for value in record:
            if hasattr(value, "item") and not isinstance(value, str | bytes):
                try:
                    value = value.item()  # numpy scalar -> Python value
                except (TypeError, ValueError):
                    pass
            if value is None or (not isinstance(value, str) and pd.isna(value)):
                row.append(None)
            elif isinstance(value, float):
                row.append(round(value, 10))
            elif isinstance(value, int | str | bool):
                row.append(value)
            else:
                row.append(str(value))
        rows.append(row)
    return rows


class SummaryStore:
    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets

    def _source(self, project_id: str, dataset_id: str):
        try:
            dataset, frame, metadata = snapshot(self.datasets, project_id, dataset_id)
        except PreparationError as error:
            raise SummaryError(str(error)) from error
        return dataset, _clean(frame, metadata), metadata

    def plan(self, project_id: str, recipe: Recipe) -> Plan:
        ProjectStore(self.datasets.database).get(project_id)
        left = self._source(project_id, str(recipe.dataset_id))
        right = self._source(project_id, str(recipe.join.dataset_id)) if recipe.join else None
        return Plan(recipe, left, right)

    @staticmethod
    def token(plan: Plan) -> str:
        identity = {
            "recipe": plan.recipe.model_dump(mode="json"),
            "sources": [(s[0].id, s[0].sha256, s[2].updated_at) for s in plan.sources],
        }
        return hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str).encode()
        ).hexdigest()

    def fields(self, project_id: str, dataset_id: str) -> list[dict]:
        _, frame, metadata = self._source(project_id, dataset_id)
        meta = {c.name: c for c in metadata.document.columns}
        return [
            {
                "name": name,
                "kind": meta[name].kind if name in meta else "text",
                "label": meta[name].label if name in meta else "",
            }
            for name in frame.columns
        ]

    def preview(self, project_id: str, recipe: Recipe) -> dict:
        started = perf_counter()
        plan = self.plan(project_id, recipe)
        frame, count = run(plan, PREVIEW_ROWS)
        warnings = []
        if recipe.join:
            right = plan.sources[1][1][recipe.join.right_key]
            if right.duplicated().any():
                warnings.append(
                    "У наборі для з'єднання ключ повторюється: рядки основного набору можуть "
                    "дублюватися. Перевірте кількість рядків результату."
                )
        result = {
            "columns": list(frame.columns),
            "rows": _json_rows(frame),
            "row_count": count,
            "sql": plan.literal_sql(),
            "preview_token": self.token(plan),
            "warnings": warnings,
            "sources": [
                {"id": s[0].id, "name": s[0].name, "rows": len(s[1])} for s in plan.sources
            ],
        }
        with connect(self.datasets.database) as connection:
            ProjectStore._event(
                connection,
                project_id,
                "summary.previewed",
                {
                    "recipe": recipe.model_dump(mode="json"),
                    "sql": result["sql"],
                    "rows": count,
                    "duration_ms": (perf_counter() - started) * 1000,
                },
            )
        return result

    def create(
        self, project_id: str, recipe: Recipe, preview_token: str, name: str | None
    ) -> Dataset:
        started = perf_counter()
        audit: dict = {"recipe": recipe.model_dump(mode="json")}
        folder: Path | None = None
        reference = None
        try:
            plan = self.plan(project_id, recipe)
            if self.token(plan) != preview_token:
                raise SummaryError(
                    "Дані, метадані або кроки змінилися. Перегляньте результат ще раз."
                )
            frame, count = run(plan)
            audit.update(sql=plan.literal_sql(), rows=count)
            if not count:
                raise SummaryError("Результат порожній — нічого зберігати.")
            content, parser = csv_snapshot(frame.reset_index(drop=True))
            if len(content) > MAX_BYTES:
                raise SummaryError("Результат перевищує ліміт 5 МБ. Додайте умови або групування.")
            dataset_id = str(uuid4())
            relative = Path(project_id) / "datasets" / dataset_id / "original.csv"
            folder = self.datasets.files_root / relative.parent
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "original.csv").write_bytes(content)
            now = datetime.now(timezone.utc).isoformat()
            base = plan.sources[0][0]
            title = (name or "").strip()[:160] or (Path(base.name).stem[:150] + " · агрегація")
            details = {
                **parser,
                "columns": profile(frame),
                "summary_recipe": recipe.model_dump(mode="json"),
                "summary_sql": plan.literal_sql(),
                "summary_sources": [{"id": s[0].id, "sha256": s[0].sha256} for s in plan.sources],
                "engine": "duckdb",
            }
            dataset = Dataset(
                dataset_id,
                project_id,
                title if title.endswith(".csv") else title + ".csv",
                "csv",
                relative.as_posix(),
                len(content),
                hashlib.sha256(content).hexdigest(),
                len(frame),
                len(frame.columns),
                details,
                now,
            )
            fresh = skeleton(frame, details)
            for target in fresh.columns:
                meaning = plan.meaning.get(target.name)
                if not meaning:
                    continue
                source = meaning["from"]
                original = None
                if source:
                    alias, column_name = source
                    metadata = plan.sources[0 if alias == "a" else 1][2]
                    original = next(
                        (c for c in metadata.document.columns if c.name == column_name), None
                    )
                if meaning["func"] is None and original is not None:
                    for key in ("label", "description", "unit", "computation", "categories",
                                "range_min", "range_max", "role"):  # fmt: skip
                        setattr(target, key, getattr(original, key))
                    if original.kind != "id":
                        target.kind = original.kind
                elif meaning["func"]:
                    func = meaning["func"]
                    what = f"«{source[1]}»" if source else "рядків"
                    groups = ", ".join(recipe.group_by) or "усіх рядках"
                    target.description = f"{FUNC_TITLES[func]} {what} за групами: {groups}."
                    target.computation = (
                        f"DuckDB: {FUNC_SQL[func].format(source[1] if source else '*')}"
                    )
                    if original is not None and func not in ("count", "count_distinct"):
                        target.unit = original.unit
            fresh.dataset.title = title
            fresh.dataset.description = (
                "Агрегація DuckDB за кроками без SQL користувача. Джерела: "
                + ", ".join(s[0].name for s in plan.sources)
                + "."
            )
            reference = self.datasets.persist(dataset, frame.reset_index(drop=True))
            with connect(self.datasets.database) as connection:
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dataset.id, project_id, dataset.name, "csv", dataset.stored_path,
                        dataset.size_bytes, dataset.sha256, dataset.row_count,
                        dataset.column_count, json.dumps(details, ensure_ascii=False), now, None,
                    ),
                )  # fmt: skip
                connection.execute(
                    "INSERT INTO dataset_metadata VALUES (?, ?, ?, ?)",
                    (dataset.id, fresh.model_dump_json(), now, None),
                )
                audit.update(
                    output_dataset_id=dataset.id,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                )
                ProjectStore._event(connection, project_id, "summary.created", audit)
            return dataset
        except BaseException as error:
            self.datasets.analytics.discard(reference)
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)  # only this operation's fresh folder
            if isinstance(error, (SummaryError, DatasetError, OSError)):
                message = (
                    str(error)
                    if not isinstance(error, OSError)
                    else "Не вдалося зберегти агрегацію."
                )
                audit.update(
                    status="error", error=message, duration_ms=(perf_counter() - started) * 1000
                )
                with connect(self.datasets.database) as connection:
                    ProjectStore._event(connection, project_id, "summary.created", audit)
                raise SummaryError(message) from error
            raise


def available() -> str | None:
    try:
        import duckdb
    except ImportError:
        return None
    return str(duckdb.__version__)


__all__ = ["Aggregate", "Filter", "Join", "Recipe", "SummaryError", "SummaryStore", "available"]
