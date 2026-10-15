"""Read-only SQL over explicitly supplied project tables in an isolated DuckDB connection."""

import hashlib
import json
from threading import Timer
from time import perf_counter

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from .analytics import AnalyticsError
from .datasets import DatasetStore
from .metadata import MetadataStore
from .missing_values import missing_masks
from .projects import ProjectStore
from .storage import connect

SQL_SECONDS = 5.0


class AnalyticalQueries:
    def __init__(self, service):
        self.service = service
        self.datasets: DatasetStore = service.datasets
        self.analytics = self.datasets.analytics

    def _dataset(self, dataset) -> dict:
        metadata = MetadataStore(self.datasets.database, self.datasets).get(dataset.id)
        self.datasets.verified_content(dataset)
        source = json.dumps(
            {
                "version": 1,
                "sha256": dataset.sha256,
                "profile": dataset.profile,
                "metadata": metadata.document.model_dump(mode="json"),
                "confirmed_at": metadata.confirmed_at,
                "updated_at": metadata.updated_at,
                "pandas": pd.__version__,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        fingerprint = self.analytics.find(dataset.project_id, "normalized", dataset.id, source)
        if fingerprint is None:
            raw = self.datasets.frame(dataset.id)
            columns = {}
            profiles = {c["name"]: c for c in dataset.profile["columns"]}
            for column in metadata.document.columns:
                if column.role == "ignore":
                    continue
                series = raw[column.name]
                base, extra, error = missing_masks(series, column.missing_codes)
                if error:
                    raise AnalyticsError(f"Поле «{column.name}»: {error}")
                series = series.mask(base | extra)
                try:
                    if column.kind == "number":
                        series = pd.to_numeric(series, errors="raise").replace(
                            [float("inf"), -float("inf")], float("nan")
                        )
                    elif column.kind == "date":
                        fmt = profiles[column.name].get("date_format")
                        if not fmt:
                            raise ValueError("date format missing")
                        series = pd.to_datetime(series.astype("string"), format=fmt, errors="raise")
                    elif column.kind in ("category", "text"):
                        series = series.astype("string")
                    elif column.kind == "bool":
                        if (
                            not ptypes.is_bool_dtype(series)
                            and not series.dropna()
                            .map(lambda value: isinstance(value, (bool, np.bool_)))
                            .all()
                        ):
                            raise ValueError("boolean values required")
                        series = series.astype("boolean")
                except (ValueError, TypeError) as conversion_error:
                    raise AnalyticsError(
                        f"Поле «{column.name}» не відповідає описаному типу. Перевірте метадані."
                    ) from conversion_error
                columns[column.name] = series
            if not columns:
                raise AnalyticsError("Для SQL немає полів: усі поля виключені з аналізу.")
            fingerprint = self.analytics.put(
                dataset.project_id, "normalized", dataset.id, pd.DataFrame(columns), source=source
            )["fingerprint"]
        reference = next(
            r
            for r in self.analytics.catalog(dataset.project_id)
            if r["kind"] == "normalized"
            and r["object_id"] == dataset.id
            and r["fingerprint"] == fingerprint
        )
        return {
            **reference,
            "alias": "d_" + dataset.id.replace("-", ""),
            "name": dataset.name,
            "fields": [
                c.model_dump(include={"name", "kind", "role", "unit", "label"})
                for c in metadata.document.columns
                if c.role != "ignore"
            ],
        }

    def schema(self, project_id: str) -> list[dict]:
        self.service.store.get(project_id)
        tables = [self._dataset(d) for d in self.datasets.list_datasets(project_id)]
        for model in self.service.model_store.list(project_id):
            model = self.service.model_store.get(project_id, model["id"])
            source = self.service.model_store._source(model["document"])
            for kind, prefix in (
                ("coefficients", "c_"),
                ("model_metrics", "m_"),
                ("predictions", "p_"),
                ("residuals", "r_"),
            ):
                fingerprint = self.analytics.find(project_id, kind, model["id"], source)
                if fingerprint:
                    table = next(
                        r
                        for r in self.analytics.catalog(project_id)
                        if r["kind"] == kind
                        and r["object_id"] == model["id"]
                        and r["fingerprint"] == fingerprint
                    )
                    tables.append(
                        {
                            **table,
                            "alias": prefix + model["id"].replace("-", ""),
                            "name": kind + " · " + model["id"],
                        }
                    )
        return tables

    def query(self, project_id: str, sql: str, limit: int) -> dict:
        import duckdb

        self.service.store.get(project_id)
        if (
            not isinstance(sql, str)
            or not sql.strip()
            or len(sql) > 10000
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 10000
        ):
            raise AnalyticsError("Вкажіть SQL SELECT і ліміт від 1 до 10000 рядків.")
        started = perf_counter()
        audit: dict = {"sql": sql, "limit": limit, "status": "error"}
        connection = None
        timer = None
        try:
            tables = self.schema(project_id)
            aliases = {t["alias"]: t for t in tables}
            temporary = self.analytics.database.parent / "tmp" / "duckdb-queries"
            temporary.mkdir(parents=True, exist_ok=True)
            connection = duckdb.connect(
                ":memory:",
                config={
                    "threads": 2,
                    "memory_limit": "512MB",
                    "temp_directory": str(temporary),
                },
            )
            connection.execute("SET enable_external_access=false")
            connection.execute("SET python_enable_replacements=false")
            timer = Timer(
                max(0.001, SQL_SECONDS - (perf_counter() - started)), connection.interrupt
            )
            timer.start()
            statements = connection.extract_statements(sql)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                raise AnalyticsError("Дозволено один SQL-запит SELECT без зміни даних.")
            # Never attach the persistent database or another project to this connection.
            mentioned = connection.get_table_names(sql)
            if not mentioned.issubset(aliases):
                raise AnalyticsError("SQL звертається до таблиць поза доступним каталогом проєкту.")
            for alias in mentioned:
                reference = aliases[alias]
                frame = self.analytics.read(
                    project_id, reference["kind"], reference["object_id"], reference["fingerprint"]
                )
                connection.register(alias, frame)
            provenance = [
                {k: aliases[a][k] for k in ("alias", "fingerprint", "kind", "object_id")}
                for a in sorted(mentioned)
            ]
            source = json.dumps(
                {
                    "version": 1,
                    "duckdb": duckdb.__version__,
                    "sql": sql,
                    "limit": limit,
                    "tables": provenance,
                },
                sort_keys=True,
            )
            object_id = "sql." + hashlib.sha256(source.encode()).hexdigest()
            result = self.analytics.result(project_id, "chart_data", object_id, source)
            reused = result is not None
            if result is None:
                relation = connection.sql(statements[0].query).limit(limit + 1)
                timer.cancel()
                timer.join()
                remaining = SQL_SECONDS - (perf_counter() - started)
                if remaining <= 0:
                    raise AnalyticsError("Перевищено час виконання SQL. Звузьте запит.")
                # Start immediately before execution: interrupts during binding are cleared
                # by DuckDB before the relation executes.
                timer = Timer(remaining, connection.interrupt)
                timer.start()
                frame = relation.df()
                truncated = len(frame) > limit
                frame = frame.head(limit)
                parsed = json.loads(
                    frame.to_json(orient="split", date_format="iso", double_precision=15)
                )
                result = {
                    "columns": parsed["columns"],
                    "rows": parsed["data"],
                    "row_count": len(frame),
                    "truncated": truncated,
                    "sources": provenance,
                }
                if len(json.dumps(result, ensure_ascii=False).encode()) > 2_000_000:
                    raise AnalyticsError("Результат SQL завеликий. Оберіть менше полів або рядків.")
                self.analytics.save_result(
                    project_id, "chart_data", object_id, source, result, [("table", frame)]
                )
            audit.update(
                status="success",
                reused=reused,
                row_count=result["row_count"],
                truncated=result["truncated"],
                tables=sorted(mentioned),
            )
            fingerprint = self.analytics.find(
                project_id, "chart_data", object_id + ".table", source
            )
            return {
                **result,
                "reused": reused,
                "reference": {
                    "kind": "chart_data",
                    "object_id": object_id + ".table",
                    "fingerprint": fingerprint,
                },
            }
        except duckdb.Error as error:
            audit["error"] = "SQL не виконано: перевірте поля, синтаксис або обсяг запиту."
            raise AnalyticsError(audit["error"]) from error
        except (AnalyticsError, ValueError) as error:
            audit["error"] = str(error)
            raise
        finally:
            if timer is not None:
                timer.cancel()
                timer.join()
            if connection is not None:
                connection.close()
            audit["duration_ms"] = (perf_counter() - started) * 1000
            with connect(self.datasets.database) as connection:
                ProjectStore._event(connection, project_id, "analytics.queried", audit)

    def save_summary(self, project_id: str, sql: str, limit: int, name: str):
        from .prepared_datasets import csv_snapshot

        result = self.query(project_id, sql, limit)
        if result["truncated"]:
            raise AnalyticsError("Агрегацію обрізано лімітом. Збільште ліміт або звузьте запит.")
        reference = result["reference"]
        frame = self.analytics.read(
            project_id, "chart_data", reference["object_id"], reference["fingerprint"]
        )
        if frame.empty:
            raise AnalyticsError("SQL не повернув рядків для нового набору.")
        for column in frame.columns:
            if ptypes.is_datetime64_any_dtype(frame[column]):
                frame[column] = frame[column].astype("string")
        content, parser = csv_snapshot(frame)
        parser.update(sql=sql, sql_sources=result["sources"], engine="duckdb")
        return self.datasets.add(
            project_id, name if name.endswith(".csv") else name + ".csv", content, parser=parser
        )
