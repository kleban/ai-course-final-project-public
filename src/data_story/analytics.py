"""Versioned analytical tables in persistent DuckDB; no model-authored SQL executes here."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import UUID

import pandas as pd

Kind = Literal[
    "dataset",
    "summary",
    "statistics",
    "model_metrics",
    "coefficients",
    "predictions",
    "residuals",
    "chart_data",
    "normalized",
]
KINDS = {
    "dataset",
    "summary",
    "statistics",
    "model_metrics",
    "coefficients",
    "predictions",
    "residuals",
    "chart_data",
    "normalized",
}
_locks: dict[Path, Any] = {}
_registry_lock = RLock()


class AnalyticsError(ValueError):
    """Analytical storage could not provide a verified result."""


def _project(value: str) -> str:
    try:
        return str(UUID(value))
    except (ValueError, TypeError, AttributeError) as error:
        raise AnalyticsError("Неправильний ідентифікатор проєкту.") from error


class AnalyticsStore:
    """Lazy storage: construction does not create a database or migrate author data."""

    def __init__(self, database: Path):
        self.database = database.resolve()
        with _registry_lock:
            self._lock = _locks.setdefault(self.database, RLock())

    @contextmanager
    def _connection(self):
        import duckdb

        with self._lock:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.database.parent / "tmp" / "duckdb-analytics"
            temporary.mkdir(parents=True, exist_ok=True)
            connection = duckdb.connect(
                str(self.database),
                config={
                    "threads": 2,
                    "memory_limit": "1GB",
                    "temp_directory": str(temporary),
                },
            )
            try:
                connection.execute("SET enable_external_access = false")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS _analytics_catalog ("
                    "project_id VARCHAR, kind VARCHAR, object_id VARCHAR, fingerprint VARCHAR, "
                    "schema_name VARCHAR, table_name VARCHAR, columns_json VARCHAR, "
                    "row_count BIGINT, created_at TIMESTAMPTZ DEFAULT current_timestamp, "
                    "PRIMARY KEY(project_id, kind, object_id, fingerprint))"
                )
                connection.execute(
                    "ALTER TABLE _analytics_catalog ADD COLUMN IF NOT EXISTS source_key VARCHAR"
                )
                connection.execute(
                    "ALTER TABLE _analytics_catalog ADD COLUMN IF NOT EXISTS dtypes_json VARCHAR"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS _analytics_results ("
                    "project_id VARCHAR, kind VARCHAR, object_id VARCHAR, source_key VARCHAR, "
                    "payload VARCHAR, PRIMARY KEY(project_id, kind, object_id, source_key))"
                )
                yield connection
            except duckdb.Error as error:
                raise AnalyticsError(
                    "Не вдалося виконати операцію аналітичного сховища."
                ) from error
            finally:
                connection.close()

    def put(
        self,
        project_id: str,
        kind: Kind,
        object_id: str,
        frame: pd.DataFrame,
        *,
        source: str,
    ) -> dict:
        project_id = _project(project_id)
        if kind not in KINDS or not object_id or len(object_id) > 200:
            raise AnalyticsError("Неправильний тип або ідентифікатор аналітичної таблиці.")
        row_limit = 5_000_000
        if frame.shape[1] == 0 or len(frame) > row_limit:
            raise AnalyticsError(f"Таблиця має містити поля та не більше {row_limit} рядків.")
        columns = list(frame.columns)
        if any(not isinstance(c, str) or not c for c in columns):
            raise AnalyticsError("Назви полів мають бути непорожніми рядками.")
        if len({c.casefold() for c in columns}) != len(columns):
            raise AnalyticsError("DuckDB потребує унікальних назв полів незалежно від регістру.")
        schema_json = json.dumps(
            {"columns": columns, "types": [str(t) for t in frame.dtypes]}, ensure_ascii=False
        )
        digest = hashlib.sha256((source + "\n" + schema_json).encode("utf-8"))
        digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
        fingerprint = digest.hexdigest()
        schema = "p_" + UUID(project_id).hex
        table = (
            "t_"
            + hashlib.sha256(
                (kind + "\n" + object_id + "\n" + fingerprint).encode("utf-8")
            ).hexdigest()
        )
        with self._connection() as connection:
            existing = connection.execute(
                "SELECT 1 FROM _analytics_catalog "
                "WHERE project_id=? AND kind=? AND object_id=? AND fingerprint=?",
                [project_id, kind, object_id, fingerprint],
            ).fetchone()
            if existing is None:
                connection.register("_incoming", frame)
                connection.execute("BEGIN TRANSACTION")
                try:
                    connection.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
                    connection.execute(
                        f'CREATE TABLE "{schema}"."{table}" AS SELECT * FROM _incoming'
                    )
                    connection.execute(
                        "INSERT INTO _analytics_catalog "
                        "(project_id, kind, object_id, fingerprint, schema_name, table_name, "
                        "columns_json, row_count, source_key, dtypes_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            project_id,
                            kind,
                            object_id,
                            fingerprint,
                            schema,
                            table,
                            json.dumps(columns, ensure_ascii=False),
                            len(frame),
                            hashlib.sha256(source.encode("utf-8")).hexdigest(),
                            json.dumps([str(t) for t in frame.dtypes]),
                        ],
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        return {
            "project_id": project_id,
            "kind": kind,
            "object_id": object_id,
            "fingerprint": fingerprint,
            "table": f"{schema}.{table}",
            "columns": columns,
            "row_count": len(frame),
            "reused": existing is not None,
        }

    def catalog(self, project_id: str) -> list[dict]:
        project_id = _project(project_id)
        if not self.database.exists():
            return []
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT kind, object_id, fingerprint, schema_name, table_name, columns_json, "
                "row_count FROM _analytics_catalog WHERE project_id=? "
                "ORDER BY created_at, kind, object_id, fingerprint",
                [project_id],
            ).fetchall()
        return [
            {
                "kind": row[0],
                "object_id": row[1],
                "fingerprint": row[2],
                "table": f"{row[3]}.{row[4]}",
                "columns": json.loads(row[5]),
                "row_count": row[6],
            }
            for row in rows
        ]

    def read(self, project_id: str, kind: Kind, object_id: str, fingerprint: str) -> pd.DataFrame:
        project_id = _project(project_id)
        if not self.database.exists():
            raise AnalyticsError("Аналітичну таблицю не знайдено в поточному проєкті.")
        with self._connection() as connection:
            row = connection.execute(
                "SELECT schema_name, table_name, dtypes_json FROM _analytics_catalog "
                "WHERE project_id=? AND kind=? AND object_id=? AND fingerprint=?",
                [project_id, kind, object_id, fingerprint],
            ).fetchone()
            if row is None:
                raise AnalyticsError("Аналітичну таблицю не знайдено в поточному проєкті.")
            frame = connection.execute(f'SELECT * FROM "{row[0]}"."{row[1]}"').df()
            if row[2]:
                frame = frame.astype(dict(zip(frame.columns, json.loads(row[2]))))
            return frame

    def find(self, project_id: str, kind: Kind, object_id: str, source: str) -> str | None:
        project_id = _project(project_id)
        if not self.database.exists():
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT fingerprint FROM _analytics_catalog "
                "WHERE project_id=? AND kind=? AND object_id=? AND source_key=? "
                "ORDER BY created_at DESC LIMIT 1",
                [project_id, kind, object_id, hashlib.sha256(source.encode("utf-8")).hexdigest()],
            ).fetchone()
            return row[0] if row else None

    def discard(self, reference: dict | None) -> None:
        """Compensate a failed SQLite creation; never discard a reused version."""
        if reference is None or reference["reused"]:
            return
        with self._connection() as connection:
            row = connection.execute(
                "SELECT schema_name, table_name FROM _analytics_catalog "
                "WHERE project_id=? AND kind=? AND object_id=? AND fingerprint=?",
                [reference[k] for k in ("project_id", "kind", "object_id", "fingerprint")],
            ).fetchone()
            if row is not None:
                connection.execute("BEGIN TRANSACTION")
                try:
                    connection.execute(f'DROP TABLE "{row[0]}"."{row[1]}"')
                    connection.execute(
                        "DELETE FROM _analytics_catalog "
                        "WHERE project_id=? AND kind=? AND object_id=? AND fingerprint=?",
                        [reference[k] for k in ("project_id", "kind", "object_id", "fingerprint")],
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise

    def result(self, project_id: str, kind: Kind, object_id: str, source: str) -> dict | None:
        project_id = _project(project_id)
        if not self.database.exists():
            return None
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM _analytics_results "
                "WHERE project_id=? AND kind=? AND object_id=? AND source_key=?",
                [project_id, kind, object_id, hashlib.sha256(source.encode("utf-8")).hexdigest()],
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_result(
        self,
        project_id: str,
        kind: Kind,
        object_id: str,
        source: str,
        payload: dict,
        tables: list[tuple[str, pd.DataFrame]],
    ) -> None:
        """Publish a JSON result only after all its numerical tables are durable."""
        project_id = _project(project_id)
        if kind not in KINDS:
            raise AnalyticsError("Неправильний тип аналітичного результату.")
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        references = []
        with self._lock:
            try:
                for suffix, frame in tables:
                    references.append(
                        self.put(project_id, kind, object_id + "." + suffix, frame, source=source)
                    )
                with self._connection() as connection:
                    connection.execute(
                        "INSERT INTO _analytics_results VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT DO NOTHING",
                        [
                            project_id,
                            kind,
                            object_id,
                            hashlib.sha256(source.encode("utf-8")).hexdigest(),
                            serialized,
                        ],
                    )
            except BaseException:
                for reference in reversed(references):
                    self.discard(reference)
                raise

    def discard_result(self, project_id: str, kind: Kind, object_id: str, source: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "DELETE FROM _analytics_results "
                "WHERE project_id=? AND kind=? AND object_id=? AND source_key=?",
                [
                    _project(project_id),
                    kind,
                    object_id,
                    hashlib.sha256(source.encode("utf-8")).hexdigest(),
                ],
            )
