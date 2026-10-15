"""One SQLite file for the whole local service: connection rules and schema migrations."""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 17

_V1 = """
    CREATE TABLE projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        purpose TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE project_events (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        operation TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        payload TEXT NOT NULL
    );
    CREATE INDEX project_events_project ON project_events(project_id, occurred_at);
"""

# v2: project type; editable analysis instructions; audit trail of configuration changes.
# Existing projects become regression projects — the only type that existed in practice.
_V2 = """
    ALTER TABLE projects ADD COLUMN project_type TEXT NOT NULL DEFAULT 'regression';
    CREATE TABLE instructions (
        project_type TEXT PRIMARY KEY,
        text TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE config_events (
        id TEXT PRIMARY KEY,
        subject TEXT NOT NULL,
        operation TEXT NOT NULL,
        occurred_at TEXT NOT NULL,
        payload TEXT NOT NULL
    );
"""

# v3: uploaded datasets. The original file lives under runtime/projects/<project>/datasets/<id>/;
# the row keeps its metadata and the column profile computed once at upload time.
_V3 = """
    CREATE TABLE datasets (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        name TEXT NOT NULL,
        file_format TEXT NOT NULL,
        stored_path TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        sha256 TEXT NOT NULL,
        row_count INTEGER NOT NULL,
        column_count INTEGER NOT NULL,
        profile TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX datasets_project ON datasets(project_id, created_at);
"""

# v4: data dictionary per dataset (one JSON document) and raw descriptions the author attached.
_V4 = """
    CREATE TABLE dataset_metadata (
        dataset_id TEXT PRIMARY KEY REFERENCES datasets(id),
        document TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        confirmed_at TEXT
    );
    CREATE TABLE metadata_sources (
        id TEXT PRIMARY KEY,
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        kind TEXT NOT NULL,
        name TEXT NOT NULL,
        stored_path TEXT,
        text TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX metadata_sources_dataset ON metadata_sources(dataset_id, created_at);
"""

# v5: a dataset can be an independent copy of a dataset from another project (stage 02c);
# the column keeps the source id for provenance only (no foreign key: the copy stands alone).
_V5 = """
    ALTER TABLE datasets ADD COLUMN copied_from TEXT;
"""


# v6: one local chart configuration, independent from projects/datasets.
_V6 = """
    CREATE TABLE chart_configuration (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        document TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
"""


# v7: project description in Markdown (D-64); later drafted with LLM help (stage 08).
_V7 = """
    ALTER TABLE projects ADD COLUMN description TEXT NOT NULL DEFAULT '';
"""


_V8 = """
    CREATE TABLE preparation_plans (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        created_at TEXT NOT NULL,
        document TEXT NOT NULL
    );
    CREATE INDEX preparation_plans_project ON preparation_plans(project_id, created_at);
"""


_V9 = """
    CREATE TABLE preparation_outputs (
        plan_id TEXT PRIMARY KEY REFERENCES preparation_plans(id),
        dataset_id TEXT NOT NULL UNIQUE REFERENCES datasets(id)
    );
"""


_V10 = """
    CREATE TABLE training_runs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        dataset_id TEXT NOT NULL REFERENCES datasets(id),
        created_at TEXT NOT NULL,
        document TEXT NOT NULL
    );
    CREATE INDEX training_runs_project ON training_runs(project_id, created_at);
"""


# v11: trained linear regression results linked to saved train/test matrices (stage 06).
_V11 = """
    CREATE TABLE model_runs (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        training_run_id TEXT NOT NULL UNIQUE REFERENCES training_runs(id),
        created_at TEXT NOT NULL,
        document TEXT NOT NULL
    );
    CREATE INDEX model_runs_project ON model_runs(project_id, created_at);
"""


# v12: editable knowledge base documents for RAG (stage 07); seeded from repository defaults.
_V12 = """
    CREATE TABLE knowledge_documents (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        source TEXT NOT NULL,
        priority INTEGER NOT NULL,
        text TEXT NOT NULL,
        is_default INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
"""


# v13: OpenAI model per pipeline task (stage 08); the key itself lives only in .env (D-76).
_V13 = """
    CREATE TABLE ai_configuration (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        document TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
"""


# v14: every OpenAI call (request without key, response, usage, error) for the AI log.
_V14 = """
    CREATE TABLE ai_calls (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        task TEXT NOT NULL,
        model TEXT NOT NULL,
        project_id TEXT,
        status TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        document TEXT NOT NULL
    );
    CREATE INDEX ai_calls_started ON ai_calls(started_at);
"""


_V15 = """
    CREATE TABLE reports (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        revision INTEGER NOT NULL,
        document TEXT NOT NULL
    );
    CREATE INDEX reports_project ON reports(project_id, created_at);
"""


# v16: conclusions block — AI draft from facts + cited knowledge fragments, edited and approved
# by the author before they enter a report.
_V16 = """
    CREATE TABLE conclusions (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        model_id TEXT NOT NULL REFERENCES model_runs(id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        revision INTEGER NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('draft', 'approved')),
        document TEXT NOT NULL
    );
    CREATE INDEX conclusions_project ON conclusions(project_id, created_at);
"""


# v17: several models per saved split (author decision, history 143): the split is built from all
# fields once, and each model chooses its own features (by hand or stepwise). Rebuilds
# model_runs without UNIQUE(training_run_id); rows and ids are kept, so conclusions stay valid.
_V17 = """
    CREATE TABLE model_runs_v17 (
        id TEXT PRIMARY KEY,
        project_id TEXT NOT NULL REFERENCES projects(id),
        training_run_id TEXT NOT NULL REFERENCES training_runs(id),
        created_at TEXT NOT NULL,
        document TEXT NOT NULL
    );
    INSERT INTO model_runs_v17 SELECT id, project_id, training_run_id, created_at, document
        FROM model_runs;
    DROP TABLE model_runs;
    ALTER TABLE model_runs_v17 RENAME TO model_runs;
    CREATE INDEX model_runs_project ON model_runs(project_id, created_at);
    CREATE INDEX model_runs_training ON model_runs(training_run_id);
"""


@contextmanager
def connect(database: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def ensure_schema(database: Path) -> Path:
    """Create or upgrade the schema in one transaction; never touch unknown or newer files."""
    database = Path(database).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    with connect(database) as connection:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (*range(SCHEMA_VERSION), SCHEMA_VERSION):
            raise RuntimeError("Непідтримувана версія бази даних; файл не змінено.")
        if version == 0:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if tables:
                raise RuntimeError("Невідома схема бази даних; файл не змінено.")
        steps = {
            0: _V1
            + _V2
            + _V3
            + _V4
            + _V5
            + _V6
            + _V7
            + _V8
            + _V9
            + _V10
            + _V11
            + _V12
            + _V13
            + _V14
            + _V15
            + _V16
            + _V17,
            1: _V2
            + _V3
            + _V4
            + _V5
            + _V6
            + _V7
            + _V8
            + _V9
            + _V10
            + _V11
            + _V12
            + _V13
            + _V14
            + _V15
            + _V16
            + _V17,
            2: _V3
            + _V4
            + _V5
            + _V6
            + _V7
            + _V8
            + _V9
            + _V10
            + _V11
            + _V12
            + _V13
            + _V14
            + _V15
            + _V16
            + _V17,
            3: _V4
            + _V5
            + _V6
            + _V7
            + _V8
            + _V9
            + _V10
            + _V11
            + _V12
            + _V13
            + _V14
            + _V15
            + _V16
            + _V17,
            4: _V5 + _V6 + _V7 + _V8 + _V9 + _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            5: _V6 + _V7 + _V8 + _V9 + _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            6: _V7 + _V8 + _V9 + _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            7: _V8 + _V9 + _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            8: _V9 + _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            9: _V10 + _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            10: _V11 + _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            11: _V12 + _V13 + _V14 + _V15 + _V16 + _V17,
            12: _V13 + _V14 + _V15 + _V16 + _V17,
            13: _V14 + _V15 + _V16 + _V17,
            14: _V15 + _V16 + _V17,
            15: _V16 + _V17,
            16: _V17,
        }.get(version)
        if steps:
            # Foreign keys are off only while tables are rebuilt (v17), then checked before
            # commit: a violation rolls the whole upgrade back and leaves the file as it was.
            connection.commit()
            connection.execute("PRAGMA foreign_keys = OFF")
            try:
                connection.executescript(f"BEGIN; {steps} PRAGMA user_version = {SCHEMA_VERSION};")
                if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    raise RuntimeError("Оновлення схеми порушило зв'язки таблиць; файл не змінено.")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.execute("PRAGMA foreign_keys = ON")
    return database
