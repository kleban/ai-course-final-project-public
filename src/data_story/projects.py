"""Project metadata and its audit trail, without UI or AI dependencies."""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .project_types import DEFAULT_TYPE, require_enabled
from .storage import SCHEMA_VERSION, connect, ensure_schema
from .validation import DESCRIPTION_LIMIT, NAME_LIMIT, PURPOSE_LIMIT, text


class ProjectNotFound(LookupError):
    """A requested project does not exist."""


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    purpose: str
    created_at: str
    project_type: str = DEFAULT_TYPE
    description: str = ""


@dataclass(frozen=True)
class ProjectEvent:
    id: str
    project_id: str
    operation: str
    occurred_at: str
    payload: dict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStore:
    """SQLite repository. Project changes and audit events commit together."""

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, database: Path):
        self.database = ensure_schema(database)

    def _connection(self):
        return connect(self.database)

    @staticmethod
    def _event(connection, project_id: str, operation: str, payload: dict) -> None:
        connection.execute(
            "INSERT INTO project_events VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), project_id, operation, _now(), json.dumps(payload, ensure_ascii=False)),
        )

    @staticmethod
    def _get(connection, project_id: str) -> Project:
        row = connection.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if row is None:
            raise ProjectNotFound("Проєкт не знайдено. Оновіть список і виберіть інший.")
        return Project(**dict(row))

    def create(self, name: str, purpose: str = "", project_type: str = DEFAULT_TYPE) -> Project:
        name = text(name, "Назва", NAME_LIMIT, required=True)
        purpose = text(purpose, "Мета", PURPOSE_LIMIT)
        kind = require_enabled(project_type)
        project = Project(str(uuid4()), name, purpose, _now(), kind)
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO projects (id, name, purpose, created_at, project_type) "
                "VALUES (?, ?, ?, ?, ?)",
                (project.id, project.name, project.purpose, project.created_at, kind),
            )
            self._event(
                connection, project.id, "project.created", {"name": name, "project_type": kind}
            )
        return project

    def list_projects(self) -> list[Project]:
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM projects ORDER BY created_at, id").fetchall()
            return [Project(**dict(row)) for row in rows]

    def get(self, project_id: str) -> Project:
        with self._connection() as connection:
            return self._get(connection, project_id)

    def save_description(
        self, project_id: str, description: str, *, expected_description: str | None = None
    ) -> Project:
        """Markdown description of the project; the change and its event commit together."""
        if not isinstance(description, str):
            raise ValueError("Опис: очікується текст.")
        if len(description) > DESCRIPTION_LIMIT:
            raise ValueError(f"Опис: не більше {DESCRIPTION_LIMIT} символів.")
        description = description.replace("\r\n", "\n").strip()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if (
                expected_description is not None
                and self._get(connection, project_id).description != expected_description
            ):
                raise ValueError("Опис змінився після створення чернетки. Створіть нову чернетку.")
            self._get(connection, project_id)
            connection.execute(
                "UPDATE projects SET description = ? WHERE id = ?", (description, project_id)
            )
            self._event(
                connection,
                project_id,
                "project.description_saved",
                {
                    "length": len(description),
                    "empty": not description,
                    **({"origin": "ai_draft"} if expected_description is not None else {}),
                },
            )
            return self._get(connection, project_id)

    def open(self, project_id: str) -> Project:
        with self._connection() as connection:
            project = self._get(connection, project_id)
            self._event(connection, project.id, "project.opened", {"name": project.name})
            return project

    def events(self, project_id: str) -> list[ProjectEvent]:
        with self._connection() as connection:
            self._get(connection, project_id)
            rows = connection.execute(
                "SELECT * FROM project_events WHERE project_id = ? ORDER BY occurred_at, rowid",
                (project_id,),
            ).fetchall()
            return [
                ProjectEvent(
                    id=row["id"],
                    project_id=row["project_id"],
                    operation=row["operation"],
                    occurred_at=row["occurred_at"],
                    payload=json.loads(row["payload"]),
                )
                for row in rows
            ]
