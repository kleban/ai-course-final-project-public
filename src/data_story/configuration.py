"""Editable configuration stored next to the projects: analysis instructions per project type.

Later stages add the AI model settings here. Every change is written
to `config_events` in the same transaction, so the configuration history stays traceable.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .project_types import PROJECT_TYPES, require_known
from .storage import connect, ensure_schema
from .validation import text

INSTRUCTION_LIMIT = 20000


@dataclass(frozen=True)
class Instruction:
    project_type: str
    text: str
    is_default: bool
    updated_at: str | None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InstructionStore:
    def __init__(self, database: Path):
        self.database = ensure_schema(database)

    @staticmethod
    def _event(connection, subject: str, operation: str, payload: dict) -> None:
        connection.execute(
            "INSERT INTO config_events VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), subject, operation, _now(), json.dumps(payload, ensure_ascii=False)),
        )

    def get(self, project_type: str) -> Instruction:
        key = require_known(project_type)
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT text, updated_at FROM instructions WHERE project_type = ?", (key,)
            ).fetchone()
        if row is None:
            return Instruction(key, PROJECT_TYPES[key].default_instruction(), True, None)
        return Instruction(key, row["text"], False, row["updated_at"])

    def save(self, project_type: str, value: str) -> Instruction:
        key = require_known(project_type)
        value = text(value, "Інструкція", INSTRUCTION_LIMIT, required=True)
        updated_at = _now()
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO instructions VALUES (?, ?, ?) ON CONFLICT(project_type) "
                "DO UPDATE SET text = excluded.text, updated_at = excluded.updated_at",
                (key, value, updated_at),
            )
            self._event(
                connection, f"instruction:{key}", "instruction.saved", {"length": str(len(value))}
            )
        return Instruction(key, value, False, updated_at)

    def reset(self, project_type: str) -> Instruction:
        key = require_known(project_type)
        with connect(self.database) as connection:
            connection.execute("DELETE FROM instructions WHERE project_type = ?", (key,))
            self._event(connection, f"instruction:{key}", "instruction.reset", {})
        return self.get(key)

    def events(self) -> list[dict]:
        with connect(self.database) as connection:
            rows = connection.execute(
                "SELECT subject, operation, occurred_at FROM config_events ORDER BY occurred_at"
            ).fetchall()
        return [dict(row) for row in rows]
