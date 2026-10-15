"""Fresh audited previews and immutable, explicitly confirmed preparation recipes."""

import hashlib
import json
from datetime import datetime, timezone
from time import perf_counter
from uuid import uuid4

from .datasets import DatasetError, DatasetStore, _cell
from .metadata import MetadataStore
from .preparation import MissingRule, PreparationError, preview_missing
from .projects import ProjectStore
from .storage import connect


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


class PreparationStore:
    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets
        self.metadata = MetadataStore(datasets.database, datasets)

    def preview(
        self,
        project_id: str,
        dataset_id: str,
        rules: list[MissingRule],
        *,
        expected_token: str | None = None,
    ) -> dict:
        """Saving recomputes the preview and rejects changed rules, file or metadata."""
        ProjectStore(self.datasets.database).get(project_id)
        started = perf_counter()
        operation = "preparation.saved" if expected_token else "preparation.previewed"
        audit: dict = {
            "dataset_id": dataset_id,
            "fields": [rule.field for rule in rules],
            "actions": [rule.action for rule in rules],
        }
        try:
            dataset = self.datasets.get(dataset_id)
            if dataset.project_id != project_id:
                raise PreparationError("Набір не належить поточному проєкту.")
            audit["dataset_name"] = dataset.name
            path = (self.datasets.files_root / dataset.stored_path).resolve()
            if not path.is_relative_to(self.datasets.files_root) or not path.is_file():
                raise PreparationError("Файл набору даних відсутній у сховищі.")
            content = path.read_bytes()
            sha = hashlib.sha256(content).hexdigest()
            audit["sha256"] = sha
            if sha != dataset.sha256:
                raise PreparationError(
                    "Файл набору змінився або пошкоджений. Завантажте його заново."
                )
            frame = self.datasets.frame(dataset_id)
            metadata = self.metadata.get(dataset_id)
            audit.update(
                metadata_updated_at=metadata.updated_at,
                metadata_state="confirmed"
                if metadata.confirmed_at
                else "draft"
                if metadata.saved
                else "none",
            )
            prepared = preview_missing(frame, metadata, rules)
            identity = {
                "version": 1,
                "project_id": project_id,
                "dataset_id": dataset_id,
                "sha256": sha,
                "metadata": metadata.document.model_dump(mode="json"),
                "metadata_updated_at": metadata.updated_at,
                "confirmed_at": metadata.confirmed_at,
                "rules": [rule.model_dump(mode="json") for rule in rules],
            }
            token = digest(identity)
            if expected_token and token != expected_token:
                raise PreparationError(
                    "Дані, метадані або правила змінилися. Повторіть попередній перегляд."
                )
            result = {
                "project_id": project_id,
                "dataset_id": dataset_id,
                "dataset_name": dataset.name,
                "preview_token": token,
                "sha256": sha,
                "metadata_updated_at": metadata.updated_at,
                "report": prepared.report,
                "columns": list(prepared.frame.columns),
                "rows": [
                    [_cell(v) for v in row]
                    for row in prepared.frame.head(10).itertuples(index=False)
                ],
                "positions": prepared.retained_positions[:10],
            }
            audit.update(
                {key: prepared.report[key] for key in ("rows_before", "rows_after", "rows_removed")}
            )
            if expected_token:
                if not rules:
                    raise PreparationError("Оберіть хоча б одну операцію для збереження плану.")
                plan_id = str(uuid4())
                now = datetime.now(timezone.utc).isoformat()
                document = {
                    **identity,
                    "preview_token": token,
                    "report": prepared.report,
                    "retained_positions": prepared.retained_positions,
                }
                with connect(self.datasets.database) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    current = connection.execute(
                        "SELECT document, updated_at, confirmed_at FROM dataset_metadata WHERE dataset_id = ?",
                        (dataset_id,),
                    ).fetchone()
                    if (current is None) != (not metadata.saved) or (
                        current is not None
                        and (
                            current["updated_at"] != metadata.updated_at
                            or current["confirmed_at"] != metadata.confirmed_at
                            or json.loads(current["document"]) != identity["metadata"]
                        )
                    ):
                        raise PreparationError(
                            "Метадані змінилися під час збереження. Повторіть попередній перегляд."
                        )
                    connection.execute(
                        "INSERT INTO preparation_plans VALUES (?, ?, ?, ?, ?)",
                        (
                            plan_id,
                            project_id,
                            dataset_id,
                            now,
                            json.dumps(document, ensure_ascii=False, allow_nan=False),
                        ),
                    )
                    audit.update(
                        status="success",
                        plan_id=plan_id,
                        duration_ms=(perf_counter() - started) * 1000,
                    )
                    ProjectStore._event(connection, project_id, operation, audit)
                return {**result, "plan_id": plan_id, "created_at": now}
            audit["status"] = "success"
            return result
        except (PreparationError, DatasetError, LookupError, OSError, ValueError) as error:
            message = (
                str(error)
                if isinstance(error, (PreparationError, DatasetError, LookupError))
                else "Не вдалося прочитати файл або метадані набору."
            )
            audit.update(status="error", error=message)
            raise PreparationError(message) from error
        finally:
            # The success event for a save is atomic with its recipe row.
            if not (expected_token and audit.get("status") == "success"):
                audit["duration_ms"] = (perf_counter() - started) * 1000
                with connect(self.datasets.database) as connection:
                    ProjectStore._event(connection, project_id, operation, audit)

    def plans(self, project_id: str) -> list[dict]:
        with connect(self.datasets.database) as connection:
            ProjectStore._get(connection, project_id)
            rows = connection.execute(
                "SELECT * FROM preparation_plans WHERE project_id = ? ORDER BY created_at DESC, id",
                (project_id,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "dataset_id": row["dataset_id"],
                "created_at": row["created_at"],
                "document": json.loads(row["document"]),
            }
            for row in rows
        ]
