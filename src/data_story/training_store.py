"""Persist exact split positions and numeric matrices for the future model block."""

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
import sklearn

from .datasets import DatasetError, DatasetStore
from .preparation import PreparationError
from .prepared_datasets import snapshot
from .projects import ProjectStore
from .storage import connect
from .training_data import TrainingChoices, prepare_training


class TrainingStore:
    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets

    def create(self, project_id: str, dataset_id: str, choices: TrainingChoices) -> dict:
        ProjectStore(self.datasets.database).get(project_id)
        started = perf_counter()
        audit: dict = {
            "dataset_id": dataset_id,
            "target": choices.target,
            "fields": choices.features,
            "test_percent": choices.test_percent,
            "seed": choices.seed,
            "encoding": choices.encoding,
            "numeric_missing": choices.numeric_missing,
            "category_missing": choices.category_missing,
        }
        folder: Path | None = None
        try:
            with connect(self.datasets.database) as connection:
                connection.execute("BEGIN IMMEDIATE")
                dataset, frame, metadata = snapshot(self.datasets, project_id, dataset_id)
                audit.update(dataset_name=dataset.name, sha256=dataset.sha256)
                matrices = prepare_training(frame, metadata, choices)
                run_id = str(uuid4())
                relative = Path(project_id) / "training" / run_id / "matrices.npz"
                folder = self.datasets.files_root / relative.parent
                folder.mkdir(parents=True, exist_ok=False)
                path = self.datasets.files_root / relative
                np.savez_compressed(
                    path,
                    x_train=matrices.x_train,
                    x_test=matrices.x_test,
                    y_train=matrices.y_train,
                    y_test=matrices.y_test,
                )
                now = datetime.now(timezone.utc).isoformat()
                document = {
                    "version": 1,
                    "dataset_name": dataset.name,
                    "sha256": dataset.sha256,
                    "metadata": metadata.document.model_dump(mode="json"),
                    "metadata_updated_at": metadata.updated_at,
                    "choices": choices.model_dump(mode="json"),
                    "report": matrices.report,
                    "train_positions": matrices.train_positions,
                    "test_positions": matrices.test_positions,
                    "matrix_path": relative.as_posix(),
                    "matrix_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "sklearn_version": sklearn.__version__,
                }
                connection.execute(
                    "INSERT INTO training_runs VALUES (?, ?, ?, ?, ?)",
                    (
                        run_id,
                        project_id,
                        dataset_id,
                        now,
                        json.dumps(document, ensure_ascii=False, allow_nan=False),
                    ),
                )
                audit.update(
                    train_count=len(matrices.train_positions),
                    test_count=len(matrices.test_positions),
                    feature_count=matrices.report["feature_count"],
                    run_id=run_id,
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                )
                ProjectStore._event(connection, project_id, "training.prepared", audit)
            return {
                "id": run_id,
                "project_id": project_id,
                "dataset_id": dataset_id,
                "created_at": now,
                "document": document,
            }
        except BaseException as error:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            if isinstance(
                error, (PreparationError, DatasetError, LookupError, OSError, ValueError)
            ):
                message = (
                    str(error)
                    if isinstance(error, (PreparationError, DatasetError, LookupError))
                    else "Не вдалося підготувати й зберегти вибірки."
                )
                audit.update(
                    status="error", error=message, duration_ms=(perf_counter() - started) * 1000
                )
                with connect(self.datasets.database) as connection:
                    ProjectStore._event(connection, project_id, "training.prepared", audit)
                raise PreparationError(message) from error
            raise

    def list(self, project_id: str) -> list[dict]:
        with connect(self.datasets.database) as connection:
            ProjectStore._get(connection, project_id)
            rows = connection.execute(
                "SELECT * FROM training_runs WHERE project_id = ? ORDER BY created_at DESC, id",
                (project_id,),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "dataset_id": r["dataset_id"],
                "created_at": r["created_at"],
                "document": json.loads(r["document"]),
            }
            for r in rows
        ]

    def matrices(self, project_id: str, run_id: str) -> dict[str, np.ndarray]:
        with connect(self.datasets.database) as connection:
            row = connection.execute(
                "SELECT document FROM training_runs WHERE project_id = ? AND id = ?",
                (project_id, run_id),
            ).fetchone()
        if row is None:
            raise PreparationError("Підготовку вибірок не знайдено в цьому проєкті.")
        document = json.loads(row["document"])
        path = (self.datasets.files_root / document["matrix_path"]).resolve()
        if not path.is_relative_to(self.datasets.files_root) or not path.is_file():
            raise PreparationError("Файл підготовлених вибірок відсутній.")
        if hashlib.sha256(path.read_bytes()).hexdigest() != document["matrix_sha256"]:
            raise PreparationError("Файл підготовлених вибірок змінився або пошкоджений.")
        try:
            with np.load(path, allow_pickle=False) as archive:
                return {key: archive[key] for key in archive.files}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise PreparationError("Файл підготовлених вибірок пошкоджений.") from error

    def get(self, project_id: str, run_id: str) -> dict:
        with connect(self.datasets.database) as connection:
            ProjectStore._get(connection, project_id)
            row = connection.execute(
                "SELECT * FROM training_runs WHERE project_id = ? AND id = ?",
                (project_id, run_id),
            ).fetchone()
        if row is None:
            raise PreparationError("Підготовку вибірок не знайдено в цьому проєкті.")
        return {
            "id": row["id"],
            "dataset_id": row["dataset_id"],
            "created_at": row["created_at"],
            "document": json.loads(row["document"]),
        }
