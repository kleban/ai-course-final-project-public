"""Stage 02c: add an existing dataset (with its description and metadata) to another project.

The result is an independent copy: new ids, its own copy of the original file and of the raw
description files, its own metadata row. Later edits in one project never reach the other;
only `copied_from` keeps the source id for provenance.
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .datasets import Dataset, DatasetError, DatasetStore
from .projects import ProjectStore
from .storage import connect


@dataclass(frozen=True)
class Candidate:
    dataset: Dataset
    project_name: str
    metadata: str  # "confirmed" | "draft" | "none"
    already_here: bool  # the same file (SHA-256) is already in the target project


class DatasetCopier:
    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets
        self.database = datasets.database

    def candidates(self, project_id: str) -> list[Candidate]:
        with connect(self.database) as connection:
            ProjectStore._get(connection, project_id)
            names = {
                r["id"]: r["name"] for r in connection.execute("SELECT id, name FROM projects")
            }
            states = {
                r["dataset_id"]: "confirmed" if r["confirmed_at"] else "draft"
                for r in connection.execute("SELECT dataset_id, confirmed_at FROM dataset_metadata")
            }
        everything = self.datasets.list_all()
        here = {d.sha256 for d in everything if d.project_id == project_id}
        return [
            Candidate(d, names.get(d.project_id, ""), states.get(d.id, "none"), d.sha256 in here)
            for d in everything
            if d.project_id != project_id
        ]

    def copy(self, source_id: str, project_id: str) -> Dataset:
        source = self.datasets.get(source_id)
        if source.project_id == project_id:
            raise DatasetError("Цей набір уже належить поточному проєкту.")
        root = self.datasets.files_root
        source_file = (root / source.stored_path).resolve()
        if not source_file.is_relative_to(root) or not source_file.is_file():
            raise DatasetError("Файл вихідного набору відсутній у сховищі.")
        with connect(self.database) as connection:
            ProjectStore._get(connection, project_id)
            source_project = ProjectStore._get(connection, source.project_id)
            meta = connection.execute(
                "SELECT * FROM dataset_metadata WHERE dataset_id = ?", (source.id,)
            ).fetchone()
            sources = connection.execute(
                "SELECT * FROM metadata_sources WHERE dataset_id = ? ORDER BY created_at, id",
                (source.id,),
            ).fetchall()

        dataset_id = str(uuid4())
        folder = Path(project_id) / "datasets" / dataset_id
        target_dir = root / folder
        now = datetime.now(timezone.utc).isoformat()
        copied = Dataset(
            id=dataset_id,
            project_id=project_id,
            name=source.name,
            file_format=source.file_format,
            stored_path=(folder / Path(source.stored_path).name).as_posix(),
            size_bytes=source.size_bytes,
            sha256=source.sha256,
            row_count=source.row_count,
            column_count=source.column_count,
            profile=source.profile,
            created_at=now,
            copied_from=source.id,
        )
        target_dir.mkdir(parents=True, exist_ok=False)  # a fresh UUID folder, never reused
        reference = None
        try:
            frame = self.datasets.frame(source.id)
            shutil.copyfile(source_file, root / copied.stored_path)
            new_sources = []
            for row in sources:
                new_id = str(uuid4())
                stored = None
                if row["stored_path"]:
                    original = (root / row["stored_path"]).resolve()
                    if original.is_relative_to(root) and original.is_file():
                        stored = (folder / "metadata" / f"{new_id}{original.suffix}").as_posix()
                        (root / stored).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(original, root / stored)
                new_sources.append(
                    # original time keeps the order in which the descriptions were added
                    (
                        new_id,
                        dataset_id,
                        row["kind"],
                        row["name"],
                        stored,
                        row["text"],
                        row["created_at"],
                    )
                )
            reference = self.datasets.persist(copied, frame)
            with connect(self.database) as connection:
                ProjectStore._get(connection, project_id)
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        copied.id,
                        copied.project_id,
                        copied.name,
                        copied.file_format,
                        copied.stored_path,
                        copied.size_bytes,
                        copied.sha256,
                        copied.row_count,
                        copied.column_count,
                        json.dumps(copied.profile, ensure_ascii=False),
                        copied.created_at,
                        copied.copied_from,
                    ),
                )
                if meta is not None:
                    connection.execute(
                        "INSERT INTO dataset_metadata VALUES (?, ?, ?, ?)",
                        (dataset_id, meta["document"], meta["updated_at"], meta["confirmed_at"]),
                    )
                connection.executemany(
                    "INSERT INTO metadata_sources VALUES (?, ?, ?, ?, ?, ?, ?)", new_sources
                )
                ProjectStore._event(
                    connection,
                    project_id,
                    "dataset.copied",
                    {
                        "dataset_id": dataset_id,
                        "name": copied.name,
                        "rows": str(copied.row_count),
                        "columns": str(copied.column_count),
                        "source_dataset_id": source.id,
                        "source_project": source_project.name,
                        "metadata": "підтверджено"
                        if meta is not None and meta["confirmed_at"]
                        else ("чернетка" if meta is not None else "немає"),
                        "descriptions": str(len(new_sources)),
                    },
                )
        except BaseException:
            self.datasets.analytics.discard(reference)
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        return copied
