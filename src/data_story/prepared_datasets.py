"""Execute a confirmed recipe into an independent, traceable dataset exactly once."""

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import pandas as pd
from pandas.api import types as ptypes

from .datasets import MAX_BYTES, Dataset, DatasetError, DatasetStore, profile, read_table
from .metadata import MetadataStore, skeleton
from .preparation import MissingRule, PreparationError, preview_missing
from .preparation_store import digest
from .projects import ProjectStore
from .storage import connect


def snapshot(datasets: DatasetStore, project_id: str, dataset_id: str):
    dataset = datasets.get(dataset_id)
    if dataset.project_id != project_id:
        raise PreparationError("Набір не належить поточному проєкту.")
    path = (datasets.files_root / dataset.stored_path).resolve()
    if not path.is_relative_to(datasets.files_root) or not path.is_file():
        raise PreparationError("Файл набору даних відсутній у сховищі.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != dataset.sha256:
        raise PreparationError("Файл набору змінився або пошкоджений. Завантажте його заново.")
    frame = datasets.frame(dataset_id)
    metadata = MetadataStore(datasets.database, datasets).get(dataset_id)
    return dataset, frame, metadata


def csv_snapshot(frame: pd.DataFrame) -> tuple[bytes, dict]:
    """Preserve textual NA/empty/numeric strings, booleans and float precision on reload."""
    token = "__NA_" + uuid4().hex[:12] + "__"
    while any(frame[c].astype("string").eq(token).any() for c in frame.columns):
        token = "__NA_" + uuid4().hex[:12] + "__"
    dtypes = {
        c: str(frame[c].dtype)
        if ptypes.is_numeric_dtype(frame[c]) or ptypes.is_bool_dtype(frame[c])
        else "str"
        for c in frame.columns
    }
    content = frame.to_csv(index=False, na_rep=token).encode("utf-8")
    meta = {
        "encoding": "utf-8",
        "delimiter": ",",
        "prepared_csv": {"dtypes": dtypes, "na_token": token},
    }
    restored, _ = read_table(content, "csv", meta=meta)
    try:
        pd.testing.assert_frame_equal(
            frame.reset_index(drop=True), restored, check_dtype=False, check_exact=True
        )
    except AssertionError as error:
        raise PreparationError("Не вдалося зберегти таблицю без зміни значень.") from error
    return content, meta


class PreparedDatasets:
    def __init__(self, datasets: DatasetStore):
        self.datasets = datasets

    def create(self, project_id: str, plan_id: str) -> Dataset:
        ProjectStore(self.datasets.database).get(project_id)
        started = perf_counter()
        audit: dict = {"plan_id": plan_id}
        folder: Path | None = None
        reference = None
        try:
            with connect(self.datasets.database) as connection:
                connection.execute("BEGIN IMMEDIATE")
                plan = connection.execute(
                    "SELECT * FROM preparation_plans WHERE id = ? AND project_id = ?",
                    (plan_id, project_id),
                ).fetchone()
                if plan is None:
                    raise PreparationError("План не належить поточному проєкту або не знайдений.")
                existing = connection.execute(
                    "SELECT dataset_id FROM preparation_outputs WHERE plan_id = ?", (plan_id,)
                ).fetchone()
                if existing:
                    return self.datasets.get(existing["dataset_id"])
                document = json.loads(plan["document"])
                source, frame, metadata = snapshot(self.datasets, project_id, plan["dataset_id"])
                audit.update(dataset_id=source.id, dataset_name=source.name, sha256=source.sha256)
                identity = {
                    key: document[key]
                    for key in ("version", "project_id", "dataset_id", "sha256", "rules")
                }
                identity.update(
                    metadata=metadata.document.model_dump(mode="json"),
                    metadata_updated_at=metadata.updated_at,
                    confirmed_at=metadata.confirmed_at,
                )
                if (
                    source.sha256 != document["sha256"]
                    or digest(identity) != document["preview_token"]
                ):
                    raise PreparationError(
                        "Дані або метадані змінилися. Створіть новий план підготовки."
                    )
                prepared = preview_missing(
                    frame, metadata, [MissingRule.model_validate(r) for r in document["rules"]]
                )
                if prepared.retained_positions != document["retained_positions"]:
                    raise PreparationError("Результат не відповідає збереженому плану.")
                content, parser = csv_snapshot(prepared.frame)
                if len(content) > MAX_BYTES:
                    raise PreparationError("Підготовлений файл перевищує ліміт 5 МБ.")
                dataset_id = str(uuid4())
                relative = Path(project_id) / "datasets" / dataset_id / "original.csv"
                folder = self.datasets.files_root / relative.parent
                folder.mkdir(parents=True, exist_ok=False)
                (folder / "original.csv").write_bytes(content)
                now = datetime.now(timezone.utc).isoformat()
                details = {
                    **parser,
                    "columns": profile(prepared.frame),
                    "prepared_from": source.id,
                    "preparation_plan_id": plan_id,
                    "source_sha256": source.sha256,
                }
                dataset = Dataset(
                    dataset_id,
                    project_id,
                    (Path(source.name).stem[:160] + " · підготовлено.csv"),
                    "csv",
                    relative.as_posix(),
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                    len(prepared.frame),
                    len(prepared.frame.columns),
                    details,
                    now,
                )
                fresh = skeleton(prepared.frame, details)
                for target, original in zip(fresh.columns, metadata.document.columns):
                    for key in (
                        "label",
                        "description",
                        "kind",
                        "unit",
                        "computation",
                        "categories",
                        "range_min",
                        "range_max",
                        "missing_codes",
                        "role",
                        "source",
                    ):
                        setattr(target, key, getattr(original, key))
                fresh.dataset = metadata.document.dataset.model_copy(deep=True)
                reference = self.datasets.persist(dataset, prepared.frame.reset_index(drop=True))
                connection.execute(
                    "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        dataset.id,
                        project_id,
                        dataset.name,
                        "csv",
                        dataset.stored_path,
                        dataset.size_bytes,
                        dataset.sha256,
                        dataset.row_count,
                        dataset.column_count,
                        json.dumps(details, ensure_ascii=False),
                        now,
                        None,
                    ),
                )
                connection.execute(
                    "INSERT INTO dataset_metadata VALUES (?, ?, ?, ?)",
                    (dataset.id, fresh.model_dump_json(), now, None),
                )
                connection.execute(
                    "INSERT INTO preparation_outputs VALUES (?, ?)", (plan_id, dataset.id)
                )
                audit.update(
                    output_dataset_id=dataset.id,
                    rows_before=len(frame),
                    rows_after=len(prepared.frame),
                    rows_removed=prepared.report["rows_removed"],
                    status="success",
                    duration_ms=(perf_counter() - started) * 1000,
                )
                ProjectStore._event(connection, project_id, "preparation.applied", audit)
            return dataset
        except BaseException as error:
            self.datasets.analytics.discard(reference)
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)  # only this operation's fresh UUID folder
            if isinstance(error, (PreparationError, DatasetError, LookupError, OSError)):
                message = (
                    str(error)
                    if not isinstance(error, OSError)
                    else "Не вдалося зберегти підготовлений набір."
                )
                audit.update(
                    status="error", error=message, duration_ms=(perf_counter() - started) * 1000
                )
                with connect(self.datasets.database) as connection:
                    ProjectStore._event(connection, project_id, "preparation.applied", audit)
                raise PreparationError(message) from error
            raise
