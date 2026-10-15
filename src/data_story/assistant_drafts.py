"""Reviewable AI proposals; only an explicit apply saves through existing domain stores."""

import hashlib
import json
from uuid import UUID

from .contracts import (
    ChartDraftQuery,
    DraftApply,
    Failure,
    MetadataDraftQuery,
    ProjectDraftQuery,
    SqlSummaryDraft,
)
from .openai_client import AiError


class DraftError(AiError):
    """A draft is invalid, belongs elsewhere, or is stale."""


def _token(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def _dataset(service, project_id: str, dataset_id: str):
    dataset = service.datasets.get(dataset_id)
    if dataset.project_id != project_id:
        raise DraftError("Набір недоступний у цьому проєкті.")
    path = (service.datasets.files_root / dataset.stored_path).resolve()
    if not path.is_relative_to(service.datasets.files_root) or not path.is_file():
        raise DraftError("Файл набору недоступний.")
    if hashlib.sha256(path.read_bytes()).hexdigest() != dataset.sha256:
        raise DraftError("Файл набору змінився або пошкоджений.")
    return dataset


def _metadata_state(service, project_id: str, dataset_id: str):
    dataset = _dataset(service, project_id, dataset_id)
    metadata = service.metadata_store.get(dataset_id)
    state = {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "sha256": dataset.sha256,
        "document": metadata.document.model_dump(),
        "saved": metadata.saved,
        "updated_at": metadata.updated_at,
        "confirmed_at": metadata.confirmed_at,
        "sources": [s.id for s in service.metadata_store.sources(dataset_id)],
    }
    return metadata, _token(state), tuple(state["sources"])


def _patch_document(metadata, query: MetadataDraftQuery) -> dict:
    document = metadata.document.model_dump()
    known = {c["name"]: c for c in document["columns"]}
    seen = set()
    for patch in query.columns:
        if patch.name not in known or patch.name in seen:
            raise DraftError("Чернетка містить невідоме або повторене поле.")
        seen.add(patch.name)
        values = patch.model_dump(exclude={"name"}, exclude_none=True)
        if not values:
            raise DraftError("Чернетка поля не містить запропонованих змін.")
        known[patch.name].update(values, source="llm")
    if query.description is not None:
        document["dataset"]["description"] = query.description
    return document


def propose_project(service, project_id: str, raw: dict) -> dict:
    query = ProjectDraftQuery.model_validate(raw)
    project = service.store.get(project_id)
    return DraftApply(
        kind="project_description",
        project_id=UUID(project_id),
        base_token=_token(vars(project)),
        text=query.text,
    ).model_dump(mode="json")


def propose_metadata(service, project_id: str, raw: dict) -> dict:
    query = MetadataDraftQuery.model_validate(raw)
    metadata, token, _ = _metadata_state(service, project_id, str(query.dataset_id))
    _patch_document(metadata, query)
    return DraftApply(
        kind="metadata", project_id=UUID(project_id), base_token=token, metadata=query
    ).model_dump(mode="json")


def propose_chart(service, project_id: str, raw: dict) -> dict:
    query = ChartDraftQuery.model_validate(raw)
    dataset_id = str(query.query.dataset_id)
    metadata, token, _ = _metadata_state(service, project_id, dataset_id)
    eligible = {
        c.name
        for c in metadata.document.columns
        if c.kind == "number" and c.role not in ("id", "ignore")
    }
    values = query.query.model_dump(mode="json")
    fields = values.get(
        "fields",
        [values["field"]]
        if "field" in values
        else [values["x"], values["y"]]
        if "x" in values
        else [],
    )
    if len(fields) != len(set(fields)) or any(f not in eligible for f in fields):
        raise DraftError("Оберіть відомі придатні числові поля без повторів.")
    # Freeze the current local theme; proposals cannot change global settings.
    if query.query.options is None:
        query.query.options = service.chart_config.get().effective()
    return DraftApply(
        kind="chart", project_id=UUID(project_id), base_token=token, chart=query
    ).model_dump(mode="json")


def sql_token(service, project_id: str) -> str:
    return _token(service.analytical_queries.schema(project_id))


def propose_sql_summary(service, project_id: str, raw: dict) -> dict:
    query = SqlSummaryDraft.model_validate(raw)
    token = sql_token(service, project_id)
    service.analytical_queries.query(project_id, query.sql, query.limit)
    return DraftApply(
        kind="sql_summary", project_id=UUID(project_id), base_token=token, sql_summary=query
    ).model_dump(mode="json")


def apply(service, project_id: str, proposal: DraftApply) -> dict:
    if str(proposal.project_id) != project_id:
        raise DraftError("Чернетка належить іншому проєкту.")
    if proposal.kind == "sql_summary":
        if proposal.sql_summary is None or any(
            v is not None for v in (proposal.chart, proposal.metadata, proposal.text)
        ):
            raise DraftError("Пропозиція агрегації має містити лише параметри SQL.")
        if sql_token(service, project_id) != proposal.base_token:
            raise DraftError("Дані або метадані змінилися. Створіть нову пропозицію агрегації.")
        query = proposal.sql_summary
        dataset = service.analytical_queries.save_summary(
            project_id, query.sql, query.limit, query.name
        )
        return {"kind": "sql_summary", "dataset_id": dataset.id, "name": dataset.name}
    if proposal.sql_summary is not None:
        raise DraftError("Чернетка містить зайві параметри агрегації.")
    if proposal.kind == "chart":
        if proposal.chart is None or proposal.metadata is not None or proposal.text is not None:
            raise DraftError("Пропозиція графіка містить зайві або відсутні параметри.")
        chart = proposal.chart
        dataset_id = str(chart.query.dataset_id)
        _, token, _ = _metadata_state(service, project_id, dataset_id)
        if token != proposal.base_token:
            raise DraftError("Дані або метадані змінилися. Створіть нову пропозицію графіка.")
        propose_chart(service, project_id, chart.model_dump(mode="json"))
        method = (
            service.missing_chart if chart.chart == "missing" else getattr(service, chart.chart)
        )
        result = method(project_id, chart.query.model_dump(mode="json"))
        if isinstance(result, Failure):
            raise DraftError(result.error.message)
        return {
            "kind": "chart",
            "chart": chart.chart,
            "query": chart.query.model_dump(mode="json"),
            "result": result.data.model_dump(mode="json"),
        }
    if proposal.chart is not None:
        raise DraftError("Чернетка містить зайві параметри графіка.")
    if proposal.kind == "project_description":
        if not proposal.text or not proposal.text.strip() or proposal.metadata is not None:
            raise DraftError("Чернетка опису має містити лише текст.")
        project = service.store.get(project_id)
        if _token(vars(project)) != proposal.base_token:
            raise DraftError("Опис змінився після створення чернетки. Створіть нову чернетку.")
        try:
            saved = service.store.save_description(
                project_id, proposal.text, expected_description=project.description
            )
        except ValueError as error:
            raise DraftError(str(error)) from error
        return {"kind": proposal.kind, "text": saved.description}
    if proposal.metadata is None or proposal.text is not None:
        raise DraftError("Чернетка метаданих має містити лише опис полів.")
    dataset_id = str(proposal.metadata.dataset_id)
    metadata, token, sources = _metadata_state(service, project_id, dataset_id)
    if token != proposal.base_token:
        raise DraftError("Метадані або їхні джерела змінилися. Створіть нову чернетку.")
    document = _patch_document(metadata, proposal.metadata)
    saved_metadata = service.metadata_store.save(
        dataset_id,
        document,
        confirm=False,
        expected_version=(metadata.saved, metadata.updated_at),
        expected_sources=sources,
    )
    return {
        "kind": proposal.kind,
        "dataset_id": dataset_id,
        "metadata_confirmed": bool(saved_metadata.confirmed_at),
    }
