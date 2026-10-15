"""Thin NiceGUI and HTTP adapters over the same project service."""

import logging
import os
import threading
import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from nicegui import app, ui

from .ai_settings import AiConfiguration
from .chart_configuration import ChartConfiguration
from .chart_screen import chart_settings_screen
from .combine import Recipe
from .config import Settings
from .contracts import (
    HTTP_STATUS,
    AnalyticalQuery,
    AssistantQuery,
    ConclusionCreate,
    ConclusionSave,
    DatasetCandidate,
    DatasetCopyRequest,
    DatasetPreview,
    DatasetView,
    DraftApply,
    EventView,
    Failure,
    HeatmapQuery,
    HeatmapView,
    HistogramQuery,
    HistogramView,
    InstructionUpdate,
    InstructionView,
    KnowledgeDocument,
    KnowledgeQuery,
    MetadataSourceView,
    MetadataText,
    MetadataUpdate,
    MetadataView,
    MissingQuery,
    MissingView,
    ModelCheckQuery,
    ModelTrainQuery,
    PreparationQuery,
    PreparationSave,
    ProjectCreate,
    ProjectDescription,
    ProjectTypeView,
    ProjectView,
    ReportCreate,
    ReportRevision,
    ReportSave,
    RowsPage,
    RowsQuery,
    ScatterQuery,
    ScatterView,
    StatisticsPage,
    StatisticsQuery,
    StepwiseQuery,
    Success,
    SummarySave,
    TrainingQuery,
)
from .datasets import MAX_BYTES
from .metadata import SOURCE_BYTES
from .projects import ProjectStore
from .screen import icons_screen, projects_screen, settings_screen
from .service import ProjectService, validation_failure


def router(service: ProjectService) -> APIRouter:
    api = APIRouter(prefix="/api")

    def response(result):
        status = HTTP_STATUS[result.error.code] if isinstance(result, Failure) else 200
        return JSONResponse(result.model_dump(mode="json"), status_code=status)

    @api.get("/projects/{project_id}/analytics", response_model=Success[list[dict]] | Failure)
    def analytics_catalog(project_id: str):
        return response(service.analytics_catalog(project_id))

    @api.get(
        "/projects/{project_id}/analytics/schema", response_model=Success[list[dict]] | Failure
    )
    def analytics_schema(project_id: str):
        return response(service.analytics_schema(project_id))

    @api.post("/projects/{project_id}/analytics/query", response_model=Success[dict] | Failure)
    def analytics_query(project_id: str, query: AnalyticalQuery):
        return response(service.analytics_query(project_id, query.model_dump()))

    @api.post(
        "/projects/{project_id}/analytics/datasets/{dataset_id}",
        response_model=Success[dict] | Failure,
    )
    def analytics_sync(project_id: str, dataset_id: str):
        return response(service.analytics_sync(project_id, dataset_id))

    @api.get("/projects/{project_id}/conclusions", response_model=Success[list[dict]] | Failure)
    def conclusion_list(project_id: str):
        return response(service.conclusion_list(project_id))

    @api.post("/projects/{project_id}/conclusions", response_model=Success[dict] | Failure)
    def conclusion_prepare(project_id: str, query: ConclusionCreate):
        return response(service.conclusion_prepare(project_id, query.model_dump(mode="json")))

    @api.get(
        "/projects/{project_id}/conclusions/{conclusion_id}",
        response_model=Success[dict] | Failure,
    )
    def conclusion_get(project_id: str, conclusion_id: str):
        return response(service.conclusion_get(project_id, conclusion_id))

    @api.put(
        "/projects/{project_id}/conclusions/{conclusion_id}",
        response_model=Success[dict] | Failure,
    )
    def conclusion_save(project_id: str, conclusion_id: str, query: ConclusionSave):
        return response(
            service.conclusion_save(project_id, conclusion_id, query.model_dump(mode="json"))
        )

    @api.post(
        "/projects/{project_id}/conclusions/{conclusion_id}/approve",
        response_model=Success[dict] | Failure,
    )
    def conclusion_approve(project_id: str, conclusion_id: str, query: ConclusionSave):
        return response(
            service.conclusion_save(
                project_id, conclusion_id, query.model_dump(mode="json"), approve=True
            )
        )

    @api.get("/projects/{project_id}/reports", response_model=Success[list[dict]] | Failure)
    def report_list(project_id: str):
        return response(service.report_list(project_id))

    @api.post("/projects/{project_id}/reports", response_model=Success[dict] | Failure)
    def report_create(project_id: str, query: ReportCreate):
        return response(service.report_create(project_id, query.model_dump(mode="json")))

    @api.get("/projects/{project_id}/reports/{report_id}", response_model=Success[dict] | Failure)
    def report_get(project_id: str, report_id: str):
        return response(service.report_get(project_id, report_id))

    @api.put("/projects/{project_id}/reports/{report_id}", response_model=Success[dict] | Failure)
    def report_save(project_id: str, report_id: str, query: ReportSave):
        return response(service.report_save(project_id, report_id, query.model_dump(mode="json")))

    @api.post(
        "/projects/{project_id}/reports/{report_id}/ai", response_model=Success[dict] | Failure
    )
    def report_ai(project_id: str, report_id: str, query: ReportRevision):
        return response(
            service.report_generate(project_id, report_id, query.model_dump(mode="json"), ai=True)
        )

    @api.post(
        "/projects/{project_id}/reports/{report_id}/render", response_model=Success[dict] | Failure
    )
    def report_render(project_id: str, report_id: str, query: ReportRevision):
        return response(
            service.report_generate(project_id, report_id, query.model_dump(mode="json"))
        )

    @api.get("/projects/{project_id}/reports/{report_id}/figures/{name}")
    def report_figure(project_id: str, report_id: str, name: str):
        result = service.report_figure(project_id, report_id, name)
        if isinstance(result, Failure):
            return response(result)
        return FileResponse(result.data, media_type="image/svg+xml")

    @api.get("/projects/{project_id}/reports/{report_id}/files/{artifact_id}/{kind}")
    def report_download(project_id: str, report_id: str, artifact_id: str, kind: str):
        result = service.report_download(project_id, report_id, artifact_id, kind)
        if isinstance(result, Failure):
            return response(result)
        if kind == "view":
            # Inline preview of the self-contained HTML (opened in the app and in a new tab).
            return FileResponse(result.data, media_type="text/html; charset=utf-8")
        return FileResponse(result.data, filename=result.data.name)

    @api.get(
        "/projects/{project_id}/summaries/fields/{dataset_id}",
        response_model=Success[list[dict]] | Failure,
    )
    def summary_fields(project_id: str, dataset_id: str):
        return response(service.summary_fields(project_id, dataset_id))

    @api.post("/projects/{project_id}/summaries/preview", response_model=Success[dict] | Failure)
    def summary_preview(project_id: str, query: Recipe):
        return response(service.summary_preview(project_id, query.model_dump(mode="json")))

    @api.post("/projects/{project_id}/summaries", response_model=Success[DatasetView] | Failure)
    def summary_create(project_id: str, query: SummarySave):
        return response(service.summary_create(project_id, query.model_dump(mode="json")))

    @api.post("/projects/{project_id}/preparation/preview", response_model=Success[dict] | Failure)
    def preparation_preview(project_id: str, query: PreparationQuery):
        return response(service.preparation_preview(project_id, query.model_dump(mode="json")))

    @api.post("/projects/{project_id}/preparation/plans", response_model=Success[dict] | Failure)
    def preparation_save(project_id: str, query: PreparationSave):
        return response(service.preparation_save(project_id, query.model_dump(mode="json")))

    @api.get(
        "/projects/{project_id}/preparation/plans", response_model=Success[list[dict]] | Failure
    )
    def preparation_plans(project_id: str):
        return response(service.preparation_plans(project_id))

    @api.post(
        "/projects/{project_id}/preparation/plans/{plan_id}/apply",
        response_model=Success[DatasetView] | Failure,
    )
    def preparation_apply(project_id: str, plan_id: str):
        return response(service.preparation_apply(project_id, plan_id))

    @api.post("/projects/{project_id}/training", response_model=Success[dict] | Failure)
    def training_prepare(project_id: str, query: TrainingQuery):
        return response(service.training_prepare(project_id, query.model_dump(mode="json")))

    @api.get("/projects/{project_id}/training", response_model=Success[list[dict]] | Failure)
    def training_runs(project_id: str):
        return response(service.training_runs(project_id))

    @api.post("/projects/{project_id}/models", response_model=Success[dict] | Failure)
    def model_train(project_id: str, query: ModelTrainQuery):
        return response(service.model_train(project_id, query.model_dump(mode="json")))

    @api.post("/projects/{project_id}/models/stepwise", response_model=Success[dict] | Failure)
    def model_stepwise(project_id: str, query: StepwiseQuery):
        return response(service.model_stepwise(project_id, query.model_dump(mode="json")))

    @api.get("/projects/{project_id}/models", response_model=Success[list[dict]] | Failure)
    def model_runs(project_id: str):
        return response(service.model_runs(project_id))

    @api.post("/projects/{project_id}/models/check", response_model=Success[dict] | Failure)
    def model_check(project_id: str, query: ModelCheckQuery):
        return response(service.model_check(project_id, query.model_dump(mode="json")))

    @api.get("/projects/{project_id}/models/{model_id}", response_model=Success[dict] | Failure)
    def model_run(project_id: str, model_id: str):
        return response(service.model_run(project_id, model_id))

    @api.get("/configuration/ai", response_model=Success[dict] | Failure)
    def ai_configuration():
        return response(service.ai_configuration())

    @api.put("/configuration/ai", response_model=Success[dict] | Failure)
    def save_ai_configuration(request: AiConfiguration):
        return response(service.save_ai_configuration(request.model_dump()))

    @api.post("/configuration/ai/reset", response_model=Success[dict] | Failure)
    def reset_ai_configuration():
        return response(service.reset_ai_configuration())

    @api.post("/projects/{project_id}/assistant", response_model=Success[dict] | Failure)
    def assistant_ask(project_id: str, query: AssistantQuery):
        return response(service.assistant_ask(project_id, query.model_dump(mode="json")))

    @api.post(
        "/projects/{project_id}/assistant/drafts/apply", response_model=Success[dict] | Failure
    )
    def assistant_apply_draft(project_id: str, query: DraftApply):
        return response(service.assistant_apply_draft(project_id, query.model_dump(mode="json")))

    @api.get("/monitoring", response_model=Success[dict] | Failure)
    def monitoring():
        return response(service.monitoring())

    @api.post("/configuration/ai/check", response_model=Success[list[dict]] | Failure)
    def ai_check_access():
        return response(service.ai_check_access())

    @api.post("/configuration/ai/test", response_model=Success[dict] | Failure)
    def ai_test_request():
        return response(service.ai_test_request())

    @api.get("/configuration/ai/calls", response_model=Success[dict] | Failure)
    def ai_calls():
        return response(service.ai_calls())

    @api.get("/knowledge/documents", response_model=Success[list[dict]] | Failure)
    def knowledge_documents():
        return response(service.knowledge_documents())

    @api.post("/knowledge/documents", response_model=Success[dict] | Failure)
    def knowledge_create(request: KnowledgeDocument):
        return response(service.knowledge_create(request.model_dump()))

    @api.put("/knowledge/documents/{document_id}", response_model=Success[dict] | Failure)
    def knowledge_save(document_id: str, request: KnowledgeDocument):
        return response(service.knowledge_save(document_id, request.model_dump()))

    @api.post("/knowledge/documents/{document_id}/reset", response_model=Success[dict] | Failure)
    def knowledge_reset(document_id: str):
        return response(service.knowledge_reset(document_id))

    @api.delete("/knowledge/documents/{document_id}", response_model=Success[dict] | Failure)
    def knowledge_delete(document_id: str):
        return response(service.knowledge_delete(document_id))

    @api.post("/knowledge/search", response_model=Success[dict] | Failure)
    def knowledge_search(query: KnowledgeQuery):
        return response(service.knowledge_search(query.model_dump()))

    @api.get("/configuration/charts", response_model=Success[ChartConfiguration] | Failure)
    def chart_configuration():
        return response(service.chart_configuration())

    @api.put("/configuration/charts", response_model=Success[ChartConfiguration] | Failure)
    def save_chart_configuration(request: ChartConfiguration):
        return response(service.save_chart_configuration(request.model_dump()))

    @api.post("/configuration/charts/reset", response_model=Success[ChartConfiguration] | Failure)
    def reset_chart_configuration():
        return response(service.reset_chart_configuration())

    @api.post(
        "/projects/{project_id}/charts/heatmap", response_model=Success[HeatmapView] | Failure
    )
    def heatmap(project_id: str, query: HeatmapQuery):
        return response(service.heatmap(project_id, query.model_dump(mode="json")))

    @api.post(
        "/projects/{project_id}/charts/missing", response_model=Success[MissingView] | Failure
    )
    def missing_chart(project_id: str, query: MissingQuery):
        return response(service.missing_chart(project_id, query.model_dump(mode="json")))

    @api.post(
        "/projects/{project_id}/charts/scatter", response_model=Success[ScatterView] | Failure
    )
    def scatter(project_id: str, query: ScatterQuery):
        return response(service.scatter(project_id, query.model_dump(mode="json")))

    @api.post(
        "/projects/{project_id}/charts/histogram", response_model=Success[HistogramView] | Failure
    )
    def histogram(project_id: str, query: HistogramQuery):
        return response(service.histogram(project_id, query.model_dump(mode="json")))

    @api.put("/projects/{project_id}/description", response_model=Success[ProjectView] | Failure)
    def save_description(project_id: str, request: ProjectDescription):
        return response(service.save_description(project_id, request.model_dump()))

    @api.get("/projects", response_model=Success[list[ProjectView]] | Failure)
    def list_projects():
        return response(service.list_projects())

    @api.post("/projects/{project_id}/statistics", response_model=Success[StatisticsPage] | Failure)
    def statistics(project_id: str, query: StatisticsQuery):
        return response(service.statistics(project_id, query.model_dump(mode="json")))

    @api.post(
        "/projects",
        response_model=Success[ProjectView] | Failure,
        responses={422: {"model": Failure}},
    )
    def create_project(request: ProjectCreate):
        return response(service.create(request))

    @api.get("/projects/{project_id}", response_model=Success[ProjectView] | Failure)
    def get_project(project_id: str):
        return response(service.get(project_id))

    @api.post("/projects/{project_id}/open", response_model=Success[ProjectView] | Failure)
    def open_project(project_id: str):
        return response(service.get(project_id, opened=True))

    @api.get("/projects/{project_id}/events", response_model=Success[list[EventView]] | Failure)
    def events(project_id: str):
        return response(service.events(project_id))

    @api.post(
        "/projects/{project_id}/datasets",
        response_model=Success[DatasetView] | Failure,
        responses={422: {"model": Failure}},
    )
    async def upload_dataset(project_id: str, file: UploadFile):
        content = await file.read(MAX_BYTES + 1)
        return response(service.upload(project_id, file.filename, content))

    @api.get("/projects/{project_id}/datasets", response_model=Success[list[DatasetView]] | Failure)
    def list_datasets(project_id: str):
        return response(service.list_datasets(project_id))

    @api.get("/datasets/{dataset_id}", response_model=Success[DatasetView] | Failure)
    def get_dataset(dataset_id: str):
        return response(service.dataset(dataset_id))

    @api.get(
        "/projects/{project_id}/dataset-candidates",
        response_model=Success[list[DatasetCandidate]] | Failure,
    )
    def dataset_candidates(project_id: str):
        return response(service.dataset_candidates(project_id))

    @api.post(
        "/projects/{project_id}/datasets/copy",
        response_model=Success[DatasetView] | Failure,
        responses={422: {"model": Failure}},
    )
    def copy_dataset(project_id: str, request: DatasetCopyRequest):
        return response(service.copy_dataset(project_id, request))

    @api.get("/datasets/{dataset_id}/preview", response_model=Success[DatasetPreview] | Failure)
    def preview_dataset(dataset_id: str):
        return response(service.preview(dataset_id))

    @api.post(
        "/datasets/{dataset_id}/rows",
        response_model=Success[RowsPage] | Failure,
        responses={422: {"model": Failure}},
    )
    def dataset_rows(dataset_id: str, request: RowsQuery):
        return response(service.rows(dataset_id, request))

    @api.get("/datasets/{dataset_id}/metadata", response_model=Success[MetadataView] | Failure)
    def get_metadata(dataset_id: str):
        return response(service.metadata(dataset_id))

    @api.put(
        "/datasets/{dataset_id}/metadata",
        response_model=Success[MetadataView] | Failure,
        responses={422: {"model": Failure}},
    )
    def save_metadata(dataset_id: str, request: MetadataUpdate):
        return response(service.save_metadata(dataset_id, request.model_dump()))

    @api.get(
        "/datasets/{dataset_id}/metadata/sources",
        response_model=Success[list[MetadataSourceView]] | Failure,
    )
    def metadata_sources(dataset_id: str):
        return response(service.metadata_sources(dataset_id))

    @api.post(
        "/datasets/{dataset_id}/metadata/sources/text",
        response_model=Success[MetadataSourceView] | Failure,
        responses={422: {"model": Failure}},
    )
    def add_metadata_text(dataset_id: str, request: MetadataText):
        return response(service.add_metadata_text(dataset_id, request.model_dump()))

    @api.post(
        "/datasets/{dataset_id}/metadata/sources/file",
        response_model=Success[MetadataSourceView] | Failure,
        responses={422: {"model": Failure}},
    )
    async def add_metadata_file(dataset_id: str, file: UploadFile):
        content = await file.read(SOURCE_BYTES + 1)
        return response(service.add_metadata_file(dataset_id, file.filename, content))

    @api.get("/project-types", response_model=Success[list[ProjectTypeView]] | Failure)
    def project_types():
        return response(service.project_types())

    @api.get(
        "/project-types/{project_type}/instruction",
        response_model=Success[InstructionView] | Failure,
    )
    def instruction(project_type: str):
        return response(service.instruction(project_type))

    @api.put(
        "/project-types/{project_type}/instruction",
        response_model=Success[InstructionView] | Failure,
        responses={422: {"model": Failure}},
    )
    def save_instruction(project_type: str, request: InstructionUpdate):
        return response(service.save_instruction(project_type, request.model_dump()))

    @api.delete(
        "/project-types/{project_type}/instruction",
        response_model=Success[InstructionView] | Failure,
    )
    def reset_instruction(project_type: str):
        return response(service.reset_instruction(project_type))

    return api


def install_error_handlers(target: FastAPI) -> None:
    @target.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError):
        result = validation_failure(error.errors())
        return JSONResponse(result.model_dump(mode="json"), status_code=422)


def configure(settings: Settings) -> None:
    service = ProjectService(ProjectStore(settings.database))
    app.include_router(router(service))
    install_error_handlers(app)
    assets = Path(__file__).parent / "assets"
    app.add_static_files("/assets", assets)

    @app.get("/health")
    def health():
        return {"application": "data-story", "instance": settings.instance, "stage": "statistics"}

    @app.post("/shutdown")
    async def shutdown(request: Request):
        if request.headers.get("X-Data-Story-Instance") != settings.instance:
            return JSONResponse({"message": "Це інший локальний екземпляр."}, status_code=409)
        app.shutdown()
        _exit_later()
        return {"status": "stopping"}

    @ui.page("/")
    def projects_page(request: Request):
        query = request.query_params
        projects_screen(
            service,
            query.get("project"),
            query.get("dataset"),
            query.get("view"),
            query.get("tool"),
            query.get("run"),
        )

    @ui.page("/settings")
    def settings_page(request: Request):
        settings_screen(service, request.query_params.get("type"))

    @ui.page("/settings/charts")
    def chart_settings_page():
        chart_settings_screen(service)

    @ui.page("/settings/ai")
    def ai_settings_page():
        from .ai_screen import ai_settings_screen

        ai_settings_screen(service)

    @ui.page("/settings/monitoring")
    def monitoring_page():
        from .monitoring_screen import monitoring_screen

        monitoring_screen(service)

    @ui.page("/settings/knowledge")
    def knowledge_page():
        from .knowledge_screen import knowledge_screen

        knowledge_screen(service)

    @ui.page("/instructions")
    def instructions_page():
        from .instructions_screen import instructions_screen

        instructions_screen(service)

    @ui.page("/icons")
    def icons_page():
        icons_screen(service)


SHUTDOWN_GRACE = 5.0


def _exit_later(seconds: float = SHUTDOWN_GRACE) -> None:
    """Fallback for /shutdown (history 132): after a browser session the port was freed, but
    the process lingered and kept the launch lock, so the next Start reported «already starting».
    SQLite commits every operation, so a hard exit after the grace period loses nothing."""

    def stop() -> None:
        time.sleep(seconds)
        logging.getLogger(__name__).warning("Shutdown took over %s s; process exits.", seconds)
        logging.shutdown()
        os._exit(0)

    threading.Thread(target=stop, name="shutdown-fallback", daemon=True).start()


def run(settings: Settings) -> None:
    configure(settings)
    mark = Path(__file__).parent / "assets" / "logo" / "logo-icon.svg"
    ui.add_head_html(
        '<link rel="apple-touch-icon" href="/assets/logo/apple-touch-icon.png">', shared=True
    )
    ui.run(
        host="127.0.0.1",
        port=settings.port,
        title="Data Story · Проєкти",
        reload=False,
        show=False,
        language="uk",
        favicon=mark if mark.is_file() else "📘",
    )
