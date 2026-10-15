"""Shared orchestration and error translation; adapters contain no domain logic."""

import logging
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .ai_settings import MODELS, STAGES, TASKS, AiConfiguration, AiSettingsStore
from .analytical_queries import AnalyticalQueries
from .analytics import AnalyticsError
from .chart_configuration import ChartConfiguration, ChartConfigurationStore
from .charts import (
    MISSING_PAGE,
    SMALL_SAMPLE,
    BinningError,
    contrast_warnings,
    diagnostic_svg,
    heatmap_svg,
    histogram_bins,
    histogram_svg,
    missing_order,
    missing_svg,
    scatter_svg,
)
from .combine import Recipe, SummaryError, SummaryStore
from .conclusions import ConclusionError, ConclusionStore
from .configuration import InstructionStore
from .contracts import (
    AnalyticalQuery,
    AssistantQuery,
    ConclusionCreate,
    ConclusionSave,
    DatasetCandidate,
    DatasetCopyRequest,
    DatasetPreview,
    DatasetView,
    DraftApply,
    ErrorInfo,
    EventView,
    Failure,
    FieldIssue,
    HeatmapQuery,
    HeatmapView,
    HistogramQuery,
    HistogramView,
    InstructionUpdate,
    InstructionView,
    KnowledgeDocument,
    KnowledgeQuery,
    MetadataIssue,
    MetadataSourceView,
    MetadataText,
    MetadataUpdate,
    MetadataView,
    MissingQuery,
    MissingRow,
    MissingView,
    ModelCheckQuery,
    ModelTrainQuery,
    PreparationQuery,
    PreparationSave,
    ProjectCreate,
    ProjectDescription,
    ProjectId,
    ProjectTypeView,
    ProjectView,
    ReportCreate,
    ReportRevision,
    ReportSave,
    RowsPage,
    RowsQuery,
    ScatterQuery,
    ScatterView,
    StatisticsField,
    StatisticsPage,
    StatisticsQuery,
    StatisticsRow,
    StepwiseQuery,
    Success,
    SummarySave,
    TrainingQuery,
)
from .dataset_copy import DatasetCopier
from .datasets import DatasetError, DatasetNotFound, DatasetStore
from .knowledge import KnowledgeError, KnowledgeStore
from .metadata import MetadataError, MetadataStore, issues
from .model import ModelError, ModelNotFound
from .model_store import ModelStore
from .openai_client import AiError, OpenAIClient
from .preparation import PreparationError
from .preparation_store import PreparationStore
from .prepared_datasets import PreparedDatasets
from .project_types import PROJECT_TYPES
from .projects import ProjectNotFound, ProjectStore
from .reports import ReportError, ReportStore
from .statistics import StatisticsError, StatisticsStore
from .training_data import TrainingChoices
from .training_store import TrainingStore

T = TypeVar("T")
SchemaT = TypeVar("SchemaT", bound=BaseModel)
LOG = logging.getLogger(__name__)


class UnknownProjectType(LookupError):
    """A requested project type does not exist."""


class InputValidation(Exception):
    def __init__(self, error: ValidationError):
        self.error = error


def validate_input(schema: type[SchemaT], raw: object) -> SchemaT:
    try:
        return schema.model_validate(raw)
    except ValidationError as error:
        raise InputValidation(error) from error


def validation_failure(errors: Sequence[Mapping[str, Any]]) -> Failure:
    issues = [
        FieldIssue(
            field=".".join(map(str, item["loc"])),
            message=str(item["msg"]).removeprefix("Value error, "),
        )
        for item in errors
    ]
    return Failure(
        error=ErrorInfo(code="invalid_input", message="Перевірте введені дані.", fields=issues)
    )


def execute(action: Callable[[], T]) -> Success[T] | Failure:
    try:
        return Success(data=action())
    except InputValidation as error:
        return validation_failure(error.error.errors(include_input=False))
    except (
        DatasetError,
        MetadataError,
        StatisticsError,
        PreparationError,
        ModelError,
        KnowledgeError,
        AiError,
        ReportError,
        ConclusionError,
        SummaryError,
        AnalyticsError,
    ) as error:
        return Failure(
            error=ErrorInfo(
                code="invalid_input",
                message=str(error),
                fields=[FieldIssue(field="file", message=str(error))],
            )
        )
    except (ProjectNotFound, UnknownProjectType, DatasetNotFound, ModelNotFound) as error:
        return Failure(error=ErrorInfo(code="not_found", message=str(error)))
    except sqlite3.Error:
        LOG.error("Project operation failed: storage_error")
        return Failure(
            error=ErrorInfo(
                code="storage_error",
                message="Не вдалося зберегти або прочитати дані. Спробуйте ще раз.",
            )
        )
    except Exception:
        LOG.error("Project operation failed: internal_error")
        return Failure(
            error=ErrorInfo(
                code="internal_error",
                message="Не вдалося виконати дію. Дані не показані як успішно змінені.",
            )
        )


class ProjectService:
    def __init__(
        self,
        store: ProjectStore,
        instructions: InstructionStore | None = None,
        datasets: DatasetStore | None = None,
    ):
        self.store = store
        self.instructions = instructions or InstructionStore(store.database)
        self.datasets = datasets or DatasetStore(store.database)
        self.analytics = self.datasets.analytics
        self.analytical_queries = AnalyticalQueries(self)
        self.metadata_store = MetadataStore(store.database, self.datasets)
        self.copier = DatasetCopier(self.datasets)
        self.statistics_store = StatisticsStore(self.datasets)
        self.chart_config = ChartConfigurationStore(store.database)
        self.preparation_store = PreparationStore(self.datasets)
        self.prepared_datasets = PreparedDatasets(self.datasets)
        self.training_store = TrainingStore(self.datasets)
        self.model_store = ModelStore(self.training_store)
        self.knowledge = KnowledgeStore(store.database)
        self.summaries = SummaryStore(self.datasets)
        self.ai_settings = AiSettingsStore(store.database)
        self.openai = OpenAIClient(store.database, self.ai_settings)
        self.reports = ReportStore(self)
        self.conclusions = ConclusionStore(self)

    def analytics_catalog(self, project_id: str) -> Success[list[dict]] | Failure:
        def action() -> list[dict]:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            self.store.get(validated)
            return self.analytics.catalog(validated)

        return execute(action)

    def analytics_schema(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.analytical_queries.schema(
                str(validate_input(ProjectId, {"id": project_id}).id)
            )
        )

    def analytics_query(self, project_id: str, raw: object) -> Success[dict] | Failure:
        def action() -> dict:
            project = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(AnalyticalQuery, raw)
            return self.analytical_queries.query(project, query.sql, query.limit)

        return execute(action)

    def analytics_sync(self, project_id: str, dataset_id: str) -> Success[dict] | Failure:
        """Explicit first-stage import; verified CSV/metadata and old analytical flows remain intact."""
        from .prepared_datasets import snapshot
        from .storage import connect

        def action() -> dict:
            project = str(validate_input(ProjectId, {"id": project_id}).id)
            dataset_key = str(validate_input(ProjectId, {"id": dataset_id}).id)
            self.store.get(project)
            audit: dict = {"dataset_id": dataset_key, "status": "error"}
            try:
                dataset, frame, _metadata = snapshot(self.datasets, project, dataset_key)
                result = self.datasets.persist(dataset, frame)
                audit.update(
                    status="success",
                    fingerprint=result["fingerprint"],
                    row_count=result["row_count"],
                    reused=result["reused"],
                )
                return result
            except (AnalyticsError, PreparationError, DatasetError, LookupError) as error:
                audit["error"] = str(error)
                raise
            finally:
                with connect(self.store.database) as connection:
                    ProjectStore._event(connection, project, "analytics.synced", audit)

        return execute(action)

    def conclusion_list(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.conclusions.project_list(
                str(validate_input(ProjectId, {"id": project_id}).id)
            )
        )

    def conclusion_summaries(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.conclusions.summaries(
                str(validate_input(ProjectId, {"id": project_id}).id)
            )
        )

    def conclusion_get(self, project_id: str, conclusion_id: str) -> Success[dict] | Failure:
        return execute(
            lambda: self.conclusions.get(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": conclusion_id}).id),
            )
        )

    def conclusion_prepare(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """Explicit AI action: facts of a saved model + cited knowledge fragments → draft."""

        def action():
            project = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(ConclusionCreate, raw)
            return self.conclusions.prepare(
                project, str(query.model_id), [str(s) for s in query.summary_ids]
            )

        return execute(action)

    def conclusion_save(
        self, project_id: str, conclusion_id: str, raw: object, *, approve: bool = False
    ) -> Success[dict] | Failure:
        def action():
            query = validate_input(ConclusionSave, raw)
            method = self.conclusions.approve if approve else self.conclusions.save
            return method(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": conclusion_id}).id),
                query.revision,
                query.markdown,
            )

        return execute(action)

    def report_create(self, project_id: str, raw: object) -> Success[dict] | Failure:
        def action():
            project = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(ReportCreate, raw)
            return self.reports.create(project, str(query.model_id), query.title)

        return execute(action)

    def report_list(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.reports.list(str(validate_input(ProjectId, {"id": project_id}).id))
        )

    def report_get(self, project_id: str, report_id: str) -> Success[dict] | Failure:
        return execute(
            lambda: self.reports.get(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": report_id}).id),
            )
        )

    def report_save(self, project_id: str, report_id: str, raw: object) -> Success[dict] | Failure:
        def action():
            query = validate_input(ReportSave, raw)
            return self.reports.update(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": report_id}).id),
                query.revision,
                query.markdown,
            )

        return execute(action)

    def report_generate(
        self, project_id: str, report_id: str, raw: object, *, ai: bool = False
    ) -> Success[dict] | Failure:
        def action():
            query = validate_input(ReportRevision, raw)
            method = self.reports.ai_draft if ai else self.reports.render
            return method(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": report_id}).id),
                query.revision,
            )

        return execute(action)

    def report_figure(self, project_id: str, report_id: str, name: str):
        return execute(
            lambda: self.reports.figure(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": report_id}).id),
                name,
            )
        )

    def report_download(self, project_id: str, report_id: str, artifact_id: str, kind: str):
        return execute(
            lambda: self.reports.download(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": report_id}).id),
                str(validate_input(ProjectId, {"id": artifact_id}).id),
                kind,
            )
        )

    def summary_fields(self, project_id: str, dataset_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.summaries.fields(
                str(validate_input(ProjectId, {"id": project_id}).id),
                str(validate_input(ProjectId, {"id": dataset_id}).id),
            )
        )

    def summary_preview(self, project_id: str, raw: object) -> Success[dict] | Failure:
        return execute(
            lambda: self.summaries.preview(
                str(validate_input(ProjectId, {"id": project_id}).id), validate_input(Recipe, raw)
            )
        )

    def summary_create(self, project_id: str, raw: object) -> Success[DatasetView] | Failure:
        def action() -> DatasetView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(SummarySave, raw)
            return DatasetView.from_dataset(
                self.summaries.create(
                    validated, request.recipe, request.preview_token, request.name
                )
            )

        return execute(action)

    def preparation_preview(self, project_id: str, raw: object) -> Success[dict] | Failure:
        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(PreparationQuery, raw)
            return self.preparation_store.preview(validated, str(request.dataset_id), request.rules)

        return execute(action)

    def preparation_save(self, project_id: str, raw: object) -> Success[dict] | Failure:
        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(PreparationSave, raw)
            return self.preparation_store.preview(
                validated,
                str(request.dataset_id),
                request.rules,
                expected_token=request.preview_token,
            )

        return execute(action)

    def preparation_plans(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.preparation_store.plans(
                str(validate_input(ProjectId, {"id": project_id}).id)
            )
        )

    def preparation_apply(self, project_id: str, plan_id: str) -> Success[DatasetView] | Failure:
        def action() -> DatasetView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            plan = str(validate_input(ProjectId, {"id": plan_id}).id)
            return DatasetView.from_dataset(self.prepared_datasets.create(validated, plan))

        return execute(action)

    def training_prepare(self, project_id: str, raw: object) -> Success[dict] | Failure:
        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(TrainingQuery, raw)
            choices = TrainingChoices.model_validate(request.model_dump(exclude={"dataset_id"}))
            return self.training_store.create(validated, str(request.dataset_id), choices)

        return execute(action)

    def training_runs(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.training_store.list(str(validate_input(ProjectId, {"id": project_id}).id))
        )

    def model_train(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """Fit LinearRegression on one explicitly chosen saved split (train part only)."""

        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(ModelTrainQuery, raw)
            return self.model_store.train(validated, str(request.training_run_id), request.features)

        return execute(action)

    def model_stepwise(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """Stepwise choice of fields on the train part, then fit and save that model."""

        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(StepwiseQuery, raw)
            return self.model_store.stepwise(
                validated, str(request.training_run_id), request.criterion
            )

        return execute(action)

    def model_check(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """One model check chart (regression line, errors or prediction vs actual) on test rows."""

        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(ModelCheckQuery, raw)
            configuration = self.chart_config.get()
            options = query.options or configuration.effective()
            view = self.model_store.check(
                validated,
                str(query.model_id),
                query.chart,
                query.feature,
                {"theme": configuration.theme, "temporary_options": query.options is not None},
            )
            document = view.pop("document")
            view["svg"] = diagnostic_svg(view, options, query.title)
            view["warnings"] = contrast_warnings(options)
            view["coefficients"] = document["coefficients"]
            view["intercept"] = document["intercept"]
            view["theme"] = configuration.theme
            view.pop("points")
            return view

        return execute(action)

    def ai_configuration(self) -> Success[dict] | Failure:
        """Task → OpenAI model table, limits, catalogue and the masked key state from .env."""

        def action() -> dict:
            return {
                "configuration": self.ai_settings.get().model_dump(),
                "key": self.ai_settings.key(),
                "stages": STAGES,
                "tasks": [vars(task) for task in TASKS.values()],
                "models": [vars(model) for model in MODELS.values()],
            }

        return execute(action)

    def save_ai_configuration(self, raw: object) -> Success[dict] | Failure:
        def action() -> dict:
            return self.ai_settings.save(validate_input(AiConfiguration, raw)).model_dump()

        return execute(action)

    def reset_ai_configuration(self) -> Success[dict] | Failure:
        return execute(lambda: self.ai_settings.reset().model_dump())

    def assistant_ask(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """AI assistant in a project: advise, execute with tools, or explain (explicit call)."""
        from .assistant import Assistant

        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(AssistantQuery, raw)
            page = {k: str(v) for k, v in query.page.model_dump().items() if v is not None}
            history = [turn.model_dump() for turn in query.history]
            return Assistant(self).ask(
                validated, query.block, query.mode, query.question, page, history
            )

        return execute(action)

    def assistant_apply_draft(self, project_id: str, raw: object) -> Success[dict] | Failure:
        """Explicit user action; no OpenAI call and no automatic metadata confirmation."""
        from .assistant_drafts import apply

        return execute(
            lambda: apply(
                self,
                str(validate_input(ProjectId, {"id": project_id}).id),
                validate_input(DraftApply, raw),
            )
        )

    def monitoring(self) -> Success[dict] | Failure:
        """Local health of every part of the service (08b); no OpenAI call."""
        from .monitoring import Monitor

        return execute(lambda: Monitor(self).run())

    def ai_check_access(self) -> Success[list[dict]] | Failure:
        """Free check: are the configured OpenAI models reachable with the key from .env."""
        return execute(self.openai.check_models)

    def ai_test_request(self) -> Success[dict] | Failure:
        """One tiny paid request, only on an explicit click, to prove the whole path works."""
        return execute(
            lambda: self.openai.respond(
                "eda_advice",
                "Відповідай українською одним коротким реченням.",
                "Підтверди, що з'єднання з Data Story працює.",
                max_output_tokens=256,
            )
        )

    def ai_calls(self) -> Success[dict] | Failure:
        return execute(
            lambda: {
                "calls": self.openai.calls(),
                "totals": self.openai.totals(),
                "daily": self.openai.daily(),
            }
        )

    def knowledge_documents(self) -> Success[list[dict]] | Failure:
        return execute(self.knowledge.documents)

    def knowledge_save(self, document_id: str, raw: object) -> Success[dict] | Failure:
        return execute(
            lambda: self.knowledge.save(document_id, validate_input(KnowledgeDocument, raw).text)
        )

    def knowledge_create(self, raw: object) -> Success[dict] | Failure:
        return execute(lambda: self.knowledge.create(validate_input(KnowledgeDocument, raw).text))

    def knowledge_reset(self, document_id: str) -> Success[dict] | Failure:
        return execute(lambda: self.knowledge.reset(document_id))

    def knowledge_delete(self, document_id: str) -> Success[dict] | Failure:
        def action() -> dict:
            self.knowledge.delete(document_id)
            return {"deleted": document_id}

        return execute(action)

    def knowledge_search(self, raw: object) -> Success[dict] | Failure:
        """Local RAG retrieval: the most similar rule fragments with source and section."""

        def action() -> dict:
            query = validate_input(KnowledgeQuery, raw)
            return self.knowledge.search(query.query, query.limit)

        return execute(action)

    def knowledge_events(self) -> Success[list[dict]] | Failure:
        return execute(self.knowledge.events)

    def model_runs(self, project_id: str) -> Success[list[dict]] | Failure:
        return execute(
            lambda: self.model_store.list(str(validate_input(ProjectId, {"id": project_id}).id))
        )

    def model_run(self, project_id: str, model_id: str) -> Success[dict] | Failure:
        def action() -> dict:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            model = str(validate_input(ProjectId, {"id": model_id}).id)
            return self.model_store.get(validated, model)

        return execute(action)

    def create(self, raw: object) -> Success[ProjectView] | Failure:
        def action() -> ProjectView:
            request = validate_input(ProjectCreate, raw)
            project = self.store.create(request.name, request.purpose, request.project_type)
            return ProjectView.model_validate(project)

        return execute(action)

    def list_projects(self) -> Success[list[ProjectView]] | Failure:
        return execute(
            lambda: [ProjectView.model_validate(project) for project in self.store.list_projects()]
        )

    def get(self, project_id: str, *, opened: bool = False) -> Success[ProjectView] | Failure:
        def action() -> ProjectView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            project = self.store.open(validated) if opened else self.store.get(validated)
            return ProjectView.model_validate(project)

        return execute(action)

    def save_description(self, project_id: str, raw: object) -> Success[ProjectView] | Failure:
        def action() -> ProjectView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(ProjectDescription, raw)
            return ProjectView.model_validate(self.store.save_description(validated, request.text))

        return execute(action)

    def events(self, project_id: str) -> Success[list[EventView]] | Failure:
        def action() -> list[EventView]:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            return [EventView.model_validate(event) for event in self.store.events(validated)]

        return execute(action)

    def project_types(self) -> Success[list[ProjectTypeView]] | Failure:
        return execute(
            lambda: [ProjectTypeView.model_validate(item) for item in PROJECT_TYPES.values()]
        )

    def instruction(self, project_type: str) -> Success[InstructionView] | Failure:
        def action() -> InstructionView:
            return InstructionView.model_validate(self.instructions.get(_known(project_type)))

        return execute(action)

    def save_instruction(
        self, project_type: str, raw: object
    ) -> Success[InstructionView] | Failure:
        def action() -> InstructionView:
            key = _known(project_type)
            request = validate_input(InstructionUpdate, raw)
            return InstructionView.model_validate(self.instructions.save(key, request.text))

        return execute(action)

    def reset_instruction(self, project_type: str) -> Success[InstructionView] | Failure:
        return execute(
            lambda: InstructionView.model_validate(self.instructions.reset(_known(project_type)))
        )

    def upload(
        self, project_id: str, filename: object, content: bytes
    ) -> Success[DatasetView] | Failure:
        def action() -> DatasetView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            return DatasetView.from_dataset(self.datasets.add(validated, filename, content))

        return execute(action)

    def list_datasets(self, project_id: str) -> Success[list[DatasetView]] | Failure:
        def action() -> list[DatasetView]:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            return [DatasetView.from_dataset(d) for d in self.datasets.list_datasets(validated)]

        return execute(action)

    def dataset_candidates(self, project_id: str) -> Success[list[DatasetCandidate]] | Failure:
        def action() -> list[DatasetCandidate]:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            return [DatasetCandidate.from_candidate(c) for c in self.copier.candidates(validated)]

        return execute(action)

    def copy_dataset(self, project_id: str, raw: object) -> Success[DatasetView] | Failure:
        def action() -> DatasetView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            request = validate_input(DatasetCopyRequest, raw)
            return DatasetView.from_dataset(self.copier.copy(str(request.dataset_id), validated))

        return execute(action)

    def dataset(self, dataset_id: str) -> Success[DatasetView] | Failure:
        def action() -> DatasetView:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            return DatasetView.from_dataset(self.datasets.get(validated))

        return execute(action)

    def preview(self, dataset_id: str) -> Success[DatasetPreview] | Failure:
        def action() -> DatasetPreview:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            return DatasetPreview.model_validate(self.datasets.preview(validated))

        return execute(action)

    def rows(self, dataset_id: str, raw: object) -> Success[RowsPage] | Failure:
        def action() -> RowsPage:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            query = validate_input(RowsQuery, raw)
            page = self.datasets.page(
                validated,
                query.start,
                query.limit,
                [item.model_dump() for item in query.sort],
                query.filter,
            )
            return RowsPage.model_validate(page)

        return execute(action)

    def _metadata_view(self, item) -> MetadataView:
        return MetadataView(
            dataset_id=item.dataset_id,
            document=item.document,
            saved=item.saved,
            updated_at=item.updated_at,
            confirmed_at=item.confirmed_at,
            issues=[
                MetadataIssue(column=i.column, level=i.level, message=i.message)
                for i in issues(item.document)
            ],
        )

    def statistics(self, project_id: str, raw: object) -> Success[StatisticsPage] | Failure:
        def action() -> StatisticsPage:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(StatisticsQuery, raw)
            result = self.statistics_store.calculate(
                validated, str(query.dataset_id), query.view, query.fields
            )
            descriptors = {
                c["name"]: {key: c[key] for key in ("name", "label", "unit", "kind", "role")}
                for c in result["columns"]
            }
            names = result.get("fields", list(descriptors))
            rows = []
            for index in range(query.start, min(query.start + query.limit, len(names))):
                name = names[index]
                column = next(c for c in result["columns"] if c["name"] == name)
                values, counts, reasons = {}, {}, {}
                if query.view == "correlations":
                    values = dict(zip(names, result["r"][index])) if result["r"] else {}
                    counts = dict(zip(names, result["n"][index])) if result["n"] else {}
                    reasons = (
                        {n: reason for n, reason in zip(names, result["reasons"][index]) if reason}
                        if result["reasons"]
                        else {}
                    )
                elif query.view == "missing":
                    values = {
                        key: column[key]
                        for key in (
                            "base_missing",
                            "additional_missing",
                            "missing",
                            "missing_percent",
                            "invalid_infinite",
                        )
                    }
                else:
                    summary = column["statistics"] or {}
                    values = {
                        key: value
                        for key, value in summary.items()
                        if not key.endswith("reason") and key != "reasons"
                    }
                    reasons = summary.get("reasons", {})
                    if summary.get("empty_reason"):
                        reasons["empty"] = summary["empty_reason"]
                rows.append(
                    StatisticsRow(
                        **descriptors[name],
                        values=values,
                        counts=counts,
                        reasons=reasons,
                        reason=column["reason"],
                        warning=" ".join(
                            [
                                column.get("warning") or "",
                                *[
                                    issue["message"]
                                    for issue in result["issues"]
                                    if issue["column"] == name
                                ],
                            ]
                        ).strip()
                        or None,
                    )
                )
            selected = result.get("fields", [])
            excluded = result.get("excluded", {})
            return StatisticsPage(
                **{
                    key: result[key]
                    for key in (
                        "project_id",
                        "dataset_id",
                        "sha256",
                        "view",
                        "method",
                        "row_count",
                        "metadata_state",
                        "metadata_updated_at",
                        "warnings",
                    )
                },
                rows=rows,
                total=len(names),
                start=query.start,
                limit=query.limit,
                fields=[StatisticsField.model_validate(descriptors[name]) for name in selected],
                eligible=[
                    StatisticsField.model_validate(d)
                    for name, d in descriptors.items()
                    if name not in excluded
                ]
                if query.view == "correlations"
                else [],
                excluded=excluded,
                rows_with_missing=result.get("rows_with_missing"),
                rows_with_missing_percent=result.get("rows_with_missing_percent"),
                reason=result.get("reason"),
            )

        return execute(action)

    def metadata(self, dataset_id: str) -> Success[MetadataView] | Failure:
        def action() -> MetadataView:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            return self._metadata_view(self.metadata_store.get(validated))

        return execute(action)

    def save_metadata(self, dataset_id: str, raw: object) -> Success[MetadataView] | Failure:
        def action() -> MetadataView:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            request = validate_input(MetadataUpdate, raw)
            saved = self.metadata_store.save(
                validated, request.document.model_dump(), confirm=request.confirm
            )
            return self._metadata_view(saved)

        return execute(action)

    def add_metadata_text(
        self, dataset_id: str, raw: object
    ) -> Success[MetadataSourceView] | Failure:
        def action() -> MetadataSourceView:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            request = validate_input(MetadataText, raw)
            source = self.metadata_store.add_text(validated, request.text)
            return MetadataSourceView.model_validate(source)

        return execute(action)

    def add_metadata_file(
        self, dataset_id: str, filename: object, content: bytes
    ) -> Success[MetadataSourceView] | Failure:
        def action() -> MetadataSourceView:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            source = self.metadata_store.add_file(validated, filename, content)
            return MetadataSourceView.model_validate(source)

        return execute(action)

    def metadata_sources(self, dataset_id: str) -> Success[list[MetadataSourceView]] | Failure:
        def action() -> list[MetadataSourceView]:
            validated = str(validate_input(ProjectId, {"id": dataset_id}).id)
            return [
                MetadataSourceView.model_validate(s) for s in self.metadata_store.sources(validated)
            ]

        return execute(action)

    def chart_configuration(self) -> Success[ChartConfiguration] | Failure:
        return execute(self.chart_config.get)

    def save_chart_configuration(self, raw: object) -> Success[ChartConfiguration] | Failure:
        return execute(lambda: self.chart_config.save(validate_input(ChartConfiguration, raw)))

    def reset_chart_configuration(self) -> Success[ChartConfiguration] | Failure:
        return execute(self.chart_config.reset)

    def chart_configuration_events(self) -> Success[list[dict]] | Failure:
        return execute(self.chart_config.events)

    def heatmap(self, project_id: str, raw: object) -> Success[HeatmapView] | Failure:
        def action() -> HeatmapView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(HeatmapQuery, raw)
            configuration = self.chart_config.get()
            options = query.options or configuration.effective()
            result = self.statistics_store.calculate(
                validated,
                str(query.dataset_id),
                "correlations",
                query.fields,
                operation="chart.built",
                audit_extra={
                    "chart": "heatmap",
                    "theme": configuration.theme,
                    "temporary_options": query.options is not None,
                    "options": options.model_dump(),
                    "title": query.title,
                },
            )
            descriptors = {
                c["name"]: {k: c[k] for k in ("name", "label", "unit", "kind", "role")}
                for c in result["columns"]
            }
            return HeatmapView(
                **{
                    k: result[k]
                    for k in (
                        "dataset_id",
                        "project_id",
                        "sha256",
                        "row_count",
                        "metadata_state",
                        "r",
                        "n",
                        "reasons",
                        "excluded",
                    )
                },
                fields=[StatisticsField(**descriptors[n]) for n in result["fields"]],
                warnings=result["warnings"] + contrast_warnings(options),
                theme=configuration.theme,
                options=options,
                svg=heatmap_svg(result, options, query.title),
            )

        return execute(action)

    def scatter(self, project_id: str, raw: object) -> Success[ScatterView] | Failure:
        def action() -> ScatterView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(ScatterQuery, raw)
            configuration = self.chart_config.get()
            options = query.options or configuration.effective()
            result = self.statistics_store.calculate(
                validated,
                str(query.dataset_id),
                "correlations",
                [query.x, query.y],
                with_pairs=True,
                operation="chart.built",
                audit_extra={
                    "chart": "scatter",
                    "theme": configuration.theme,
                    "temporary_options": query.options is not None,
                    "options": options.model_dump(),
                    "title": query.title,
                },
            )
            pairs = result.pop("pairs", [])
            x, y = _descriptor(result, query.x), _descriptor(result, query.y)
            r, reason = result["r"][1][0], result["reasons"][1][0]
            if not pairs:
                reason = "Немає жодного рядка з обома значеннями: графік не будується."
            return ScatterView(
                **{
                    k: result[k]
                    for k in ("dataset_id", "project_id", "sha256", "row_count", "metadata_state")
                },
                warnings=result["warnings"] + contrast_warnings(options),
                x=StatisticsField(**x),
                y=StatisticsField(**y),
                n=len(pairs),
                r=r,
                reason=reason,
                x_range=[min(p[0] for p in pairs), max(p[0] for p in pairs)] if pairs else [],
                y_range=[min(p[1] for p in pairs), max(p[1] for p in pairs)] if pairs else [],
                theme=configuration.theme,
                options=options,
                svg=scatter_svg(pairs, x, y, r, options, query.title) if pairs else "",
            )

        return execute(action)

    def missing_chart(self, project_id: str, raw: object) -> Success[MissingView] | Failure:
        def action() -> MissingView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(MissingQuery, raw)
            configuration = self.chart_config.get()
            options = query.options or configuration.effective()
            result = self.statistics_store.calculate(
                validated,
                str(query.dataset_id),
                "missing",
                operation="chart.built",
                audit_extra={
                    "chart": "missing",
                    "theme": configuration.theme,
                    "temporary_options": query.options is not None,
                    "options": options.model_dump(),
                    "title": query.title,
                    "start": query.start,
                },
            )
            rows = [
                {
                    "field": {k: c[k] for k in ("name", "label", "unit", "kind", "role")},
                    "missing": c["missing"],
                    "percent": c["missing_percent"],
                    "invalid_infinite": c["invalid_infinite"],
                    "reason": c["error"]
                    or (None if c["missing_percent"] is not None else "Набір не містить рядків."),
                }
                for c in result["columns"]
            ]
            ordered = missing_order(rows, options.missing_order)
            shares = [row["percent"] for row in rows if row["percent"] is not None]
            top = max(shares) if shares and max(shares) > 0 else None
            if query.start >= max(len(ordered), 1):
                raise StatisticsError("Такої сторінки полів немає.")
            page = ordered[query.start : query.start + MISSING_PAGE]
            return MissingView(
                **{
                    k: result[k]
                    for k in ("dataset_id", "project_id", "sha256", "row_count", "metadata_state")
                },
                warnings=result["warnings"] + contrast_warnings(options),
                rows=[MissingRow(**row) for row in page],
                start=query.start,
                total=len(ordered),
                rows_with_missing=result["rows_with_missing"],
                rows_with_missing_percent=result["rows_with_missing_percent"],
                theme=configuration.theme,
                options=options,
                svg=missing_svg(page, top, options, query.title),
            )

        return execute(action)

    def histogram(self, project_id: str, raw: object) -> Success[HistogramView] | Failure:
        def action() -> HistogramView:
            validated = str(validate_input(ProjectId, {"id": project_id}).id)
            query = validate_input(HistogramQuery, raw)
            configuration = self.chart_config.get()
            options = query.options or configuration.effective()
            bins = None if query.bin_width is not None else query.bins or options.histogram_bins
            result = self.statistics_store.calculate(
                validated,
                str(query.dataset_id),
                "descriptive",
                values_for=query.field,
                operation="chart.built",
                audit_extra={
                    "chart": "histogram",
                    "theme": configuration.theme,
                    "temporary_options": query.options is not None,
                    "options": options.model_dump(),
                    "title": query.title,
                    "bins": bins,
                    "bin_width": query.bin_width,
                },
                finish=lambda result: bin_values(result, bins, query.bin_width),
            )
            values = result.pop("values")
            binned = result.pop("histogram")
            column = next(c for c in result["columns"] if c["name"] == query.field)
            stats = column["statistics"] or {}
            binned["median"] = stats.get("median")
            descriptor = {k: column[k] for k in ("name", "label", "unit", "kind", "role")}
            warnings = result["warnings"] + contrast_warnings(options)
            small = 0 < len(values) < SMALL_SAMPLE
            if small:
                warnings.append(
                    f"Малий набір: n={len(values)} (< {SMALL_SAMPLE}); форма розподілу нестійка."
                )
            return HistogramView(
                **{
                    k: result[k]
                    for k in ("dataset_id", "project_id", "sha256", "row_count", "metadata_state")
                },
                warnings=warnings,
                field=StatisticsField(**descriptor),
                n=len(values),
                missing=column["missing"],
                invalid_infinite=column["invalid_infinite"],
                mode=binned["mode"],
                edges=binned["edges"],
                counts=binned["counts"],
                width=binned["width"],
                minimum=stats.get("min"),
                maximum=stats.get("max"),
                median=binned["median"],
                small_sample=small,
                constant=binned["constant"],
                reason=binned["reason"],
                theme=configuration.theme,
                options=options,
                svg=histogram_svg(binned, descriptor, options, query.title) if values else "",
            )

        return execute(action)


def _descriptor(result: dict[str, Any], name: str) -> dict[str, Any]:
    column = next(c for c in result["columns"] if c["name"] == name)
    return {k: column[k] for k in ("name", "label", "unit", "kind", "role")}


def bin_values(result: dict[str, Any], bins: int | None, width: float | None) -> dict[str, Any]:
    """Bin inside the audited statistics call; returns what the chart event records."""
    try:
        binned = histogram_bins(result["values"], bins, width)
    except BinningError as error:
        raise StatisticsError(str(error)) from error
    result["histogram"] = binned
    return {"bin_count": len(binned["counts"]), "bin_counts": binned["counts"]}


def _known(project_type: str) -> str:
    if project_type not in PROJECT_TYPES:
        raise UnknownProjectType("Тип проєкту не знайдено.")
    return project_type
