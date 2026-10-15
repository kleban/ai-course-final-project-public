"""One public contract for UI, HTTP adapters, and future tool calls."""

from typing import Annotated, Any, Generic, Literal, TypeVar
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)

from .chart_configuration import ChartOptions, Theme
from .combine import Recipe
from .configuration import INSTRUCTION_LIMIT
from .metadata import MetadataDocument
from .preparation import MissingRule
from .project_types import DEFAULT_TYPE, ProjectTypeKey, require_enabled
from .statistics import View
from .training_data import TrainingChoices
from .validation import DESCRIPTION_LIMIT, NAME_LIMIT, PURPOSE_LIMIT, text

T = TypeVar("T")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class ProjectCreate(StrictModel):
    name: Annotated[str, Field(strict=True, min_length=1, max_length=NAME_LIMIT)]
    purpose: Annotated[str, Field(strict=True, max_length=PURPOSE_LIMIT)] = ""
    project_type: ProjectTypeKey = DEFAULT_TYPE

    @field_validator("project_type", mode="before")
    @classmethod
    def type_value(cls, value: object) -> str:
        return require_enabled(value)

    @field_validator("name", mode="before")
    @classmethod
    def name_value(cls, value: object) -> str:
        return text(value, "Назва", NAME_LIMIT, required=True)

    @field_validator("purpose", mode="before")
    @classmethod
    def purpose_value(cls, value: object) -> str:
        return text(value, "Мета", PURPOSE_LIMIT)


class ProjectId(StrictModel):
    id: UUID


class ProjectView(StrictModel):
    id: UUID
    name: Annotated[str, Field(strict=True, min_length=1, max_length=NAME_LIMIT)]
    purpose: Annotated[str, Field(strict=True, max_length=PURPOSE_LIMIT)]
    created_at: AwareDatetime
    project_type: ProjectTypeKey
    description: str = ""


class ProjectDescription(StrictModel):
    text: Annotated[str, Field(strict=True, max_length=DESCRIPTION_LIMIT)]


class ProjectTypeView(StrictModel):
    key: ProjectTypeKey
    title: str
    description: str
    icon: str
    enabled: bool


class InstructionView(StrictModel):
    project_type: ProjectTypeKey
    text: str
    is_default: bool
    updated_at: AwareDatetime | None


class InstructionUpdate(StrictModel):
    text: Annotated[str, Field(strict=True, min_length=1, max_length=INSTRUCTION_LIMIT)]

    @field_validator("text", mode="before")
    @classmethod
    def text_value(cls, value: object) -> str:
        return text(value, "Інструкція", INSTRUCTION_LIMIT, required=True)


class EventView(StrictModel):
    id: UUID
    project_id: UUID
    operation: Literal[
        "project.created",
        "project.opened",
        "project.description_saved",
        "dataset.uploaded",
        "dataset.copied",
        "metadata.saved",
        "metadata.confirmed",
        "metadata.source_added",
        "statistics.calculated",
        "chart.built",
        "preparation.previewed",
        "preparation.saved",
        "preparation.applied",
        "training.prepared",
        "model.trained",
        "model.checked",
        "assistant.asked",
        "report.created",
        "report.saved",
        "report.ai",
        "report.rendered",
        "summary.previewed",
        "summary.created",
        "analytics.synced",
        "analytics.queried",
        "conclusions.prepared",
        "conclusions.saved",
        "conclusions.approved",
    ]
    occurred_at: AwareDatetime
    payload: dict[str, Any]


class ColumnProfile(StrictModel):
    name: str
    kind: Literal["number", "text", "date", "bool"]
    missing: int
    unique: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    date_format: str | None = None
    date_ambiguous: bool = False
    first: str | None = None
    last: str | None = None


class DatasetView(StrictModel):
    id: UUID
    project_id: UUID
    name: str
    file_format: Literal["csv"]
    size_bytes: int
    row_count: int
    column_count: int
    columns: list[ColumnProfile]
    source: dict[str, str]
    created_at: AwareDatetime
    copied_from: UUID | None = None

    @classmethod
    def from_dataset(cls, dataset) -> "DatasetView":
        details = dict(dataset.profile)
        columns = details.pop("columns", [])
        return cls(
            id=dataset.id,
            project_id=dataset.project_id,
            name=dataset.name,
            file_format=dataset.file_format,
            size_bytes=dataset.size_bytes,
            row_count=dataset.row_count,
            column_count=dataset.column_count,
            columns=[ColumnProfile.model_validate(c) for c in columns],
            source={k: str(v) for k, v in details.items()},
            created_at=dataset.created_at,
            copied_from=dataset.copied_from,
        )


class DatasetCandidate(StrictModel):
    """A dataset of another project that can be copied into the current one (stage 02c)."""

    dataset: DatasetView
    project_name: str
    metadata: Literal["confirmed", "draft", "none"]
    already_here: bool

    @classmethod
    def from_candidate(cls, candidate) -> "DatasetCandidate":
        return cls(
            dataset=DatasetView.from_dataset(candidate.dataset),
            project_name=candidate.project_name,
            metadata=candidate.metadata,
            already_here=candidate.already_here,
        )


class DatasetCopyRequest(StrictModel):
    dataset_id: UUID


class DatasetPreview(StrictModel):
    columns: list[str]
    rows: list[list[str | None]]
    total_rows: int


class SortItem(StrictModel):
    colId: Annotated[str, Field(strict=True, pattern=r"^c\d{1,4}$")]
    sort: Literal["asc", "desc"]


class RowsQuery(StrictModel):
    """One page request from the grid (AG Grid infinite row model): rows start..start+limit
    after the grid's filter model and sort model are applied on the server."""

    start: Annotated[int, Field(ge=0)] = 0
    limit: Annotated[int, Field(ge=1, le=500)] = 100
    sort: Annotated[list[SortItem], Field(max_length=50)] = []
    filter: dict[str, dict[str, Any]] = {}


class RowsPage(StrictModel):
    rows: list[dict[str, int | float | str | None]]
    total: int
    all_rows: int


class StatisticsQuery(StrictModel):
    dataset_id: UUID
    view: View = "descriptive"
    fields: list[Annotated[str, Field(strict=True, min_length=1)]] | None = None
    start: Annotated[int, Field(strict=True, ge=0)] = 0
    limit: Annotated[int, Field(strict=True, ge=1, le=50)] = 10


class StatisticsField(StrictModel):
    name: str
    label: str
    unit: str
    kind: str
    role: str


class StatisticsRow(StatisticsField):
    values: dict[str, int | FiniteFloat | str | bool | None]
    counts: dict[str, int] = Field(default_factory=dict)
    reasons: dict[str, str] = Field(default_factory=dict)
    reason: str | None = None
    warning: str | None = None


class StatisticsPage(StrictModel):
    project_id: UUID
    dataset_id: UUID
    sha256: str
    view: View
    method: str
    row_count: int
    metadata_state: Literal["confirmed", "draft", "none"]
    metadata_updated_at: AwareDatetime | None
    warnings: list[str]
    rows: list[StatisticsRow]
    total: int
    start: int
    limit: int
    fields: list[StatisticsField] = Field(default_factory=list)
    eligible: list[StatisticsField] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    rows_with_missing: int | None = None
    rows_with_missing_percent: FiniteFloat | None = None
    reason: str | None = None


class MetadataIssue(StrictModel):
    column: str
    level: Literal["warning", "info"]
    message: str


class MetadataView(StrictModel):
    dataset_id: UUID
    document: MetadataDocument
    saved: bool
    updated_at: AwareDatetime | None
    confirmed_at: AwareDatetime | None
    issues: list[MetadataIssue]


class MetadataUpdate(StrictModel):
    document: MetadataDocument
    confirm: bool = False


class MetadataText(StrictModel):
    text: Annotated[str, Field(strict=True, min_length=1, max_length=50_000)]


class MetadataSourceView(StrictModel):
    id: UUID
    dataset_id: UUID
    kind: Literal["file", "text"]
    name: str
    text: str
    created_at: AwareDatetime


class FieldIssue(StrictModel):
    field: str
    message: str


class ErrorInfo(StrictModel):
    code: Literal["invalid_input", "not_found", "storage_error", "internal_error"]
    message: str
    fields: list[FieldIssue] = Field(default_factory=list)


class Success(StrictModel, Generic[T]):
    status: Literal["ok"] = "ok"
    data: T


class Failure(StrictModel):
    status: Literal["error"] = "error"
    error: ErrorInfo


HTTP_STATUS = {"invalid_input": 422, "not_found": 404, "storage_error": 503, "internal_error": 500}


class HeatmapQuery(StrictModel):
    dataset_id: UUID
    fields: list[Annotated[str, Field(strict=True, min_length=1)]] = Field(
        min_length=2, max_length=12
    )
    options: ChartOptions | None = None
    title: Annotated[str, Field(strict=True, max_length=180)] = ""


class HeatmapView(StrictModel):
    dataset_id: UUID
    project_id: UUID
    sha256: str
    row_count: int
    metadata_state: str
    warnings: list[str]
    fields: list[StatisticsField]
    r: list[list[FiniteFloat | None]]
    n: list[list[int]]
    reasons: list[list[str | None]]
    excluded: dict[str, str]
    theme: Theme
    options: ChartOptions
    svg: str


class HistogramQuery(StrictModel):
    """One numeric field; either a bin count (5-60) or an explicit bin width, not both."""

    dataset_id: UUID
    field: Annotated[str, Field(strict=True, min_length=1)]
    bins: Annotated[int, Field(strict=True, ge=5, le=60)] | None = None
    bin_width: Annotated[FiniteFloat, Field(gt=0)] | None = None
    options: ChartOptions | None = None
    title: Annotated[str, Field(strict=True, max_length=180)] = ""

    @model_validator(mode="after")
    def one_mode(self) -> "HistogramQuery":
        if self.bins is not None and self.bin_width is not None:
            raise ValueError("Оберіть або кількість інтервалів, або їх ширину.")
        return self


class ScatterQuery(StrictModel):
    """Two different numeric fields; all pairwise complete rows are drawn."""

    dataset_id: UUID
    x: Annotated[str, Field(strict=True, min_length=1)]
    y: Annotated[str, Field(strict=True, min_length=1)]
    options: ChartOptions | None = None
    title: Annotated[str, Field(strict=True, max_length=180)] = ""

    @model_validator(mode="after")
    def different(self) -> "ScatterQuery":
        if self.x == self.y:
            raise ValueError("Оберіть два різні поля.")
        return self


class ScatterView(StrictModel):
    dataset_id: UUID
    project_id: UUID
    sha256: str
    row_count: int
    metadata_state: str
    warnings: list[str]
    x: StatisticsField
    y: StatisticsField
    n: int
    r: FiniteFloat | None
    reason: str | None
    x_range: list[FiniteFloat]
    y_range: list[FiniteFloat]
    theme: Theme
    options: ChartOptions
    svg: str


class MissingQuery(StrictModel):
    """All fields of the dataset, ten per page; order comes from the chart options."""

    dataset_id: UUID
    start: Annotated[int, Field(strict=True, ge=0)] = 0
    options: ChartOptions | None = None
    title: Annotated[str, Field(strict=True, max_length=180)] = ""


class MissingRow(StrictModel):
    field: StatisticsField
    missing: int | None
    percent: FiniteFloat | None
    invalid_infinite: int
    reason: str | None


class MissingView(StrictModel):
    dataset_id: UUID
    project_id: UUID
    sha256: str
    row_count: int
    metadata_state: str
    warnings: list[str]
    rows: list[MissingRow]
    start: int
    total: int
    rows_with_missing: int | None
    rows_with_missing_percent: FiniteFloat | None
    theme: Theme
    options: ChartOptions
    svg: str


class HistogramView(StrictModel):
    dataset_id: UUID
    project_id: UUID
    sha256: str
    row_count: int
    metadata_state: str
    warnings: list[str]
    field: StatisticsField
    n: int
    missing: int | None
    invalid_infinite: int
    mode: Literal["count", "width"]
    edges: list[FiniteFloat]
    counts: list[int]
    width: FiniteFloat | None
    minimum: FiniteFloat | None
    maximum: FiniteFloat | None
    median: FiniteFloat | None
    small_sample: bool
    constant: bool
    reason: str | None
    theme: Theme
    options: ChartOptions
    svg: str


class PreparationQuery(StrictModel):
    dataset_id: UUID
    rules: list[MissingRule] = Field(default_factory=list, max_length=200)


class PreparationSave(PreparationQuery):
    preview_token: str = Field(pattern=r"^[a-f0-9]{64}$")


class TrainingQuery(TrainingChoices):
    dataset_id: UUID = Field(strict=False)


class ModelTrainQuery(StrictModel):
    training_run_id: UUID
    # Fields of the saved split to use; omitted = every field (history 143).
    features: list[Annotated[str, Field(strict=True, min_length=1, max_length=200)]] | None = Field(
        default=None, min_length=1, max_length=30
    )


class StepwiseQuery(StrictModel):
    training_run_id: UUID
    criterion: Literal["adj_r2", "aic", "bic"] = "adj_r2"


class ModelCheckQuery(StrictModel):
    model_id: UUID
    chart: Literal["regression", "residuals", "actual"] = "regression"
    feature: Annotated[str, Field(strict=True, min_length=1, max_length=200)] | None = None
    options: ChartOptions | None = None
    title: Annotated[str, Field(strict=True, max_length=120)] = ""


class KnowledgeQuery(StrictModel):
    query: Annotated[str, Field(strict=True, min_length=1, max_length=500)]
    limit: Annotated[int, Field(strict=True, ge=1, le=10)] = 3


class AssistantPage(StrictModel):
    """Where the user is: selected dataset, chart tool, saved split or model."""

    dataset_id: UUID | None = None
    chart: dict[str, Any] | None = None
    tool: Annotated[str, Field(strict=True, max_length=40)] | None = None
    run_id: UUID | None = None


class AssistantTurn(StrictModel):
    role: Literal["user", "assistant"]
    text: Annotated[str, Field(strict=True, max_length=4000)]


class AssistantQuery(StrictModel):
    block: Literal["data", "eda", "preparation", "model", "validation"]
    mode: Literal["advise", "execute", "explain"]
    question: Annotated[str, Field(strict=True, min_length=1, max_length=2000)]
    page: AssistantPage = Field(default_factory=AssistantPage)
    history: list[AssistantTurn] = Field(default_factory=list, max_length=20)


class ProjectDraftQuery(StrictModel):
    text: Annotated[str, Field(strict=True, min_length=1, max_length=DESCRIPTION_LIMIT)]


class MetadataColumnDraft(StrictModel):
    name: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    label: Annotated[str, Field(strict=True, max_length=200)] | None = None
    description: Annotated[str, Field(strict=True, max_length=2000)] | None = None
    unit: Annotated[str, Field(strict=True, max_length=100)] | None = None
    kind: Literal["number", "category", "text", "date", "bool", "id"] | None = None
    role: Literal["feature", "target", "id", "ignore"] | None = None


class MetadataDraftQuery(StrictModel):
    dataset_id: UUID
    description: Annotated[str, Field(strict=True, max_length=4000)] | None = None
    columns: list[MetadataColumnDraft] = Field(min_length=1, max_length=200)


class ChartDraftQuery(StrictModel):
    chart: Literal["histogram", "scatter", "heatmap", "missing"]
    query: HistogramQuery | ScatterQuery | HeatmapQuery | MissingQuery
    explanation: Annotated[str, Field(strict=True, min_length=1, max_length=1000)]

    @model_validator(mode="after")
    def correct_query(self) -> "ChartDraftQuery":
        expected = {
            "histogram": HistogramQuery,
            "scatter": ScatterQuery,
            "heatmap": HeatmapQuery,
            "missing": MissingQuery,
        }[self.chart]
        if not isinstance(self.query, expected):
            raise ValueError("Параметри не відповідають типу графіка.")
        return self


class AnalyticalQuery(StrictModel):
    sql: Annotated[str, Field(strict=True, min_length=1, max_length=10000)]
    limit: Annotated[int, Field(strict=True, ge=1, le=10000)] = 100


class SqlSummaryDraft(AnalyticalQuery):
    name: Annotated[str, Field(strict=True, min_length=1, max_length=160)]
    explanation: Annotated[str, Field(strict=True, min_length=1, max_length=1000)]


class DraftApply(StrictModel):
    kind: Literal["project_description", "metadata", "chart", "sql_summary"]
    project_id: UUID
    base_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    text: Annotated[str, Field(strict=True, min_length=1, max_length=DESCRIPTION_LIMIT)] | None = (
        None
    )
    metadata: MetadataDraftQuery | None = None
    chart: ChartDraftQuery | None = None
    sql_summary: SqlSummaryDraft | None = None


class KnowledgeDocument(StrictModel):
    text: Annotated[str, Field(strict=True, min_length=1, max_length=20000)]


class ReportCreate(StrictModel):
    model_id: UUID
    title: Annotated[str, Field(strict=True, min_length=1, max_length=180)] = "Аналітичний звіт"


class ReportRevision(StrictModel):
    revision: Annotated[int, Field(strict=True, ge=1)]


class ReportSave(ReportRevision):
    markdown: Annotated[str, Field(strict=True, min_length=1, max_length=100000)]


class ConclusionCreate(StrictModel):
    model_id: UUID
    summary_ids: Annotated[list[UUID], Field(max_length=3)] = []


class ConclusionSave(ReportRevision):
    # Approve without a text keeps the saved wording; saving a draft always sends the text.
    markdown: Annotated[str | None, Field(strict=True, min_length=1, max_length=8000)] = None


class SummarySave(StrictModel):
    recipe: Recipe
    preview_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    name: Annotated[str, Field(max_length=160)] | None = None
