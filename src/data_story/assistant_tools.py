"""Model operations described for the future AI assistant (stage 08–09).

No OpenAI calls here. Each tool has a JSON schema of its arguments in function-calling form and
calls the same ``ProjectService`` method as the UI and HTTP API, so an assistant can later
advise («Порадь»), execute an explicit task («Виконай») or explain a saved result
(«Поясни результат») without new domain logic.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from .contracts import (
    AnalyticalQuery,
    ChartDraftQuery,
    ErrorInfo,
    Failure,
    KnowledgeQuery,
    MetadataDraftQuery,
    ModelCheckQuery,
    ModelTrainQuery,
    ProjectDraftQuery,
    SqlSummaryDraft,
    StepwiseQuery,
    Success,
)
from .model import explain
from .service import ProjectService, execute, validate_input

Mode = Literal["advise", "execute", "explain"]


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelResultQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str


@dataclass(frozen=True)
class AssistantTool:
    name: str
    mode: Mode
    description: str
    arguments: type[BaseModel]
    changes_data: bool
    handler: Callable[[ProjectService, str, dict], Success | Failure]

    def schema(self) -> dict:
        parameters = self.arguments.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }


def _training_runs(service: ProjectService, project_id: str, _args: dict) -> Success | Failure:
    runs = service.training_runs(project_id)
    models = service.model_runs(project_id)
    if isinstance(runs, Failure):
        return runs
    trained: dict[str, list[dict]] = {}
    for m in models.data if isinstance(models, Success) else []:
        trained.setdefault(m["training_run_id"], []).append(
            {"model_id": m["id"], "features": m["document"]["choices"]["features"]}
        )
    return Success(
        data=[
            {
                "training_run_id": r["id"],
                "dataset_name": r["document"]["dataset_name"],
                "target": r["document"]["choices"]["target"],
                "features": r["document"]["choices"]["features"],
                "train_count": r["document"]["report"]["train_count"],
                "test_count": r["document"]["report"]["test_count"],
                "feature_count": r["document"]["report"]["feature_count"],
                "created_at": r["created_at"],
                "models": trained.get(r["id"], []),
            }
            for r in runs.data
        ]
    )


def _train(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    return service.model_train(project_id, args)


def _stepwise(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    return service.model_stepwise(project_id, args)


def _explain(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    found = service.model_run(project_id, str(args.get("model_id", "")))
    if isinstance(found, Failure):
        return found
    document = found.data["document"]
    return Success(
        data={
            "model_id": found.data["id"],
            "target": document["choices"]["target"],
            "metrics": document["metrics"],
            "coefficients": document["coefficients"],
            "intercept": document["intercept"],
            "explanation": explain(document),
            "warnings": document["warnings"] + document["preparation_warnings"],
            "limitations": document["limitations"],
            "provenance": {
                key: document[key]
                for key in (
                    "dataset_name",
                    "source_sha256",
                    "training_run_id",
                    "matrix_sha256",
                    "sklearn_version",
                )
            },
        }
    )


def _check(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    result = service.model_check(project_id, args)
    if isinstance(result, Success):
        result.data.pop("svg", None)  # the assistant gets numbers and text, the UI draws
    return result


def _knowledge(service: ProjectService, _project_id: str, args: dict) -> Success | Failure:
    return service.knowledge_search(args)


def _project_draft(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    from .assistant_drafts import propose_project

    return execute(
        lambda: {
            "proposal": propose_project(
                service, project_id, validate_input(ProjectDraftQuery, args).model_dump(mode="json")
            )
        }
    )


def _metadata_draft(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    from .assistant_drafts import propose_metadata

    return execute(
        lambda: {
            "proposal": propose_metadata(
                service,
                project_id,
                validate_input(MetadataDraftQuery, args).model_dump(mode="json"),
            )
        }
    )


def _chart_draft(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    from .assistant_drafts import propose_chart

    return execute(
        lambda: {
            "proposal": propose_chart(
                service, project_id, validate_input(ChartDraftQuery, args).model_dump(mode="json")
            )
        }
    )


def _analytics_schema(service: ProjectService, project_id: str, _args: dict) -> Success | Failure:
    return service.analytics_schema(project_id)


def _analytics_query(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    result = service.analytics_query(project_id, args)
    if isinstance(result, Success):
        result.data["preview_only"] = len(result.data["rows"]) > 20
        result.data["rows"] = result.data["rows"][:20]
    return result


def _sql_draft(service: ProjectService, project_id: str, args: dict) -> Success | Failure:
    from .assistant_drafts import propose_sql_summary

    return execute(
        lambda: {
            "proposal": propose_sql_summary(
                service, project_id, validate_input(SqlSummaryDraft, args).model_dump()
            )
        }
    )


TOOLS: dict[str, AssistantTool] = {
    tool.name: tool
    for tool in (
        AssistantTool(
            "propose_project_description",
            "execute",
            "Підготувати українську Markdown-чернетку опису проєкту з його мети. "
            "Показує пропозицію для перегляду; нічого не зберігає. Користувач застосує її кнопкою.",
            ProjectDraftQuery,
            False,
            _project_draft,
        ),
        AssistantTool(
            "propose_metadata",
            "execute",
            "Запропонувати опис набору та зміни описів, назв, одиниць, типів і ролей відомих полів "
            "на основі наданих сирих описів. Невідомі одиниці й значення не вигадувати. "
            "Лише чернетка для перегляду; нічого не зберігає і не підтверджує.",
            MetadataDraftQuery,
            False,
            _metadata_draft,
        ),
        AssistantTool(
            "propose_eda_chart",
            "execute",
            "Запропонувати параметри графіка обраного набору для перегляду і кнопки Застосувати. "
            "Типи: histogram (одне числове field, bins або bin_width), scatter (числові x,y), "
            "heatmap (2–12 числових fields), missing (start сторінки). "
            "Для зміни оформлення збережи всі поточні options і зміни лише замовлені. "
            "explanation — коротко поясни вибір за правилами. Не будує графік до застосування.",
            ChartDraftQuery,
            False,
            _chart_draft,
        ),
        AssistantTool(
            "list_training_runs",
            "advise",
            "Показати збережені навчальні/тестові вибірки проєкту й чи навчена на них модель.",
            NoArguments,
            False,
            _training_runs,
        ),
        AssistantTool(
            "train_linear_regression",
            "execute",
            "Навчити LinearRegression (scikit-learn) на навчальній частині обраних збережених "
            "вибірок і оцінити на тестовій. features — назви полів вибірок для моделі "
            "(без features — усі поля). Ті самі поля повертають збережену модель.",
            ModelTrainQuery,
            True,
            _train,
        ),
        AssistantTool(
            "stepwise_linear_regression",
            "execute",
            "Обрати поля моделі методом stepwise (в обидва боки, лише навчальна частина) за "
            "критерієм adj_r2 (скоригований R², максимум), aic або bic (мінімум), навчити й "
            "зберегти модель; повертає кроки відбору й метрики на тестовій частині.",
            StepwiseQuery,
            True,
            _stepwise,
        ),
        AssistantTool(
            "explain_model_result",
            "explain",
            "Повернути збережені показники, коефіцієнти, попередження, обмеження й походження "
            "моделі для пояснення результату без нових обчислень.",
            ModelResultQuery,
            False,
            _explain,
        ),
        AssistantTool(
            "search_knowledge",
            "advise",
            "Знайти в базі знань правила (візуалізація, підготовка даних, показники й "
            "коефіцієнти регресії, перевірка моделі) з джерелом і розділом для поради чи "
            "пояснення. Пошук локальний.",
            KnowledgeQuery,
            False,
            _knowledge,
        ),
        AssistantTool(
            "build_model_chart",
            "execute",
            "Побудувати або замінити графік перевірки моделі на тестовій частині: "
            "regression (ціль від обраної числової ознаки з лінією моделі), residuals "
            "(похибки проти прогнозу) або actual (прогноз проти факту); параметри стилю "
            "необов'язкові. Дані не змінює.",
            ModelCheckQuery,
            False,
            _check,
        ),
        AssistantTool(
            "analytics_schema",
            "advise",
            "Отримати каталог SQL-таблиць поточного проєкту: alias, поля, кількість рядків. "
            "Використовуй alias у SQL. Коди пропусків стають NULL; ignore-поля виключені. "
            "d_ — набори, c_ — коефіцієнти, m_ — метрики, p_ — прогнози, r_ — залишки.",
            NoArguments,
            False,
            _analytics_schema,
        ),
        AssistantTool(
            "query_analytics",
            "advise",
            "Виконати один SELECT у DuckDB для поточного проєкту. Спочатку отримай analytics_schema. "
            "Можна JOIN, GROUP BY, CTE, агрегати; жодного запису чи доступу до файлів. "
            "Повертає числа й до 20 рядків для пояснення; row_count і truncated показують обмеження.",
            AnalyticalQuery,
            False,
            _analytics_query,
        ),
        AssistantTool(
            "propose_sql_summary",
            "execute",
            "Запропонувати незалежну агрегацію за SQL SELECT для графіків/аналізу. "
            "Отримай analytics_schema, склади SQL, вкажи назву й пояснення. "
            "Сервіс перевірить запит. Користувач перегляне й застосує чернетку; до цього набору не додаємо. "
            "Не використовуй обрізаний результат; limit до 10000.",
            SqlSummaryDraft,
            False,
            _sql_draft,
        ),
    )
}


def run_tool(service: ProjectService, project_id: str, name: str, args: dict) -> Success | Failure:
    tool = TOOLS.get(name)
    if tool is None:
        return Failure(error=ErrorInfo(code="not_found", message="Інструмент не знайдено."))
    return tool.handler(service, project_id, args)
