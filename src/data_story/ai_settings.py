"""AI settings (stage 08): which OpenAI model serves which task of the analysis pipeline.

The key is never stored or accepted here: it is read only from ``.env`` (``OPENAI_API_KEY``,
D-76) and reported masked. Model choices live in SQLite without secrets; every change is
written to ``config_events``. Task keys are the contract for the future assistant (D-75).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import ROOT
from .configuration import InstructionStore, _now
from .storage import connect, ensure_schema

ENV_FILE = ROOT / ".env"
KEY_NAME = "OPENAI_API_KEY"


@dataclass(frozen=True)
class Model:
    id: str
    title: str
    kind: str  # "text" | "embedding" | "local"


# Candidates (OpenAI catalogue read 2026-10-10); not verified for
# the author's account. "local" keeps a task without OpenAI.
MODELS: dict[str, Model] = {
    m.id: m
    for m in (
        Model("gpt-6-luna", "gpt-6-luna · економна, для частих коротких задач", "text"),
        Model("gpt-6.1-sol", "gpt-6.1-sol · сильніша, для складних рішень", "text"),
        Model("text-embedding-3-small", "text-embedding-3-small · вектори для пошуку", "embedding"),
        Model("local", "Без OpenAI · локальний розрахунок", "local"),
    )
}


@dataclass(frozen=True)
class Task:
    key: str
    stage: str
    title: str
    purpose: str
    mode: str  # «Порадь» | «Виконай» | «Поясни результат» | «Пошук»
    kind: str  # allowed model kind
    default: str


STAGES = [
    "Проєкт і дані",
    "Оглядовий аналіз",
    "Підготовка даних",
    "Модель і перевірка",
    "Висновки та звіт",
    "База знань",
]

TASKS: dict[str, Task] = {
    t.key: t
    for t in (
        Task(
            "project_description",
            "Проєкт і дані",
            "Опис проєкту",
            "Чернетка мети й опису проєкту",
            "Виконай",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "metadata_draft",
            "Проєкт і дані",
            "Метадані полів",
            "Опис полів, ролі, одиниці з сирого опису",
            "Виконай",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "eda_advice",
            "Оглядовий аналіз",
            "Поради з огляду",
            "Які поля й графіки переглянути далі",
            "Порадь",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "chart_command",
            "Оглядовий аналіз",
            "Графік за командою",
            "Перебудова графіка за текстовою командою",
            "Виконай",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "eda_explain",
            "Оглядовий аналіз",
            "Пояснення графіків",
            "Що видно на статистиці й графіках",
            "Поясни результат",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "preparation_advice",
            "Підготовка даних",
            "Поради з підготовки",
            "Пропуски, категорії, поділ",
            "Порадь",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "model_advice",
            "Модель і перевірка",
            "Рекомендації щодо моделі",
            "Ціль, ознаки, ризики перед навчанням",
            "Порадь",
            "text",
            "gpt-6.1-sol",
        ),
        Task(
            "model_explain",
            "Модель і перевірка",
            "Пояснення моделі",
            "Показники, коефіцієнти, p-value, обмеження",
            "Поясни результат",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "validation_explain",
            "Модель і перевірка",
            "Пояснення перевірки",
            "Графіки похибок і прогнозу проти факту",
            "Поясни результат",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "next_steps",
            "Модель і перевірка",
            "Наступні кроки",
            "Що зробити далі в аналізі",
            "Порадь",
            "text",
            "gpt-6.1-sol",
        ),
        Task(
            "conclusions",
            "Висновки та звіт",
            "Висновки",
            "Висновки з доказами та непевністю",
            "Поясни результат",
            "text",
            "gpt-6.1-sol",
        ),
        Task(
            "report_draft",
            "Висновки та звіт",
            "Чернетка звіту",
            "Текст звіту зі збережених результатів",
            "Виконай",
            "text",
            "gpt-6-luna",
        ),
        Task(
            "knowledge_search",
            "База знань",
            "Пошук правил",
            "Пошук фрагментів бази знань (RAG)",
            "Пошук",
            "embedding",
            "local",
        ),
    )
}


MAX_LEVELS = 3

# Prices in USD per 1M tokens, Standard tier, short context (D-106). Source and date are shown in
# the UI; the author edits them on the settings page when OpenAI changes the rate card.
PRICES_SOURCE = "https://developers.openai.com/api/docs/pricing"
PRICES_CHECKED = "2026-10-14"
LONG_CONTEXT = 272_000  # longer prompts use another rate; cost is then reported as unknown


class Price(BaseModel):
    """One model's rate card: every input token is billed at exactly one of three input rates."""

    model_config = ConfigDict(extra="forbid", strict=True)
    input: Annotated[float, Field(ge=0, le=1000)]
    cached_input: Annotated[float, Field(ge=0, le=1000)]
    cache_write: Annotated[float, Field(ge=0, le=1000)]
    output: Annotated[float, Field(ge=0, le=1000)]


def default_prices() -> dict[str, "Price"]:
    return {
        "gpt-6-luna": Price(input=0.10, cached_input=0.01, cache_write=0.125, output=0.50),
        "gpt-6.1-sol": Price(input=2.00, cached_input=0.10, cache_write=2.50, output=10.00),
    }


def cost_of(usage: dict | None, price: "Price | None") -> dict:
    """Cost in USD from actual usage; ``usd`` is None with a reason when it cannot be known.

    Uncached input = input − cached − cache writes; reasoning tokens are already in output.
    """
    usage = usage or {}
    inp, out = usage.get("input_tokens"), usage.get("output_tokens")
    if price is None:
        return {"usd": None, "reason": "немає тарифу моделі"}
    if inp is None or out is None:
        return {"usd": None, "reason": "OpenAI не повернув токени"}
    if inp > LONG_CONTEXT:
        return {"usd": None, "reason": "довгий контекст — інший тариф"}
    cached = usage.get("cached_tokens") or 0
    write = usage.get("cache_write_tokens") or 0
    plain = max(inp - cached - write, 0)
    usd = (
        plain * price.input
        + cached * price.cached_input
        + write * price.cache_write
        + out * price.output
    ) / 1_000_000
    return {"usd": usd, "reason": None}


def format_usd(usd: float | None) -> str:
    if usd is None:
        return "невідомо"
    return f"${usd:.4f}" if usd < 1 else f"${usd:.2f}"


def default_fallbacks() -> dict[str, list[str]]:
    """Level 2 for text tasks: the other text model (escalate luna → sol, back up sol → luna)."""
    other = {"gpt-6-luna": "gpt-6.1-sol", "gpt-6.1-sol": "gpt-6-luna"}
    return {k: [other[t.default]] if t.default in other else [] for k, t in TASKS.items()}


class AiConfiguration(BaseModel):
    """Level 1 model per task (``models``), up to two extra levels (``fallbacks``) and retries.

    A level is tried with ``retries`` repeats on transient failures (timeout, network, 429,
    5xx); an unavailable model, an empty or cut answer moves to the next level (D-79).
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    models: dict[str, str] = Field(default_factory=lambda: {k: t.default for k, t in TASKS.items()})
    fallbacks: dict[str, list[str]] = Field(default_factory=default_fallbacks)
    retries: Annotated[int, Field(ge=0, le=5)] = 2
    max_output_tokens: Annotated[int, Field(ge=256, le=32000)] = 2000
    max_agent_steps: Annotated[int, Field(ge=1, le=20)] = 6
    prices: dict[str, Price] = Field(default_factory=default_prices)
    prices_checked: Annotated[str, Field(max_length=40)] = PRICES_CHECKED

    @model_validator(mode="after")
    def known(self) -> "AiConfiguration":
        def check(key: str, model: str) -> None:
            if key not in TASKS:
                raise ValueError(f"Невідома задача: {key}.")
            if model not in MODELS:
                raise ValueError(f"Невідома модель: {model}.")
            allowed = {TASKS[key].kind, "local"}
            if MODELS[model].kind not in allowed:
                raise ValueError(f"Модель {model} не підходить для задачі «{TASKS[key].title}».")

        for key, model in self.models.items():
            check(key, model)
        for key, chain in self.fallbacks.items():
            if len(chain) > MAX_LEVELS - 1:
                raise ValueError(f"Не більше {MAX_LEVELS} рівнів моделей для задачі.")
            for model in chain:
                check(key, model)
                if model == "local":
                    raise ValueError("Запасний рівень має бути моделлю OpenAI.")
        for model in self.prices:
            if model not in MODELS or MODELS[model].kind != "text":
                raise ValueError(f"Тариф можна задати лише текстовій моделі OpenAI: {model}.")
        for model, price in default_prices().items():  # older saved settings get the defaults
            self.prices.setdefault(model, price)
        for key, task in TASKS.items():  # new tasks get their default
            self.models.setdefault(key, task.default)
            self.fallbacks.setdefault(key, default_fallbacks()[key])
        return self

    def cost(self, model: str, usage: dict | None) -> dict:
        return cost_of(usage, self.prices.get(model))

    def chain(self, task: str) -> list[str]:
        """Distinct models in level order: level 1, then the configured fallbacks."""
        result: list[str] = []
        for model in [self.models[task], *self.fallbacks.get(task, [])]:
            if model not in result:
                result.append(model)
        return result


def key_status(env_file: Path = ENV_FILE) -> dict:
    """Presence and masked tail of the key in .env; the key itself never leaves this function."""
    if not env_file.is_file():
        return {"state": "missing_file", "source": ".env", "masked": None,
                "message": "Файл .env не знайдено. Скопіюйте .env.example у .env і впишіть ключ."}  # fmt: skip
    value = (dotenv_values(env_file).get(KEY_NAME) or "").strip()
    if not value:
        return {"state": "empty", "source": ".env", "masked": None,
                "message": f"У .env немає значення {KEY_NAME}."}  # fmt: skip
    if not value.startswith("sk-") or len(value) < 20:
        return {"state": "suspicious", "source": ".env", "masked": "…" + value[-4:],
                "message": "Значення не схоже на ключ OpenAI (очікується sk-…)."}  # fmt: skip
    return {"state": "found", "source": ".env", "masked": "sk-…" + value[-4:],
            "message": "Ключ знайдено у .env. Доступ до OpenAI ще не перевірявся."}  # fmt: skip


class AiSettingsStore:
    def __init__(self, database: Path, env_file: Path = ENV_FILE):
        self.database = ensure_schema(database)
        self.env_file = env_file

    def get(self) -> AiConfiguration:
        with connect(self.database) as connection:
            row = connection.execute("SELECT document FROM ai_configuration WHERE id=1").fetchone()
        return AiConfiguration.model_validate_json(row["document"]) if row else AiConfiguration()

    def save(self, value: AiConfiguration) -> AiConfiguration:
        value = AiConfiguration.model_validate(value.model_dump())
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO ai_configuration VALUES (1, ?, ?) ON CONFLICT(id) "
                "DO UPDATE SET document=excluded.document, updated_at=excluded.updated_at",
                (value.model_dump_json(), _now()),
            )
            InstructionStore._event(connection, "ai", "ai.configuration.saved", value.model_dump())
        return value

    def reset(self) -> AiConfiguration:
        with connect(self.database) as connection:
            connection.execute("DELETE FROM ai_configuration WHERE id=1")
            InstructionStore._event(connection, "ai", "ai.configuration.reset", {})
        return self.get()

    def key(self) -> dict:
        return key_status(self.env_file)

    def model_for(self, task: str) -> str:
        """The configured model for a task; the future adapter reads only this."""
        if task not in TASKS:
            raise KeyError(task)
        return self.get().models[task]
