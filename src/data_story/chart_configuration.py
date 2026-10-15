"""Local chart profiles; defaults from lecture theme, never overwrite built-in profiles."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .configuration import InstructionStore, _now
from .storage import connect, ensure_schema

Theme = Literal["storytelling", "academic", "custom"]
THEMES = {"storytelling": "Storytelling", "academic": "Академічна", "custom": "Власна"}
FONTS = ["Open Sans", "IBM Plex Sans", "STIX Two Text"]
Color = Annotated[str, Field(strict=True, pattern=r"^#[0-9a-fA-F]{6}$")]


class ChartOptions(BaseModel):
    """Flat option set. New fields need defaults so earlier saved custom themes stay valid."""

    model_config = ConfigDict(extra="forbid", strict=True)
    font: Literal["Open Sans", "IBM Plex Sans", "STIX Two Text"] = "Open Sans"
    title_size: Annotated[int, Field(ge=16, le=32)] = 22
    label_size: Annotated[int, Field(ge=12, le=22)] = 14
    decimals: Annotated[int, Field(ge=1, le=4)] = 2
    positive: Color = "#1B4F8A"
    negative: Color = "#E07B39"
    neutral: Color = "#F6F5F1"
    text: Color = "#202124"
    background: Color = "#FFFFFF"
    numbers: bool = True
    lower_triangle: bool = True
    frame: bool = False
    threshold_enabled: bool = False
    threshold: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)] = 0.3
    # Shared axes/colours for charts with axes (from history 052, histogram first).
    accent: Color = "#1B4F8A"
    axis_labels: bool = True
    grid_x: bool = False
    grid_y: bool = True
    # Histogram: default bin count (5-60), median line, bar opacity.
    histogram_bins: Annotated[int, Field(ge=5, le=60)] = 20
    histogram_median: bool = True
    histogram_opacity: Annotated[float, Field(ge=0.2, le=1, allow_inf_nan=False)] = 0.85
    # Scatter: point radius, opacity (overlap stays visible), outline.
    scatter_size: Annotated[int, Field(ge=1, le=10)] = 3
    scatter_opacity: Annotated[float, Field(ge=0.05, le=1, allow_inf_nan=False)] = 0.6
    scatter_outline: bool = False
    # Missing values: grey context with accent on the largest share, direct labels, order.
    context: Color = "#C3C6CA"
    missing_labels: bool = True
    missing_order: Literal["desc", "source"] = "desc"


def preset(theme: Literal["storytelling", "academic"]) -> ChartOptions:
    academic = theme == "academic"
    return ChartOptions(lower_triangle=not academic, frame=academic, grid_x=academic)


class ChartConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    theme: Theme = "storytelling"
    custom: ChartOptions | None = None

    @model_validator(mode="after")
    def require_custom(self) -> Self:
        if self.theme == "custom" and self.custom is None:
            raise ValueError("Для власної теми потрібні налаштування.")
        return self

    def effective(self) -> ChartOptions:
        if self.theme == "custom" and self.custom is not None:
            return self.custom.model_copy(deep=True)
        return preset("academic" if self.theme == "academic" else "storytelling")


class ChartConfigurationStore:
    def __init__(self, database: Path):
        self.database = ensure_schema(database)

    def get(self) -> ChartConfiguration:
        with connect(self.database) as connection:
            row = connection.execute(
                "SELECT document FROM chart_configuration WHERE id=1"
            ).fetchone()
        return (
            ChartConfiguration.model_validate_json(row["document"]) if row else ChartConfiguration()
        )

    def save(self, value: ChartConfiguration) -> ChartConfiguration:
        # Validate even when invoked directly with a constructed model.
        value = ChartConfiguration.model_validate(value.model_dump())
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO chart_configuration VALUES (1, ?, ?) ON CONFLICT(id) "
                "DO UPDATE SET document=excluded.document, updated_at=excluded.updated_at",
                (value.model_dump_json(), _now()),
            )
            InstructionStore._event(
                connection, "charts", "charts.configuration.saved", value.model_dump()
            )
        return value

    def reset(self) -> ChartConfiguration:
        with connect(self.database) as connection:
            connection.execute("DELETE FROM chart_configuration WHERE id=1")
            InstructionStore._event(connection, "charts", "charts.configuration.reset", {})
        return self.get()

    def events(self) -> list[dict]:
        with connect(self.database) as connection:
            rows = connection.execute(
                "SELECT operation, occurred_at, payload FROM config_events WHERE subject='charts' "
                "ORDER BY occurred_at DESC LIMIT 10"
            ).fetchall()
        return [dict(row, payload=json.loads(row["payload"])) for row in rows]
