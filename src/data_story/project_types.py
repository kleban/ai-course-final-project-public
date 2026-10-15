"""Kinds of analysis a project can be. Each kind has an editable instruction (prompt) for the agent.

Default instructions ship with the code as Markdown in `instructions/<key>.md`; edited versions
are stored in SQLite (see configuration.py) and can be reset to the default.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ProjectTypeKey = Literal["regression", "classification", "clustering"]
DEFAULT_TYPE: ProjectTypeKey = "regression"
INSTRUCTIONS = Path(__file__).parent / "instructions"


@dataclass(frozen=True)
class ProjectType:
    key: ProjectTypeKey
    title: str
    description: str
    icon: str
    enabled: bool

    def default_instruction(self) -> str:
        return (INSTRUCTIONS / f"{self.key}.md").read_text(encoding="utf-8").strip()


PROJECT_TYPES: dict[str, ProjectType] = {
    t.key: t
    for t in (
        ProjectType(
            "regression",
            "Регресія",
            "Пояснити й передбачити числовий показник за іншими змінними.",
            "regression",
            True,
        ),
        ProjectType(
            "classification",
            "Класифікація",
            "Віднести спостереження до однієї з категорій.",
            "classification",
            False,
        ),
        ProjectType(
            "clustering",
            "Кластеризація",
            "Знайти групи схожих спостережень без готових міток.",
            "clustering",
            False,
        ),
    )
}


def require_enabled(key: object) -> ProjectTypeKey:
    """Validated key of a type that can be chosen for a new project."""
    if not isinstance(key, str) or key not in PROJECT_TYPES:
        raise ValueError("Тип проєкту: невідомий тип.")
    project_type = PROJECT_TYPES[key]
    if not project_type.enabled:
        raise ValueError(f"Тип проєкту: «{project_type.title}» ще недоступний.")
    return project_type.key


def require_known(key: object) -> ProjectTypeKey:
    if not isinstance(key, str) or key not in PROJECT_TYPES:
        raise ValueError("Тип проєкту: невідомий тип.")
    return PROJECT_TYPES[key].key
