"""Analysis blocks of a project (D-62, D-63): ordered stages per project type.

A block appears in the project menu as a link once it is implemented (``view`` is set);
planned blocks are shown greyed without a link. New blocks are added here as they appear.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    key: str
    title: str
    icon: str
    purpose: str
    view: str | None  # project page view; None = planned, no link yet

    @property
    def available(self) -> bool:
        return self.view is not None


BLOCKS: dict[str, Block] = {
    b.key: b
    for b in (
        Block("data", "Дані", "data", "Набори, таблиця й метадані: ціль, ознаки, ролі полів.", ""),
        Block(
            "eda",
            "Оглядовий аналіз",
            "compare",
            "Статистика, пропуски, розподіли й зв’язки — таблиці та графіки.",
            "eda",
        ),
        Block(
            "preparation",
            "Підготовка даних",
            "methods",
            "Пропуски, кодування категорій і поділ на навчальну та тестову вибірки.",
            "preparation",
        ),
        Block(
            "model",
            "Модель",
            "regression",
            "Лінійна регресія: коефіцієнти, R², MAE, RMSE.",
            "model",
        ),
        Block(
            "validation",
            "Перевірка моделі",
            "check",
            "Графік регресії за ознакою, похибки, прогноз проти факту.",
            "validation",
        ),
        Block(
            "conclusions",
            "Висновки",
            "spark",
            "Висновки AI з фактів моделі й посиланнями на правила та книжки; затверджує автор.",
            "conclusions",
        ),
        Block("report", "Звіт", "doc", "Збирання результатів аналізу у звіт.", "report"),
    )
}

# Classification and clustering are not enabled yet; their own blocks come later.
TYPE_BLOCKS: dict[str, list[str]] = {
    "regression": [
        "data",
        "eda",
        "preparation",
        "model",
        "validation",
        "conclusions",
        "report",
    ],
    "classification": ["data", "eda", "preparation", "conclusions", "report"],
    "clustering": ["data", "eda", "preparation", "conclusions", "report"],
}

# Tools inside the exploratory analysis block; the first one opens by default.
EDA_TOOLS: dict[str, str] = {
    "statistics": "Статистика",
    "heatmap": "Теплова карта",
    "histogram": "Гістограма",
    "scatter": "Розсіювання",
    "missing": "Графік пропусків",
}


def blocks_for(project_type: str) -> list[Block]:
    return [BLOCKS[key] for key in TYPE_BLOCKS.get(project_type, TYPE_BLOCKS["regression"])]


def block_link(block: Block, project_id: str) -> str | None:
    if block.view is None:
        return None
    return f"/?project={project_id}" + (f"&view={block.view}" if block.view else "")


def eda_link(project_id: str, tool: str, dataset_id: str | None = None) -> str:
    dataset = f"&dataset={dataset_id}" if dataset_id else ""
    return f"/?project={project_id}{dataset}&view=eda&tool={tool}"
