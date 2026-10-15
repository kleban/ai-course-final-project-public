"""Explicit live acceptance of the assistant, with Melbourne in an isolated workspace.

Uses the root .env and configured models. Calls OpenAI and consumes tokens.
Run deliberately: python tools/assistant_live_check.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story.ai_settings import AiSettingsStore  # noqa: E402
from data_story.config import Settings  # noqa: E402
from data_story.contracts import Failure  # noqa: E402
from data_story.openai_client import env_key  # noqa: E402
from data_story.projects import ProjectStore  # noqa: E402
from data_story.service import ProjectService  # noqa: E402


def main() -> int:
    if not env_key():
        print("Key missing in root .env")
        return 1
    folder = ROOT / "tmp" / ("084-assistant-live-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    folder.mkdir(parents=True, exist_ok=False)
    service = ProjectService(ProjectStore(folder / "data-story.sqlite3"))
    service.ai_settings.save(AiSettingsStore(Settings().database).get())
    project = service.store.create("Melbourne — live assistant acceptance")
    dataset = service.datasets.add(
        project.id, "melb_data.csv", (ROOT / "samples" / "melb_data.csv").read_bytes()
    )

    def data(result):
        if isinstance(result, Failure):
            raise RuntimeError(result.error.message)
        return result.data

    choices = {
        "dataset_id": dataset.id,
        "target": "Price",
        "features": ["Rooms", "Distance", "Bathroom"],
        "numeric_missing": "median",
    }
    first = data(service.training_prepare(project.id, choices))
    second = data(service.training_prepare(project.id, {**choices, "seed": 17}))
    model = data(service.model_train(project.id, {"training_run_id": first["id"]}))
    records = []

    def ask(name, query):
        result = service.assistant_ask(project.id, query)
        record = {"scenario": name, "query": query, "result": result.model_dump(mode="json")}
        records.append(record)
        (folder / "results.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        answer = data(result)
        print(
            json.dumps(
                {
                    "scenario": name,
                    "text": answer["text"],
                    "question": answer["question"],
                    "tools": answer["tools"],
                    "tokens": answer["usage_total"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return answer

    explained = ask(
        "explain",
        {
            "block": "model",
            "mode": "explain",
            "question": "Поясни тестові R², MAE і RMSE збереженої моделі Price. Назви фактичні числа та одне обмеження.",
            "page": {"run_id": first["id"]},
        },
    )
    clarify_query = {
        "block": "model",
        "mode": "execute",
        "question": "Навчи модель, але я ще не обрав, яку зі збережених вибірок використати. Запитай мене.",
    }
    clarified = ask("clarify", clarify_query)
    trained = ask(
        "train",
        {
            "block": "model",
            "mode": "execute",
            "question": f"Так, підтверджую: навчи лінійну регресію на вибірках {second['id']}. Виконай інструмент навчання.",
            "page": {"run_id": second["id"]},
            "history": [
                {"role": "user", "text": clarify_query["question"]},
                {"role": "assistant", "text": clarified["text"]},
            ],
        },
    )
    chart = ask(
        "residual_chart",
        {
            "block": "validation",
            "mode": "execute",
            "question": f"Підтверджую: побудуй графік похибок (residuals) моделі {model['id']} і поясни його за фактичними результатами інструмента.",
            "page": {"run_id": model["id"], "tool": "residuals"},
        },
    )
    checks = {
        "explanation_has_text": bool(explained["text"].strip()),
        "explanation_has_sources": bool(explained["sources"]),
        "clarification_has_options": bool(
            clarified["question"] and clarified["question"]["options"]
        ),
        "clarification_did_not_execute": clarified["tools"] == [],
        "train_tool_succeeded": {"name": "train_linear_regression", "ok": True} in trained["tools"],
        "new_model_saved": len(data(service.model_runs(project.id))) == 2,
        "chart_tool_succeeded": {"name": "build_model_chart", "ok": True} in chart["tools"],
        "secret_absent_from_logs": all(
            env_key() not in json.dumps(call) for call in service.openai.calls(100000)
        ),
    }
    summary = {
        "checks": checks,
        "totals": service.openai.totals(),
        "model_metrics": model["document"]["metrics"],
        "workspace": str(folder),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
