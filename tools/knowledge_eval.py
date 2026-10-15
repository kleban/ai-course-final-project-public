"""Compare knowledge search methods on control, paraphrased and unrelated Ukrainian questions.

Runs on an isolated temporary database inside tmp/ (the author's data is not touched) and prints
hit@3 for TF-IDF alone and for the hybrid search, plus raw scores for calibration. The result is
also written to tmp/knowledge-eval.json.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_knowledge import CONTROL  # noqa: E402

from data_story import embeddings  # noqa: E402
from data_story.knowledge import KnowledgeStore  # noqa: E402

# Same meaning as a rule, but different words: the case TF-IDF is weak at.
PARAPHRASES = [
    ("наскільки добре модель пояснює розкид цін", "R² (R-квадрат) — коефіцієнт детермінації"),
    ("на скільки гривень у середньому помиляється прогноз", "MAE — середня абсолютна помилка"),
    (
        "модель чудово працює на тренуванні, але погано на нових даних",
        "Навчальна і тестова частини",
    ),
    ("чи можна вірити передбаченню для будинку на 20 кімнат", "Екстраполяція"),
    ("дві ознаки майже дублюють одна одну", "Мультиколінеарність"),
    ("що робити з порожніми клітинками в таблиці", "Пропуски: вилучити чи заповнити"),
    ("чи можна рахувати середнє по всій таблиці до поділу", "Витік даних"),
    ("як перетворити текстові значення типу району на числа", "Категорії і one-hot кодування"),
    ("номер замовлення варто брати в модель", "Ідентифікатори і дати"),
    ("на що вказують розкидані віялом залишки", "Графік похибок"),
    (
        "наскільки малоймовірний такий вплив, якщо насправді зв'язку немає",
        "p-value і зірочки значущості",
    ),
    ("ціна зростає, коли кімнат більше — на скільки саме", "Що означає коефіцієнт"),
    ("як показати форму розподілу цін", "Вибір графіка: Розподіл"),
    ("забагато кольорів і рамок на діаграмі", "Прибирання зайвого"),
    ("модель краща за просте середнє значення?", "Порівняння з простим прогнозом"),
]
# Held-out paraphrases (history 128): written before results were seen, never used for tuning.
NEW_PARAPHRASES = [
    ("яку частку мінливості пояснює модель", "R² (R-квадрат) — коефіцієнт детермінації"),
    ("середня помилка прогнозу в одиницях цілі", "MAE — середня абсолютна помилка"),
    (
        "чому великі помилки сильніше впливають на метрику",
        "RMSE — корінь середнього квадрата помилки",
    ),
    ("модель погіршилась на нових даних порівняно з навчальними", "Навчальна і тестова частини"),
    ("як прогнозувати для значень, яких не було в даних", "Екстраполяція"),
    ("ознаки корелюють і коефіцієнти нестабільні", "Мультиколінеарність"),
    ("чи варто прибрати рядки з пропущеними значеннями", "Пропуски: вилучити чи заповнити"),
    ("чи можна використати айді клієнта як ознаку", "Ідентифікатори і дати"),
    ("як показати частини цілого", "Вибір графіка: Склад"),
    ("як зобразити зміну показника в часі", "Вибір графіка: Динаміка"),
    ("чи потрібна вісь від нуля для барів", "Стовпці починаються з нуля"),
    ("що показують точки на діаграмі прогноз-факт", "Прогноз проти факту"),
    ("чи справедливі p-value, якщо похибки нерівномірні", "Припущення для p-value"),
    ("тестових рядків дуже мало, чи можна вірити метрикам", "Мала тестова вибірка"),
    ("один колір для головного, решта сіра", "Сірий контекст і один акцент"),
    ("скільки рядків відкласти для перевірки моделі", "Поділ на навчальну й тестову частини"),
    ("що означає, що зірочок біля коефіцієнта немає", "p-value і зірочки значущості"),
    ("як вибрати графік для порівняння категорій", "Вибір графіка: Порівняння"),
    ("чи брати дату продажу як число в модель", "Ідентифікатори і дати"),
    ("чим замінити відсутні значення числової ознаки", "Пропуски: вилучити чи заповнити"),
    ("на скільки зміниться прогноз, якщо ознака більша на одиницю", "Що означає коефіцієнт"),
    ("чому одна категорія зникла після кодування", "Категорії і лінійна залежність"),
    ("як показати зв'язок двох числових змінних", "Вибір графіка: Зв'язок"),
    ("модель гірша за середнє, що це означає", "Порівняння з простим прогнозом"),
    ("кольори мають бути зрозумілі людям з порушенням кольорозору", "Доступність"),
]
UNRELATED = [
    "рецепт борщу",
    "яка погода завтра у Львові",
    "як налаштувати wi-fi роутер",
    "хто виграв чемпіонат світу з футболу",
    "порадь фільм на вечір",
    "як вивчити англійську швидко",
    "скільки коштує квиток до Варшави",
    "напиши вірш про осінь",
]


def run(store: KnowledgeStore, use_model: bool) -> dict:
    context = patch.object(embeddings, "embedder", return_value=None) if not use_model else None
    if context:
        context.start()
    try:
        rows = []
        for group, items in (
            ("control", CONTROL),
            ("paraphrase", PARAPHRASES),
            ("paraphrase_new", NEW_PARAPHRASES),
            ("unrelated", [(q, None) for q in UNRELATED]),
        ):
            for question, expected in items:
                started = time.perf_counter()
                result = store.search(question, 3)
                sections = [h["section"] for h in result["hits"]]
                rows.append(
                    {
                        "group": group,
                        "question": question,
                        "expected": expected,
                        "ok": (expected in sections) if expected else not sections,
                        "hits": [
                            (h["section"], h["score"], h["lexical"], h["semantic"])
                            for h in result["hits"]
                        ],
                        "method": result["method"],
                        "ms": round((time.perf_counter() - started) * 1000),
                    }
                )
        return {"method": rows[0]["method"], "rows": rows}
    finally:
        if context:
            context.stop()


def raw_scores(store: KnowledgeStore) -> list[dict]:
    """Semantic cosine of the expected fragment and the best other one, for calibration."""
    model = embeddings.embedder()
    fragments = store.fragments()
    _, vectors = store._semantic(store._ensure_index()[0], fragments)
    result = []
    for group, items in (("control", CONTROL), ("paraphrase", PARAPHRASES), ("paraphrase_new", NEW_PARAPHRASES), ("unrelated", [(q, None) for q in UNRELATED])):  # fmt: skip
        for question, expected in items:
            cos = vectors @ model.encode([question], "query")[0]
            best = int(cos.argmax())
            index = next((i for i, f in enumerate(fragments) if f.section == expected), None)
            result.append(
                {
                    "group": group,
                    "question": question,
                    "expected_cos": None if index is None else round(float(cos[index]), 4),
                    "best": fragments[best].section,
                    "best_cos": round(float(cos[best]), 4),
                }
            )
    return result


BOOK_QUESTIONS = [
    "як інтерпретувати коефіцієнт множинної регресії",
    "що робити, якщо залишки мають форму лійки",
    "як писати висновки аналітичного звіту",
    "чому кореляція не означає причинність",
    "навіщо крос-валідація",
    "як масштабувати ознаки перед моделлю",
    "що таке стандартне відхилення",
    "як подати результати керівництву",
]


def book_scores(store: KnowledgeStore) -> list[dict]:
    """Top book fragments for typical questions and unrelated ones (library of the author)."""
    from data_story.knowledge_sources import SourceLibrary

    store.library = SourceLibrary(ROOT / "runtime" / "knowledge-sources")
    rows = []
    for group, items in (("book", BOOK_QUESTIONS), ("unrelated", UNRELATED)):
        for question in items:
            vector = embeddings.embedder().encode([question], "query")[0]
            chunks, books, vectors = store.library.load()
            if vectors is None:
                return [{"error": store.library.status()["reason"]}]
            scores = vectors @ vector
            top = np.argsort(-scores)[:3]
            rows.append(
                {
                    "group": group,
                    "question": question,
                    "top": [
                        (round(float(scores[i]), 4), chunks[i]["book"], chunks[i]["section"],
                         chunks[i]["pages"])
                        for i in top
                    ],
                }
            )  # fmt: skip
    for row in rows:
        best = row["top"][0]
        print("book", row["group"], best[0], row["question"], "|", best[1], best[2], best[3])
    return rows


def main() -> int:
    folder = ROOT / "tmp"
    folder.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        store = KnowledgeStore(Path(temp) / "eval.sqlite3")
        print("Status of embeddings:", embeddings.status() or "ready", flush=True)
        report = {"tfidf": run(store, False)}
        if embeddings.status() is None:
            started = time.perf_counter()
            report["hybrid"] = run(store, True)
            report["hybrid_seconds"] = round(time.perf_counter() - started, 1)
            report["raw"] = raw_scores(store)
            report["books"] = book_scores(store)
        for name in ("tfidf", "hybrid"):
            if name not in report:
                continue
            rows = report[name]["rows"]
            print(f"\n== {report[name]['method']} ==")
            for group in ("control", "paraphrase", "paraphrase_new", "unrelated"):
                part = [r for r in rows if r["group"] == group]
                print(f"{group}: {sum(r['ok'] for r in part)}/{len(part)}")
                for r in part:
                    if not r["ok"]:
                        print("   miss:", r["question"], "->", [h[0] for h in r["hits"]])
        for r in report.get("raw", []):
            print(
                "raw", r["group"], r["expected_cos"], r["best_cos"], r["question"], "|", r["best"]
            )
        (folder / "knowledge-eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    print("\nSaved tmp/knowledge-eval.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
