"""Service monitoring (stage 08b, D-49): which parts of Data Story work, checked locally.

Every check is cheap and offline: no OpenAI call, no network. Results are not cached; each run
is written to ``config_events`` (subject ``monitoring``) as a short summary without secrets.
"""

import importlib.metadata
import json
import os
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .ai_settings import key_status
from .config import ROOT
from .storage import SCHEMA_VERSION, connect

OK, WARNING, ERROR, NOT_CONFIGURED, NOT_IMPLEMENTED = (
    "ok",
    "warning",
    "error",
    "not_configured",
    "not_implemented",
)
STATE_TITLES = {
    OK: "Працює",
    WARNING: "Працює із зауваженнями",
    ERROR: "Недоступна",
    NOT_CONFIGURED: "Не налаштована",
    NOT_IMPLEMENTED: "Ще не реалізована",
}
PACKAGES = ["nicegui", "fastapi", "pandas", "numpy", "scikit-learn", "scipy", "httpx", "pydantic"]
ASSETS = Path(__file__).parent / "assets"
LOCAL_ASSETS = {
    "Шрифти": ASSETS / "fonts" / "OpenSans.ttf",
    "MathJax": ASSETS / "vendor" / "mathjax" / "tex-svg.js",
    "Редактор Markdown": ASSETS / "vendor" / "toastui",
    "Стилі": ASSETS / "style.css",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# The main view shows the parts the service runs on (author feedback 5, history 097); the
# rest are technical details of the whole project, shown collapsed.
SERVICES = ("server", "database", "duckdb", "knowledge", "reports", "openai_access")


def _row(key: str, title: str, state: str, detail: str) -> dict:
    return {"key": key, "title": title, "state": state, "state_title": STATE_TITLES[state],
            "detail": detail, "group": "service" if key in SERVICES else "detail"}  # fmt: skip


def lock_versions(lock: Path = ROOT / "requirements.lock") -> dict[str, str]:
    if not lock.is_file():
        return {}
    pins = {}
    for line in lock.read_text(encoding="utf-8").splitlines():
        if "==" in line:
            name, version = line.strip().split("==", 1)
            pins[name.lower().replace("_", "-")] = version
    return pins


class Monitor:
    def __init__(self, service, lock: Path = ROOT / "requirements.lock"):
        self.service = service
        self.database: Path = service.store.database
        self.lock = lock

    def server(self) -> dict:
        return _row("server", "Вебсервер і API", OK,
                    "Сторінку й API обслуговує цей процес; слухає лише 127.0.0.1.")  # fmt: skip

    def database_check(self) -> dict:
        with connect(self.database) as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            check = connection.execute("PRAGMA quick_check").fetchone()[0]
            projects = connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        size = self.database.stat().st_size / 1024 / 1024
        detail = f"Схема v{version}, проєктів {projects}, розмір {size:.1f} МБ."
        if check != "ok":
            return _row(
                "database", "База SQLite", ERROR, f"Перевірка цілісності: {check}. {detail}"
            )
        if version != SCHEMA_VERSION:
            return _row("database", "База SQLite", WARNING,
                        f"Очікувана схема v{SCHEMA_VERSION}. {detail}")  # fmt: skip
        return _row("database", "База SQLite", OK, detail)

    def files(self) -> dict:
        root: Path = self.service.datasets.files_root
        if not root.exists():
            return _row("files", "Файли наборів", OK, "Наборів ще немає; каталог створиться.")
        if not os.access(root, os.W_OK):
            return _row("files", "Файли наборів", ERROR, f"Немає права запису: {root.name}.")
        with connect(self.database) as connection:
            stored = [r[0] for r in connection.execute("SELECT stored_path FROM datasets")]
        missing = sum(1 for path in stored if not (root / path).is_file())
        detail = f"Наборів {len(stored)}, відсутніх файлів {missing}."
        return _row("files", "Файли наборів", WARNING if missing else OK, detail)

    def packages(self) -> dict:
        pins = lock_versions(self.lock)
        problems, found = [], []
        for name in PACKAGES:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                problems.append(f"{name} не встановлено")
                continue
            found.append(f"{name} {version}")
            pinned = pins.get(name)
            if pinned and pinned != version:
                problems.append(f"{name} {version} ≠ lock {pinned}")
        if any("не встановлено" in p for p in problems):
            return _row("packages", "Python-залежності", ERROR, "; ".join(problems))
        if problems:
            return _row("packages", "Python-залежності", WARNING,
                        "Версії відрізняються від requirements.lock: " + "; ".join(problems))  # fmt: skip
        return _row("packages", "Python-залежності", OK, ", ".join(found))

    def assets(self) -> dict:
        missing = [name for name, path in LOCAL_ASSETS.items() if not path.exists()]
        if missing:
            return _row("assets", "Локальні ресурси інтерфейсу", ERROR,
                        "Відсутні: " + ", ".join(missing))  # fmt: skip
        return _row("assets", "Локальні ресурси інтерфейсу", OK,
                    "Шрифти, MathJax, редактор і стилі на місці; CDN не потрібен.")  # fmt: skip

    def knowledge(self) -> dict:
        documents = self.service.knowledge.documents()
        fragments = sum(d["fragments"] for d in documents)
        self.service.knowledge._ensure_index()  # builds the index without logging a search
        if not fragments:
            return _row(
                "knowledge", "Пошук у базі знань (RAG)", NOT_CONFIGURED, "Документів немає."
            )
        from . import embeddings

        reason = embeddings.status()
        library = self.service.knowledge.library.status()
        books = (
            f" Книжки: {len(library['books'])}, фрагментів {library['fragments']}"
            + ("" if library["indexed"] else " (не проіндексовано)")
            + "."
            if library["fragments"]
            else ""
        )
        if reason:
            return _row("knowledge", "Пошук у базі знань (RAG)", WARNING,
                        f"Документів {len(documents)}, фрагментів {fragments}; лише TF-IDF. "
                        + reason + books)  # fmt: skip
        return _row("knowledge", "Пошук у базі знань (RAG)", OK,
                    f"Документів {len(documents)}, фрагментів {fragments}; гібридний пошук: "
                    f"embeddings {embeddings.MODEL} + TF-IDF, локально." + books)  # fmt: skip

    def openai_access(self) -> dict:
        # The key itself is not shown here (author feedback 5); only whether AI can work.
        status = key_status(self.service.ai_settings.env_file)
        if status["state"] != "found":
            state = WARNING if status["state"] == "suspicious" else NOT_CONFIGURED
            return _row("openai_access", "Доступ до OpenAI", state,
                        "Ключ OpenAI не налаштовано або він некоректний: «Моделі AI».")  # fmt: skip
        checks = [c for c in self.service.openai.calls(limit=200) if c["task"] == "access_check"]
        if not checks:
            return _row("openai_access", "Доступ до OpenAI", NOT_CONFIGURED,
                        "Ще не перевірявся: кнопка «Перевірити доступ» у «Моделі AI».")  # fmt: skip
        last_run = checks[0]["run_id"]
        latest = [c for c in checks if c["run_id"] == last_run]
        bad = [c for c in latest if c["status"] != "success"]
        when = latest[0]["started_at"][:16].replace("T", " ")
        if bad:
            return _row("openai_access", "Доступ до OpenAI", ERROR,
                        f"Остання перевірка {when} UTC: {bad[0].get('error', 'помилка')}")  # fmt: skip
        return _row("openai_access", "Доступ до OpenAI", OK,
                    f"Остання перевірка {when} UTC: доступно моделей {len(latest)}.")  # fmt: skip

    def errors(self) -> dict:
        since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        with connect(self.database) as connection:
            events = connection.execute(
                "SELECT payload FROM project_events WHERE occurred_at >= ?", (since,)
            ).fetchall()
            ai = connection.execute(
                "SELECT COUNT(*) FROM ai_calls WHERE started_at >= ? AND status != 'success'",
                (since,),
            ).fetchone()[0]
        failed = sum(1 for e in events if json.loads(e[0]).get("status") == "error")
        detail = f"За 24 год: помилок операцій {failed}, невдалих спроб AI {ai}."
        return _row("errors", "Журнал помилок", WARNING if failed or ai else OK, detail)

    def duckdb(self) -> dict:
        try:
            version = importlib.metadata.version("duckdb")
        except importlib.metadata.PackageNotFoundError:
            return _row("duckdb", "DuckDB", ERROR,
                        "Не встановлено: агрегація й з'єднання наборів недоступні. "
                        "Запустіть Setup.cmd.")  # fmt: skip
        analytics = self.service.analytics
        if not analytics.database.exists():
            return _row("duckdb", "DuckDB", OK,
                        f"Пакет {version}; база створиться після завантаження даних. "
                        "SQL від AI виконує сервіс у межах поточного проєкту.")  # fmt: skip
        try:
            with analytics._connection() as connection:
                tables = connection.execute("SELECT count(*) FROM _analytics_catalog").fetchone()[0]
                results = connection.execute("SELECT count(*) FROM _analytics_results").fetchone()[
                    0
                ]
                missing = connection.execute(
                    "SELECT count(*) FROM _analytics_catalog c LEFT JOIN information_schema.tables t "
                    "ON c.schema_name=t.table_schema AND c.table_name=t.table_name "
                    "WHERE t.table_name IS NULL"
                ).fetchone()[0]
        except Exception:
            return _row("duckdb", "DuckDB", ERROR,
                        "Аналітичну базу не вдалося прочитати. Перевірте файл і доступ до нього.")  # fmt: skip
        size = analytics.database.stat().st_size / 1024 / 1024
        return _row("duckdb", "DuckDB", ERROR if missing else OK,
                    f"Пакет {version}; таблиць {tables}, результатів {results}, "
                    f"відсутніх таблиць {missing}; {size:.1f} МБ. "
                    "SQL від AI виконує сервіс у межах поточного проєкту.")  # fmt: skip

    def matrices(self) -> dict:
        """Saved train/test matrices exist and match their SHA-256; models link to them."""
        with connect(self.database) as connection:
            runs = connection.execute("SELECT project_id, id FROM training_runs").fetchall()
            models = connection.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0]
        broken = 0
        for project_id, run_id in runs:
            try:
                self.service.training_store.matrices(project_id, run_id)
            except Exception:
                broken += 1
        detail = f"Вибірок {len(runs)}, пошкоджених або відсутніх {broken}; моделей {models}."
        return _row("matrices", "Вибірки й моделі", ERROR if broken else OK, detail)

    def storage(self) -> dict:
        """What is stored where and how big: database, dataset files, matrices, logs."""
        root: Path = self.service.datasets.files_root

        def size(paths) -> tuple[int, float]:
            files = [p for p in paths if p.is_file()]
            return len(files), sum(p.stat().st_size for p in files) / 1024 / 1024

        data_n, data_mb = size(root.glob("*/datasets/**/*")) if root.exists() else (0, 0.0)
        npz_n, npz_mb = size(root.glob("*/training/*/*.npz")) if root.exists() else (0, 0.0)
        db_mb = self.database.stat().st_size / 1024 / 1024
        analytical = self.service.analytics.database
        analytics_mb = analytical.stat().st_size / 1024 / 1024 if analytical.exists() else 0.0
        with connect(self.database) as connection:
            calls = connection.execute("SELECT COUNT(*) FROM ai_calls").fetchone()[0]
            events = connection.execute("SELECT COUNT(*) FROM project_events").fetchone()[0]
        detail = (
            f"SQLite {db_mb:.1f} МБ · DuckDB {analytics_mb:.1f} МБ · файли наборів {data_n} ({data_mb:.1f} МБ) · матриці {npz_n} "
            f"({npz_mb:.1f} МБ) · подій журналу {events} · викликів AI {calls}. "
            f"Усе в {self.database.parent.name}/ всередині проєкту."
        )
        return _row("storage", "Сховище даних", OK, detail)

    def planned(self) -> list[dict]:
        from .quarto_runtime import version

        installed = version()
        return [
            _row(
                "reports",
                "Генерація звітів (Quarto)",
                OK if installed else NOT_CONFIGURED,
                "Локальний Quarto " + installed if installed else "Запустіть Setup.cmd.",
            )
        ]

    def run(self) -> dict:
        started = perf_counter()
        checks: list[Callable[[], dict]] = [
            self.server, self.database_check, self.files, self.packages, self.assets,
            self.storage, self.matrices, self.knowledge, self.openai_access,
            self.errors, self.duckdb,
        ]  # fmt: skip
        rows = []
        for check in checks:
            try:
                rows.append(check())
            except Exception as error:  # a broken part must not hide the others
                name = check.__name__
                rows.append(
                    _row(name, name, ERROR, f"Перевірка не виконалась: {type(error).__name__}.")
                )
        rows += self.planned()
        result = {
            "checked_at": _now(),
            "rows": rows,
            "summary": {
                state: sum(r["state"] == state for r in rows if r["group"] == "service")
                for state in STATE_TITLES
            },
            "detail_problems": sum(
                r["state"] in (WARNING, ERROR) for r in rows if r["group"] == "detail"
            ),
            "duration_ms": (perf_counter() - started) * 1000,
        }
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO config_events VALUES (?, ?, ?, ?, ?)",
                (str(uuid4()), "monitoring", "monitoring.checked", result["checked_at"],
                 json.dumps({"summary": result["summary"],
                             "states": {r["key"]: r["state"] for r in rows},
                             "duration_ms": result["duration_ms"]}, ensure_ascii=False)),
            )  # fmt: skip
        return result
