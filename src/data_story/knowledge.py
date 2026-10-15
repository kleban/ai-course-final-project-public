"""Knowledge base for RAG (stage 07): editable rule documents, fragments and local search.

Documents live in SQLite, seeded once from the repository defaults in ``knowledge/``; a future
editor changes the rows, never the defaults. Each ``##`` section is one fragment with its
document, section, source and priority. Search is fully local: TF-IDF over character n-grams
(scikit-learn), which tolerates Ukrainian word endings, combined with local neural embeddings
(``embeddings.py``) that match meaning rather than letters. Without the model the search falls
back to TF-IDF alone and says so in ``method``.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import embeddings
from .knowledge_sources import BOOK_MIN, BOOK_STRICT, SourceLibrary
from .storage import connect, ensure_schema

DEFAULTS = Path(__file__).parent / "knowledge"
METHOD = "tfidf-char_wb-3-5"
HYBRID = "hybrid-bge-m3+tfidf"
MIN_SCORE = 0.08
# Hybrid score = bge-m3 cosine stretched above SEMANTIC_FLOOR plus a small TF-IDF share. A
# fragment is returned when its meaning is close enough (SEMANTIC_MIN; unrelated questions
# peaked at 0.39) or its words match strongly (LEXICAL_MIN). Calibrated in history 091 on 15
# control, 15 paraphrased and 8 unrelated questions (tools/knowledge_eval.py).
SEMANTIC_FLOOR = 0.3
SEMANTIC_MIN = 0.42
LEXICAL_MIN = 0.2
LEXICAL_WEIGHT = 0.25
MAX_QUERY = 500
MAX_DOCUMENT = 20000
MAX_DOCUMENTS = 50


class KnowledgeError(ValueError):
    """The knowledge base cannot answer this request."""


@dataclass(frozen=True)
class Fragment:
    id: str
    document_id: str
    title: str
    section: str
    source: str
    priority: int
    text: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse(document_id: str, text: str) -> tuple[dict, list[Fragment]]:
    """Header ``# title``, ``Джерело:``, ``Пріоритет:``; every ``## section`` is a fragment."""
    lines = text.strip().splitlines()
    if not lines or not lines[0].startswith("# "):
        raise KnowledgeError("Документ бази знань має починатися із заголовка «# Назва».")
    title = lines[0][2:].strip()
    source, priority = "", 3
    for line in lines[1:4]:
        if line.startswith("Джерело:"):
            source = line.split(":", 1)[1].strip()
        if line.startswith("Пріоритет:"):
            priority = int(line.split(":", 1)[1].strip() or 3)
    fragments = []
    for number, block in enumerate(re.split(r"^## ", text, flags=re.M)[1:], 1):
        heading, _, body = block.partition("\n")
        body = " ".join(body.split())
        if body:
            fragments.append(
                Fragment(
                    f"{document_id}#{number}",
                    document_id,
                    title,
                    heading.strip(),
                    source,
                    priority,
                    body,
                )
            )
    if not fragments:
        raise KnowledgeError("Документ бази знань не містить розділів «## …».")
    meta = {"id": document_id, "title": title, "source": source, "priority": priority}
    return meta, fragments


# Fragment vectors shared by all stores in this process (key: model revision + fragment texts),
# so a new store or a test database does not re-embed the same rules.
_VECTORS: dict[str, np.ndarray] = {}


def passage(fragment: Fragment) -> str:
    return f"{fragment.title}. {fragment.section}. {fragment.text}"


class KnowledgeStore:
    def __init__(self, database: Path, defaults: Path = DEFAULTS):
        self.database = ensure_schema(database)
        self.defaults = defaults
        self._index: tuple[str, TfidfVectorizer, np.ndarray, list[Fragment]] | None = None
        self._vectors: tuple[str, np.ndarray] | None = None
        # Selected book chapters next to the database (runtime/knowledge-sources, outside Git).
        self.library = SourceLibrary(self.database.parent / "knowledge-sources")

    def _defaults(self) -> dict[str, str]:
        return {p.stem: p.read_text(encoding="utf-8") for p in sorted(self.defaults.glob("*.md"))}

    def _seed(self, connection) -> None:
        """Insert missing defaults and refresh unedited defaults; edited rows stay untouched."""
        rows = {
            r["id"]: r
            for r in connection.execute("SELECT id, text, is_default FROM knowledge_documents")
        }
        for document_id, text in self._defaults().items():
            row = rows.get(document_id)
            if row is not None and (not row["is_default"] or row["text"] == text):
                continue
            meta, _ = parse(document_id, text)
            connection.execute(
                "INSERT INTO knowledge_documents VALUES (?, ?, ?, ?, ?, 1, ?) ON CONFLICT(id) "
                "DO UPDATE SET title=excluded.title, source=excluded.source, "
                "priority=excluded.priority, text=excluded.text, updated_at=excluded.updated_at",
                (document_id, meta["title"], meta["source"], meta["priority"], text, _now()),
            )

    def _event(self, connection, operation: str, payload: dict) -> None:
        connection.execute(
            "INSERT INTO config_events VALUES (?, ?, ?, ?, ?)",
            (str(uuid4()), "knowledge", operation, _now(), json.dumps(payload, ensure_ascii=False)),
        )

    def _validated(self, document_id: str, text: str) -> tuple[dict, list[Fragment], str]:
        text = str(text).replace("\r\n", "\n").strip() + "\n"
        if len(text) > MAX_DOCUMENT:
            raise KnowledgeError(f"Документ задовгий: до {MAX_DOCUMENT} символів.")
        try:
            meta, fragments = parse(document_id, text)
        except KnowledgeError:
            raise
        except ValueError as error:  # a non-numeric priority
            raise KnowledgeError("Пріоритет має бути цілим числом від 1 до 9.") from error
        if not meta["title"]:
            raise KnowledgeError("Вкажіть назву документа в рядку «# Назва».")
        if not 1 <= meta["priority"] <= 9:
            raise KnowledgeError("Пріоритет має бути цілим числом від 1 до 9.")
        return meta, fragments, text

    def document(self, document_id: str) -> dict:
        found = next((d for d in self.documents() if d["id"] == document_id), None)
        if found is None:
            raise KnowledgeError("Документ бази знань не знайдено.")
        return found

    def save(self, document_id: str, text: str) -> dict:
        """Replace one document after validation; the previous text goes to the audit log."""
        previous = self.document(document_id)
        meta, fragments, text = self._validated(document_id, text)
        default = self._defaults().get(document_id)
        same_as_default = default is not None and default.strip() == text.strip()
        with connect(self.database) as connection:
            connection.execute(
                "UPDATE knowledge_documents SET title=?, source=?, priority=?, text=?, "
                "is_default=?, updated_at=? WHERE id=?",
                (meta["title"], meta["source"], meta["priority"], text, int(same_as_default),
                 _now(), document_id),
            )  # fmt: skip
            self._event(connection, "knowledge.saved", {
                "id": document_id, "title": meta["title"], "fragments": len(fragments),
                "previous_text": previous["text"],
            })  # fmt: skip
        return self.document(document_id)

    def create(self, text: str) -> dict:
        with connect(self.database) as connection:
            self._seed(connection)
            count = connection.execute("SELECT COUNT(*) FROM knowledge_documents").fetchone()[0]
        if count >= MAX_DOCUMENTS:
            raise KnowledgeError(f"У базі знань може бути до {MAX_DOCUMENTS} документів.")
        document_id = "user-" + uuid4().hex[:8]
        meta, fragments, text = self._validated(document_id, text)
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO knowledge_documents VALUES (?, ?, ?, ?, ?, 0, ?)",
                (document_id, meta["title"], meta["source"], meta["priority"], text, _now()),
            )
            self._event(connection, "knowledge.created",
                        {"id": document_id, "title": meta["title"], "fragments": len(fragments)})  # fmt: skip
        return self.document(document_id)

    def reset(self, document_id: str) -> dict:
        """Return a built-in document to its repository text."""
        default = self._defaults().get(document_id)
        if default is None:
            raise KnowledgeError("Це ваш документ: типової версії немає, його можна видалити.")
        previous = self.document(document_id)
        meta, _ = parse(document_id, default)
        with connect(self.database) as connection:
            connection.execute(
                "UPDATE knowledge_documents SET title=?, source=?, priority=?, text=?, "
                "is_default=1, updated_at=? WHERE id=?",
                (meta["title"], meta["source"], meta["priority"], default, _now(), document_id),
            )
            self._event(connection, "knowledge.reset",
                        {"id": document_id, "previous_text": previous["text"]})  # fmt: skip
        return self.document(document_id)

    def delete(self, document_id: str) -> None:
        """Delete a user document; built-in documents can only be reset."""
        if document_id in self._defaults():
            raise KnowledgeError("Типовий документ не видаляється — його можна повернути.")
        previous = self.document(document_id)
        with connect(self.database) as connection:
            connection.execute("DELETE FROM knowledge_documents WHERE id=?", (document_id,))
            self._event(connection, "knowledge.deleted",
                        {"id": document_id, "previous_text": previous["text"]})  # fmt: skip

    def documents(self) -> list[dict]:
        defaults = self._defaults()
        with connect(self.database) as connection:
            self._seed(connection)
            rows = connection.execute(
                "SELECT * FROM knowledge_documents ORDER BY priority, id"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "title": r["title"],
                "source": r["source"],
                "priority": r["priority"],
                "text": r["text"],
                "is_default": bool(r["is_default"]),
                "builtin": r["id"] in defaults,
                "updated_at": r["updated_at"],
                "fragments": len(parse(r["id"], r["text"])[1]),
            }
            for r in rows
        ]

    def fragments(self) -> list[Fragment]:
        result: list[Fragment] = []
        for document in self.documents():
            result.extend(parse(document["id"], document["text"])[1])
        return result

    def _ensure_index(self) -> tuple[str, TfidfVectorizer, np.ndarray, list[Fragment]]:
        fragments = self.fragments()
        fingerprint = hashlib.sha256(
            json.dumps([(f.id, f.section, f.text) for f in fragments], ensure_ascii=False).encode()
        ).hexdigest()
        if self._index is None or self._index[0] != fingerprint:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), lowercase=True, sublinear_tf=True
            )
            matrix = vectorizer.fit_transform(
                [f"{f.title}. {f.section}. {f.section}. {f.text}" for f in fragments]
            )
            self._index = (fingerprint, vectorizer, matrix, fragments)
        return self._index

    def _semantic(self, fingerprint: str, fragments: list[Fragment]):
        """Fragment vectors (cached in memory and next to the database) or None without a model."""
        model = embeddings.embedder()
        if model is None:
            return None, None
        key = f"{model.version}-{fingerprint[:16]}"
        if self._vectors is None or self._vectors[0] != key:
            cache = self.database.parent / "knowledge-index" / f"{key}.npy"
            vectors = _VECTORS.get(key)
            if vectors is None and cache.is_file():
                try:
                    vectors = np.load(cache)
                    if vectors.shape[0] != len(fragments):
                        vectors = None
                except (OSError, ValueError):
                    vectors = None
            if vectors is None:
                vectors = model.encode([passage(f) for f in fragments], "passage")
                try:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, vectors)
                except OSError:
                    pass  # the cache is an optimisation only
            _VECTORS[key] = vectors
            self._vectors = (key, vectors)
        return model, self._vectors[1]

    def search(self, query: str, limit: int = 3, books: int = 2) -> dict:
        query = " ".join(str(query).split())
        if not query:
            raise KnowledgeError("Введіть запитання для пошуку в базі знань.")
        if len(query) > MAX_QUERY:
            raise KnowledgeError(f"Запитання задовге: до {MAX_QUERY} символів.")
        started = perf_counter()
        fingerprint, vectorizer, matrix, fragments = self._ensure_index()
        lexical = (matrix @ vectorizer.transform([query.lower()]).T).toarray().ravel()
        query_vector = None
        semantic_reason = None
        try:
            model, vectors = self._semantic(fingerprint, fragments)
            if model is not None and vectors is not None:
                query_vector = model.encode([query], "query")[0]
                semantic = vectors @ query_vector
                if not np.isfinite(semantic).all():
                    raise ValueError("Embeddings повернула нескінченні або порожні значення.")
        except Exception as error:
            semantic_reason = embeddings.failed(error)
            self._vectors = None
            model, vectors, query_vector = None, None, None
        if model is None or vectors is None:
            method, semantic = METHOD, None
            scores = lexical
            accepted = lexical >= MIN_SCORE
        else:
            method = HYBRID
            stretched = np.clip((semantic - SEMANTIC_FLOOR) / (1 - SEMANTIC_FLOOR), 0, None)
            scores = stretched + LEXICAL_WEIGHT * lexical
            accepted = (semantic >= SEMANTIC_MIN) | (lexical >= LEXICAL_MIN)
        order = sorted(range(len(fragments)), key=lambda i: (-scores[i], fragments[i].priority, i))
        hits = [
            {
                "id": fragments[i].id,
                "document_id": fragments[i].document_id,
                "title": fragments[i].title,
                "section": fragments[i].section,
                "source": fragments[i].source,
                "priority": fragments[i].priority,
                "text": fragments[i].text,
                "score": round(float(scores[i]), 4),
                "lexical": round(float(lexical[i]), 4),
                "semantic": None if semantic is None else round(float(semantic[i]), 4),
            }
            for i in [i for i in order if accepted[i]][:limit]
        ]
        result = {
            "query": query,
            "method": method,
            "semantic_reason": (semantic_reason or embeddings.status())
            if semantic is None
            else None,
            "index": fingerprint[:12],
            "fragment_count": len(fragments),
            "hits": hits,
            "reason": None if hits else "У базі знань не знайдено достатньо схожих правил.",
        }
        book_hits, books_reason = (
            self.library.search(query_vector, books, BOOK_MIN if hits else BOOK_STRICT)
            if books > 0
            else ([], None)
        )
        result["book_hits"] = book_hits
        result["books_reason"] = books_reason
        with connect(self.database) as connection:
            connection.execute(
                "INSERT INTO config_events VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid4()),
                    "knowledge",
                    "knowledge.searched",
                    _now(),
                    json.dumps(
                        {
                            "query": query,
                            "limit": limit,
                            "method": method,
                            "index": fingerprint[:12],
                            "hits": [{"id": h["id"], "score": h["score"]} for h in hits],
                            "book_hits": [{"id": h["id"], "score": h["score"]} for h in book_hits],
                            "duration_ms": (perf_counter() - started) * 1000,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        return result

    def events(self, limit: int = 20) -> list[dict]:
        with connect(self.database) as connection:
            rows = connection.execute(
                "SELECT operation, occurred_at, payload FROM config_events "
                "WHERE subject = 'knowledge' ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "operation": r["operation"],
                "occurred_at": r["occurred_at"],
                **json.loads(r["payload"]),
            }
            for r in rows
        ]
