"""Book library for RAG: selected chapters as fragments, searched by meaning with bge-m3.

Fragments (``chunks.jsonl``) and book descriptions (``books.json``) are built from the author's
PDFs by ``tools/build_sources.py``; vectors are computed once by ``tools/index_sources.py`` on the
machine with the model. Everything lives next to the database in ``knowledge-sources/`` (under
``runtime/``, outside Git: the books are copyrighted). Without vectors or the model the library
answers with a reason instead of hits.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from .. import embeddings

SELECTION = Path(__file__).with_name("selection.json")
# bge-m3 cosine for a book fragment to count as relevant (calibrated in history 093 on 8 typical
# and 8 unrelated questions: relevant 0.51-0.63, unrelated up to 0.50). The stricter level applies
# when no rule of the knowledge base matched, i.e. the question is probably off-topic.
BOOK_MIN = 0.48
BOOK_STRICT = 0.55
BATCH = 8


def passage(chunk: dict, books: dict) -> str:
    title = books.get(chunk["book"], {}).get("title", chunk["book"])
    return f"{title}. {chunk['section']}. {chunk['text']}"


class SourceLibrary:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._lock = threading.Lock()
        self._cache: tuple[str, list[dict], dict, np.ndarray | None] | None = None

    def _files(self) -> tuple[Path, Path]:
        return self.folder / "chunks.jsonl", self.folder / "books.json"

    def fingerprint(self) -> str | None:
        chunks, books = self._files()
        if not chunks.is_file() or not books.is_file():
            return None
        return hashlib.sha256(chunks.read_bytes()).hexdigest()

    def vector_path(self, fingerprint: str) -> Path:
        version = embeddings.CONFIG["revision"][:12]
        return self.folder / f"vectors-{version}-{fingerprint[:16]}.npy"

    def load(self) -> tuple[list[dict], dict, np.ndarray | None]:
        fingerprint = self.fingerprint()
        if fingerprint is None:
            return [], {}, None
        with self._lock:
            if self._cache is None or self._cache[0] != fingerprint:
                chunks_file, books_file = self._files()
                chunks = [
                    json.loads(line)
                    for line in chunks_file.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                books = {
                    b["id"]: b for b in json.loads(books_file.read_text(encoding="utf-8"))["books"]
                }
                vectors = None
                path = self.vector_path(fingerprint)
                if path.is_file():
                    try:
                        loaded = np.load(path)
                        if loaded.shape[0] == len(chunks):
                            vectors = loaded
                    except (OSError, ValueError):
                        vectors = None
                self._cache = (fingerprint, chunks, books, vectors)
            return self._cache[1], self._cache[2], self._cache[3]

    def status(self) -> dict:
        chunks, books, vectors = self.load()
        reason = None
        if not chunks:
            reason = "Книжки ще не додано до бази знань."
        elif vectors is None:
            reason = "Книжки не проіндексовано: запустіть tools/index_sources.py."
        elif embeddings.status():
            reason = embeddings.status()
        return {
            "books": [
                {k: b.get(k) for k in ("id", "title", "authors", "year", "why", "chunks")}
                for b in books.values()
            ],
            "fragments": len(chunks),
            "indexed": vectors is not None,
            "reason": reason,
        }

    def search(
        self, query_vector: np.ndarray | None, limit: int = 2, minimum: float = BOOK_MIN
    ) -> tuple[list[dict], str | None]:
        chunks, books, vectors = self.load()
        if not chunks:
            return [], None
        if vectors is None or query_vector is None:
            return [], self.status()["reason"] or embeddings.status()
        scores = vectors @ query_vector
        order = np.argsort(-scores)
        hits: list[dict] = []
        for index in order[: max(limit * 4, limit)]:
            if scores[index] < minimum or len(hits) >= limit:
                break
            chunk = chunks[int(index)]
            book = books.get(chunk["book"], {})
            first, last = chunk["pages"]
            pages = f"с. {first}" if first == last else f"с. {first}–{last}"
            hits.append(
                {
                    "id": chunk["id"],
                    "kind": "book",
                    "document_id": chunk["book"],
                    "title": book.get("title", chunk["book"]),
                    "section": chunk["section"],
                    "source": f"{book.get('authors', '')}, {book.get('year', '')}; {pages} (PDF)",
                    "priority": 5,
                    "text": chunk["text"],
                    "score": round(float(scores[index]), 4),
                    "semantic": round(float(scores[index]), 4),
                    "lexical": None,
                }
            )
        return hits, None


def index(folder: Path, report=print) -> Path:
    """Compute and save fragment vectors with the installed model, with progress output."""
    library = SourceLibrary(folder)
    chunks, books, vectors = library.load()
    fingerprint = library.fingerprint()
    if not chunks or fingerprint is None:
        raise RuntimeError(f"Немає фрагментів у {folder}. Спершу tools/build_sources.py.")
    target = library.vector_path(fingerprint)
    if vectors is not None:
        report(f"Already indexed: {len(chunks)} fragments ({target.name}).")
        return target
    model = embeddings.embedder()
    if model is None:
        raise RuntimeError(embeddings.status() or "Модель embeddings недоступна.")
    texts = [passage(c, books) for c in chunks]
    parts, started, shown = [], time.monotonic(), -5
    for start in range(0, len(texts), BATCH):
        parts.append(model.encode(texts[start : start + BATCH], "passage"))
        done = min(start + BATCH, len(texts))
        percent = done * 100 // len(texts)
        if percent >= shown + 5 or done == len(texts):
            shown = percent
            elapsed = time.monotonic() - started
            left = elapsed / done * (len(texts) - done)
            report(f"  {percent}% ({done}/{len(texts)}), залишилось ~{left / 60:.1f} хв")
    matrix = np.vstack(parts).astype(np.float32)
    partial = target.with_suffix(".part.npy")
    np.save(partial, matrix)
    partial.replace(target)
    report(f"Indexed {len(texts)} fragments in {(time.monotonic() - started) / 60:.1f} min.")
    return target
