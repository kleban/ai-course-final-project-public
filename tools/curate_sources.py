"""Clean the existing book corpus and subset unchanged vectors; keep a local backup."""

import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story.knowledge_sources import SourceLibrary  # noqa: E402
from data_story.knowledge_sources.curation import MANIFEST, reviewed_chunks  # noqa: E402


def curate(folder: Path) -> dict:
    folder = folder.resolve()
    if not folder.is_relative_to(ROOT):
        raise ValueError("Корпус має бути всередині проєкту.")
    library = SourceLibrary(folder)
    chunks, _, vectors = library.load()
    before = library.fingerprint()
    kept = reviewed_chunks(chunks)
    if not kept:
        raise ValueError("Немає перевірених уривків; оригінальний корпус не змінено.")
    if len({row["id"] for row in chunks}) != len(chunks):
        raise ValueError("Повторені ID уривків; корпус не змінено.")
    if kept == chunks:
        return {"before": len(chunks), "after": len(kept), "changed": False}
    backup = Path(tempfile.mkdtemp(prefix="corpus-before-099-", dir=ROOT / "tmp"))
    for name in ("chunks.jsonl", "books.json"):
        shutil.copy2(folder / name, backup / name)
    selected = {row["id"] for row in kept}
    indices = [i for i, row in enumerate(chunks) if row["id"] in selected]
    if vectors is not None:
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError("Некоректна матриця векторів; корпус не змінено.")
        shutil.copy2(library.vector_path(before), backup)
    counts = Counter(row["book"] for row in kept)
    topics = json.loads(MANIFEST.read_text(encoding="utf-8"))["topics"]
    metadata = json.loads((folder / "books.json").read_text(encoding="utf-8"))
    metadata["books"] = [
        {**book, "chunks": counts[book["id"]], "why": topics[book["id"]]}
        for book in metadata["books"]
        if counts[book["id"]]
    ]
    metadata["curation"] = {
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "before": len(chunks),
        "after": len(kept),
        "source_fingerprint": before,
    }
    pending = backup / "cleaned"
    pending.mkdir()
    (pending / "chunks.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8"
    )
    (pending / "books.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    after = SourceLibrary(pending).fingerprint()
    new_path = library.vector_path(after)
    if vectors is not None:
        np.save(new_path, vectors[indices])
    (pending / "books.json").replace(folder / "books.json")
    (pending / "chunks.jsonl").replace(folder / "chunks.jsonl")
    # Old matrices contain the removed examples; keep them only in the backup.
    for old in folder.glob("vectors-*.npy"):
        if old != new_path:
            if not (backup / old.name).exists():
                shutil.copy2(old, backup)
            old.unlink()
    result = {
        "before": len(chunks),
        "after": len(kept),
        "removed": len(chunks) - len(kept),
        "books": len(counts),
        "vectors_preserved": vectors is not None,
        "backup": str(backup),
        "changed": True,
    }
    (backup / "verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime/knowledge-sources"
    print(json.dumps(curate(folder), ensure_ascii=False, indent=2))
