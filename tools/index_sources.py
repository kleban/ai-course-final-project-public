"""Compute bge-m3 vectors for the book fragments in runtime/knowledge-sources (with progress)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story.knowledge_sources import index  # noqa: E402

if __name__ == "__main__":
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runtime" / "knowledge-sources"
    index(folder, lambda text: print(text, flush=True))
