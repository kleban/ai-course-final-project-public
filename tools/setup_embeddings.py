"""Install the pinned local embeddings model for the knowledge base inside this repository."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_story.embeddings import install  # noqa: E402

if __name__ == "__main__":
    install()
