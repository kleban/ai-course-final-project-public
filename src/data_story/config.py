"""Runtime location is derived from this repository, never the launch directory."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    database: Path = ROOT / "runtime" / "data-story.sqlite3"
    port: int = 8765

    @property
    def instance(self) -> str:
        return hashlib.sha256(str(self.database.resolve()).encode()).hexdigest()[:20]

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"
