"""Explicitly reviewed book passages; unreviewed or changed text is excluded."""

import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name("reviewed.json")


def text_hash(chunk: dict) -> str:
    return hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()


def reviewed_chunks(chunks: list[dict], manifest: dict | None = None) -> list[dict]:
    if manifest is None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    approved = {row["id"]: row for row in manifest["fragments"]}
    return [
        chunk
        for chunk in chunks
        if chunk["id"] in approved
        and chunk["book"] == approved[chunk["id"]]["book"]
        and text_hash(chunk) == approved[chunk["id"]]["text_sha256"]
    ]
