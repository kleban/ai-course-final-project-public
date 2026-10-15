"""Build knowledge-base fragments from selected book chapters (PDF -> text -> chunks).

Needs the ``pdftotext`` command (poppler). Reads the selection in
``src/data_story/knowledge_sources/selection.json`` and PDFs from a folder (default ``sources/``),
writes ``runtime/knowledge-sources/chunks.jsonl`` and ``books.json``. Texts stay outside Git.
Vectors are computed separately on the machine with the model: ``tools/index_sources.py``.

Usage: python tools/build_sources.py [pdf_folder] [output_folder]
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story.knowledge_sources.curation import reviewed_chunks  # noqa: E402

SELECTION = ROOT / "src" / "data_story" / "knowledge_sources" / "selection.json"
CODE = re.compile(
    r"^(In|Out)\[\d*\]|^\s*(>>>|\.\.\.|#|from \w+ import|import \w+|print\()"
    r"|\w+\s*=\s*[\w.]+\(|\.fit\(|\.predict\(|dtype:|^(Training|Test) set score"
    r"|\.indd\s+\d+|^\s*(Figure|Table|Box)\s+[\d.-]+\s*$"
)


def page_texts(pdf: Path) -> list[str]:
    result = subprocess.run(
        ["pdftotext", "-enc", "UTF-8", str(pdf), "-"], capture_output=True, check=True
    )
    return result.stdout.decode("utf-8", "replace").split("\f")


def useful(line: str) -> bool:
    text = line.strip()
    if not text or CODE.search(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 3 or letters / max(len(text), 1) < 0.55:
        return False
    words = re.findall(r"[A-Za-zÀ-ÿ]{2,}", text)
    symbols = sum(ch in "=()[]{}_<>#;" for ch in text)
    return not (symbols / max(len(text), 1) > 0.08 and len(words) < 6)


def heading(line: str, following: str) -> bool:
    text = line.strip()
    words = text.split()
    if not 1 <= len(words) <= 10 or len(text) > 80 or text[-1:] in ".,;:?)":
        return False
    if ":" in text or re.match(r"^(Figure|Table|Box|Example|Source)\b", text):
        return False
    if sum(ch.isdigit() for ch in text) > 6 or "=" in text:
        return False
    if re.match(r"^\d+(\.\d+)+\s+[A-Z]", text):
        return True
    return text[:1].isupper() and len(following.split()) >= 8 and not text.isupper()


def clean_page(text: str, running: set[str]) -> list[tuple[str, bool]]:
    lines = [line.rstrip() for line in text.splitlines()]
    out: list[tuple[str, bool]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped in running or re.fullmatch(r"\d{1,4}|[ivxlc]{1,6}|\|", stripped):
            continue
        if not useful(line):
            continue
        following = next((n for n in lines[index + 1 :] if n.strip()), "")
        out.append((stripped, heading(stripped, following)))
    return out


def running_headers(pages: list[str]) -> set[str]:
    """Short lines repeated on many pages (book, chapter and section running titles)."""
    counts: dict[str, int] = {}
    for page in pages:
        for line in {line.strip() for line in page.splitlines() if line.strip()}:
            counts[line] = counts.get(line, 0) + 1
    limit = max(4, len(pages) // 8)
    return {line for line, count in counts.items() if count >= limit and len(line) < 90}


def chunks_for(book: dict, pages: list[str], size: int, overlap: int) -> list[dict]:
    result: list[dict] = []
    for start, end, chapter in book["ranges"]:
        selected = pages[start - 1 : end]
        running = running_headers(selected)
        words: list[tuple[str, int]] = []
        sections: list[tuple[int, str]] = []
        section = chapter
        for number, text in enumerate(selected, start):
            for line, is_heading in clean_page(text, running):
                if is_heading:
                    section = re.sub(r"\s+", " ", line)
                    sections.append((len(words), section))
                    continue
                line = re.sub(r"-$", "-", line)
                for word in line.split():
                    words.append((word, number))
        # Join hyphenated line breaks ("hyper-" + "plane").
        merged: list[tuple[str, int]] = []
        for word, number in words:
            if merged and merged[-1][0].endswith("-") and word[:1].islower():
                merged[-1] = (merged[-1][0][:-1] + word, merged[-1][1])
            else:
                merged.append((word, number))
        position = 0
        while position < len(merged):
            part = merged[position : position + size]
            # End on a sentence boundary when one is near.
            cut = len(part)
            for back in range(len(part) - 1, max(len(part) - 60, 0), -1):
                if part[back][0].endswith((".", "?", "!")):
                    cut = back + 1
                    break
            part = part[:cut]
            text = " ".join(w for w, _ in part)
            if len(part) >= 40:
                title = next((s for at, s in reversed(sections) if at <= position), chapter)
                pages_used = sorted({n for _, n in part})
                result.append(
                    {
                        "book": book["id"],
                        "chapter": chapter,
                        "section": title,
                        "pages": [pages_used[0], pages_used[-1]],
                        "text": text,
                    }
                )
            if position + cut >= len(merged):
                break
            position += max(cut - overlap, 1)
    for index, chunk in enumerate(result):
        chunk["id"] = f"{book['id']}-{index:04d}"
    return result


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "sources"
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "runtime" / "knowledge-sources"
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    books, chunks = [], []
    for book in selection["books"]:
        pdf = source / book["file"]
        if not pdf.is_file():
            print(f"skip {book['id']}: {pdf.name} not found", flush=True)
            continue
        pages = page_texts(pdf)
        made = chunks_for(book, pages, selection["chunk_words"], selection["overlap_words"])
        extracted = len(made)
        made = reviewed_chunks(made)
        if not made:
            print(f"skip {book['id']}: no reviewed passages in {extracted} fragments", flush=True)
            continue
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        meta = {k: book[k] for k in ("id", "title", "authors", "year", "why", "ranges")}
        books.append({**meta, "pdf_sha256": digest, "chunks": len(made)})
        chunks.extend(made)
        print(f"{book['id']}: {len(made)}/{extracted} reviewed fragments", flush=True)
    with (output / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    (output / "books.json").write_text(
        json.dumps({"books": books, "skipped": selection["skipped"]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"total {len(chunks)} fragments -> {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
