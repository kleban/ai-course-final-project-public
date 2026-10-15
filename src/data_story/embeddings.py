"""Local neural embeddings for RAG: a pinned multilingual model run with ONNX Runtime.

The model (BAAI/bge-m3, MIT, multilingual incl. Ukrainian; chosen in history 091 over
multilingual-e5-small/base and multilingual-mpnet on control, paraphrased and unrelated
questions) is downloaded once by Setup into ``runtime/dependencies/embeddings`` with SHA-256 checks; afterwards everything is
offline. onnxruntime and tokenizers are imported lazily, so the service still starts (and the
knowledge base falls back to TF-IDF) when the model or the packages are missing.
"""

import hashlib
import json
import logging
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from .config import ROOT

CONFIG = json.loads(Path(__file__).with_name("embeddings.json").read_text(encoding="utf-8"))
MODEL = CONFIG["model"]
INSTALL = (ROOT / CONFIG["installation"]).resolve()
if not INSTALL.is_relative_to(ROOT):
    raise RuntimeError("Модель embeddings має бути всередині проєкту.")
BATCH = 16
RETRY_SECONDS = 120
LOG = logging.getLogger(__name__)


def _path(name: str) -> Path:
    return INSTALL / Path(name).name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def missing() -> str | None:
    """Why embeddings cannot run (cheap check: files and packages), or None when ready."""
    absent = [Path(n).name for n in CONFIG["files"] if not _path(n).is_file()]
    if absent:
        return "Модель embeddings не встановлена (" + ", ".join(absent) + "). Запустіть Setup.cmd."
    try:
        import onnxruntime  # noqa: F401
        import tokenizers  # noqa: F401
    except ImportError:
        return "Пакети onnxruntime/tokenizers не встановлені. Запустіть Setup.cmd."
    return None


class Embedder:
    """L2-normalised sentence vectors: the model's pooled output, else CLS or mean pooling."""

    def __init__(self) -> None:
        from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions
        from tokenizers import Tokenizer

        options = SessionOptions()
        options.log_severity_level = 3
        # Memory-lean loading (history 095): the FP32 model is 2.3 GB, and full graph
        # optimisation, weight pre-packing and the memory arena copied weights, so the server
        # process hit "bad allocation". Basic optimisation keeps the results the same.
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.add_session_config_entry("session.disable_prepacking", "1")
        self.session = InferenceSession(
            str(_path("onnx/model.onnx")), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.tokenizer = Tokenizer.from_file(str(_path("onnx/tokenizer.json")))
        self.tokenizer.enable_truncation(CONFIG["max_tokens"])
        pad = self.tokenizer.token_to_id("<pad>")
        self.tokenizer.enable_padding(pad_id=0 if pad is None else pad, pad_token="<pad>")
        self.model = MODEL
        self.version = CONFIG["revision"][:12]

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        # Multiple UI clients must not allocate large inference batches simultaneously.
        with _encode_lock:
            return self._encode(texts, kind)

    def _encode(self, texts: list[str], kind: str) -> np.ndarray:
        prefix = CONFIG["query_prefix"] if kind == "query" else CONFIG["passage_prefix"]
        parts = []
        for start in range(0, len(texts), BATCH):
            encoded = self.tokenizer.encode_batch(
                [prefix + t for t in texts[start : start + BATCH]]
            )
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            results = dict(zip(self.outputs, self.session.run(None, feeds)))
            name = CONFIG.get("output") if CONFIG.get("output") in results else self.outputs[0]
            hidden = np.asarray(results[name])
            if hidden.ndim == 2:
                vectors = hidden
            elif CONFIG.get("pooling") == "cls":
                vectors = hidden[:, 0]
            else:
                weights = mask[..., None].astype(np.float32)
                vectors = (hidden * weights).sum(1) / np.maximum(weights.sum(1), 1.0)
            parts.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        if not parts:
            return np.zeros((0, CONFIG["dimension"]), dtype=np.float32)
        return np.vstack(parts).astype(np.float32)


def failure_reason(error: Exception) -> str:
    """The type and the first line of ONNX Runtime's own message, so the cause is visible."""
    detail = " ".join(str(error).split())[:300]
    return f"Модель embeddings не запустилась: {type(error).__name__}" + (
        f" — {detail}" if detail else "."
    )


_lock = threading.Lock()
_encode_lock = threading.Lock()
_state: dict = {}


def failed(error: Exception) -> str:
    """Release a failed session and keep its visible reason during the retry cooldown."""
    reason = failure_reason(error)
    with _lock:
        _state.update(embedder=None, reason=reason, at=time.monotonic())
    LOG.error("Embeddings inference failed: %s", reason)
    return reason


def embedder() -> Embedder | None:
    """Shared embedder for the process; None (with the reason in ``status()``) if unavailable."""
    with _lock:
        if "embedder" in _state and (
            _state["embedder"] is not None or time.monotonic() - _state["at"] < RETRY_SECONDS
        ):
            return _state["embedder"]
        reason = missing()
        instance = None
        if reason is None:
            try:
                instance = Embedder()
            except Exception as error:  # a damaged model file, an incompatible runtime, no memory
                reason = failure_reason(error)
                LOG.exception("Embeddings model %s failed to load", MODEL)
        # A failure (e.g. temporarily not enough memory) is retried after RETRY_SECONDS.
        _state.update(embedder=instance, reason=reason, at=time.monotonic())
        return instance


def status() -> str | None:
    embedder()
    return _state.get("reason")


def reset() -> None:
    with _lock:
        _state.clear()


def verify() -> list[str]:
    """Files whose SHA-256 differs from the pinned value (slow: reads ~490 MB)."""
    return [
        Path(name).name
        for name, sha in CONFIG["files"].items()
        if not _path(name).is_file() or _sha256(_path(name)) != sha
    ]


def install() -> None:
    """Setup step: download the pinned model files (~2.3 GB) with progress and SHA-256 checks."""
    INSTALL.mkdir(parents=True, exist_ok=True)
    print(f"Embeddings model {MODEL} -> {INSTALL}", flush=True)
    for name, sha in CONFIG["files"].items():
        path = _path(name)
        if path.is_file() and _sha256(path) == sha:
            print(f"  {path.name}: already ready", flush=True)
            continue
        url = f"{CONFIG['download_base']}/{MODEL}/resolve/{CONFIG['revision']}/{name}"
        partial = path.with_suffix(path.suffix + ".part")
        started = time.monotonic()
        with urllib.request.urlopen(url, timeout=300) as response, partial.open("wb") as output:
            total = int(response.headers.get("Content-Length") or 0)
            done, shown = 0, -10
            while chunk := response.read(1 << 20):
                output.write(chunk)
                done += len(chunk)
                percent = done * 100 // total if total else 100
                if percent >= shown + 10:
                    shown = percent
                    print(f"  {path.name}: {percent}% ({done >> 20} MB)", flush=True)
        if _sha256(partial) != sha:
            partial.unlink()
            raise RuntimeError(
                f"Контрольна сума {path.name} не збігається. Запустіть Setup ще раз."
            )
        partial.replace(path)
        print(f"  {path.name}: ready in {time.monotonic() - started:.0f}s", flush=True)
    reset()
    if missing() is not None:
        raise RuntimeError(missing())
    vectors = Embedder().encode(["перевірка моделі"], "query")
    print(f"Embeddings ready: vector size {vectors.shape[1]}.", flush=True)
