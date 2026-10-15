"""Load the embeddings model exactly as the service does and print the full cause of a failure.

Usage (Windows): tools\\windows\\Check-Embeddings.cmd -> log in tmp\\check-embeddings-load.log
"""

import ctypes
import platform
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_story import embeddings  # noqa: E402


def memory() -> str:
    if platform.system() != "Windows":
        return "n/a"

    class Status(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("load", ctypes.c_ulong),
            ("total", ctypes.c_ulonglong),
            ("available", ctypes.c_ulonglong),
            ("total_page", ctypes.c_ulonglong),
            ("available_page", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("available_virtual", ctypes.c_ulonglong),
            ("extended", ctypes.c_ulonglong),
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined]
    gb = 1024**3
    return (
        f"RAM {status.available / gb:.1f} of {status.total / gb:.1f} GB free; "
        f"commit {status.available_page / gb:.1f} of {status.total_page / gb:.1f} GB free"
    )


def peak() -> str:
    """Peak working set and commit of this process (Windows), i.e. what loading really needed."""
    if platform.system() != "Windows":
        return "n/a"

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("faults", ctypes.c_ulong),
            ("peak_working_set", ctypes.c_size_t),
            ("working_set", ctypes.c_size_t),
            ("peak_paged_pool", ctypes.c_size_t),
            ("paged_pool", ctypes.c_size_t),
            ("peak_nonpaged_pool", ctypes.c_size_t),
            ("nonpaged_pool", ctypes.c_size_t),
            ("pagefile", ctypes.c_size_t),
            ("peak_pagefile", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(Counters)
    process = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
    ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
        process, ctypes.byref(counters), counters.cb
    )
    gb = 1024**3
    return (
        f"process peak RAM {counters.peak_working_set / gb:.1f} GB, "
        f"peak commit {counters.peak_pagefile / gb:.1f} GB"
    )


def main() -> int:
    print(f"Python {sys.version.split()[0]} ({platform.architecture()[0]}), {platform.platform()}")
    try:
        import onnxruntime

        print(
            f"onnxruntime {onnxruntime.__version__}; providers {onnxruntime.get_available_providers()}"
        )
    except ImportError as error:
        print(f"onnxruntime import failed: {error}")
        return 1
    print(f"Model folder: {embeddings.INSTALL}")
    for name in embeddings.CONFIG["files"]:
        path = embeddings._path(name)
        size = f"{path.stat().st_size:,} bytes" if path.is_file() else "MISSING"
        print(f"  {path.name}: {size}")
    print("Before load:", memory())
    started = time.monotonic()
    try:
        model = embeddings.Embedder()
        print(f"Loaded in {time.monotonic() - started:.1f} s. After load:", memory())
        print("Peak:", peak())
        vector = model.encode(["перевірка моделі"], "query")
        print(f"Encode OK: shape {vector.shape}, norm {float((vector**2).sum()) ** 0.5:.3f}")
    except Exception as error:
        print(f"LOAD FAILED after {time.monotonic() - started:.1f} s:", memory(), "|", peak())
        print(embeddings.failure_reason(error))
        traceback.print_exc(file=sys.stdout)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
