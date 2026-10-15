"""Run component acceptance checks without installed packages."""

import argparse
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from data_story import embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-embeddings",
        action="store_true",
        help="Also allocate the installed ONNX model; requires substantial RAM.",
    )
    args = parser.parse_args()
    mode = (
        nullcontext()
        if args.real_embeddings
        else patch.object(
            embeddings,
            "missing",
            return_value="Автотести: TF-IDF; реальну модель перевіряють окремо.",
        )
    )
    print(
        "Embeddings: "
        + ("installed model" if args.real_embeddings else "isolated TF-IDF and mocks"),
        flush=True,
    )
    with mode:
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
