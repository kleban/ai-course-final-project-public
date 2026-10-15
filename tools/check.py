"""One entry point for offline quality gates and stage acceptance tests."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    checks = [
        ("Dependencies", [sys.executable, "-m", "pip", "check"]),
        ("Code rules", [sys.executable, "-m", "ruff", "check", "src", "tests", "tools"]),
        (
            "Formatting",
            [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "tools"],
        ),
        ("Types", [sys.executable, "-m", "mypy"]),
        ("Offline contracts, domain and API", [sys.executable, "-B", "tools/check_projects.py"]),
    ]
    for name, command in checks:
        print(f"\n{name}", flush=True)
        if subprocess.run(command, cwd=ROOT, check=False).returncode:
            return 1
    print("\nAll offline checks passed. Browser acceptance is recorded separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
