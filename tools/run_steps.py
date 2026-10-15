"""Run named steps with live output and a log in tmp/<name>.log (for checks on Windows).

Usage: python tools/run_steps.py <name> "Title::command" ["Title::command" ...]
"""

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    name, steps = sys.argv[1], [s.split("::", 1) for s in sys.argv[2:]]
    log = ROOT / "tmp" / f"{name}.log"
    log.parent.mkdir(exist_ok=True)
    log.write_text("", encoding="utf-8")

    def say(text: str) -> None:
        line = time.strftime("%H:%M:%S ") + text
        print(line, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    say(f"=== {name}: кроків {len(steps)}. Не закривайте вікно. ===")
    ok = True
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for number, (title, command) in enumerate(steps, 1):
        say(f"[{number}/{len(steps)}] {title} ...")
        process = subprocess.Popen(
            command, cwd=ROOT, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )  # fmt: skip
        assert process.stdout is not None
        for line in process.stdout:
            print("    " + line.rstrip(), flush=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write("    " + line.rstrip() + "\n")
        code = process.wait()
        say(f"[{number}/{len(steps)}] {title}: {'OK' if code == 0 else f'ПОМИЛКА, код {code}'}")
        ok = ok and code == 0
    say("=== ГОТОВО: усе успішно. ===" if ok else f"=== ЗАВЕРШЕНО З ПОМИЛКАМИ: див. {log} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
