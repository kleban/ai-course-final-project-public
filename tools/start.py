"""Single-process launch, duplicate detection, readiness, and graceful stop."""

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from data_story.config import Settings  # noqa: E402

OPENER = build_opener(ProxyHandler({}))


def running(settings: Settings) -> bool:
    try:
        with OPENER.open(settings.url + "/health", timeout=1) as response:
            data = json.load(response)
        return data.get("application") == "data-story" and data.get("instance") == settings.instance
    except (URLError, OSError, ValueError):
        return False


def port_occupied(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


@contextmanager
def launch_lock(settings: Settings):
    """The OS releases this lock even if the console is closed."""
    import msvcrt

    lock_path = settings.database.parent / f"start-{settings.port}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError(
                "Застосунок уже запускається. Зачекайте кілька секунд і відкрийте Start.cmd ще раз."
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def ready_browser(settings: Settings) -> None:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if running(settings):
            webbrowser.open(settings.url)
            return
        time.sleep(0.25)
    print("Браузер не відкрито автоматично. Перегляньте повідомлення запуску.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path, default=ROOT / "runtime" / "data-story.sqlite3")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    arguments = parser.parse_args()
    database = arguments.database.resolve()
    if not database.is_relative_to(ROOT) or not 1024 <= arguments.port <= 65535:
        print("База має бути всередині репозиторію; порт — у межах 1024–65535.")
        return 1
    settings = Settings(database=database, port=arguments.port)
    try:
        if arguments.stop:
            if not running(settings):
                print("Наш застосунок на цьому порту не запущений. Чужі процеси не змінено.")
                return 0
            request = Request(
                settings.url + "/shutdown",
                data=b"",
                method="POST",
                headers={"X-Data-Story-Instance": settings.instance},
            )
            with OPENER.open(request, timeout=5):
                pass
            deadline = time.monotonic() + 10
            while port_occupied(settings.port) and time.monotonic() < deadline:
                time.sleep(0.1)
            if port_occupied(settings.port):
                raise RuntimeError("Сервер ще завершує роботу. Перевірте вікно запуску.")
            print("Data Story зупинено.")
            return 0
        if running(settings):
            if not arguments.no_browser:
                webbrowser.open(settings.url)
            print("Data Story вже працює; повторний сервер не створено.")
            return 0
        if port_occupied(settings.port):
            raise RuntimeError(
                "Порт зайнятий іншим застосунком. Його не змінено; звільніть порт 8765 або зверніться до розробника."
            )
        with launch_lock(settings):
            settings.database.parent.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(
                filename=settings.database.parent / "application.log",
                level=logging.INFO,
                encoding="utf-8",
            )
            # UI state (e.g. collapsed menu) is stored next to the database, not in the launch folder
            os.environ.setdefault(
                "NICEGUI_STORAGE_PATH", str(settings.database.parent / "nicegui-storage")
            )
            from data_story.web import run

            if not arguments.no_browser:
                threading.Thread(target=ready_browser, args=(settings,), daemon=True).start()
            print(
                f"Data Story: {settings.url}\nДля зупинки відкрийте Stop.cmd або закрийте це вікно."
            )
            run(settings)
        return 0
    except (RuntimeError, OSError, ImportError) as error:
        print(f"Не вдалося запустити Data Story: {error}")
        print("Якщо залежності не встановлено, відкрийте Setup.cmd.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
