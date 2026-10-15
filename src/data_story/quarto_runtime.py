"""Pinned, portable Quarto; installation is explicit Setup, daily render stays local."""

import hashlib
import json
import os
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .config import ROOT

CONFIG = json.loads(Path(__file__).with_name("quarto.json").read_text(encoding="utf-8"))
VERSION = CONFIG["version"]
SHA256 = CONFIG["windows_zip_sha256"]
INSTALL = (ROOT / CONFIG["installation"]).resolve()
if not INSTALL.is_relative_to(ROOT):
    raise RuntimeError("Quarto має бути всередині проєкту.")


def executable() -> Path:
    name = "quarto.exe" if os.name == "nt" else "quarto"
    matches = sorted(INSTALL.glob(f"**/bin/{name}"))
    return matches[0] if matches else INSTALL / "bin" / name


def version() -> str | None:
    path = executable()
    if not path.is_file():
        return None
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def install() -> None:
    if version() == VERSION:
        print(f"Quarto {VERSION} already ready.", flush=True)
        return
    if os.name != "nt":
        raise RuntimeError("Цей Setup Quarto призначений для Windows.")
    cache = ROOT / "tmp" / "install"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f"quarto-{VERSION}-win.zip"
    if not archive.is_file() or hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        print(f"Downloading portable Quarto {VERSION}…", flush=True)
        url = f"{CONFIG['download_base']}/v{VERSION}/{archive.name}"
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Контрольна сума Quarto не збігається. Запустіть Setup повторно.")
    INSTALL.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for entry in package.infolist():
            if not (INSTALL / entry.filename).resolve().is_relative_to(INSTALL.resolve()):
                raise RuntimeError("Некоректний шлях в архіві Quarto.")
        package.extractall(INSTALL)
    if version() != VERSION:
        raise RuntimeError("Quarto встановлено, але перевірка запуску не пройшла.")
    (INSTALL / "installation.json").write_text(
        json.dumps({"version": VERSION, "sha256": SHA256}), encoding="utf-8"
    )
    print(f"Quarto {VERSION} ready inside the project.", flush=True)
