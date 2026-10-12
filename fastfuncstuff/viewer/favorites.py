"""Favourite directories, kept across sessions in the user's config.

One JSON file, ``$XDG_CONFIG_HOME/fastfuncstuff/viewer.json`` (``~/.config``
by default), so every viewer on the machine shares one list and anything
else the toolbox wants to remember can sit beside it under its own key.

No Qt: the window draws the list, this only reads and writes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

KEY = "favorite_dirs"


def config_path() -> Path:
    root = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(root).expanduser() / "fastfuncstuff" / "viewer.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        # Unreadable: set aside rather than overwritten, in case it holds
        # something someone wrote by hand.
        try:
            path.replace(path.with_suffix(".json.bad"))
        except OSError:
            pass
        return {}
    return data if isinstance(data, dict) else {}


def load(path: Path | None = None) -> list[Path]:
    """The favourites, in the order they were added."""
    entries = _read(path or config_path()).get(KEY, [])
    return [Path(e) for e in entries if isinstance(e, str)]


def save(directories: list[Path], path: Path | None = None) -> None:
    path = path or config_path()
    data = _read(path)
    data[KEY] = [str(d) for d in directories]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def toggle(directory: str | Path, path: Path | None = None) -> bool:
    """Add ``directory`` to the favourites, or take it off; returns whether it is one now."""
    d = Path(directory).expanduser().resolve()
    current = load(path)
    if d in current:
        save([f for f in current if f != d], path)
        return False
    save([*current, d], path)
    return True


__all__ = ["config_path", "load", "save", "toggle"]
