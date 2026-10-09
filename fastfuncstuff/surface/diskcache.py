"""Derived surface arrays kept on disk between sessions, keyed by what they were made of.

A ribbon map or a sphere.reg lookup costs seconds and depends only on its input
arrays, so the key is a hash of those arrays' bytes, not of file names or mtimes:
an edited mesh misses, a copied subject hits. Entries are plain ``.npz`` under
``$XDG_CACHE_HOME/fastfuncstuff/<kind>``, the oldest-used dropped past a count. A
cache that cannot be read or written is a cache miss, never an error.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

__all__ = ["content_key", "load", "save"]

#: Entries kept per kind; a ribbon map on a 0.8 mm grid is ~10 MB.
MAX_ENTRIES = 32


def _dir(kind: str) -> Path:
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "fastfuncstuff" / kind


def content_key(*parts) -> str:
    """A hash of arrays (dtype, shape and bytes) and plain values, in order."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(f"{a.dtype.str}{a.shape}".encode())
            h.update(memoryview(a).cast("B"))
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:24]


def load(kind: str, key: str) -> dict[str, np.ndarray] | None:
    path = _dir(kind) / f"{key}.npz"
    try:
        with np.load(path) as z:
            out = {k: z[k] for k in z.files}
        os.utime(path)  # most recently used survives the prune
        return out
    except (OSError, ValueError, KeyError):
        return None


def save(kind: str, key: str, **arrays: np.ndarray) -> None:
    d = _dir(kind)
    try:
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / f".{key}.{os.getpid()}.npz"
        np.savez(tmp, **arrays)
        tmp.replace(d / f"{key}.npz")  # atomic: a reader never sees half a file
        # pathlib's glob matches dot files: leave another process's temp alone.
        done = [p for p in d.glob("*.npz") if not p.name.startswith(".")]
        entries = sorted(done, key=lambda p: p.stat().st_mtime)
        for old in entries[:-MAX_ENTRIES]:
            old.unlink(missing_ok=True)
    except OSError:
        pass
