"""Derived surface arrays on disk, keyed by the arrays they were made of."""

from __future__ import annotations

import os

import numpy as np

from fastfuncstuff.surface import diskcache


def test_roundtrip_and_any_input_change_misses():
    v = np.arange(12.0).reshape(4, 3)
    key = diskcache.content_key("t-1", v, (3, 4))
    diskcache.save("t", key, a=np.arange(5), b=np.ones((2, 2), np.float32))
    got = diskcache.load("t", key)
    assert got is not None
    np.testing.assert_array_equal(got["a"], np.arange(5))
    assert got["b"].dtype == np.float32
    moved = v.copy()
    moved[0, 0] += 1e-9  # an edited vertex is a different mesh
    assert diskcache.content_key("t-1", moved, (3, 4)) != key
    assert diskcache.content_key("t-1", v.astype(np.float32), (3, 4)) != key
    assert diskcache.content_key("t-1", v, (3, 5)) != key
    assert diskcache.load("t", diskcache.content_key("t-1", moved, (3, 4))) is None


def test_prune_keeps_the_most_recently_used(monkeypatch):
    monkeypatch.setattr(diskcache, "MAX_ENTRIES", 3)
    keys = [diskcache.content_key("p", i) for i in range(4)]
    for i, k in enumerate(keys[:3]):
        diskcache.save("p", k, x=np.array([i]))
    folder = diskcache._dir("p")
    for i, k in enumerate(keys[:3]):  # distinct, ordered mtimes
        os.utime(folder / f"{k}.npz", (1e9 + i, 1e9 + i))
    assert diskcache.load("p", keys[0]) is not None  # a hit refreshes the oldest
    diskcache.save("p", keys[3], x=np.array([3]))
    assert diskcache.load("p", keys[1]) is None  # now the least recently used
    assert all(diskcache.load("p", k) is not None for k in (keys[0], keys[2], keys[3]))


def test_unwritable_cache_is_a_miss_not_an_error(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.setattr(diskcache, "_dir", lambda kind: blocker / kind)
    diskcache.save("x", "k", a=np.zeros(1))
    assert diskcache.load("x", "k") is None
