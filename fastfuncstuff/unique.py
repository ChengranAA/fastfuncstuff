"""``np.unique`` for integer arrays, by sorting.

NumPy 2.4's ``np.unique`` takes a hash-based path for integer input that is
pathologically slow once there are many distinct values: 3.1 s for 4M int64
against 57 ms for a sort and an adjacent-difference mask (numpy 2.4.6,
measured 2026-10-16), with ``return_index`` / ``return_inverse`` 5-10x slower
too.  Mesh edge lists, voxel and vertex index sets are exactly that input --
building a 150k-vertex hemisphere's topology spent 1 s in it.

:func:`sorted_unique` is a drop-in for the ``axis=None`` call on integer and bool
arrays; anything else (floats, where NaN handling matters; strings; ``axis=``)
goes to ``np.unique`` unchanged.  Results match ``np.unique`` exactly, including
first-occurrence indices and an inverse shaped like the input.
"""

from __future__ import annotations

from typing import Any


def sorted_unique(
    ar: Any,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    **kwargs: Any,
):
    """``np.unique(ar, ...)``, sorting rather than hashing for integer input."""
    import numpy as np  # here, not at import: CLI modules import this on their -help path

    a = np.asarray(ar)
    if kwargs or not (np.issubdtype(a.dtype, np.integer) or a.dtype == np.bool_):
        return np.unique(
            ar,
            return_index=return_index,
            return_inverse=return_inverse,
            return_counts=return_counts,
            **kwargs,
        )
    flat = a.ravel()
    want_order = return_index or return_inverse
    if want_order:
        # Stable, so the first index kept per value is its first occurrence.
        order = np.argsort(flat, kind="stable")
        s = flat[order]
    else:
        s = np.sort(flat)
    first = np.empty(s.shape, dtype=bool)
    if s.size:
        first[0] = True
        np.not_equal(s[1:], s[:-1], out=first[1:])
    out = [s[first]]
    if return_index:
        out.append(order[first].astype(np.intp, copy=False))
    if return_inverse:
        inverse = np.empty(flat.shape, dtype=np.intp)
        inverse[order] = np.cumsum(first) - 1
        out.append(inverse.reshape(a.shape))
    if return_counts:
        starts = np.flatnonzero(first)
        out.append(np.diff(np.append(starts, s.size)).astype(np.intp, copy=False))
    return out[0] if len(out) == 1 else tuple(out)
