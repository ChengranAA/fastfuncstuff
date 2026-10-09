"""Blurring data TO a smoothness: the master, its time-point sample, and the kernel search.

Geometry-free: the surface (heat kernel, mesh FWHM) and the volume (Gaussian, 3dFWHMx
1-difference) each supply "a smoother of width k" and "the FWHM of this master", and
:func:`bisect_kernel` does the rest. The smoother is linear, so the search runs on a
small master alone and the chosen kernel is then applied to the data once; AFNI's
3dBlurToFWHM / SurfSmooth instead re-estimate after every incremental pass.

**The master** is the residual of the model's own OLS fit (SurfSmooth's advice for
``-blurmaster``): detrended data still carry the task, which is spatially smooth and
inflates the measured FWHM, so the data come out under-blurred. With one design shared
by every voxel, the spatial smoother ``S`` and the residual-forming matrix ``R`` act on
different axes and commute, ``R (Y S) = (R Y) S``: residuals of the unblurred data
measure exactly what the blurred data's residuals will be, with no refit. Per-voxel
regressors (``-dsort``) break that, and are left out of the master.

**The sample**: a FWHM estimate averages over every edge or voxel pair, so a few dozen
time points pin it (3dBlurToFWHM uses at most 32). They are spread over the runs in
proportion to each run's uncensored length, because smoothness differs between runs
(motion, distortion), and censored points never enter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["bisect_kernel", "residual_master", "sample_columns"]


def sample_columns(
    run_lengths: list[int],
    n_columns: int,
    good: np.ndarray | list[int] | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Sorted time indices (into the concatenation), ``n_columns`` of them spread over
    the runs in proportion to each run's kept length, never a censored one."""
    total = int(sum(run_lengths))
    keep = np.ones(total, bool)
    if good is not None:
        keep[:] = False
        keep[np.asarray(good, np.int64)] = True
    starts = np.r_[0, np.cumsum(run_lengths)[:-1]].astype(np.int64)
    pools = [np.flatnonzero(keep[s : s + n]) + s for s, n in zip(starts, run_lengths, strict=True)]
    n_kept = np.array([len(p) for p in pools])
    want = min(int(n_columns), int(n_kept.sum()))
    if want == int(n_kept.sum()):
        return np.sort(np.concatenate(pools))
    # Largest-remainder apportionment, then a run never asks for more than it has.
    share = want * n_kept / n_kept.sum()
    take = np.floor(share).astype(int)
    for i in np.argsort(-(share - take))[: want - take.sum()]:
        take[i] += 1
    take = np.minimum(take, n_kept)
    rng = np.random.default_rng(seed)
    picks = [rng.choice(p, size=k, replace=False) for p, k in zip(pools, take, strict=True) if k]
    return np.sort(np.concatenate(picks))


def residual_master(
    data: np.ndarray,
    design: np.ndarray,
    columns: np.ndarray,
    good: np.ndarray | list[int] | None = None,
    chunk: int = 16384,
) -> np.ndarray:
    """``(V, k)`` OLS residuals of ``data`` (``(V, T)``) against ``design`` (``(T, p)``),
    at the time points ``columns`` only. Censored rows (outside ``good``) take no part
    in the fit. Never copies ``data``: one ``(k, T)`` residual-forming block is built in
    float64 and applied a voxel chunk at a time."""
    x = np.asarray(design, np.float64)
    t = x.shape[0]
    keep = np.arange(t) if good is None else np.asarray(good, np.int64)
    pinv = np.linalg.pinv(x[keep])  # (p, G): rank-deficient designs are fine
    cols = np.asarray(columns, np.int64)
    rows = np.zeros((len(cols), t))
    rows[:, keep] = -(x[cols] @ pinv)
    rows[np.arange(len(cols)), cols] += 1.0
    out = np.empty((data.shape[0], len(cols)), np.float32)
    for v0 in range(0, data.shape[0], chunk):
        block = np.asarray(data[v0 : v0 + chunk], np.float64)
        out[v0 : v0 + chunk] = block @ rows.T
    return out


def bisect_kernel(
    measure: Callable[[float], tuple[float, Any]],
    target: float,
    tol: float = 0.02,
) -> tuple[Any, float, float]:
    """The smallest kernel that brings a master to ``target`` FWHM.

    ``measure(k)`` returns ``(fwhm of the master smoothed by k, the smoother)``, with
    ``k = 0`` meaning unsmoothed (its smoother is ignored). Returns ``(smoother,
    kernel, achieved)``; ``smoother`` is None when the master already measures at
    least ``target`` -- nothing to add, as 3dBlurToFWHM reports.
    """
    start, _ = measure(0.0)
    if not np.isfinite(start) or start >= target:
        return None, 0.0, float(start)
    lo, hi = 0.0, float(target)
    fhi, sm_hi = measure(hi)
    while fhi < target and hi < 8 * target:
        lo, hi = hi, hi * 1.5
        fhi, sm_hi = measure(hi)
    best = (hi, fhi, sm_hi)
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        fmid, sm_mid = measure(mid)
        if fmid >= target:
            hi, best = mid, (mid, fmid, sm_mid)
        else:
            lo = mid
        if abs(best[1] - target) <= tol * target:
            break
    kernel, achieved, smoother = best
    return smoother, float(kernel), float(achieved)
