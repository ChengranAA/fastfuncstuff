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

__all__ = [
    "MaskedGaussian",
    "bisect_kernel",
    "residual_master",
    "sample_columns",
    "volume_blur_to_fwhm",
    "volume_fwhm",
]


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


class MaskedGaussian:
    """A Gaussian of ``fwhm`` mm applied *within* a mask, as 3dBlurInMask does it:
    ``blur(x m) / blur(m)``. Nothing outside the brain bleeds into the rim (which
    would read as smoothness to the estimate, and as dropout to everything else), and
    voxels outside the mask pass through untouched. Linear, so it commutes with the
    OLS residual like the surface heat kernel.

    Called on ``(V, T)`` rows over the whole ``volume_shape`` grid (C order), numpy in,
    numpy out; time points go through the device in memory-sized chunks.
    """

    def __init__(self, mask: np.ndarray, voxdims, fwhm: float, device=None):
        import torch

        from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched
        from fastfuncstuff.utils import get_device

        self.device = device if device is not None else get_device()
        self.fwhm = float(fwhm)
        self.shape = tuple(int(n) for n in mask.shape)
        self._sigma = fwhm_mm_to_sigma_vox(self.fwhm, tuple(abs(float(v)) for v in voxdims))
        self._idx = torch.as_tensor(np.flatnonzero(mask), device=self.device)
        self._m = torch.as_tensor(mask.astype(np.float32), device=self.device)[None]
        self._denom = gaussian3d_batched(self._m, self._sigma).clamp_min(1e-6)
        self._blur = gaussian3d_batched

    def __call__(self, data: np.ndarray) -> np.ndarray:
        import torch

        from fastfuncstuff.memory import get_available_memory

        x = np.asarray(data)
        out = np.array(x, dtype=np.float32, copy=True)
        if self.fwhm <= 0:
            return out
        n_t = x.shape[1]
        # The block, it masked, the padded working copy and the result.
        per_t = 5 * int(np.prod(self.shape)) * 4
        chunk = max(1, min(n_t, int(get_available_memory(self.device) // per_t)))
        idx = self._idx.cpu().numpy()
        for t0 in range(0, n_t, chunk):
            t1 = min(n_t, t0 + chunk)
            block = torch.as_tensor(
                np.ascontiguousarray(x[:, t0:t1].T, np.float32), device=self.device
            ).reshape(t1 - t0, *self.shape)
            sm = self._blur(block * self._m, self._sigma) / self._denom
            out[idx, t0:t1] = sm.reshape(t1 - t0, -1)[:, self._idx].T.cpu().numpy()
            del block, sm
        return out


def volume_fwhm(x: np.ndarray, mask: np.ndarray, voxdims, device=None) -> float:
    """Combined FWHM (mm) of ``(V, k)`` time series on ``mask``'s grid, from the mean
    temporal correlation of in-mask neighbours along each axis.

    This is 3dFWHMx's classic ``-unif`` estimate (Forman's ``1 - dvar / 2 var`` *is* the
    neighbour correlation once every voxel has unit variance, which is what ``-unif``
    is for, and what ffs_reml's post-fit ``-save_acf`` uses), computed so that no
    per-voxel scale has to be estimated: a correlation is scale-free. Normalising each
    voxel by a MAD from a few dozen points instead leaves ~10% spatially white scale
    error, which reads as roughness -- a synthetic search then overshot by 14%. The
    surface estimate (``surface_fwhm``) is per-edge correlation for the same reason.
    """
    import torch

    from fastfuncstuff.stats.fwhmx import _geom_combine

    device = device if device is not None else torch.device("cpu")
    m = torch.as_tensor(np.asarray(mask, bool), device=device)
    z = torch.as_tensor(np.asarray(x, np.float32), device=device)
    z = z - z.mean(1, keepdim=True)
    sd = z.norm(dim=1, keepdim=True)
    z = torch.where(sd > 0, z / sd.clamp_min(1e-30), torch.zeros_like(z))
    z = z.T.reshape(z.shape[1], *m.shape)  # (k, s0, s1, s2), unit norm over time
    ok = m & (sd.reshape(m.shape) > 0)
    out = []
    for ax, width in enumerate(abs(float(v)) for v in voxdims):
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        pair = ok[tuple(lo)] & ok[tuple(hi)]
        if int(pair.sum()) < 6:
            out.append(0.0)
            continue
        corr = (z[(slice(None), *lo)] * z[(slice(None), *hi)]).sum(0)[pair]
        rho = float(corr.double().mean())
        out.append(2.35482 * width * np.sqrt(-1.0 / (4.0 * np.log(rho))) if 0 < rho < 1 else 0.0)
    return _geom_combine(*out)


def volume_blur_to_fwhm(
    master: np.ndarray,
    mask: np.ndarray,
    voxdims,
    target: float,
    device=None,
    tol: float = 0.02,
) -> tuple[MaskedGaussian | None, float, float]:
    """The in-mask Gaussian that brings ``master`` (``(V, k)`` on ``mask``'s grid) to
    ``target`` mm by :func:`volume_fwhm`."""
    from fastfuncstuff.utils import get_device

    device = device if device is not None else get_device()
    mask = np.asarray(mask, bool)

    def measure(kernel: float) -> tuple[float, MaskedGaussian | None]:
        if kernel <= 0:
            return volume_fwhm(master, mask, voxdims, device), None
        g = MaskedGaussian(mask, voxdims, kernel, device=device)
        return volume_fwhm(g(master), mask, voxdims, device), g

    return bisect_kernel(measure, target, tol)
