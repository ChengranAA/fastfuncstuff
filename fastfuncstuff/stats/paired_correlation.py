"""Voxel-to-voxel temporal correlation BETWEEN datasets on the same grid.

AFNI's ``3dTcorrelate``, generalised to N datasets: each voxel's time course in one
series is correlated with the SAME voxel's time course in another -- two echoes of
one run, two runs of one task, a series before and after a correction -- giving one
r-map per pair.

Each dataset's voxel chunk is detrended and normalised to unit length ONCE, after
which every pair is a row-wise dot product. N datasets therefore cost N
normalisations and ``N(N-1)/2`` dot products; at that point reading the data is the
expensive part, so all-pairs is no dearer than one pair in practice.

Drift is removed with Legendre polynomials (projected out with an orthonormal basis)
before correlating. Shared slow drift is the classic way two series correlate for
reasons that have nothing to do with their content, and echoes of one run share all
of it.

Caveat for anyone grading a correction with this: thermal noise is independent
across echoes and runs, and ANY resampling averages it down while leaving the shared
signal, so blur alone raises these correlations. Read a rise in r alongside a
sharpness measure, never on its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from tqdm import tqdm

from fastfuncstuff.memory import estimate_chunk_size


def pair_list(n: int, mode: str = "all") -> list[tuple[int, int]]:
    """Index pairs for ``n`` datasets: ``all`` (i<j), ``adjacent`` (i, i+1), ``ref:K``.

    ``ref:K`` pairs dataset K (1-based) with every other one, in input order.
    """
    if n < 2:
        raise ValueError(f"need at least 2 datasets to correlate, got {n}")
    if mode == "all":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if mode == "adjacent":
        return [(i, i + 1) for i in range(n - 1)]
    if mode.startswith("ref:"):
        try:
            k = int(mode[4:]) - 1
        except ValueError as exc:
            raise ValueError(f"-pairs ref:K needs an integer K, got {mode!r}") from exc
        if not 0 <= k < n:
            raise ValueError(f"-pairs ref:K needs 1 <= K <= {n}, got {k + 1}")
        return [(k, j) for j in range(n) if j != k]
    raise ValueError(f"pairs must be 'all', 'adjacent' or 'ref:K', got {mode!r}")


def _detrend_basis(n_t: int, polort: int, device: torch.device) -> torch.Tensor:
    """Orthonormal ``(T, polort+1)`` Legendre basis; ``polort < 0`` gives ``(T, 0)``."""
    from fastfuncstuff.glm.core import construct_polynomial_matrix

    if polort < 0:
        return torch.zeros(n_t, 0, device=device)
    poly = construct_polynomial_matrix(n_t, polort, torch.device("cpu"), dtype=torch.float64)
    q, _ = torch.linalg.qr(poly)
    return q.float().to(device)


def paired_voxel_correlation(
    series: Sequence[np.ndarray | torch.Tensor],
    pairs: Sequence[tuple[int, int]] | None = None,
    *,
    polort: int = 0,
    mask: np.ndarray | torch.Tensor | None = None,
    fisher: bool = False,
    device: torch.device | None = None,
    verbose: bool = True,
) -> torch.Tensor:
    """Per-voxel Pearson r between paired datasets, ``(nx, ny, nz, P)`` float32 on CPU.

    ``series`` are ``(nx, ny, nz, T)`` arrays on one grid with one T; they stay where
    they are (host memory) and are streamed through ``device`` in x-slabs, so neither
    an F-ordered nibabel array nor a long run is ever copied whole. ``pairs`` defaults
    to all ``i < j``. ``polort`` is the Legendre drift degree removed first (0 = the
    mean only, as a plain Pearson r; -1 = nothing). ``mask`` restricts the work to its
    voxels; elsewhere, and wherever either side is constant, r is 0. ``fisher`` returns
    ``atanh(r)``.
    """
    if device is None:
        device = torch.device("cpu")
    n = len(series)
    pairs = list(pairs) if pairs is not None else pair_list(n, "all")
    if n < 2:
        raise ValueError(f"need at least 2 datasets to correlate, got {n}")
    shape = tuple(series[0].shape)
    if len(shape) != 4:
        raise ValueError(f"each dataset must be 4-D (nx, ny, nz, T), got {shape}")
    for k, s in enumerate(series):
        if tuple(s.shape) != shape:
            raise ValueError(f"dataset {k + 1} has shape {tuple(s.shape)}, dataset 1 {shape}")
    for a, b in pairs:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"bad pair ({a}, {b}) for {n} datasets")
    nx, ny, nz, n_t = shape
    if n_t < polort + 3:
        raise ValueError(f"{n_t} timepoints cannot carry a degree-{polort} detrend and an r")

    keep = None
    if mask is not None:
        keep = torch.as_tensor(np.asarray(mask) if not torch.is_tensor(mask) else mask) != 0
        if tuple(keep.shape) != (nx, ny, nz):
            raise ValueError(f"mask shape {tuple(keep.shape)} != grid {(nx, ny, nz)}")

    basis = _detrend_basis(n_t, polort, device)
    chunk = estimate_chunk_size(nx * ny * nz, n_t, 0, device, operation="paired_corr", n_trials=n)
    slab = max(1, chunk // (ny * nz))
    out = torch.zeros(nx, ny, nz, len(pairs))

    def _slab(s, x0: int, x1: int) -> torch.Tensor:
        block = s[x0:x1]
        if not torch.is_tensor(block):
            block = torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))
        return block.reshape(-1, n_t).to(device=device, dtype=torch.float32)

    for x0 in tqdm(
        range(0, nx, slab),
        desc="voxel correlation",
        unit="slab",
        leave=True,
        disable=not verbose or nx <= slab,
    ):
        x1 = min(nx, x0 + slab)
        sel = None if keep is None else keep[x0:x1].reshape(-1).to(device)
        if sel is not None and not bool(sel.any()):
            continue
        unit = []
        for s in series:
            x = _slab(s, x0, x1)
            if sel is not None:
                x = x[sel]
            scale = x.norm(dim=1, keepdim=True)
            if basis.shape[1]:
                x = x - (x @ basis) @ basis.T
            norm = x.norm(dim=1, keepdim=True)
            # A constant voxel has no direction to correlate; r = 0 there, not NaN. The
            # test is RELATIVE: float32 detrending leaves ~1e-7 of a constant behind,
            # which would otherwise normalise to a unit vector of rounding error.
            flat = norm <= 1e-5 * scale
            unit.append(torch.where(flat, torch.zeros_like(x), x / norm.clamp_min(1e-30)))
        r = torch.stack([(unit[a] * unit[b]).sum(dim=1) for a, b in pairs], dim=1)
        r = r.clamp(-1.0, 1.0)
        if fisher:
            r = torch.atanh(r.clamp(-0.999999, 0.999999))
        r = r.cpu()
        if sel is None:
            out[x0:x1] = r.reshape(x1 - x0, ny, nz, len(pairs))
        else:
            dst = torch.zeros((x1 - x0) * ny * nz, len(pairs))
            dst[sel.cpu()] = r
            out[x0:x1] = dst.reshape(x1 - x0, ny, nz, len(pairs))
        del unit
    return out
