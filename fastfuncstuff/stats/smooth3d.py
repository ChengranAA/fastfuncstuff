"""
Batched 3D Gaussian smoothing for [[ffs_perm]]'s ``-vsmooth`` (pseudo-t).

Implements **mask-aware** separable Gaussian smoothing — the standard
``3dBlurInMask``-style correction so out-of-mask zeros don't bleed into
mask voxels.  ``smooth_var_per_perm`` is the public entry point: takes a
``[P, V_in_mask]`` variance matrix, scatters to a 4-D ``[P, X, Y, Z]``
volume, smooths each perm independently with a shared kernel, and
gathers back to ``[P, V_in_mask]``.

Performance: separable 1-D passes along each axis, each a shifted-slab
accumulation (``blur_axis``) that works on any memory layout.
"""

from __future__ import annotations

import math

import numpy as np
import torch

# FWHM ↔ sigma constant: FWHM = sigma · 2√(2 ln 2)
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def fwhm_mm_to_sigma_vox(
    fwhm_mm: float,
    voxel_size_mm: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Convert isotropic FWHM in mm to per-axis Gaussian σ in voxels."""
    sx = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[0]
    sy = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[1]
    sz = fwhm_mm * FWHM_TO_SIGMA / voxel_size_mm[2]
    return (sx, sy, sz)


def _gaussian_kernel_1d(
    sigma: float,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Truncated 1-D Gaussian kernel, normalised to sum to 1.

    Truncation radius = 4σ (matches scipy.ndimage default ``truncate=4.0``).
    """
    radius = max(1, int(math.ceil(4.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return k / k.sum()


def blur_axis(
    x: torch.Tensor, kernel: torch.Tensor, axis: int, *, reflect: bool = False
) -> torch.Tensor:
    """Correlate ``x`` with a symmetric 1-D ``kernel`` along ``axis``, any layout.

    Shifted-slab accumulation -- one strided ``add_`` per tap -- rather than
    ``conv1d``/``conv3d``. A convolution wants the filtered axis contiguous, so
    on a ``(x, y, z, t)`` run it costs a transpose per axis per volume; and
    oneDNN's single-channel ``conv3d`` measured 7x slower than this on CPU.
    Every op here is elementwise over whatever strides ``x`` has, so a C- or
    F-ordered array is blurred in place of its layout, on any device.

    ``reflect`` mirrors without repeating the edge sample (torch's
    ``reflect``), which needs ``radius < x.shape[axis]``; otherwise the
    border is zero-padded, as ``conv1d(padding=r)`` would.
    """
    n = x.shape[axis]
    taps = [float(w) for w in kernel.tolist()]
    r = (len(taps) - 1) // 2
    out = x * taps[r]
    for j in range(1, r + 1):
        w = taps[r + j]
        if j < n:
            out.narrow(axis, 0, n - j).add_(x.narrow(axis, j, n - j), alpha=w)
            out.narrow(axis, j, n - j).add_(x.narrow(axis, 0, n - j), alpha=w)
        if reflect:
            if j >= n:
                raise ValueError(f"reflect padding needs radius < {n}, got {r}")
            # Past the high edge, i + j lands on 2(n-1) - (i + j); past the
            # low edge, i - j lands on j - i. Both are j samples, mirrored.
            out.narrow(axis, n - j, j).add_(x.narrow(axis, n - 1 - j, j).flip(axis), alpha=w)
            out.narrow(axis, 0, j).add_(x.narrow(axis, 1, j).flip(axis), alpha=w)
    return out


def gaussian3d_batched(
    vol: torch.Tensor,  # [P, X, Y, Z]
    sigma_vox: tuple[float, float, float],
) -> torch.Tensor:
    """Separable 3-D Gaussian, batched over the leading dim.

    Padding is reflect so the mask-normalised denominator stays well
    behaved at the brain boundary.
    """
    x = vol
    for ax, s in enumerate(sigma_vox):
        if s > 0:
            x = blur_axis(x, _gaussian_kernel_1d(s, x.dtype, x.device), ax + 1, reflect=True)
    return x


def smooth_var_per_perm(
    var_pv: torch.Tensor,  # [P, V_in_mask], float32
    mask: np.ndarray,  # [X, Y, Z] bool
    sigma_vox: tuple[float, float, float],
    perm_chunk: int = 256,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Mask-aware Gaussian smoothing of a per-perm variance matrix.

    Mask-aware = divide ``smooth(var · 1_mask)`` by ``smooth(1_mask)``
    so out-of-mask zeros don't dilute voxels near the boundary.  This is
    the same correction used by ``3dBlurInMask``.

    Streams in chunks of ``perm_chunk`` perms so peak memory stays
    bounded regardless of P.
    """
    p, v_mask = var_pv.shape
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    nx, ny, nz = mask.shape
    flat_idx = torch.from_numpy(np.flatnonzero(mask.ravel())).to(dev, dtype=torch.int64)
    assert flat_idx.numel() == v_mask, "mask voxel count must equal V_in_mask"
    n_total = nx * ny * nz

    # Pre-compute mask-smooth denominator once (1-channel, broadcast).
    mask_3d = torch.from_numpy(mask.astype(np.float32)).to(dev).unsqueeze(0)
    mask_smooth = gaussian3d_batched(mask_3d, sigma_vox).clamp_min(1e-10)  # [1, X, Y, Z]

    out = torch.empty_like(var_pv)
    for p0 in range(0, p, perm_chunk):
        p1 = min(p0 + perm_chunk, p)
        pc = p1 - p0
        chunk_cpu = var_pv[p0:p1]
        chunk = chunk_cpu.to(dev, non_blocking=True)
        vol_flat = torch.zeros((pc, n_total), dtype=chunk.dtype, device=dev)
        vol_flat[:, flat_idx] = chunk
        vol = vol_flat.view(pc, nx, ny, nz)
        smoothed = gaussian3d_batched(vol, sigma_vox)  # [Pc, X, Y, Z]
        smoothed = smoothed / mask_smooth  # mask-aware
        gathered = smoothed.view(pc, -1)[:, flat_idx]
        out[p0:p1] = gathered.cpu() if dev.type != "cpu" else gathered
    return out
