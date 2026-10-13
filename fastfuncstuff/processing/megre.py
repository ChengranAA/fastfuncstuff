"""Multi-echo GRE anatomy: relaxometry, field, SWI and vein contrast.

The fit itself is :mod:`multiecho` (T2*/S0, optimal combination) and the field
is ROMEO via :func:`b0fmap.run_romeo`; this module owns the image-space steps
that turn those into vein contrast:

- :func:`masked_highpass` removes the background field (air/tissue boundaries,
  shim) by subtracting a mask-normalised Gaussian blur of the field, leaving the
  local field that venous susceptibility produces.
- :func:`swi_phase_mask` / :func:`minimum_intensity_projection` are the SWI of
  Haacke et al. (2004): the magnitude darkened where the local phase is
  vein-signed, then minimum-projected over a slab so vessels read as lines.
- The R2*·S0 product is the vein contrast Stanley et al. (2021) segmented
  ds003427's veins from.

Tensors are ``(nz, ny, nx)``, as :func:`io.load_image` returns them, so the
voxel sizes passed here are in that order too.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ..stats.smooth3d import FWHM_TO_SIGMA, _gaussian_kernel_1d, blur_axis


def _blur(vol: Tensor, fwhm_mm: float, voxel_mm: tuple[float, float, float]) -> Tensor:
    out = vol
    for axis, vs in enumerate(voxel_mm):
        sigma = fwhm_mm * FWHM_TO_SIGMA / vs
        if sigma < 0.25:  # under a quarter voxel the kernel is a delta
            continue
        out = blur_axis(out, _gaussian_kernel_1d(sigma, vol.dtype, vol.device), axis)
    return out


def masked_highpass(
    field: Tensor, mask: Tensor, fwhm_mm: float, voxel_mm: tuple[float, float, float]
) -> Tensor:
    """``field`` minus its Gaussian low-pass, the blur normalised by the blurred
    mask so the brain edge is not pulled toward the zeros outside it. Zero
    outside ``mask``.

    A high-pass is the simplest background-field removal (the homodyne filter of
    SWI); it also removes the low spatial frequencies of the local field, which
    for veins (a few voxels across) is the part that does not matter."""
    m = mask.to(field.dtype)
    low = _blur(field * m, fwhm_mm, voxel_mm) / _blur(m, fwhm_mm, voxel_mm).clamp_min(1e-3)
    return (field - low) * m


def swi_phase_mask(local_phase: Tensor, sign: int = -1, power: int = 4) -> Tensor:
    """Haacke's phase mask, raised to ``power``: 1 where the phase has the
    non-venous sign, falling linearly to 0 at a phase of ``sign * pi``.

    ``sign`` is the sign venous phase takes in this data's convention (-1 for the
    right-handed convention Haacke used; scanners differ, so check the output)."""
    p = (sign * local_phase).clamp(0.0, math.pi)
    return (1.0 - p / math.pi) ** power


def minimum_intensity_projection(vol: Tensor, n: int, axis: int = 0) -> Tensor:
    """Sliding minimum over ``n`` consecutive slices along ``axis`` (centred,
    truncated at the slab ends), so the output keeps the input's shape."""
    if n <= 1:
        return vol.clone()
    half = n // 2
    pad_lo, pad_hi = half, n - 1 - half
    x = vol.movedim(axis, -1)
    big = torch.finfo(vol.dtype).max
    xp = torch.nn.functional.pad(x, (pad_lo, pad_hi), value=big)
    return xp.unfold(-1, n, 1).amin(dim=-1).movedim(-1, axis)
