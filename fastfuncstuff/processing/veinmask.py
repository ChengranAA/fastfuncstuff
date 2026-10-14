"""Vein candidates from an EPI mean and the anat's tissue fractions.

Where to let spatial phase regression (ffs_phasereg -spr -spr_mask) borrow phase,
without a GRE/SWI. The anat says what each EPI voxel holds (GM/WM/CSF fractions);
the EPI mean, divided by a smooth bias field, should then look like
``mu_gm*c1 + mu_wm*c2 + mu_csf*c3``. A vein in a GM voxel dephases at the EPI's TE
and comes out darker than that prediction; a local-mean comparison alone cannot
tell it from white matter or a tissue boundary, which the tissue model can.

Unvalidated: the first ds003427 results for this mask used tissue maps 2-7 mm off the
EPI and are void. Rescored with aligned maps, sPR inside it removed tissue response
too (2026-10-14). See the fmri_wiki ffs_veinmask and ffs_phasereg notes.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fastfuncstuff.stats.smooth3d import _gaussian_kernel_1d, blur_axis


def masked_blur(x: Tensor, weight: Tensor, sigma_vox: tuple[float, float, float]) -> Tensor:
    """Gaussian average of ``x`` over ``weight`` (normalised convolution).

    Voxels with zero weight neither contribute nor drag the result toward zero, so
    the smooth field is defined up to the mask edge."""
    num, den = x * weight, weight.to(x.dtype)
    for ax, s in enumerate(sigma_vox):
        if s > 0:
            k = _gaussian_kernel_1d(s, x.dtype, x.device)
            num, den = blur_axis(num, k, ax), blur_axis(den, k, ax)
    return num / den.clamp_min(1e-3)


def tissue_darkness(
    mean: Tensor,
    tissue: Tensor,
    fit_mask: Tensor,
    sigma_vox: tuple[float, float, float],
    n_iter: int = 8,
    trim: float = 2.5,
) -> tuple[Tensor, Tensor, Tensor]:
    """``mean / (bias * tissue prediction)``, centred so typical tissue reads 1.

    ``tissue`` is ``(3, *mean.shape)``: GM, WM, CSF fractions on the EPI grid. Alternates
    the three tissue levels (least squares) with a smooth multiplicative bias
    (``sigma_vox``). Voxels further than ``trim`` robust SDs from the median log
    ratio leave the next fit on BOTH sides: trimming only the dark tail (the veins)
    biases the levels up every round, so typical GM would read darker than 1.

    Returns ``(ratio, expected, levels)``: ``expected = bias * prediction`` in EPI
    units (what the voxel should look like), ``levels`` the fitted GM/WM/CSF
    intensities before bias."""
    dt = torch.float64
    m = mean.to(dt)
    c = tissue.to(dt)
    flat_c = c.reshape(3, -1)
    use = fit_mask & (m > 0) & (c.sum(0) > 0.5)
    w = use.clone()
    bias = torch.ones_like(m)
    levels = torch.zeros(3, dtype=dt, device=m.device)
    expected = torch.ones_like(m)
    log_r = torch.zeros_like(m)
    for _ in range(n_iter):
        sel = w.reshape(-1)
        a = flat_c[:, sel].T
        b = (m / bias).reshape(-1)[sel]
        levels = torch.linalg.lstsq(a, b[:, None]).solution[:, 0].clamp_min(0)
        pred = torch.einsum("k,k...->...", levels, c).clamp_min(1e-6)
        wf = w.to(dt)
        bias = masked_blur(m * pred, wf, sigma_vox) / masked_blur(
            pred * pred, wf, sigma_vox
        ).clamp_min(1e-12)
        expected = (bias * pred).clamp_min(1e-6)
        log_r = torch.log(m.clamp_min(1e-6) / expected)
        r = log_r[use]
        med = r.median()
        mad = 1.4826 * (r - med).abs().median()
        w = use & ((log_r - med).abs() <= trim * mad)
    med = log_r[use].median()
    expected = expected * torch.exp(med)
    ratio = torch.where(use, torch.exp(log_r - med), torch.ones_like(m))
    return ratio.to(mean.dtype), expected.to(mean.dtype), levels


def _dilate6(x: Tensor) -> Tensor:
    out = x.clone()
    for ax in range(3):
        for s in (1, -1):
            sh = torch.roll(x, s, ax)
            idx = [slice(None)] * 3
            idx[ax] = slice(0, 1) if s == 1 else slice(-1, None)
            sh[tuple(idx)] = False
            out |= sh
    return out


def vein_region(tissue: Tensor, brain: Tensor, gm_min: float = 0.3, gm_core: float = 0.5) -> Tensor:
    """Where a vein that matters for BOLD can be: GM (fraction >= ``gm_min``), or CSF
    touching cortex (CSF the largest class, next to a voxel with GM >= ``gm_core``):
    the pial veins in the sulci. WM is left out on purpose -- deep WM veins carry no
    cortical response, and dark WM is the confound this model exists to avoid."""
    gm, wm, csf = tissue[0], tissue[1], tissue[2]
    pial = (csf > torch.maximum(gm, wm)) & _dilate6(gm >= gm_core)
    return brain & ((gm >= gm_min) | pial)


def vein_mask(ratio: Tensor, region: Tensor, darker: float, dilate: int = 1) -> Tensor:
    """Region voxels darker than predicted by more than ``darker`` (0.25 = a quarter
    darker), grown ``dilate`` voxels (6-connected) to forgive residual EPI-anat
    misregistration, as the GRE mask it stands in for needed."""
    m = region & (ratio < 1.0 - darker)
    for _ in range(dilate):
        m = _dilate6(m)
    return m
