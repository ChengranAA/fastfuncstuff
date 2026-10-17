"""Reference-anchored phase unwrapping of a whole time series on the GPU.

MEDIC (warpkit) and ROMEO's ``-t epi`` mode spend nearly all their time unwrapping
every frame spatially with ROMEO's region growing, which is sequential and cannot
use a GPU. A frame differs from its neighbours in time by a small field change,
so this module does the spatial unwrap ONCE and anchors every frame to it:

1. **Reference.** The temporal-mean complex image (high SNR, typical anatomy),
   unwrapped by warpkit's own per-frame routine (MCPC-3D-S phase offset + C++ ROMEO
   + global-mode correction) for multi-echo, or ROMEO 3-D for a single echo. Where
   the field is steep, motion decoheres the mean (later echoes drop out), and its
   MCPC-3D-S offset and 2π counts go wrong; a few ROMEO'd single frames
   (``anchor_frames``) supply the offset (their circular mean) and outvote the
   reference's shortest-echo count.
2. **Change, per voxel, all frames at once.** Each frame's wrapped difference from
   the reference. The shortest echo's change is wrap-free up to ``1000/(2 TE1)`` Hz.
   Where a voxel's change really leaves that range (motion through a steep field,
   drift; a time-varying voxel), the anchored value flips by 2π between frames; the
   change is then followed along time instead (``temporal_path``), but only in
   voxels whose frame-to-frame steps are small, so noise cannot random-walk. The
   mean cannot speak for such a voxel, so its level is set by the anchor frames at
   their own time points.
3. **Echo ladder.** Each later echo's 2π count from the weighted through-origin fit
   of the earlier echoes' change (warpkit's temporal-consistency step, per voxel),
   then once more on the total phase: rounding the reference and the change
   separately lets their residuals add past π. From the third echo on, the count is
   then checked against a line WITH intercept, which no offset error can bias.
4. **Field.** Magnitude²-weighted fit through the origin (warpkit's convention).

Everything per voxel, so the series is streamed in slabs along its slowest spatial
axis: each slab holds every frame, which is what the temporal path needs.

:func:`svd_filter_fields` is warpkit's border-aware + global SVD field denoising,
ported to torch (warpkit is MIT; Van et al. 2026, doi:10.1162/IMAG.a.1262).
"""

from __future__ import annotations

import os
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

TWO_PI = 2.0 * np.pi

# A good voxel's shortest-echo change moves far less than π/2 between frames (noise
# ~1/SNR rad); in a noise-dominated voxel ~half of the steps exceed it.
_ROUGH_STEP = np.pi / 2


@dataclass
class UnwrapResult:
    field_hz: np.ndarray  # (x, y, z, t), unfiltered, 0 outside the mask
    mask: np.ndarray  # (x, y, z) int8: 0 outside, 2 core, 1 dilated border (warpkit's)
    phase_offset: np.ndarray  # (x, y, z) radians removed before unwrapping (0 for 1 echo)
    temporal_path: np.ndarray  # (x, y, z) bool, voxels whose change was followed in time
    unwrapped: np.ndarray | None = None  # (x, y, z, e, t) radians, offset removed


def raw_phase_to_radians(phase: np.ndarray, frame: int = 0) -> np.ndarray:
    """Map raw scanner phase onto [-π, π) the way warpkit does (min/max mode over echoes).

    ``phase`` is ``(x, y, z, e, t)``; the range is read from one frame per echo and
    the most common min and max win, so a single odd echo cannot set the scale.
    """
    from scipy.stats import mode

    lo = mode([float(phase[..., e, frame].min()) for e in range(phase.shape[3])]).mode
    hi = mode([float(phase[..., e, frame].max()) for e in range(phase.shape[3])]).mode
    out = (phase - lo) * np.float32(TWO_PI / (hi - lo)) - np.float32(np.pi)
    return out.astype(np.float32, copy=False)


def _wrap(x: Tensor) -> Tensor:
    return torch.remainder(x + np.pi, TWO_PI) - np.pi


def _slab_axis(a: np.ndarray) -> int:
    """The spatial axis that is slowest in memory: slabs along it are (near-)contiguous
    whether the series came C-ordered from np.stack or as a view of (t, z, y, x)."""
    return int(np.argmax([abs(st) for st in a.strides[:3]]))


def _slab_width(n_plane: int, ne: int, nt: int, device: torch.device) -> int:
    """Planes per slab: phase + magnitude + ~6 working copies of (e, t) per voxel."""
    from ..memory import get_available_memory

    per_plane = max(1, n_plane * ne * nt * 4 * 8)
    return max(1, int(get_available_memory(device) // per_plane))


def signal_frames(mag: np.ndarray) -> np.ndarray:
    """Frames carrying signal: noise volumes (scanner off, a few % of the median
    intensity) are often appended to a run and must not set the reference."""
    level = mag[..., 0, :].mean(axis=(0, 1, 2))
    return level > 0.3 * np.median(level)


def _complex_mean(
    phase: np.ndarray, mag: np.ndarray, keep: np.ndarray, device: torch.device
) -> np.ndarray:
    ax = _slab_axis(phase)
    out = np.empty(phase.shape[:4], np.complex64)
    pv, mv, ov = (np.moveaxis(a, ax, 0) for a in (phase, mag, out))
    sx = _slab_width(int(np.prod(pv.shape[1:3])), phase.shape[3], phase.shape[4], device)
    k = torch.from_numpy(keep).to(device)
    for a in range(0, pv.shape[0], sx):
        p = torch.from_numpy(np.ascontiguousarray(pv[a : a + sx])).to(device)
        m = torch.from_numpy(np.ascontiguousarray(mv[a : a + sx])).to(device)
        ov[a : a + sx] = torch.polar(m, p)[..., k].mean(-1).cpu().numpy()
    return out


def _single_echo_mask(mag_mean: np.ndarray, dilation: int, device: torch.device) -> np.ndarray:
    """Core automask (2) + ``dilation`` voxels of border (1), warpkit's mask convention."""
    import torch.nn.functional as F

    from .mask import automask

    vol = torch.from_numpy(np.ascontiguousarray(mag_mean.transpose(2, 1, 0))).float().to(device)
    core = automask(vol, dilate_extra=0, device=device).bool()
    x = core.float()[None, None]
    for _ in range(max(0, dilation)):
        x = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    m = core.to(torch.int8) + (x[0, 0] > 0.5).to(torch.int8)
    return m.cpu().numpy().transpose(2, 1, 0).copy()


def _romeo_volume(
    phase: np.ndarray, mag: np.ndarray, tes: np.ndarray, se_mask: np.ndarray | None, dilation: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROMEO one ``(x, y, z, e)`` volume: (unwrapped, mask 0/1/2, offset), no ladder.

    Multi-echo: warpkit's per-frame routine (MCPC-3D-S offset + ROMEO + global mode).
    Single echo: ROMEO 3-D in ``se_mask``; the offset cannot be separated, so it stays in.
    """
    if phase.shape[3] >= 2:
        from warpkit.unwrap import unwrap_phase

        work = phase.astype(np.float32).copy()  # warpkit removes the offset in place
        u, mask = unwrap_phase(
            work,
            mag.astype(np.float32),
            tes.astype(np.float32),
            np.ones(mag.shape[:3], bool),
            automask=True,
            automask_dilation=dilation,
        )
        offset = np.angle(np.exp(1j * (phase[..., 0] - u[..., 0]))).astype(np.float32)
        offset[mask == 0] = 0
        return u.astype(np.float32), mask.astype(np.int8), offset
    from warpkit.warpkit_cpp import romeo_unwrap3d

    assert se_mask is not None
    u = romeo_unwrap3d(
        phase=np.ascontiguousarray(phase[..., 0], np.float32),
        weights="romeo",
        mag=np.ascontiguousarray(mag[..., 0], np.float32),
        mask=se_mask > 0,
        correct_global=True,
    )
    u = np.asarray(u, np.float32)[..., None]
    u[se_mask == 0] = 0
    return u, se_mask, np.zeros(se_mask.shape, np.float32)


def unwrap_reference(
    ref_phase: np.ndarray,
    ref_mag: np.ndarray,
    tes: np.ndarray,
    *,
    anchors: Sequence[tuple[np.ndarray, np.ndarray]] = (),
    automask_dilation: int = 3,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatially unwrap the reference ``(x, y, z, e)`` volume: (unwrapped, mask, offset).

    ROMEO on the reference alone can settle on a different 2π count than ROMEO on
    the individual frames where the field is steep: motion there decoheres the
    temporal mean. Each of ``anchors`` (single frames' phase, magnitude) is ROMEO'd
    too, carried onto the reference by its wrapped difference, and where most of
    them put the shortest echo in another count, the reference follows the majority.
    Later echoes then come from a per-voxel ladder (warpkit's global-mode correction
    is one count per echo for the whole volume).

    Also returns each anchor's shortest-echo unwrapped phase on the reference's
    offset, ``(x, y, z, n_anchors)``, NaN outside that frame's mask (or for an anchor
    ROMEO failed on): the per-time check for voxels whose field moves too far for
    the reference to speak for them.
    """
    ne = ref_phase.shape[3]
    se_mask = None if ne >= 2 else _single_echo_mask(ref_mag[..., 0], automask_dilation, device)
    u, mask, offset = _romeo_volume(ref_phase, ref_mag, tes, se_mask, automask_dilation)
    inside = mask > 0
    pairs = []
    done: list = [None] * len(anchors)
    if anchors:
        from concurrent.futures import ThreadPoolExecutor

        # romeo releases the GIL around its C++ kernel, so threads parallelise it
        def romeo_or_none(pm):
            try:
                return _romeo_volume(pm[0], pm[1], tes, se_mask, automask_dilation)
            except (IndexError, ValueError) as exc:  # e.g. warpkit finds no head in it
                warnings.warn(f"anchor frame skipped, ROMEO failed: {exc}", stacklevel=2)
                return None

        with ThreadPoolExecutor(min(len(anchors), os.cpu_count() or 1)) as ex:
            done = list(ex.map(romeo_or_none, anchors))
        pairs = [(a, d) for a, d in zip(anchors, done, strict=True) if d is not None]
        if ne >= 2 and pairs:
            # MCPC-3D-S on the decohered mean is biased where later echoes drop out;
            # the frames' own offsets (circular mean) are what warpkit removes.
            z = sum(np.exp(1j * oa) * (ma > 0) for _a, (_u, ma, oa) in pairs)
            offset = np.where(np.abs(z) > 0, np.angle(z), offset).astype(np.float32)
            # keep each echo's 2pi count, re-base its fraction on the new offset
            u = u + np.angle(np.exp(1j * ((ref_phase - offset[..., None]) - u)))
    if pairs:
        ref0 = ref_phase[..., 0] - offset
        counts = []
        for (p, _m), (ua, ma, oa) in pairs:
            carried = ua[..., 0] + np.angle(np.exp(1j * (ref0 - (p[..., 0] - oa))))
            k = np.round((carried - u[..., 0]) / TWO_PI)
            counts.append(np.where(ma > 0, k, np.nan))
        counts = np.stack(counts, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN outside every mask
            vote = np.round(np.nanmedian(counts, -1))
        agree = (counts == vote[..., None]).sum(-1) > 0.5 * len(pairs)
        move = np.where(inside & agree & np.isfinite(vote), vote, 0.0)
        # an echo-1 count is 1000/TE1 Hz of field: carry it to every echo, then re-ladder
        u = u + (TWO_PI * move)[..., None] * (tes / tes[0])
    w = ref_mag.astype(np.float32) ** 2
    for e in range(1, ne):
        num = (w[..., :e] * u[..., :e] * tes[:e]).sum(-1)
        den = np.maximum((w[..., :e] * tes[:e] ** 2).sum(-1), 1e-12)
        u[..., e] += TWO_PI * np.round((num / den * tes[e] - u[..., e]) / TWO_PI)
    u[~inside] = 0
    at_anchor = np.full((*mask.shape, len(anchors)), np.nan, np.float32)
    for i, d in enumerate(done):
        if d is not None:
            ua, ma, oa = d
            shifted = ua[..., 0] + np.angle(np.exp(1j * (oa - offset)))
            at_anchor[..., i] = np.where(ma > 0, shifted, np.nan)
    return u.astype(np.float32), mask, offset, at_anchor


def _gather(slab: np.ndarray, shape: tuple[int, ...], idx: Tensor) -> Tensor:
    """A host slab, flattened to voxels, on ``idx``'s device and restricted to ``idx``."""
    return torch.from_numpy(np.ascontiguousarray(slab)).to(idx.device).reshape(shape)[idx]


def _offset_free_step(u: Tensor, w: Tensor, te: Tensor) -> Tensor:
    """-1, 0 or +1 for the last echo of ``u`` (V, e, t): whichever 2π step leaves the
    smallest weighted residual about a line WITH intercept through all its echoes.

    The through-origin prediction inherits any error in the removed phase offset,
    scaled by TE_e/TE_1 - 1 (4x for a typical third echo), which is enough to flip a
    low-signal voxel's count in the odd frame; a free intercept is blind to it.
    """
    t = te[None, :, None]
    sw, st, stt = w.sum(1), (w * t).sum(1), (w * t * t).sum(1)
    det = (sw * stt - st * st).clamp_min(1e-12)
    best = torch.full_like(u[:, 0], float("inf"))
    step = torch.zeros_like(u[:, 0])
    for k in (0.0, -1.0, 1.0):
        y = u.clone()
        y[:, -1] += TWO_PI * k
        sy, sty, syy = (w * y).sum(1), (w * t * y).sum(1), (w * y * y).sum(1)
        a = (stt * sy - st * sty) / det
        b = (sw * sty - st * sy) / det
        ssr = syy - a * sy - b * sty
        better = ssr < best - 1e-6 * syy.abs()
        best = torch.where(better, ssr, best)
        step = torch.where(better, torch.full_like(step, k), step)
    return step


def _anchor_slab(
    P: Tensor,
    M: Tensor,
    uref: Tensor,
    offset: Tensor,
    inside: Tensor,
    te: Tensor,
    *,
    temporal_path: bool,
    rough_max: float,
    at_anchor: Tensor | None = None,
    anchor_times: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """One slab, voxels flattened: P, M (V, e, t). Returns (U (V,e,t), field (V,t), path (V,))."""
    ne = P.shape[1]
    D = _wrap(P - offset[:, None, None] - uref[:, :, None])
    c = D[:, 0]
    path = torch.zeros(c.shape[0], dtype=torch.bool, device=c.device)
    if temporal_path and c.shape[1] > 2:
        step = _wrap(torch.diff(c, dim=1))
        u = torch.cat([c[:, :1], c[:, :1] + torch.cumsum(step, 1)], 1)
        u = u - TWO_PI * torch.round(u.median(1, keepdim=True).values / TWO_PI)
        moved = (torch.round((u - c) / TWO_PI) != 0).any(1)
        smooth = (step.abs() > _ROUGH_STEP).float().mean(1) <= rough_max
        path = moved & smooth & inside
        if at_anchor is not None and anchor_times is not None and at_anchor.shape[1]:
            # A voxel that moves this far decoheres the temporal mean, so the
            # reference's count there is unreliable. Set the path's level by majority
            # against the ROMEO'd frames at their own time points instead.
            k = torch.round((uref[:, :1] + u[:, anchor_times] - at_anchor) / TWO_PI)
            valid = torch.isfinite(k)
            vote = torch.nanmedian(torch.where(valid, k, torch.nan), 1).values
            agree = ((k == vote[:, None]) & valid).sum(1) > 0.5 * valid.sum(1)
            u = u - TWO_PI * torch.where(path & agree, torch.nan_to_num(vote), 0.0)[:, None]
        c = torch.where(path[:, None], u, c)
    w = M * M
    dphi = torch.empty_like(D)
    dphi[:, 0] = c
    for e in range(1, ne):
        sl = (w[:, :e] * dphi[:, :e] * te[None, :e, None]).sum(1) / (
            w[:, :e] * te[None, :e, None] ** 2
        ).sum(1).clamp_min(1e-12)
        dphi[:, e] = D[:, e] + TWO_PI * torch.round((sl * te[e] - D[:, e]) / TWO_PI)
    U = uref[:, :, None] + dphi
    for e in range(1, ne):
        sl = (w[:, :e] * U[:, :e] * te[None, :e, None]).sum(1) / (
            w[:, :e] * te[None, :e, None] ** 2
        ).sum(1).clamp_min(1e-12)
        U[:, e] += TWO_PI * torch.round((sl * te[e] - U[:, e]) / TWO_PI)
        if e >= 2:
            U[:, e] += TWO_PI * _offset_free_step(U[:, : e + 1], w[:, : e + 1], te[: e + 1])
    U = U * inside[:, None, None]
    field = (w * U * te[None, :, None]).sum(1) / (w * te[None, :, None] ** 2).sum(1).clamp_min(
        1e-12
    )
    return U, field * (1000.0 / TWO_PI), path


def unwrap_timeseries(
    phase: np.ndarray,
    mag: np.ndarray,
    tes: Sequence[float],
    *,
    device: torch.device,
    temporal_path: bool = True,
    rough_max: float = 0.05,
    anchor_frames: int = 8,
    automask_dilation: int = 3,
    return_unwrapped: bool = False,
    verbose: bool = True,
) -> UnwrapResult:
    """Unwrap ``(x, y, z, e, t)`` phase (radians, [-π, π)) against one reference.

    ``mag`` matches ``phase``; ``tes`` in ms, shortest first. ``rough_max`` is the
    largest fraction of frame-to-frame steps above π/2 a voxel may have and still
    have its shortest-echo change followed in time (see the module docstring).
    """
    if phase.shape != mag.shape or phase.ndim != 5:
        raise ValueError(
            f"phase/mag must share an (x, y, z, e, t) shape: {phase.shape} vs {mag.shape}"
        )
    te_np = np.asarray(tes, np.float32)
    if te_np.size != phase.shape[3]:
        raise ValueError(f"{te_np.size} echo times for {phase.shape[3]} echoes")
    if np.any(np.diff(te_np) <= 0):
        raise ValueError(f"echo times must increase: {te_np.tolist()}")
    nx, ny, nz, ne, nt = phase.shape
    t0 = time.perf_counter()
    keep = signal_frames(mag)
    zbar = _complex_mean(phase, mag, keep, device)
    good = np.flatnonzero(keep)
    picks = good[
        np.unique(np.linspace(0, good.size - 1, min(anchor_frames, good.size)).round().astype(int))
    ]
    uref, mask, offset, at_anchor = unwrap_reference(
        np.angle(zbar).astype(np.float32),
        np.abs(zbar).astype(np.float32),
        te_np,
        anchors=[(phase[..., t], mag[..., t]) for t in picks] if anchor_frames > 0 else (),
        automask_dilation=automask_dilation,
        device=device,
    )
    del zbar
    t_ref = time.perf_counter() - t0

    field = np.zeros((nx, ny, nz, nt), np.float32)
    path_map = np.zeros((nx, ny, nz), bool)
    unwrapped = np.zeros(phase.shape, np.float32) if return_unwrapped else None
    te = torch.from_numpy(te_np).to(device)
    ax = _slab_axis(phase)
    views = [
        np.moveaxis(a, ax, 0) for a in (phase, mag, uref, offset, mask, field, path_map, at_anchor)
    ]
    pv, mv, uv, ofv, mkv, fv, pmv, av = views
    n_anchor = at_anchor.shape[-1]
    t_anchor = torch.from_numpy(picks[:n_anchor]).to(device) if n_anchor else None
    unv = np.moveaxis(unwrapped, ax, 0) if unwrapped is not None else None
    d1, d2 = pv.shape[1:3]
    sx = _slab_width(d1 * d2, ne, nt, device)
    slabs = range(0, pv.shape[0], sx)
    for a in tqdm(
        slabs, desc="  anchored unwrap", leave=True, disable=not verbose or len(slabs) < 2
    ):
        b = min(pv.shape[0], a + sx)
        n = (b - a) * d1 * d2
        inside = torch.from_numpy(np.ascontiguousarray(mkv[a:b] > 0)).to(device).reshape(n)
        idx = inside.nonzero()[:, 0]
        if idx.numel() == 0:
            continue
        U, f, p = _anchor_slab(
            _gather(pv[a:b], (n, ne, nt), idx),
            _gather(mv[a:b], (n, ne, nt), idx),
            _gather(uv[a:b], (n, ne), idx),
            _gather(ofv[a:b], (n,), idx),
            torch.ones(idx.numel(), dtype=torch.bool, device=device),
            te,
            temporal_path=temporal_path,
            rough_max=rough_max,
            at_anchor=_gather(av[a:b], (n, n_anchor), idx) if n_anchor else None,
            anchor_times=t_anchor,
        )
        full = torch.zeros(n, nt, device=device)
        full[idx] = f
        fv[a:b] = full.reshape(b - a, d1, d2, nt).cpu().numpy()
        pm = torch.zeros(n, dtype=torch.bool, device=device)
        pm[idx] = p
        pmv[a:b] = pm.reshape(b - a, d1, d2).cpu().numpy()
        if unv is not None:
            full = torch.zeros(n, ne, nt, device=device)
            full[idx] = U
            unv[a:b] = full.reshape(b - a, d1, d2, ne, nt).cpu().numpy()
        del U, f, full
    if verbose:
        dt = time.perf_counter() - t0
        n_in = int((mask > 0).sum())
        print(
            f"  anchored unwrap: {nt} frames x {ne} echo(es) in {dt:.1f}s "
            f"(reference {t_ref:.1f}s); {n_in:,} voxels, "
            f"{int(path_map.sum()):,} followed in time"
        )
    return UnwrapResult(
        field_hz=field,
        mask=mask,
        phase_offset=offset,
        temporal_path=path_map,
        unwrapped=unwrapped,
    )


def _gaussian_symmetric(vols: Tensor, sigma: float) -> Tensor:
    """Separable Gaussian of ``(T, x, y, z)`` volumes, scipy's ``reflect`` edges, truncate 4."""
    r = int(4.0 * sigma + 0.5)
    k = torch.arange(-r, r + 1, device=vols.device, dtype=vols.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    out = vols
    for ax in (1, 2, 3):
        n = out.shape[ax]
        idx = torch.arange(-r, n + r, device=vols.device)
        # scipy 'reflect' = (d c b a | a b c d | d c b a): mirror including the edge sample
        period = 2 * n
        idx = torch.remainder(idx, period)
        idx = torch.where(idx >= n, period - 1 - idx, idx)
        padded = out.index_select(ax, idx)
        moved = padded.movedim(ax, -1)
        sh = moved.shape
        conv = torch.nn.functional.conv1d(moved.reshape(-1, 1, sh[-1]), k[None, None])
        out = conv.reshape(*sh[:-1], n).movedim(-1, ax)
    return out


def _low_rank(a: Tensor, k: int) -> Tensor:
    """Rank-``k`` reconstruction of a tall (V, T) matrix via its T x T Gram matrix."""
    from ..utils import factor_device

    g = (a.T.double() @ a.double()).to(factor_device(a.device))
    _, vecs = torch.linalg.eigh(g)
    v = vecs[:, -k:].to(a.device, a.dtype)
    return (a @ v) @ v.T


def svd_filter_fields(
    field: np.ndarray,
    mask: np.ndarray,
    voxel_size: float,
    *,
    border_filt: tuple[int, int] = (1, 5),
    svd_filt: int = 10,
    device: torch.device,
) -> np.ndarray:
    """warpkit's field denoising: border voxels (mask 1) from a smoothed low-rank fit,
    then every masked voxel from a rank-``svd_filt`` fit. ``mask`` is 3-D or 4-D."""
    nt = field.shape[-1]
    m4 = mask if mask.ndim == 4 else np.broadcast_to(mask[..., None], field.shape)
    union = (m4 > 0).any(-1)
    out = field.copy()
    flat_union = torch.from_numpy(union).to(device)
    border = torch.from_numpy(np.ascontiguousarray((m4 == 1)[union])).to(device)  # (V, T)
    inside = torch.from_numpy(np.ascontiguousarray((m4 > 0)[union])).to(device)
    f = torch.from_numpy(out).to(device)  # (x, y, z, T)
    if m4.max() == 2 and nt >= max(border_filt):
        sigma = (4.0 / voxel_size) / 2.355
        sm = _gaussian_symmetric(f.permute(3, 0, 1, 2), sigma).permute(1, 2, 3, 0)
        vals = f[flat_union]
        rec = _low_rank(sm[flat_union], border_filt[0])
        vals = torch.where(border, rec, vals)
        rec = _low_rank(vals, border_filt[1])
        vals = torch.where(border, rec, vals)
        f[flat_union] = vals
        del sm
    if nt >= svd_filt:
        vals = f[flat_union]
        rec = _low_rank(vals, svd_filt)
        f[flat_union] = torch.where(inside, rec, vals)
    return f.cpu().numpy()
