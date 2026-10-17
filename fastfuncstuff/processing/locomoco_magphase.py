"""Joint magnitude + phase estimate of the TE-scaled partition wiggle (EXPERIMENTAL).

The multi-echo partition shift is ``d_e(r, t) = alpha_e * w(r, t)`` with ``alpha_e =
TE_e / TE_1`` fixed. A genuine displacement moves the complex image, so it shows in the
magnitude AND in the phase. What each channel also carries that is NOT displacement is
different, and multi-echo data can tell the two apart per voxel:

* phase, as a field in Hz: a field change (respiration, drift, phase BOLD) is the SAME
  number of Hz at every echo, while a displacement's footprint ``-d_e * dHz_e/dz`` grows
  with TE. Projecting out one echo-constant term per voxel and frame makes the phase
  field-blind without any spatial high-pass.
* log-magnitude: an S0 change is echo-constant, a BOLD/R2* change is proportional to TE.
  Projecting those out ("s0r2") makes the magnitude BOLD-blind. The leading term of the
  displacement footprint is also proportional to TE, so this costs magnitude most of its
  information; the phase then carries the estimate, which is the point.

Each channel's per-voxel nuisance is projected out (weighted by its noise) before the
Lucas-Kanade sums are pooled over the ``window_sigma`` Gaussian. On real 7T 5-echo data
the magnitude-only estimate reported 2-4x more displacement than this at the slab ends,
the brain edge and where the I/S field gradient is steep, and applying it to the phase
there made the phase worse -- i.e. it was not a displacement of the image.

The phase must be UNWRAPPED and resampled with exactly the transforms the magnitude got
(one resample, no wrapped values ever interpolated); :mod:`.phase_input` is the guard.

Single echo: nothing per voxel separates a field change from a displacement, so the
phase instead gets a mask-aware spatial high-pass (``phase_hpf_sigma``), which removes
the smooth field fluctuation and keeps tissue-scale texture. Not field-blind -- a field
change with structure at the texture scale still reads as shift -- but on the 7T pilot
the high-passed phase recovered injected shifts without bias (slope ~1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

MAG_NUISANCE_CHOICES = ("none", "s0", "s0r2")


@dataclass
class MagPhaseFlow:
    w_field: torch.Tensor  # (nx,ny,nz,T) shared field; echo e moves by alpha_e * w (voxels)
    alpha: torch.Tensor  # (E,) TE_e / TE_1
    phase_share: torch.Tensor  # (nx,ny,nz) fraction of the pooled information from phase
    phase_valid: torch.Tensor  # (nx,ny,nz) bool, where phase entered the solve
    brain: torch.Tensor  # (nx,ny,nz) bool, the voxels whose observations were used


def nuisance_columns(kind: str, te: torch.Tensor) -> torch.Tensor:
    """Per-echo nuisance design ``(E, q)``: 'flat' / 's0' = 1, 'r2' adds TE (scaled),
    'flat_offset' adds TE1/TE (a receiver-phase change at TE 0, seen in Hz)."""
    cols = []
    if kind in ("flat", "flat_offset", "s0", "s0r2"):
        cols.append(torch.ones_like(te))
    if kind == "s0r2":
        cols.append(te / te.mean())
    if kind == "flat_offset":
        cols.append(te[0] / te)
    if not cols:
        return torch.zeros(te.numel(), 0)
    return torch.stack(cols, 1)


def _weighted_design(
    x: torch.Tensor, wts: torch.Tensor, n_cols: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``M x`` and ``x' M x`` per voxel, ``M = W - W N (N'WN)^-1 N'W``.

    ``x`` and ``wts`` are ``(E, V)``. ``M`` is the weighted residual-maker of the
    nuisance block, so ``x' M y`` is the displacement evidence left after each voxel's
    own nuisance has been fitted out of ``y``. M is never materialised: ``M x`` is all
    the LK sums need, because ``x' M y = (M x)' y``.
    """
    from ..memory import bytes_per_voxel_magphase_design, get_available_memory

    e, v = x.shape
    q = n_cols.shape[1]
    if q == 0:
        mx = wts * x
        return mx, (x * mx).sum(0)
    n = n_cols.to(device, x.dtype)
    chunk = max(1, get_available_memory(device) // bytes_per_voxel_magphase_design(e, q))
    mx = torch.empty_like(x)
    for s in range(0, v, chunk):
        wv = wts[:, s : s + chunk].T  # (c,E)
        xv = x[:, s : s + chunk].T
        wn = wv[:, :, None] * n[None]  # (c,E,q)
        nwn = torch.einsum("eq,cer->cqr", n, wn)
        # A voxel with no valid observation in this channel has W = 0, so N'WN = 0; the
        # tiny ridge makes its solve return 0 instead of NaN, and M x is then 0 too.
        ridge = 1e-6 * nwn.diagonal(dim1=1, dim2=2).sum(1).clamp_min(1e-30)
        nwn = nwn + ridge[:, None, None] * torch.eye(q, device=device, dtype=x.dtype)
        nwx = torch.einsum("eq,ce->cq", n, wv * xv)
        c = torch.linalg.solve(nwn, nwx[..., None])[..., 0]
        mx[:, s : s + chunk] = (wv * xv - torch.einsum("ceq,cq->ce", wn, c)).T
    return mx, (x * mx).sum(0)


def _masked_highpass(
    vols: list[np.ndarray], mask: np.ndarray, sigma: float, device: torch.device
) -> list[np.ndarray]:
    """``v - G(v m) / G(m)`` inside ``m``, 0 outside, per frame.

    Normalising by the blurred mask keeps the smooth estimate from being dragged toward
    the zero fill at the support edge, which would leave a ring of fake texture there;
    filling 0 outside (the high-pass residual is ~zero-mean) avoids a step at the edge.
    """
    from .locomoco import _blur3d_b

    m = torch.from_numpy(mask).to(device).float()
    gm = _blur3d_b(m[None], sigma)[0].clamp_min(1e-3)
    out = []
    for vol in vols:
        hp = np.empty_like(vol, dtype=np.float32)
        for t in range(vol.shape[3]):
            v = torch.from_numpy(np.ascontiguousarray(vol[..., t])).to(device).float() * m
            hp[..., t] = ((v - _blur3d_b(v[None], sigma)[0] / gm) * m).cpu().numpy()
        out.append(hp)
    return out


def _pe_erode(mask: np.ndarray, axis: int, n: int) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    st = np.zeros((3, 3, 3), bool)
    idx = [1, 1, 1]
    for k in (0, 1, 2):
        idx[axis] = k
        st[tuple(idx)] = True
    return binary_erosion(mask, structure=st, iterations=n) if n > 0 else mask


def estimate_partition_flow_magphase(
    mags: list[np.ndarray],
    phases_hz: list[np.ndarray],
    echo_times: list[float],
    pe_axis: int,
    *,
    window_sigma: float = 2.0,
    n_iters: int = 4,
    refine_rounds: int = 0,
    mag_nuisance: str = "s0r2",
    phase_nuisance: str = "auto",
    phase_hpf_sigma: float = 0.0,
    warp_interp: str = "lanczos",
    warp_radius: int = 3,
    device: torch.device | None = None,
    verbose: bool = True,
) -> MagPhaseFlow:
    """Shared TE-scaled partition field from magnitude AND unwrapped phase (Hz).

    ``mags`` / ``phases_hz``: one ``(nx, ny, nz, T)`` array per echo, same grid, same
    motion/distortion resample. ``mag_nuisance`` picks what the magnitude may change by
    in place at each voxel and frame: 'none' (plain), 's0' (echo-constant), 's0r2' (plus
    proportional to TE; BOLD-blind, needs 3+ echoes). With 2+ echoes the phase gets the
    echo-constant field term, and by default (``phase_nuisance='auto'``) with 3+ a
    receiver-offset term too (TE1/TE in Hz): a drifting offset otherwise reads as a
    TE-dependent shift (one 3-echo run drifted 0.08 rad). ``phase_hpf_sigma`` (voxels, required for a single echo,
    optional otherwise) high-passes it inside its support first. Gauss-Newton: ``n_iters`` pull-warp updates per frame
    against the per-echo temporal-mean templates; ``refine_rounds`` rebuilds those
    templates from the corrected frames and re-solves (the uncorrected mean still
    carries the wiggle and biases the field low).
    """
    from .locomoco import _blur3d_b, _shift3d_axis
    from .mask import automask

    if device is None:
        device = torch.device("cpu")
    e = len(mags)
    if e < 1 or len(phases_hz) != e or len(echo_times) != e:
        raise ValueError("mag+phase needs one phase series and one TE per magnitude echo.")
    if e == 1 and mag_nuisance != "none":
        raise ValueError("a single echo has no per-voxel magnitude nuisance to fit; use 'none'.")
    if phase_nuisance not in ("auto", "none", "flat", "flat_offset"):
        raise ValueError(
            f"phase_nuisance must be auto/none/flat/flat_offset, got {phase_nuisance!r}"
        )
    if phase_nuisance == "flat_offset" and e < 3:
        raise ValueError(
            "phase_nuisance 'flat_offset' fits two terms per voxel and needs 3+ echoes."
        )
    if e == 1 and phase_hpf_sigma <= 0:
        raise ValueError(
            "single-echo phase needs phase_hpf_sigma > 0: without the TE law the smooth field "
            "fluctuation is indistinguishable from displacement."
        )
    if mag_nuisance not in MAG_NUISANCE_CHOICES:
        raise ValueError(f"mag_nuisance must be one of {MAG_NUISANCE_CHOICES}")
    if mag_nuisance == "s0r2" and e < 3:
        raise ValueError("mag_nuisance 's0r2' fits two terms per voxel; it needs >= 3 echoes.")
    shape = mags[0].shape
    nx, ny, nz, nt = (int(s) for s in shape)
    te = torch.tensor([float(t) for t in echo_times], dtype=torch.float32)
    alpha = te / te[0]
    alpha_d = alpha.to(device)[:, None, None, None]

    echo_mean = np.mean([m.mean(axis=3) for m in mags], axis=0)
    brain_np = (
        automask(
            torch.from_numpy(echo_mean.transpose(2, 1, 0).copy()).float(),
            device=torch.device("cpu"),
        )
        .numpy()
        .transpose(2, 1, 0)
        .astype(bool)
    )
    # Phase is trusted only where every echo has a value in (nearly) every frame, and far
    # enough from the edge of that support that neither the PE derivative nor the warp
    # kernel reads the zero fill outside it -- that step would be a huge fake gradient.
    have = np.ones(shape[:3], bool)
    for p in phases_hz:
        have &= np.count_nonzero(np.isfinite(p) & (p != 0), axis=3) >= 0.99 * nt
    reach = (warp_radius if warp_interp == "lanczos" else 2) + 1
    valid_np = _pe_erode(have & brain_np, pe_axis, reach)
    brain = torch.from_numpy(brain_np).to(device)
    valid_in_brain = torch.from_numpy(valid_np[brain_np]).to(device)
    v = int(brain.sum())

    if phase_hpf_sigma > 0:
        phases_hz = _masked_highpass(phases_hz, have & brain_np, phase_hpf_sigma, device)

    floor = float(echo_mean[brain_np].mean()) * 1e-3
    chans = {"mag": mags, "hz": phases_hz}
    if phase_nuisance == "auto":
        # an offset change needs a third echo to be told apart from field + shift
        phase_nuisance = "flat_offset" if e >= 3 else ("flat" if e >= 2 else "none")
    kinds = {"mag": mag_nuisance, "hz": phase_nuisance}

    def frame(ch: str, t: int, w: torch.Tensor | None) -> torch.Tensor:
        vol = torch.stack(
            [torch.from_numpy(np.ascontiguousarray(a[..., t])) for a in chans[ch]]
        ).to(device)
        if w is not None:
            vol = _shift3d_axis(
                vol, alpha_d * w[None], pe_axis, mode=warp_interp, radius=warp_radius
            )
        if ch == "mag":
            vol = torch.log(vol.clamp_min(floor))
        return vol

    def deriv(vol: torch.Tensor) -> torch.Tensor:
        return (torch.roll(vol, -1, pe_axis + 1) - torch.roll(vol, 1, pe_axis + 1)) / 2

    def templates(w_all: torch.Tensor | None) -> dict[str, torch.Tensor]:
        acc = {ch: torch.zeros(e, nx, ny, nz, device=device) for ch in chans}
        for t in range(nt):
            wt = None if w_all is None else w_all[..., t].to(device)
            for ch in chans:
                acc[ch] += frame(ch, t, wt)
        return {ch: a / nt for ch, a in acc.items()}

    # Noise weights: temporal variance of each observation's frame-to-frame change once
    # its channel's nuisance is projected out (unweighted), so a channel is trusted in
    # proportion to what its displacement-relevant part actually fluctuates by.
    weights = {}
    for ch in chans:
        n = nuisance_columns(kinds[ch], te).to(device)
        proj = torch.eye(e, device=device) - (n @ torch.linalg.pinv(n) if n.shape[1] else 0)
        acc = torch.zeros(e, v, device=device)
        prev = frame(ch, 0, None)[:, brain]
        for t in range(1, nt):
            cur = frame(ch, t, None)[:, brain]
            r = proj @ (cur - prev)
            acc += r * r
            prev = cur
        var = acc / (2 * max(nt - 1, 1))
        wt = 1.0 / var.clamp_min(float(var.median()) * 1e-3)
        if ch == "hz":
            wt = wt * valid_in_brain[None]
        weights[ch] = wt

    def to_vol(vals: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(nx, ny, nz, device=device)
        out[brain] = vals
        return out

    w_all = torch.zeros(nx, ny, nz, nt)
    share = None
    for rnd in range(refine_rounds + 1):
        tpl = templates(None if rnd == 0 else w_all)
        mx, sxx_tot, info = {}, torch.zeros(v, device=device), {}
        for ch in chans:
            x = -alpha.to(device)[:, None] * deriv(tpl[ch])[:, brain]
            mx[ch], sxx = _weighted_design(x, weights[ch], nuisance_columns(kinds[ch], te), device)
            info[ch] = _blur3d_b(to_vol(sxx)[None], window_sigma)[0]
            sxx_tot += sxx
        g_sxx = _blur3d_b(to_vol(sxx_tot)[None], window_sigma)[0]
        lam = float(g_sxx[brain].median()) * 1e-3
        share = (info["hz"] / (info["hz"] + info["mag"]).clamp_min(1e-30)).cpu()
        tpl_b = {ch: tpl[ch][:, brain] for ch in chans}
        desc = (
            "mag+phase" if refine_rounds == 0 else f"mag+phase pass {rnd + 1}/{refine_rounds + 1}"
        )
        for t in tqdm(
            range(nt), desc=desc, unit="frame", leave=True, disable=nt < 3 or not verbose
        ):
            w = w_all[..., t].to(device)
            for _ in range(n_iters):
                sxy = torch.zeros(v, device=device)
                for ch in chans:
                    y = frame(ch, t, w)[:, brain] - tpl_b[ch]
                    sxy += (mx[ch] * y).sum(0)
                # The pooled ratio decays smoothly to 0 outside the brain (sxy -> 0 while
                # lam stays), so the field needs no hard mask edge.
                w = w + _blur3d_b(to_vol(sxy)[None], window_sigma)[0] / (g_sxx + lam)
            w_all[..., t] = w.cpu()

    if verbose:
        ap = w_all[torch.from_numpy(brain_np)].abs()
        print(
            f"🧲 mag+phase (experimental): {e} echo{'es' if e > 1 else ''}, mag nuisance {mag_nuisance}, "
            f"phase {'field-blind' if e > 1 else 'high-passed'}"
            f"{f' (hpf σ {phase_hpf_sigma:.2g} vox)' if phase_hpf_sigma > 0 else ''}; "
            f"phase valid in {int(valid_np.sum())}/{int(brain_np.sum())} brain voxels, "
            f"median phase share of information {float(share[torch.from_numpy(brain_np)].median()):.2f}; "
            f"|w| median {float(ap.median()):.4f} vox (echo {e}: x{float(alpha[-1]):.2f})"
        )
    return MagPhaseFlow(
        w_field=w_all,
        alpha=alpha,
        phase_share=share,
        phase_valid=torch.from_numpy(valid_np),
        brain=torch.from_numpy(brain_np),
    )


def magphase_me_result(
    datas: list[np.ndarray],
    flow: MagPhaseFlow,
    echo_times: list[float],
    pe_axis: int,
    slice_axis: int,
    *,
    want_corrected: bool = True,
    warp_interp: str = "lanczos",
    warp_radius: int = 3,
    device: torch.device | None = None,
):
    """Wrap a :class:`MagPhaseFlow` as the multi-echo result the CLI writes from.

    Same geometry derivation and per-echo layout as the magnitude-only estimator, so the
    warps, flows and corrected series are written by the unchanged output code.
    """
    from .locomoco import LocomocoResult, MultiEchoLocomocoResult, _resample_once

    if device is None:
        device = torch.device("cpu")
    nx, ny, nz, nt = (int(s) for s in datas[0].shape)
    disp_slice = slice_axis if slice_axis != pe_axis else next(a for a in (0, 1, 2) if a != pe_axis)
    a0, a1 = sorted(a for a in (0, 1, 2) if a != disp_slice)
    perm4 = [3, disp_slice, a0, a1]
    per_echo = []
    for j, data in enumerate(datas):
        field = flow.alpha[j] * flow.w_field
        corrected = (
            _resample_once(
                data,
                [field],
                [pe_axis],
                warp_interp=warp_interp,
                warp_radius=warp_radius,
                device=device,
                desc=f"resample e{j + 1}",
            )
            if want_corrected
            else None
        )
        canon = field.permute(perm4).contiguous()
        zero = torch.zeros_like(canon)
        per_echo.append(
            LocomocoResult(
                u_canon=canon if pe_axis == a1 else zero,
                v_canon=canon if pe_axis == a0 else zero,
                corrected_canon=zero,
                perm=perm4,
                pe_flow_is_u=pe_axis == a1,
                pe_axis=pe_axis,
                slice_axis=disp_slice,
                orig_shape=(nx, ny, nz, nt),
                a0=a0,
                a1=a1,
                corrected_nifti=corrected,
            )
        )
    return MultiEchoLocomocoResult(
        per_echo=per_echo,
        alpha=flow.alpha,
        echo_times=torch.tensor([float(t) for t in echo_times]),
        w_field=flow.w_field,
        pe_axis=pe_axis,
        linearity_r2=1.0,
    )
