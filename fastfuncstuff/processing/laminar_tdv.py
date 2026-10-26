"""Column sampling, continuous vascular inversion, and voxel residual preservation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra
from tqdm.auto import tqdm

from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.glm.xval import project_out_nuisance_per_run
from fastfuncstuff.laminar.params import P0, ModelSpec
from fastfuncstuff.laminar.timeseries import (
    dynamic_inverse,
    linearized_vasculature,
    vascular_transfer,
)
from fastfuncstuff.memory import estimate_chunk_size


def depth_patches(
    mask: np.ndarray, depth: np.ndarray, spacing: np.ndarray, radius_mm: float = 5.0
) -> np.ndarray:
    """Disjoint local patches reached through GM, avoiding jumps across sulci.

    Labels are flat C-order within mask. With segmentation-only depths, genuine
    touching-bank bridges cannot be resolved. Prefer surface column labels when
    available; this fallback does not claim to trace anatomical vessels.
    """
    if radius_mm <= 0 or np.any(np.asarray(spacing) <= 0):
        raise ValueError("radius and voxel spacing must be positive")
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        return np.empty(0, np.int64)
    index = np.full(mask.shape, -1, np.int64)
    index.reshape(-1)[ids] = np.arange(ids.size)
    rows, cols, weights = [], [], []
    for ax in range(3):
        lo, hi = [slice(None)] * 3, [slice(None)] * 3
        lo[ax], hi[ax] = slice(None, -1), slice(1, None)
        a, b = index[tuple(lo)], index[tuple(hi)]
        good = (a >= 0) & (b >= 0)
        rows.extend([a[good], b[good]])
        cols.extend([b[good], a[good]])
        weights.extend([np.full(good.sum(), spacing[ax])] * 2)
    graph = sp.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
        shape=(ids.size, ids.size),
    )
    # Mid-ribbon seeds give patches a chance to span depth before boundary seeds.
    order = np.argsort(np.abs(depth.reshape(-1)[ids] - 0.5), kind="stable")
    labels = np.full(ids.size, -1, np.int64)
    next_label = 0
    for seed in order:
        if labels[seed] >= 0:
            continue
        distance = dijkstra(graph, indices=int(seed), limit=radius_mm, directed=False)
        take = (labels < 0) & np.isfinite(distance)
        labels[take] = next_label
        next_label += 1
    return labels


def depth_interpolation(depth: np.ndarray, K: int) -> np.ndarray:
    """Point-sample weights (V,K), 0 pial, 1 white; rows sum to one.

    This estimates bin-sharing blur exactly for these weights. It cannot infer
    subvoxel cortical geometry from a segmentation's single mean-depth number.
    """
    if K < 2 or not np.isfinite(depth).all() or np.any((depth < 0) | (depth > 1)):
        raise ValueError("K>=2 and finite depths between 0 and 1 required")
    pos = np.clip(np.asarray(depth) * K - 0.5, 0, K - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, K - 1)
    weights = np.zeros((len(depth), K), np.float32)
    weights[np.arange(len(depth)), lo] = 1 - (pos - lo)
    weights[np.arange(len(depth)), hi] += pos - lo
    return weights


@dataclass
class ColumnProfiles:
    """Each valid patch's detrended PSC depth profile and how its bins were read."""

    members: list[np.ndarray]  # voxel indices per valid patch
    profiles: torch.Tensor  # R x U x T x K
    sampling: torch.Tensor  # U x K x K, physiological depths -> measured bins
    noise_cov: torch.Tensor  # U x K x K, bin covariance from shared voxels
    read: np.ndarray  # V x K point-sample weights
    nuisance: list[torch.Tensor]  # per run, T x n_poly


def column_profiles(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    K: int,
    polort: int,
    device: torch.device,
) -> ColumnProfiles:
    """Percent change, per-run Legendre drift removed, averaged into K bins per patch.

    Patches that do not observe every model depth are left out (they pass through).
    """
    if not runs or any(r.ndim != 2 or r.shape != runs[0].shape for r in runs):
        raise ValueError("runs must have the same (voxels,time) shape")
    V, T = runs[0].shape
    if len(labels) != V or len(depth) != V or T < 4:
        raise ValueError("depth/labels must match voxels and runs need >=4 samples")
    if not all(torch.isfinite(r).all() and (r.mean(-1) > 0).all() for r in runs):
        raise ValueError("runs need finite data and positive voxel means")
    R = len(runs)
    read = depth_interpolation(depth, K)
    unique, inverse = np.unique(labels, return_inverse=True)
    members, samples, covs, sampling = [], [], [], []
    psc = torch.cat([100 * (r / r.mean(-1, keepdim=True) - 1) for r in runs], -1)
    starts = [i * T for i in range(R)]
    nuisance = [construct_polynomial_matrix(T, polort, device, torch.float32) for _ in runs]
    psc, _ = project_out_nuisance_per_run(
        psc, torch.empty(R * T, 0, device=device), nuisance, starts, device=device, verbose=False
    )
    # Spatial means use CPU storage; only the depth inverse is sent to compute.
    psc_np = psc.cpu().numpy().reshape(V, R, T).transpose(1, 0, 2)
    del psc
    for unit in range(len(unique)):
        ix = np.flatnonzero(inverse == unit)
        W = read[ix].T
        mass = W.sum(-1)
        # Every model depth must be observed; interpolating missing depths would
        # manufacture the independent information the inverse needs.
        if ix.size < 2 * K or mass.min() < 1.0:
            continue
        S = W / mass[:, None]
        members.append(ix)
        sampling.append(S @ read[ix])
        covs.append(S @ S.T)
        samples.append(np.stack([S @ psc_np[r, ix] for r in range(R)]).transpose(0, 2, 1))
    if not samples:
        raise ValueError("no patch spans every model depth; enlarge patches or reduce -depths")
    return ColumnProfiles(
        members,
        torch.as_tensor(np.stack(samples, axis=1)),
        torch.as_tensor(np.stack(sampling)),
        torch.as_tensor(np.stack(covs)),
        read,
        nuisance,
    )


@dataclass
class PhysiologyCV:
    """Leave-one-run-out design prediction for each candidate physiology."""

    candidates: list[P0]
    sse: np.ndarray  # P x U held-out whitened SSE
    total: np.ndarray  # U

    @property
    def pooled_r2(self) -> np.ndarray:
        return 1 - self.sse.sum(1) / self.total.sum()

    @property
    def patch_r2(self) -> np.ndarray:
        return 1 - self.sse / np.maximum(self.total, 1e-30)

    @property
    def best(self) -> int:
        return int(np.argmax(self.pooled_r2))


def physiology_crossval(
    columns: ColumnProfiles,
    candidates: list[P0],
    spec: ModelSpec,
    design_for: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    params: dict[str, torch.Tensor] | None = None,
    microtime_dt: float = 0.1,
    tail_seconds: float = 60.0,
    verbose: bool = True,
) -> PhysiologyCV:
    """Score candidate fixed physiologies by how well each run's events predict it.

    ``design_for(cbf_curve)`` returns per-run ``(T, C)`` regressors for a CBF
    response sampled every ``microtime_dt`` s. Every candidate is scored on the
    same profiles, so the comparison is paired. See ``design_crossval`` for what
    the free amplitudes can and cannot tell apart.
    """
    from fastfuncstuff.laminar.timeseries import cbf_kernel, design_crossval

    R, U, T, K = columns.profiles.shape
    n_fft = T + int(np.ceil(tail_seconds / spec.TR))
    freq = torch.fft.rfftfreq(n_fft, spec.TR)
    sse = np.empty((len(candidates), U))
    total = np.empty(U)
    for i, p0 in enumerate(
        tqdm(
            candidates, desc="Physiology CV", leave=True, disable=not verbose or len(candidates) < 2
        )
    ):
        cand = replace(spec, N=K, p0=p0)
        full, _ = vascular_transfer(linearized_vasculature(cand, params), freq)
        designs = design_for(cbf_kernel(cand, params, microtime_dt))
        err, tot = design_crossval(
            columns.profiles,
            designs,
            full,
            n_fft,
            nuisance=columns.nuisance,
            sampling=columns.sampling,
            noise_cov=columns.noise_cov,
            device=spec.device,
        )
        sse[i], total[:] = err.numpy(), tot.numpy()
    return PhysiologyCV(list(candidates), sse, total)


@dataclass
class VolumeTdvResult:
    corrected: list[torch.Tensor]  # input units, V x T per run, original means retained
    removed_psc: list[torch.Tensor]  # V x T
    smooth_psc: list[torch.Tensor]  # full-model reconstruction, V x T, smoothing control
    regularization: np.ndarray  # V; NaN for patches not spanning enough depths
    cv_r2: np.ndarray
    noise_gain: np.ndarray
    valid: np.ndarray
    profiles: torch.Tensor  # R x U x T x K, GE only


def devein_volume_runs(
    runs: list[torch.Tensor],
    depth: np.ndarray,
    labels: np.ndarray,
    spec: ModelSpec,
    *,
    params: dict[str, torch.Tensor] | None = None,
    target: str = "local",
    transfer_mode: str = "dynamic",
    strengths: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    pad_seconds: float = 60.0,
    cutoff_hz: float = 0.1,
    polort: int = 2,
    columns: ColumnProfiles | None = None,
    verbose: bool = True,
) -> VolumeTdvResult:
    """Estimate patch transport jointly, subtract from individual voxel data.

    Multi-strength selection assumes matched task timing across equal-length
    runs. SE data must never enter this function or select its hyperparameters.
    Insufficient-depth patches pass through unchanged and are marked invalid.
    ``columns`` reuses profiles already built (with the same ``polort``).
    """
    if pad_seconds < 0:
        raise ValueError("padding must be nonnegative")
    device = spec.device
    R, K = len(runs), spec.K
    if columns is None:
        columns = column_profiles(runs, depth, labels, K, polort, device)
    V, T = runs[0].shape
    members, read, nuisance = columns.members, columns.read, columns.nuisance
    means = [r.mean(-1, keepdim=True) for r in runs]
    starts = [i * T for i in range(R)]
    removed = [torch.zeros_like(r) for r in runs]
    smooth = [torch.zeros_like(r) for r in runs]
    diagnostics = [np.full(V, np.nan, np.float32) for _ in range(3)]
    valid = np.zeros(V, bool)
    profiles = columns.profiles
    pad = min(int(round(pad_seconds / spec.TR)), T - 1)
    system = linearized_vasculature(replace(spec, N=K), params)
    frequencies = torch.fft.rfftfreq(T + 2 * pad, spec.TR)
    if transfer_mode not in ("dynamic", "static"):
        raise ValueError("transfer_mode must be dynamic or static")
    if transfer_mode == "static":
        frequencies = torch.zeros_like(frequencies)
    full, desired = vascular_transfer(system, frequencies, target)
    result = dynamic_inverse(
        profiles,
        full,
        desired,
        sampling=columns.sampling,
        noise_cov=columns.noise_cov,
        strengths=strengths,
        tr=spec.TR,
        pad=pad,
        cutoff_hz=cutoff_hz,
        device=device,
        verbose=verbose,
    )
    chunk = estimate_chunk_size(V, T, K, device, operation="paired_corr", n_trials=4)
    for u in tqdm(
        range(len(members)),
        desc="Map correction to voxels",
        leave=True,
        disable=not verbose or len(members) < 10,
    ):
        ix = members[u]
        rem = result.grid_correction[:, u].to(device)
        fit = result.grid_predicted[:, u].to(device)
        for start in range(0, len(ix), chunk):
            j = ix[start : start + chunk]
            w = torch.as_tensor(read[j], device=device)
            for r in range(R):
                removed[r][j] = (w @ rem[r].T).to(removed[r])
                smooth[r][j] = (w @ fit[r].T).to(smooth[r])
        valid[ix] = True
        for out, val in zip(
            diagnostics,
            (result.regularization[u], result.cv_r2[u], result.noise_gain[u]),
            strict=True,
        ):
            out[ix] = float(val)
    # The correction acts on deviations. Keep the acquired baseline and the
    # user-specified polynomial subspace exactly, including reflected-end effects.
    rem_concat, _ = project_out_nuisance_per_run(
        torch.cat(removed, -1),
        torch.empty(R * T, 0, device=device),
        nuisance,
        starts,
        device=device,
        verbose=False,
    )
    removed = list(rem_concat.split(T, dim=-1))
    corrected = [r - rem * m / 100 for r, rem, m in zip(runs, removed, means, strict=True)]
    return VolumeTdvResult(corrected, removed, smooth, *diagnostics, valid, profiles)
