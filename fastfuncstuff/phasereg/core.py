"""Core phase regression pipeline.

Removes macrovascular BOLD contamination from gradient-echo fMRI by
regressing magnitude on phase.  Oriented vessels (pial/cerebral veins)
produce correlated magnitude and phase changes; randomly oriented
microvasculature produces only magnitude changes.  Subtracting the
phase-predicted magnitude component yields a microvascular-weighted signal.

Pipeline
--------
1. Phase conditioning: temporal unwrap; optional per-volume global field removal.
2. Polynomial (+ nuisance) detrending, per run, of magnitude and phase.
3. Optional task model (TENT/FIR or canonical HRF) — used for the phi estimate
   and, only with slope_on="residual", for the slope fit.
4. Signal-based voxel masking (skip air / skull / low-SNR).
5. Optional Savitzky-Golay filtering of phase (Barry & Gore 2014).
6. Estimate variance ratio phi (FFT band or task-residual variance).
7. Deming (or OLS) regression -> per-voxel slope.
8. Apply correction; return corrected magnitude + diagnostics.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
from tqdm.auto import tqdm

from fastfuncstuff.phasereg.deming import deming_regression, ols_regression
from fastfuncstuff.phasereg.noise import estimate_variance_ratio
from fastfuncstuff.processing.filters import savgol_filter_1d, savgol_filter_explore
from fastfuncstuff.stats.task_coupling import _orthonormal_basis


@dataclass
class PhaseRegResult:
    """Results from phase regression.

    Attributes
    ----------
    magnitude_corrected : Tensor (n_voxels, n_timepoints)
        Macrovascular-suppressed magnitude signal.  Preserves the original
        mean and slow trends — only the phase-correlated macrovascular
        fluctuations are removed.  Feed directly into a GLM.
    macrovascular_component : Tensor (n_voxels, n_timepoints)
        The removed component: slope * (phase - mean(phase)).
    slope : Tensor (n_voxels,)
        Deming/OLS regression slope per voxel.  Large |slope| indicates
        strong macrovascular contribution.
    intercept : Tensor (n_voxels,)
        Regression intercept per voxel.
    phi : Tensor (n_voxels,)
        Variance ratio used for Deming regression.
    r2_phase : Tensor (n_voxels,)
        Fraction of magnitude variance explained by phase (ODR-shrinkage metric,
        matches phaseprep / Stanley 2021).
    r2_naive : Tensor (n_voxels,) or None
        Naive R²: ``1 - SS_res(observed) / SS_tot`` without ODR shrinkage.
        Highlights voxels where phase regression had the largest raw effect
        (useful for QC/visualisation).  None unless r2_mode='naive' or 'both'.
    voxel_mask : Tensor (n_voxels,), bool
        True for voxels where regression was attempted.
    mag_detrended : Tensor (n_voxels, n_timepoints) or None
        Magnitude after polynomial (+ nuisance) detrending.  None unless
        save_intermediates=True.
    pha_detrended : Tensor (n_voxels, n_timepoints) or None
        Phase after polynomial (+ nuisance) detrending.  None unless
        save_intermediates=True.
    pha_detrended_filt : Tensor (n_voxels, n_timepoints) or None
        Phase after detrending and optional SGF filtering (the series used for
        slope estimation and correction).  None unless save_intermediates=True.
    mag_residual : Tensor (n_voxels, n_timepoints) or None
        Magnitude after polynomial + task removal (used for slope fit).
        Identical to mag_detrended when task_removal='none'.  None unless
        save_intermediates=True.
    pha_residual_filt : Tensor (n_voxels, n_timepoints) or None
        Phase after polynomial + task removal + SGF (used for phi estimation
        and slope fit).  None unless save_intermediates=True.
    sgf_window_map, sgf_order_map : Tensor (n_voxels,) or None
        Per-voxel SGF window length and polynomial order chosen by the
        data-driven search.  Only populated when phase_filter='explore'.
    spr_donor : Tensor (n_voxels,) or None
        Index of the neighbouring voxel whose phase was used as the regressor
        (== own index where the voxel's own phase won, i.e. standard PR).
        Only populated when spr=True.
    spr_donor_corr : Tensor (n_voxels,) or None
        Signed corr(magnitude_i, phase_donor) at the selected donor — the
        z-scored sPR slope of Vu & Gallant 2015 Eq. 12.  Only when spr=True.
    spr_donor_offset : Tensor (n_voxels,) or None
        Euclidean distance in voxels from the target to its donor (0 = self).
        Handy as a QC map: a rim of non-zero offsets tracing a vessel is sPR
        doing exactly what it is meant to do.  Only when spr=True.
    coupling_r : Tensor (n_voxels,) or None
        Pearson correlation between the magnitude and phase series the slope was
        fit on.  Only when vein_mask=True.
    vein_p : Tensor (n_voxels,) or None
        Two-sided p-value that ``coupling_r`` exceeds chance, Sidak-corrected for
        the sPR donor argmax.  Only when vein_mask=True.
    vein_exclude : Tensor (n_voxels,), bool, or None
        True where the voxel is flagged as vessel-dominated at the requested FDR
        q — i.e. the voxels to EXCLUDE from a layer profile.  Only when
        vein_mask=True.
    """

    magnitude_corrected: torch.Tensor
    macrovascular_component: torch.Tensor
    slope: torch.Tensor
    intercept: torch.Tensor
    phi: torch.Tensor
    r2_phase: torch.Tensor
    r2_naive: torch.Tensor | None
    voxel_mask: torch.Tensor
    mag_detrended: torch.Tensor | None = None
    pha_detrended: torch.Tensor | None = None
    pha_detrended_filt: torch.Tensor | None = None
    mag_residual: torch.Tensor | None = None
    pha_residual_filt: torch.Tensor | None = None
    sgf_window_map: torch.Tensor | None = None
    sgf_order_map: torch.Tensor | None = None
    spr_donor: torch.Tensor | None = None
    spr_donor_corr: torch.Tensor | None = None
    spr_donor_offset: torch.Tensor | None = None
    coupling_r: torch.Tensor | None = None
    vein_p: torch.Tensor | None = None
    vein_exclude: torch.Tensor | None = None


def _project_out(data: torch.Tensor, design: torch.Tensor) -> torch.Tensor:
    """Project design out of data via QR decomposition.

    Parameters
    ----------
    data : (n_timepoints, n_voxels)
    design : (n_timepoints, n_regressors)

    Returns
    -------
    residual : (n_timepoints, n_voxels)
    """
    Q = _orthonormal_basis(design)
    return data - Q @ (Q.T @ data) if Q.shape[1] else data


def _voxel_coords(
    volume_shape: tuple[int, int, int],
    mask_flat: torch.Tensor | None,
) -> torch.Tensor:
    """(x, y, z) grid coordinates of each voxel in the masked data array."""
    nx, ny, nz = volume_shape
    grid = torch.stack(
        torch.meshgrid(torch.arange(nx), torch.arange(ny), torch.arange(nz), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    if mask_flat is None:
        return grid
    return grid[torch.as_tensor(mask_flat).reshape(-1).bool()]


def _build_task_design_tent(
    onsets_per_condition: list[list],
    tr: float,
    n_timepoints_per_run: list[int],
    window: float,
    device: torch.device,
) -> list[torch.Tensor]:
    """Build per-run TENT design matrices for task removal."""
    from fastfuncstuff.design.builder import build_per_run_task_designs

    n_conditions = len(onsets_per_condition)
    return build_per_run_task_designs(
        [[np.asarray(o, dtype=np.float64) for o in runs] for runs in onsets_per_condition],
        list(n_timepoints_per_run),
        tr,
        basis="TENT",
        fir_window_s=[(0.0, float(window))] * n_conditions,
        device=device,
    ).per_run


def _build_task_design_canonical(
    onsets_per_condition: list[list],
    tr: float,
    n_timepoints_per_run: list[int],
    device: torch.device,
    durations: list[float] | None = None,
) -> list[torch.Tensor]:
    """Per-run SPMG1 designs, each event a duration boxcar convolved at 0.1 s.

    Bug of record: this used to place an impulse at each onset, so a 15 s block
    was modelled as a 0 s event and the task residual (hence phi) kept most of
    the block response."""
    from fastfuncstuff.design.hrf import get_spmg1_hrf
    from fastfuncstuff.design.matrices import build_event_design_microtime

    n_conditions = len(onsets_per_condition)
    durs = list(durations) if durations is not None else [0.0] * n_conditions
    design = build_event_design_microtime(
        [[np.asarray(o, dtype=np.float64) for o in runs] for runs in onsets_per_condition],
        durs,
        get_spmg1_hrf(0.1, device=device),
        list(n_timepoints_per_run),
        tr=tr,
        microtime_dt=0.1,
        device=device,
    )
    assert isinstance(design, torch.Tensor)
    return list(design.to(torch.float32).split(list(n_timepoints_per_run), dim=0))


def _apply_sgf(
    phase_detrended: torch.Tensor,
    sgf_mode: str,
    sgf_window: int,
    sgf_order: int,
    device: torch.device,
    chunk_size: int = 50000,
    verbose: bool = False,
    mag_detrended: torch.Tensor | None = None,
    sgf_window_max: int | None = None,
    sgf_order_max: int | None = None,
    sgf_step: int = 4,
    run_lengths: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Apply Savitzky-Golay filtering to phase on GPU, chunked.

    Parameters
    ----------
    phase_detrended : Tensor (n_timepoints, n_voxels)
    sgf_mode : {"none", "sgf", "explore"}
    sgf_window, sgf_order : int
        Parameters for fixed-window ("sgf") mode.
    device : torch.device
    chunk_size : int
    verbose : bool
    sgf_window_max, sgf_order_max, sgf_step : int or None
        Explore-mode grid bounds (max window, max order, window step).
        None = n_timepoints // 2, max_window // 4, and 4 (Barry & Gore 2014).
    run_lengths : list of int, optional
        Runs concatenated along time; each is filtered on its own.

    Returns
    -------
    filtered : Tensor (n_timepoints, n_voxels), on the input's device
    window_map, order_map : Tensor (n_voxels,) or None
        Per-voxel chosen (window, order) in "explore" mode; None otherwise.
    """
    if sgf_mode == "none":
        return phase_detrended, None, None

    n_tp = phase_detrended.shape[0]
    n_vox = phase_detrended.shape[1]
    result = phase_detrended.clone()

    if sgf_mode == "sgf":
        if verbose:
            print(f"  SGF: window={sgf_window}, order={sgf_order}")
        for start in range(0, n_vox, chunk_size):
            end = min(start + chunk_size, n_vox)
            chunk = phase_detrended[:, start:end].to(device)
            filtered = savgol_filter_1d(chunk.T, sgf_window, sgf_order, run_lengths).T
            result[:, start:end] = filtered.to(result.device)
        return result, None, None

    if sgf_mode == "explore":
        if verbose:
            print("  SGF: data-driven per-voxel parameter search (Barry & Gore 2014)...")
        window_map = torch.zeros(n_vox, dtype=torch.int64, device=result.device)
        order_map = torch.zeros(n_vox, dtype=torch.int64, device=result.device)
        for start in range(0, n_vox, chunk_size):
            end = min(start + chunk_size, n_vox)
            chunk = phase_detrended[:, start:end].T.to(device)

            if mag_detrended is not None:
                mag_chunk = mag_detrended[:, start:end].T.to(device)
                mag_c = mag_chunk - mag_chunk.mean(dim=-1, keepdim=True)
                mag_ss = (mag_c**2).sum(dim=-1).clamp(min=1e-30)

                # mag_c/mag_ss bound as defaults: the closure is consumed within
                # this iteration (passed to savgol_filter_explore below), and the
                # binding makes that capture explicit rather than late.
                def _metric(filt: torch.Tensor, mag_c=mag_c, mag_ss=mag_ss) -> torch.Tensor:
                    f_c = filt - filt.mean(dim=-1, keepdim=True)
                    cross = (f_c * mag_c).sum(dim=-1)
                    f_ss = (f_c**2).sum(dim=-1).clamp(min=1e-30)
                    return (cross / torch.sqrt(f_ss * mag_ss)).abs()
            else:

                def _metric(filt: torch.Tensor) -> torch.Tensor:
                    return -filt.var(dim=-1)

            filtered, win_c, ord_c = savgol_filter_explore(
                chunk,
                n_timepoints=n_tp,
                device=device,
                metric_fn=_metric,
                max_window=sgf_window_max,
                max_order=sgf_order_max,
                step=sgf_step,
                return_params=True,
                segment_lengths=run_lengths,
            )
            result[:, start:end] = filtered.T.to(result.device)
            window_map[start:end] = win_c.to(result.device)
            order_map[start:end] = ord_c.to(result.device)
        return result, window_map, order_map

    raise ValueError(
        f"Unknown phase_filter/sgf_mode: {sgf_mode!r}. Use 'none', 'sgf', or 'explore'."
    )


def phase_regress(
    magnitude: list[torch.Tensor] | torch.Tensor,
    phase: list[torch.Tensor] | torch.Tensor,
    tr: float,
    task_removal: str = "none",
    onsets_per_condition: list[list] | None = None,
    nuisance_per_run: list[torch.Tensor] | None = None,
    max_poly_degree: int = 3,
    durations_per_condition: list[float] | None = None,
    phi: float | torch.Tensor | None = None,
    phi_method: str = "fft",
    phi_freq_range: tuple[float, float | None] = (0.15, None),
    regression: str = "deming",
    max_slope_ratio: float | None = None,
    slope_on: str = "data",
    tent_window: float = 20.0,
    phase_filter: str = "none",
    sgf_window: int | None = None,
    sgf_order: int = 3,
    sgf_window_max: int | None = None,
    sgf_order_max: int | None = None,
    sgf_step: int = 4,
    signal_thresh: float = 0.03,
    phase_unwrap: bool = True,
    phase_global_order: int = -1,
    keep_drift: bool = False,
    shrink_mode: str = "slope",
    vein_mask: bool = False,
    vein_fdr_q: float = 0.05,
    spr: bool = False,
    spr_connectivity: int = 6,
    spr_select_run: int | None = None,
    spr_select_on: str = "data",
    spr_allow: torch.Tensor | None = None,
    spr_regression: str | None = "ols",
    phi_calib: torch.Tensor | None = None,
    volume_shape: tuple[int, int, int] | None = None,
    mask_flat: torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    chunk_size: int | None = None,
    verbose: bool = False,
    r2_mode: str = "odr",
    save_intermediates: bool = False,
) -> PhaseRegResult:
    """Phase regression to suppress macrovascular BOLD contamination.

    Parameters
    ----------
    magnitude : list of Tensor or Tensor
        Magnitude time series per run, each (n_voxels, n_timepoints),
        or a single tensor for single-run data.
    phase : list of Tensor or Tensor
        Phase time series (radians, unwrapped), same shapes as magnitude.
    tr : float
        Repetition time in seconds.
    task_removal : {"none", "tent", "canonical", "fit[:N]"}
        Task model. "fit:N" = canonical plus N time derivatives per condition. Feeds the phi estimate (``phi_method="residual"``) and,
        only with ``slope_on="residual"``, the slope fit.
    onsets_per_condition : list of list of ndarray, optional
        Onset times structured as [condition][run] -> ndarray.
    durations_per_condition : list of float, optional
        Event duration (s) per condition for ``task_removal="canonical"``;
        None models every event as an impulse.
    nuisance_per_run : list of Tensor, optional
        Additional per-run nuisance regressors (e.g. motion parameters),
        each (n_timepoints, n_regressors).
    max_poly_degree : int
        Polynomial drift order for detrending.  Set to -1 to skip.
    phi : float, Tensor, or None
        Fixed variance ratio for Deming regression.  If None, estimated
        automatically.
    phi_method : {"fft", "residual", "thermal"}
        How to estimate phi when phi=None. "thermal": phi = c * mean(M)^2 per
        voxel, the thermal-noise physics (magnitude SD sigma, phase SD sigma/M)
        with one global factor c for the phase noise in excess of thermal,
        measured as the median of var(mag res)/var(phase res)/M^2 over
        ``phi_calib`` voxels. Not circular: c comes from voxels without vein
        coupling, never from the voxel being fit.
    phi_calib : Tensor (n_voxels,) bool or None
        Calibration voxels for phi_method="thermal": ones where magnitude and
        phase should not share a vein signal (deep white matter away from
        veins). None: every fitted voxel, a weaker estimate.
    phi_freq_range : tuple (lo_hz, hi_hz)
        Frequency range for FFT-based noise estimation. Default 0.15 Hz to
        Nyquist, as ffs_phasereg -freq_range (Stanley 2021, phaseprep).
    regression : {"deming", "ols"}
        Regression method.
    max_slope_ratio : float or None
        Cap |slope| at this multiple of the voxel's OLS slope (Deming only).
        Default None: no cap, as in the literature and phaseprep; the cap is
        ours and kept for comparison. At 2x OLS the residual variance of
        ``mag - A*phase`` is back to the input's, so a binding cap means no net
        correction. Uncapped, the Deming slope diverges (diff/SXY as SXY -> 0)
        wherever magnitude and phase are uncorrelated but SYY > phi*SXX: the
        cure is a phi that is right, not a bound on A.
    slope_on : {"data", "residual"}
        Series the slope is fit on. "data" (default; Menon 2002, Stanley
        2021, Knudsen 2023): detrended, task preserved, because a vein's
        task-locked magnitude/phase covariance is what pins its slope.
        "residual": after task removal, which guards against over-suppression
        at the price of a far noisier slope.
    tent_window : float
        TENT window duration in seconds.
    phase_filter : {"none", "sgf", "explore"}
        Phase filtering before regression:
        - "none": no filtering.
        - "sgf": Savitzky-Golay with sgf_window/sgf_order.
        - "explore": per-voxel data-driven SGF search (Barry & Gore 2014).
    sgf_window : int or None
        SGF window length (odd) for phase_filter="sgf". None = round(20 s/TR),
        made odd, min 5.
    sgf_order : int
        SGF polynomial order for phase_filter="sgf".
    sgf_window_max, sgf_order_max, sgf_step : int or None
        Grid bounds for phase_filter="explore" (max window, max order, and
        window step). None = n_timepoints // 2, max_window // 4, and 4
        (Barry & Gore 2014's grid).
    signal_thresh : float
        Minimum mean-signal fraction (of the brain maximum) for a voxel to be
        fit (default 0.03, phaseprep). Voxels below it get slope 0, so
        corrected = input (or its mean without ``keep_drift``). 0 fits all.
    phase_unwrap : bool
        Unwrap each run's phase along time about the voxel's circular mean
        before anything else (``conditioning.temporal_unwrap``). A no-op up to
        a constant on already-unwrapped phase; required for wrapped phase.
    phase_global_order : int
        -1 (default) off. 0 removes a robust per-volume brain-wide phase
        offset; N >= 1 a per-volume spatial Legendre polynomial of degree N
        (needs ``volume_shape``). A fallback for data that did not go through
        ``ffs_phasedrift``, which removes the same field term per slice in
        native space and far more of it.
    keep_drift : bool
        False (default, phaseprep): the corrected magnitude is re-centred on
        its mean with the ``max_poly_degree`` drift dropped. True: corrected
        = input - macro, drift kept for a downstream GLM.
    shrink_mode : {"slope", "odr", "none"}
        ODR's factor s = 1/(1 + A²/φ) is ~1 where the slope is believable and
        ~0 where it is an ill-conditioned fit to phase noise (A² ≫ φ).
        "slope" (default): subtract s·A·phase and leave the residual intact,
        so an unreliable slope is damped toward zero (s·A → φ/A) instead of
        exploding; no cap, no deletion. On ds003427 (2026-10-14): 0 explosions,
        an even 7–15% suppression. "odr" (phaseprep/Stanley parity): the
        published output, the residual scaled by s, which flattens every voxel
        whose slope is unreliable, task response included (~73% of the
        response removed, veins and tissue alike, on ds003427). "none": subtract
        A·phase in full; with weakly coupled voxels the slope explodes.
    vein_mask : bool
        Also emit a vein *exclusion* mask: voxels whose magnitude covaries with
        phase beyond chance (oriented vessels). For layer workflows that drop
        such voxels. Does not alter the corrected series.
    vein_fdr_q : float
        Benjamini-Hochberg q for the mask.  The correlation is Sidak-corrected
        for the sPR donor argmax before FDR.
    spr : bool
        Source-localized phase regression (Vu & Gallant 2015): regress each
        voxel's magnitude on the phase of the neighbour (or itself) that
        tracks it best, for veins whose own phase is ~0 (cancellation inside
        a voxel-sized vein, or the magic angle). Requires ``volume_shape``.
        Vu used OLS, which suppresses less; for surviving veins keep
        ``regression="deming"``.
    spr_connectivity : {6, 18, 26}
        Donor search neighbourhood.  6 (face-adjacent, Vu & Gallant's choice)
        is the default; larger neighbourhoods search harder but increase the
        selection bias described under ``spr_select_run``.
    spr_select_run : int or None
        0-based run used *only* to select donors, then applied to every run
        (Vu & Gallant held out a run). None (default) selects on all runs:
        in-sample, biased toward over-suppression.
    spr_select_on : {"data", "residual"}
        Series the donor argmax is taken on. "data" (Vu & Gallant): the
        detrended magnitude and phase, task included. "residual": both with the
        task model projected out (needs ``task_removal``), so a neighbour wins by
        sharing the voxel's spontaneous and physiological coupling rather than
        merely carrying the task, which on task-locked data makes the argmax
        pick task-responsive phase anywhere and suppress tissue response.
    spr_allow : Tensor (n_voxels,) bool or None
        Voxels allowed to borrow a neighbour's phase, e.g. an independent vein
        mask (GRE R2* or SWI). Elsewhere the voxel keeps its own phase (plain PR).
        On ds003427 the coupling statistic could not tell a vein from the tissue
        it drains, so where to borrow has to come from outside the coupling.
    spr_regression : {"ols", "deming"} or None
        Slope rule inside ``spr_allow`` (the plausible veins). "ols" (default):
        least squares on the voxel's own or borrowed phase, no ODR damping --
        Vu & Gallant's choice for sPR, and on ds003427 the only literature rule
        that reduced the pial GE/SE excess without the invented 2x cap (2026-10-14).
        Outside the mask ``regression`` and ``shrink_mode`` apply as usual, so the
        correction is conservative where there is no vein evidence. None: the
        global rule everywhere.
    volume_shape : (nx, ny, nz) or None
        Shape of the 3-D grid the voxel axis was flattened from.  Required for
        ``spr``; ignored otherwise.
    mask_flat : Tensor (nx*ny*nz,) bool or None
        Which volume voxels are present in the data array, so sPR can rebuild
        spatial adjacency after masking.  None means the data are unmasked.
    device : str or torch.device
        PyTorch device for computation.
    chunk_size : int or None
        Voxels per device chunk. None sizes it from free device memory
        (memory.bytes_per_voxel_phasereg).
    verbose : bool
        Print progress information.
    r2_mode : {"odr", "naive", "both"}
        "odr" (default): the ODR-shrinkage R² of phaseprep/Stanley. "naive":
        ``1 - SS_res / SS_tot`` without shrinkage, a where-did-it-act map.
    save_intermediates : bool
        If True, populate the intermediate-data fields of PhaseRegResult
        (mag_detrended, pha_detrended, pha_detrended_filt, mag_residual,
        pha_residual_filt).  Off by default to avoid storing extra 4D arrays.

    Returns
    -------
    PhaseRegResult
        Corrected magnitude and diagnostic maps.
    """
    device = torch.device(device)

    if isinstance(magnitude, torch.Tensor):
        magnitude = [magnitude]
        phase = [phase]

    n_runs = len(magnitude)
    n_tp_per_run = [m.shape[1] for m in magnitude]
    n_tp = sum(n_tp_per_run)
    n_voxels = magnitude[0].shape[0]
    run_bounds = [(int(a), int(b)) for a, b in itertools.pairwise(np.cumsum([0, *n_tp_per_run]))]

    if verbose:
        print(f"Phase regression: {n_runs} run(s), {n_voxels:,} voxels, {n_tp} total timepoints")
        print(f"  Task removal: {task_removal}")
        cap = f", |A| capped at {max_slope_ratio:g}x OLS" if max_slope_ratio else ""
        print(f"  Regression: {regression}{cap if regression == 'deming' else ''}")
        print(f"  Phi method: {phi_method}")
        print(f"  Phase filter: {phase_filter}")
        print(f"  Slope fit on: {slope_on}; residual shrink: {shrink_mode}")
        print(f"  Drift: {'kept in the output' if keep_drift else 'removed from the output'}")
        if spr:
            where = "inside -spr_mask" if spr_allow is not None else "everywhere (no mask)"
            print(f"  sPR: borrow a neighbour's phase {where}")
        else:
            print("  sPR: off (each voxel uses its own phase)")

        sample = magnitude[0][0]
        print(f"  Magnitude scale: mean={sample.mean().item():.1f}, std={sample.std().item():.1f}")
        sample_p = phase[0][0]
        print(f"  Phase scale: mean={sample_p.mean().item():.4f}, std={sample_p.std().item():.4f}")

    if regression not in ("deming", "ols"):
        raise ValueError(f"Unknown regression: {regression!r}")
    if slope_on not in ("data", "residual"):
        raise ValueError(f"Unknown slope_on: {slope_on!r}. Use 'data' or 'residual'.")
    if spr_regression not in (None, "ols", "deming"):
        raise ValueError(f"Unknown spr_regression: {spr_regression!r}. Use 'ols' or 'deming'.")
    if shrink_mode not in ("slope", "odr", "none"):
        raise ValueError(f"Unknown shrink_mode: {shrink_mode!r}. Use 'slope', 'odr' or 'none'.")

    # ── 1. Keep magnitude in raw scanner units ───────────────────────────
    # phaseprep and Stanley fit ODR on raw-unit magnitude. With raw mag and
    # radian phase, var_mag/var_phase is huge → phi clamps to its max →
    # Deming → OLS, which is the regime phaseprep operates in. Earlier
    # versions normalised mag by its per-voxel mean; that put phi in the
    # tens-of-thousandths range, which sent Deming toward orthogonal
    # regression and produced exploding slopes / negative R² that the
    # downstream `r2 < 0` filter zeroed across the brain.
    orig_mean = torch.stack([m.mean(dim=1) for m in magnitude]).mean(dim=0).cpu()
    vox_mask = orig_mean > signal_thresh * orig_mean.max()

    # ── 1b. Global field removal needs every voxel of a volume at once ───
    # Everything else below is per voxel and runs in device-sized chunks.
    unwrap_in_chunks = phase_unwrap
    if phase_global_order >= 0:
        from fastfuncstuff.phasereg.conditioning import remove_global_phase, temporal_unwrap

        pha_all = torch.cat([p.T for p in phase], dim=0).float()
        if phase_unwrap:
            pha_all = temporal_unwrap(pha_all, n_tp_per_run)
        coords = None
        if phase_global_order >= 1:
            if volume_shape is None:
                raise ValueError("phase_global_order >= 1 needs volume_shape")
            coords = _voxel_coords(volume_shape, mask_flat)
        pha_all, g = remove_global_phase(
            pha_all, phase_global_order, vox_mask, coords, n_tp_per_run
        )
        if verbose:
            print(
                f"  Global phase (order {phase_global_order}) removed: "
                f"per-volume term std={g.std().item():.4f} rad"
            )
        phase = [p.T for p in torch.split(pha_all, n_tp_per_run, dim=0)]
        unwrap_in_chunks = False
        del pha_all

    # ── 2. Per-run polynomial + nuisance, and task, projection bases ─────
    from fastfuncstuff.design.builder import legendre_polynomials

    poly_per_run: list[torch.Tensor] = []
    for run_idx, n_tp_run in enumerate(n_tp_per_run):
        parts = []
        if max_poly_degree >= 0:
            poly = legendre_polynomials(n_tp_run, max_poly_degree)
            parts.append(torch.tensor(poly, dtype=torch.float32, device=device))
        if nuisance_per_run is not None and run_idx < len(nuisance_per_run):
            parts.append(nuisance_per_run[run_idx].to(device))
        if parts:
            poly_per_run.append(torch.cat(parts, dim=1))
        else:
            poly_per_run.append(torch.zeros(n_tp_run, 0, device=device))

    task_per_run: list[torch.Tensor | None] = [None] * n_runs
    if task_removal == "tent":
        if onsets_per_condition is None:
            raise ValueError("task_removal='tent' requires onsets_per_condition")
        task_per_run = _build_task_design_tent(
            onsets_per_condition,
            tr,
            n_tp_per_run,
            tent_window,
            device,
        )
    elif task_removal == "canonical":
        if onsets_per_condition is None:
            raise ValueError("task_removal='canonical' requires onsets_per_condition")
        task_per_run = _build_task_design_canonical(
            onsets_per_condition,
            tr,
            n_tp_per_run,
            device,
            durations_per_condition,
        )
    elif task_removal == "fit" or task_removal.startswith("fit:"):
        # locomoco's -detask fit:N model: the canonical design plus N time
        # derivatives per condition (one DoF each), via the shared builder. phi is a
        # ratio of two residual variances from the SAME design, per voxel, so its
        # degrees of freedom cancel; a FIR/TENT model would spend many more.
        if onsets_per_condition is None:
            raise ValueError(f"task_removal={task_removal!r} requires onsets_per_condition")
        from fastfuncstuff.stats.task_coupling import design_fit_basis

        n_deriv = int(task_removal.split(":", 1)[1]) if ":" in task_removal else 0
        canon = _build_task_design_canonical(
            onsets_per_condition, tr, n_tp_per_run, device, durations_per_condition
        )
        task_per_run = [
            None
            if t is None
            else design_fit_basis(t, max(max_poly_degree, 0), n_deriv=n_deriv, device=device).to(
                torch.float32
            )
            for t in canon
        ]
    elif task_removal != "none":
        raise ValueError(
            f"Unknown task_removal: {task_removal!r}. Use 'none', 'tent', 'canonical' or 'fit[:N]'."
        )
    has_task = any(t is not None for t in task_per_run)

    def _basis(design: torch.Tensor) -> torch.Tensor | None:
        # Rank-revealing: QR keeps an arbitrary orthonormal direction for an
        # all-zero column (a condition with no events in this run) or a
        # duplicate, and projecting that out removed real signal.
        q = _orthonormal_basis(design)
        return q if q.shape[1] else None

    q_drift = [_basis(p) for p in poly_per_run]
    q_full = [
        q_drift[r]
        if t is None
        else _basis(torch.cat([poly_per_run[r], t], dim=1) if poly_per_run[r].shape[1] else t)
        for r, t in enumerate(task_per_run)
    ]

    def _project_runs(data: torch.Tensor, bases: list[torch.Tensor | None]) -> torch.Tensor:
        out = data.clone()
        for (a, b), q in zip(run_bounds, bases, strict=True):
            if q is not None:
                out[a:b] -= q @ (q.T @ data[a:b])
        return out

    n_good = int(vox_mask.sum().item())
    if verbose:
        n_bad = n_voxels - n_good
        print(
            f"  Voxel mask: {n_good:,} / {n_voxels:,} voxels above "
            f"{signal_thresh:.0%} signal threshold ({n_bad:,} excluded)"
        )
        if signal_thresh == 0:
            print(
                "  WARNING: signal_thresh=0 → no signal-based gating. "
                "Output is safe (ODR shrinkage caps all voxels), but the "
                "R² map will show bright air/noise voxels."
            )

    if sgf_window is None:
        sgf_window = round(20.0 / tr)
        if sgf_window % 2 == 0:
            sgf_window += 1
        sgf_window = max(sgf_window, 5)
    if verbose and phase_filter == "sgf":
        print(f"  SGF: window={sgf_window}, order={sgf_order}")
    elif verbose and phase_filter == "explore":
        print("  SGF: data-driven per-voxel parameter search (Barry & Gore 2014)...")

    if chunk_size is None:
        from fastfuncstuff.memory import estimate_chunk_size

        chunk_size = estimate_chunk_size(
            n_voxels,
            n_tp,
            0,
            device,
            operation="phasereg",
            n_trials=2 if phase_filter == "explore" else 1,
        )
    chunks = [(lo, min(lo + chunk_size, n_voxels)) for lo in range(0, n_voxels, chunk_size)]

    def _rows(
        arrays: list[torch.Tensor], lo: int, hi: int, idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Voxel rows [lo, hi) (or ``idx``) of per-run (voxels, time) arrays → (time,
        voxels) on device."""
        sel = slice(lo, hi) if idx is None else idx
        block = torch.cat([a[sel].to(device, torch.float32) for a in arrays], dim=1)
        return block.T.contiguous()

    def _host(x: torch.Tensor) -> torch.Tensor:
        """(time, voxels) on device → (voxels, time) on the host, transposed device-side."""
        return x.T.contiguous().cpu()

    sgf_window_map = torch.zeros(n_voxels, dtype=torch.int64) if phase_filter == "explore" else None
    sgf_order_map = torch.zeros(n_voxels, dtype=torch.int64) if phase_filter == "explore" else None

    # ── 3–5. Unwrap, detrend (and task-residualise), SGF — per chunk ─────
    # The residual phase is filtered the same way as the detrended phase so
    # the slope is estimated and applied on consistently filtered series.
    def _prepare(lo: int, hi: int, idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        pha = _rows(phase, lo, hi, idx)
        if unwrap_in_chunks:
            from fastfuncstuff.phasereg.conditioning import temporal_unwrap

            pha = temporal_unwrap(pha, n_tp_per_run)
        mag_raw = _rows(magnitude, lo, hi, idx)
        mag_dt = _project_runs(mag_raw, q_drift)
        pha_dt = _project_runs(pha, q_drift)
        sgf_kw = dict(
            device=device,
            chunk_size=hi - lo if idx is None else int(idx.numel()),
            sgf_window_max=sgf_window_max,
            sgf_order_max=sgf_order_max,
            sgf_step=sgf_step,
            run_lengths=n_tp_per_run,
        )
        pha_filt, win, order = _apply_sgf(
            pha_dt, phase_filter, sgf_window, sgf_order, mag_detrended=mag_dt, **sgf_kw
        )
        if idx is None and sgf_window_map is not None and sgf_order_map is not None:
            assert win is not None and order is not None  # explore mode returns both
            sgf_window_map[lo:hi] = win.cpu()
            sgf_order_map[lo:hi] = order.cpu()
        out = {"mag_dt": mag_dt, "pha_dt": pha_dt, "pha_filt": pha_filt}
        if has_task:
            mag_res = _project_runs(mag_raw, q_full)
            pha_res = _project_runs(pha, q_full)
            out["mag_res"] = mag_res
            out["pha_res_filt"] = _apply_sgf(
                pha_res, phase_filter, sgf_window, sgf_order, mag_detrended=mag_res, **sgf_kw
            )[0]
        return out

    # sPR has to see every voxel's prepared series before any voxel is fit
    # (donors cross chunk boundaries), and the intermediates are returned
    # whole; otherwise each chunk goes straight from preparation to the fit.
    stored: dict[str, torch.Tensor] = {}
    if spr or save_intermediates:
        keys = ["mag_dt", "pha_filt"]
        if has_task:
            keys += ["mag_res", "pha_res_filt"]
        if save_intermediates:
            keys.append("pha_dt")
        stored = {k: torch.empty((n_voxels, n_tp), dtype=torch.float32) for k in keys}
        for lo, hi in tqdm(chunks, desc="Phase prep", disable=not verbose or len(chunks) < 2):
            prepared = _prepare(lo, hi)
            for k in keys:
                stored[k][lo:hi] = _host(prepared[k])
            del prepared

    # ── 5b. sPR: swap in each voxel's best neighbouring phase regressor ───
    # By default (spr_select_on="data") donor selection runs on the DETRENDED series, not the task-residualised
    # one: the signal that identifies a vein is precisely the task-related
    # phase change, so residualising the task out first would erase the thing
    # being searched for. The slope is still fit on whatever series
    # `task_removal` dictates.
    spr_donor: torch.Tensor | None = None
    spr_donor_corr: torch.Tensor | None = None
    spr_donor_offset: torch.Tensor | None = None
    if spr:
        from fastfuncstuff.phasereg.spr import build_neighbor_index, select_phase_donor

        if volume_shape is None:
            raise ValueError("spr=True requires volume_shape=(nx, ny, nz)")

        neighbors = build_neighbor_index(volume_shape, mask_flat, spr_connectivity)
        if neighbors.shape[0] != n_voxels:
            raise ValueError(
                f"sPR neighbour index has {neighbors.shape[0]} voxels but data has "
                f"{n_voxels}; volume_shape/mask_flat do not match the data array"
            )

        lo_t, hi_t = 0, n_tp
        if spr_select_run is not None:
            if not 0 <= spr_select_run < n_runs:
                raise ValueError(
                    f"spr_select_run={spr_select_run} out of range for {n_runs} run(s)"
                )
            lo_t, hi_t = run_bounds[spr_select_run]

        if verbose:
            where = f"run {spr_select_run}" if spr_select_run is not None else "all runs"
            print(f"  sPR: selecting phase donors ({spr_connectivity}-connected, on {where})...")

        # Time-major views of the voxel-major stores, so each donor lookup is
        # a contiguous row gather.
        if spr_select_on not in ("data", "residual"):
            raise ValueError(f"spr_select_on must be 'data' or 'residual', got {spr_select_on!r}")
        if spr_select_on == "residual" and not has_task:
            raise ValueError("spr_select_on='residual' needs a task model (task_removal)")
        sel_mag, sel_pha = (
            ("mag_res", "pha_res_filt") if spr_select_on == "residual" else ("mag_dt", "pha_filt")
        )
        spr_donor, spr_donor_corr = select_phase_donor(
            stored[sel_mag][:, lo_t:hi_t].T,
            stored[sel_pha][:, lo_t:hi_t].T,
            neighbors,
            device=device,
            chunk_size=chunk_size,
        )

        if spr_allow is not None:
            if spr_allow.shape[0] != n_voxels:
                raise ValueError(
                    f"spr_allow has {spr_allow.shape[0]} voxels but data has {n_voxels}"
                )
            self_idx = torch.arange(n_voxels)
            spr_donor = torch.where(spr_allow.cpu().bool(), spr_donor, self_idx)

        # Substituting the donor series here means phi, the slope fit, and the
        # applied correction all use the donor phase with no further changes.
        stored["pha_filt"] = stored["pha_filt"][spr_donor]
        if has_task:
            stored["pha_res_filt"] = stored["pha_res_filt"][spr_donor]

        borrowed = spr_donor != torch.arange(n_voxels)
        coords = _voxel_coords(volume_shape, mask_flat)
        spr_donor_offset = (coords[spr_donor] - coords).float().norm(dim=1)

        if verbose:
            n_borrow = int((borrowed & vox_mask).sum().item())
            print(
                f"  sPR: {n_borrow:,} / {n_good:,} masked voxels "
                f"({100.0 * n_borrow / max(n_good, 1):.1f}%) borrowed a neighbour's phase"
            )
            gain = spr_donor_corr[vox_mask].abs()
            print(f"  sPR: |corr| at chosen donor: median={gain.median().item():.3f}")

    # ── 6–10. phi, slope, R², correction — per chunk ─────────────────────
    if phi is None and verbose:
        print("  Estimating variance ratio phi...")
    phi_all = torch.zeros(n_voxels)
    slope_all = torch.zeros(n_voxels)
    intercept_all = torch.zeros(n_voxels)
    r2 = torch.zeros(n_voxels)
    r2_naive = torch.zeros(n_voxels) if r2_mode in ("naive", "both") else None
    inflation_all = torch.zeros(n_voxels)
    coupling_r = torch.zeros(n_voxels) if vein_mask else None
    n_unstable = 0
    n_capped = 0
    mag_corrected = torch.empty((n_voxels, n_tp), dtype=torch.float32)
    macro_out = torch.empty((n_voxels, n_tp), dtype=torch.float32)

    # Phase variance floor: in voxels where the phase barely varies
    # (air, skull, or genuinely flat phase), Var(phase) → 0 makes the
    # slope = Cov/Var explode to ±billions.  Floor at 1e-6 radians²
    # (std ≈ 0.001 rad ≈ 0.06°) — well below any genuine BOLD phase
    # change, catches dead voxels.
    pha_var_floor = 1e-6

    phi_thermal_c: float | None = None
    if phi is None and phi_method == "thermal":
        cal = vox_mask if phi_calib is None else (phi_calib.to(vox_mask.device) & vox_mask)
        idx = torch.nonzero(cal).squeeze(1)
        if idx.numel() < 100:
            raise ValueError(f"phi_method='thermal': only {int(idx.numel())} calibration voxels")
        if idx.numel() > 20000:  # a median needs no more; keep the pre-pass cheap
            g = torch.Generator().manual_seed(0)
            idx = idx[torch.randperm(idx.numel(), generator=g)[:20000]].sort().values
        pc = _prepare(0, 0, idx)
        m_res = pc.get("mag_res", pc["mag_dt"])
        p_res = pc.get("pha_res_filt", pc["pha_filt"])
        m_bar = _rows(magnitude, 0, 0, idx).mean(dim=0)
        ratio = (
            m_res.var(dim=0) / p_res.var(dim=0).clamp_min(1e-12) / (m_bar * m_bar).clamp_min(1e-12)
        )
        phi_thermal_c = float(ratio[torch.isfinite(ratio)].median())
        if verbose:
            src = "every fitted voxel" if phi_calib is None else "-phi_calib voxels"
            print(
                f"  Phi (thermal): c = {phi_thermal_c:.4g} (median over {int(idx.numel()):,} "
                f"{src}). Pure thermal noise gives c = 1; c < 1 means the phase noise "
                f"is {1 / max(phi_thermal_c, 1e-12) ** 0.5:.2f}x thermal, relative to the magnitude noise"
            )
        del pc, m_res, p_res

    for lo, hi in tqdm(chunks, desc="Phase regression", disable=not verbose or len(chunks) < 2):
        if stored:
            s_ = {k: v[lo:hi].to(device).T for k, v in stored.items()}
        else:
            s_ = _prepare(lo, hi)
        mag_dt, pha_filt = s_["mag_dt"], s_["pha_filt"]
        mag_res = s_.get("mag_res", mag_dt)
        pha_res_filt = s_.get("pha_res_filt", pha_filt)
        mag = _rows(magnitude, lo, hi)
        good = vox_mask[lo:hi].to(device)

        if phi_thermal_c is not None:
            # The regressor's noise is the phase's: sigma/M of the voxel the phase came
            # from. Under sPR that is the donor, and a dark vein borrowing from brighter
            # tissue would otherwise get the vein's M^2 (thermal sigma taken as locally
            # smooth, so the magnitude's own noise is unchanged).
            src = mag if spr_donor is None else _rows(magnitude, 0, 0, spr_donor[lo:hi])
            m_bar = src.mean(dim=0)
            phi_c = (phi_thermal_c * m_bar * m_bar).clamp_min(1e-6)
        elif phi is None:
            phi_c = estimate_variance_ratio(
                mag_res, pha_res_filt, tr, method=phi_method, freq_range=phi_freq_range
            )
        elif isinstance(phi, (int, float)):
            phi_c = torch.full((hi - lo,), float(phi), device=device)
        else:
            phi_c = phi[lo:hi].to(device)

        fit_mag, fit_pha = (mag_dt, pha_filt) if slope_on == "data" else (mag_res, pha_res_filt)
        # Gate on the series the slope is FIT on, not the one it is applied to:
        # with task removal active they differ, and a voxel whose residual
        # phase is flat (all its variance absorbed by the task model) would
        # otherwise reach a Cov/Var slope that explodes.
        flat = fit_pha.var(dim=0) < pha_var_floor
        n_unstable += int((flat & good).sum().item())

        if regression == "deming":
            s, b = deming_regression(fit_pha, fit_mag, phi_c)
            if max_slope_ratio is not None:
                s_ols, _ = ols_regression(fit_pha, fit_mag)
                capped = s.abs() > max_slope_ratio * s_ols.abs()
                n_capped += int((capped & good).sum().item())
                s = torch.sign(s) * torch.minimum(s.abs(), max_slope_ratio * s_ols.abs())
                b = fit_mag.mean(dim=0) - s * fit_pha.mean(dim=0)
        else:
            s, b = ols_regression(fit_pha, fit_mag)
        in_veins = None
        if spr_allow is not None and spr_regression is not None and spr_regression != regression:
            in_veins = spr_allow[lo:hi].to(device).bool()
            if spr_regression == "ols":
                s_alt, b_alt = ols_regression(fit_pha, fit_mag)
            else:
                s_alt, b_alt = deming_regression(fit_pha, fit_mag, phi_c)
            s = torch.where(in_veins, s_alt, s)
            b = torch.where(in_veins, b_alt, b)
        bad = ~good | flat
        s = torch.where(bad, torch.zeros_like(s), s)
        b = torch.where(bad, torch.zeros_like(b), b)

        if coupling_r is not None:
            mc = fit_mag - fit_mag.mean(dim=0, keepdim=True)
            pc = fit_pha - fit_pha.mean(dim=0, keepdim=True)
            denom = (mc.norm(dim=0) * pc.norm(dim=0)).clamp(min=1e-30)
            coupling_r[lo:hi] = ((mc * pc).sum(dim=0) / denom).cpu()
            del mc, pc

        # ── ODR-style residual + R² (matches phaseprep / Stanley) ──
        # R² is measured on the detrended data the slope was estimated from.
        # phaseprep's "corrected" output is the ODR residual eps, not the raw
        # observed-x residual. Algebra: with sx=stdp, sy=stdm and slope A,
        # ODR's optimal per-timepoint x-correction delta is
        #     delta_i = A·sx²·r_i / (A²sx² + sy²),
        # giving an effective output residual of
        #     eps_i = r_i / (1 + A²/φ),    where r_i = mag_dt - A·phase_dt.
        # The (1 + A²/φ) factor *shrinks* the correction in ill-conditioned
        # voxels (A² ≫ φ). Without it, a 900-magnitude slope from low-correlation
        # noise data subtracts a 90-amplitude phantom macro from a 280-mean
        # signal and produces speckle. With it, those voxels collapse to ≈ mean.
        # Both R² and the corrected output use this shrinkage so they agree
        # voxelwise with phaseprep.
        pha_centered = pha_filt - pha_filt.mean(dim=0)
        inflation = 1.0 + (s * s) / phi_c.clamp(min=1e-12)
        shrink = 1.0 / inflation if shrink_mode != "none" else torch.ones_like(inflation)
        if in_veins is not None and spr_regression == "ols":
            # ODR's factor belongs to the Deming fit; an OLS slope is used as fit
            shrink = torch.where(in_veins, torch.ones_like(shrink), shrink)

        ss_total = ((mag_dt - mag_dt.mean(dim=0)) ** 2).sum(dim=0)
        r_obs = mag_dt - (b.unsqueeze(0) + s.unsqueeze(0) * pha_centered)
        # "slope" keeps the residual whole: only the damped slope acts.
        eps = r_obs * shrink.unsqueeze(0) if shrink_mode == "odr" else r_obs
        zero = torch.zeros_like(ss_total)
        has_var = ss_total > 1e-30
        r2[lo:hi] = torch.where(has_var, 1.0 - (eps**2).sum(dim=0) / ss_total, zero).cpu()
        # Naive R²: the observed residual without ODR shrinkage — where phase
        # regression had the greatest raw effect regardless of conditioning.
        if r2_naive is not None:
            r2_naive[lo:hi] = torch.where(
                has_var, 1.0 - (r_obs**2).sum(dim=0) / ss_total, zero
            ).cpu()

        # ── Apply the correction ──
        # ``removed`` is what comes out of the detrended series: s·A·phase for
        # "slope" (and A·phase for "none"), plus (1 − s) of the residual for
        # "odr". keep_drift=True subtracts it from the input (drift kept for a
        # GLM); False re-centres the detrended series on the voxel mean, which for
        # "odr" is phaseprep's mean + eps. mag − corrected ≡ macro in every mode.
        if shrink_mode == "odr":
            # phaseprep: the detrended series minus A·phase, residual scaled by s
            removed = s.unsqueeze(0) * pha_centered + r_obs - eps
        else:
            removed = (s * shrink).unsqueeze(0) * pha_centered
        if keep_drift:
            corrected = mag - removed
        else:
            corrected = mag.mean(dim=0, keepdim=True) + (mag_dt - b.unsqueeze(0)) - removed
        macro = mag - corrected
        mag_corrected[lo:hi] = _host(corrected)
        macro_out[lo:hi] = _host(macro)

        phi_all[lo:hi] = phi_c.cpu()
        slope_all[lo:hi] = s.cpu()
        intercept_all[lo:hi] = b.cpu()
        inflation_all[lo:hi] = inflation.cpu()
        del s_, mag_dt, pha_filt, mag_res, pha_res_filt, mag, r_obs, eps, corrected, macro

    phi_tensor = phi_all if not isinstance(phi, torch.Tensor) else phi

    if verbose:
        phi_good = phi_tensor[vox_mask]
        print(
            f"  Phi: median={phi_good.median().item():.2f}, "
            f"range=[{phi_good.min().item():.2f}, {phi_good.max().item():.2f}]"
        )
        if n_unstable:
            print(
                f"  Phase variance floor: {n_unstable:,} voxels with "
                f"Var(phase) < {pha_var_floor:.0e} (slopes zeroed)"
            )
        if n_capped:
            print(
                f"  Slope cap: {n_capped:,} / {int(vox_mask.sum()):,} voxels "
                f"({100.0 * n_capped / max(int(vox_mask.sum()), 1):.1f}%) hit "
                f"{max_slope_ratio:g}x OLS (Deming wanted more; usually weak coupling)"
            )
        n_sig = (slope_all[vox_mask].abs() > 0.01).sum().item()
        print(
            f"  Slope: median={slope_all[vox_mask].median().item():.4f}, "
            f"{n_sig:,} voxels with |slope| > 0.01"
        )

    # ── 7b. Vein exclusion mask from magnitude-phase coupling ────────────
    vein_p: torch.Tensor | None = None
    vein_exclude: torch.Tensor | None = None
    if coupling_r is not None:
        from fastfuncstuff.phasereg.veinmask import coupling_pvalue, fdr_threshold

        # Every regressor projected out of both series costs a degree of
        # freedom, as do the fitted slope and intercept.
        n_nuisance = sum(p.shape[1] for p in poly_per_run)
        if slope_on == "residual":
            n_nuisance += sum(t.shape[1] for t in task_per_run if t is not None)
        df = n_tp - n_nuisance - 2

        n_cand = (spr_connectivity + 1) if spr else 1
        vein_p = coupling_pvalue(coupling_r, df, n_candidates=n_cand)
        vein_p = torch.where(vox_mask, vein_p, torch.ones_like(vein_p))

        p_crit = fdr_threshold(vein_p, vox_mask, vein_fdr_q)
        vein_exclude = (vein_p <= p_crit) & vox_mask

        if verbose:
            n_ex = int(vein_exclude.sum().item())
            print(
                f"  Vein mask: df={df}, FDR q={vein_fdr_q} → p<={p_crit:.3g}; "
                f"{n_ex:,} / {n_good:,} voxels flagged as vessel "
                f"({100.0 * n_ex / max(n_good, 1):.1f}%)"
            )
            if n_cand > 1:
                print(f"    (Sidak-corrected for {n_cand} sPR donor candidates)")

    if verbose:
        r2_good = r2[vox_mask]
        print(
            f"  R2 (phase, ODR-style): median={r2_good.median().item():.4f}, "
            f"mean={r2_good.mean().item():.4f}, "
            f"max={r2_good.max().item():.4f}"
        )
        if r2_naive is not None:
            r2n_good = r2_naive[vox_mask]
            print(
                f"  R2 (naive, no shrinkage): median={r2n_good.median().item():.4f}, "
                f"mean={r2n_good.mean().item():.4f}, "
                f"max={r2n_good.max().item():.4f}"
            )
        infl_good = inflation_all[vox_mask]
        print(
            f"  ODR inflation (1 + A²/φ): median={infl_good.median().item():.3f}, "
            f"max={infl_good.max().item():.3f}, "
            f">2: {int((infl_good > 2).sum().item()):,} voxels, "
            f">10: {int((infl_good > 10).sum().item()):,} voxels"
        )
        if shrink_mode == "none":
            print(
                "    shrink_mode='none': applying the FULL correction; the "
                "inflation numbers above are what parity mode would discard."
            )
        else:
            frac = (1.0 / infl_good).median().item()
            what = (
                "of the fitted macro component is left IN the output (slope damped)"
                if shrink_mode == "slope"
                else "of the residual (micro signal + noise) is removed WITH the macro"
            )
            print(f"    shrink_mode={shrink_mode!r}: median {100.0 * (1 - frac):.0f}% {what}.")
        print(f"  R2 distribution over {n_good:,} masked voxels:")
        for thresh in (0.01, 0.05, 0.10, 0.20, 0.30, 0.50):
            n_above = int((r2_good > thresh).sum().item())
            pct = 100.0 * n_above / max(n_good, 1)
            print(f"    R2 > {thresh:.2f}:  {n_above:>8,} voxels ({pct:5.2f}%)")

    return PhaseRegResult(
        magnitude_corrected=mag_corrected,
        macrovascular_component=macro_out,
        slope=slope_all,
        intercept=intercept_all,
        phi=phi_tensor,
        r2_phase=r2,
        r2_naive=r2_naive,
        voxel_mask=vox_mask,
        mag_detrended=stored.get("mag_dt") if save_intermediates else None,
        pha_detrended=stored.get("pha_dt") if save_intermediates else None,
        pha_detrended_filt=stored.get("pha_filt") if save_intermediates else None,
        mag_residual=stored.get("mag_res", stored.get("mag_dt")) if save_intermediates else None,
        pha_residual_filt=(
            stored.get("pha_res_filt", stored.get("pha_filt")) if save_intermediates else None
        ),
        sgf_window_map=sgf_window_map,
        sgf_order_map=sgf_order_map,
        spr_donor=spr_donor,
        spr_donor_corr=spr_donor_corr,
        spr_donor_offset=spr_donor_offset,
        coupling_r=coupling_r,
        vein_p=vein_p,
        vein_exclude=vein_exclude,
    )
