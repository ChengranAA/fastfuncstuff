#!/usr/bin/env python3

"""
ffs_phasereg - phase regression for macrovascular BOLD suppression.

Regresses each voxel's magnitude on its phase (Deming, errors-in-variables;
Menon 2002) and subtracts the phase-explained part: oriented veins change both,
randomly oriented microvasculature only magnitude. The user documentation is
the -h text built below; method notes and measurements are in the wiki
(software/ffs_phasereg.md, concepts/Phase regression.md).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import (
    add_bids_events_args,
    add_device_arg,
    add_ortvec_arguments,
    add_verbose_arg,
    check_bids_event_args,
    resolve_event_filters,
    spinner,
)

_DESCRIPTION = """\
Remove the phase-explained (macrovascular) part of magnitude BOLD.

Each voxel's magnitude is regressed on its phase with a Deming (errors-in-
variables) fit, capped at twice the OLS slope, and A*phase is subtracted. Runs passed together share one slope
per voxel (drift removed per run) and are written back one file per run.

UPSTREAM, in order (phase in radians, voxel-matched to the magnitude):
  NORDIC on complex data -> temporal unwrap (ROMEO) -> ffs_phasedrift (per-slice
  field drift, native space) -> resample magnitude and phase through the same
  warps. ffs_autoproc -phase_proc does all four. Skipping ffs_phasedrift leaves
  phase noise several times thermal and weakens every slope.

RECIPES
  GLM input (the usual case):
    ffs_phasereg -magnitude run*_mag.nii.zst -phase run*_phase.nii.zst \\
                 -prefix pr -keep_drift
  Veins survive, sub-mm 7T: add -spr -spr_select_run 0 (see -spr).
  Drop vein voxels instead of correcting them: add -vein_mask.
  phaseprep / Stanley 2021 parity:  -polort 1 -shrink odr -no_phase_unwrap \
                                    -max_slope_ratio 0
"""

# Printed at the end of -h: what each output is for and how to read it. Flag
# behaviour lives in the flag help, not here. A literal percent sign must be
# doubled (%%): argparse %-formats this text.
_EPILOG = """\
OUTPUTS  (4D files are written per run as prefix_<name>_run-NN when several
runs go in; -concat_output gives one file. 3D maps are one per fit.)

  prefix_corrected (4D)  The deliverable: magnitude minus A*phase. Drift is
      dropped and the series re-centred on its mean, unless -keep_drift.
  prefix_macro_std (3D)  Temporal std of what was subtracted: THE QC overlay.
      It should trace pial veins and sinuses, not a diffuse haze; a haze means
      the fit is chasing phase noise (check upstream phase processing first).
  prefix_slope     (3D)  A, magnitude units per radian. Read it with phi.
  prefix_phi       (3D)  Var(magnitude noise)/Var(phase noise) used by Deming.
  prefix_r2        (3D)  ODR R^2, for phaseprep parity only. It saturates at
      0.5-0.8 brain-wide in raw-unit data, so it does not show where the
      correction acted; use macro_std (or -r2_mode naive).
  prefix_mask      (3D)  Voxels fitted.
  [-save_macro] prefix_macro (4D): what was subtracted, input - corrected.
  [-spr]       prefix_spr_donor_offset (0 = own phase), prefix_spr_donor_corr
  [-vein_mask] prefix_vein_mask, prefix_vein_keep, prefix_vein_p, prefix_coupling_r
  [-phase_filter explore]  prefix_sgf_window, prefix_sgf_order

IF VEINS SURVIVE, read prefix_slope at the vein:
  * Slope ~0: the voxel's own phase carries nothing (phase cancels within a
    voxel-sized vein, or the vessel is near the magic angle). Use -spr.
  * Slope huge (1e3-1e5): an ill-conditioned fit to phase noise. Under -shrink
    odr the voxel collapses to its mean, which looks like suppression but is
    not; with the default -shrink none it shows as added noise. -spr gives it
    a real regressor.
  * Slopes small everywhere: phase noise dominates the regressor. Run
    ffs_phasedrift upstream.
"""


def create_parser() -> argparse.ArgumentParser:
    parser = FfsArgumentParser(
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=FfsHelpFormatter,
    )

    req = parser.add_argument_group("Required Arguments")
    req.add_argument(
        "-magnitude",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Magnitude series, one file per run.",
    )
    req.add_argument(
        "-phase",
        nargs="+",
        metavar="FILE",
        required=True,
        help="Phase series in radians, one per run, in the same order and on the same "
        "grid as -magnitude.",
    )
    req.add_argument(
        "-prefix",
        required=True,
        metavar="OUTPUT",
        help="Output prefix. Outputs take the first magnitude input's format (.nii.zst "
        "in, .nii.zst out) unless the prefix names one (pr.nii.gz).",
    )

    reg = parser.add_argument_group("Regression")
    reg.add_argument(
        "-regression",
        choices=["deming", "ols"],
        default="deming",
        help="'deming': errors-in-variables, equivalent to scipy.odr (Menon 2002, "
        "Stanley 2021, phaseprep). 'ols': no phase-noise term, so noisy phase "
        "attenuates the slope and suppresses less (Vu & Gallant 2015's choice, "
        "against over-correction).",
    )
    reg.add_argument(
        "-max_slope_ratio",
        type=float,
        default=2.0,
        metavar="R",
        help="Cap the Deming |A| at R times the voxel's OLS slope; 0 = no cap. "
        "Residual variance is lowest at the OLS slope and back to the input's at "
        "2x, so R=2 never leaves a voxel noisier than it was. Uncapped, Deming "
        "goes to ~1e9 where magnitude and phase are uncorrelated (5.8%% of "
        "MindsEye voxels had a >10 sd removal). MindsEye split-half, share of the "
        "task effect kept: veins 0.84 OLS / 0.71 R=2 / 0.65 R=3 / 0.59 R=10; "
        "controls 0.98 / 0.97 / 0.96 / 0.94.",
    )
    reg.add_argument(
        "-shrink",
        choices=["odr", "none"],
        default="none",
        help="'none': subtract A*phase, leave the residual intact. 'odr': phaseprep "
        "parity. It also applies ODR's factor s = 1/(1+A^2/phi): by default the "
        "residual (micro BOLD included) is scaled by s, so betas shrink most near "
        "veins; with -keep_drift only s*A*phase is removed, leaving (1-s) of the "
        "macro in.",
    )
    reg.add_argument(
        "-phi",
        type=float,
        default=None,
        metavar="VALUE",
        help="Fixed variance ratio for Deming. Default: estimated per voxel (-phi_method).",
    )
    reg.add_argument(
        "-phi_method",
        choices=["fft", "residual"],
        default="fft",
        help="'fft': noise power above -freq_range (Curtis 2014, phaseprep); safe for "
        "task and rest. 'residual': variance after detrend, nuisance and task "
        "removal (Knudsen 2023); on task data it needs -task_removal, or the task "
        "counts as noise.",
    )
    reg.add_argument(
        "-freq_range",
        nargs="+",
        type=float,
        default=[0.15],
        metavar="HZ",
        help="Band for -phi_method fft: LO (to Nyquist) or LO HI. Default 0.15 Hz "
        "(Stanley 2021, phaseprep).",
    )
    reg.add_argument(
        "-slope_on",
        choices=["data", "residual"],
        default="data",
        help="'data': fit the slope on the detrended, task-preserved series (Menon "
        "2002, Stanley 2021, Knudsen 2023); a vein's task response is what pins its "
        "slope. 'residual': fit after task removal (needs -task_removal).",
    )

    task = parser.add_argument_group(
        "Task model (only used for phi with -phi_method residual, or with -slope_on residual)"
    )
    task.add_argument(
        "-task_removal",
        choices=["none", "tent", "canonical"],
        default="none",
        help="'tent': FIR basis, model-free. 'canonical': SPM HRF.",
    )
    task.add_argument(
        "-onsets",
        nargs="+",
        metavar="FILE",
        help="AFNI timing files, one per condition, one row per run.",
    )
    add_bids_events_args(
        task,
        events_help="BIDS events TSVs, one per run (instead of -onsets).",
    )
    task.add_argument(
        "-tent_window",
        type=float,
        default=20.0,
        metavar="SECONDS",
        help="TENT window for -task_removal tent.",
    )
    task.add_argument(
        "-durations",
        nargs="+",
        metavar="SECONDS",
        help="Per-condition durations for -onsets (-events reads them from the TSV).",
    )

    pcond = parser.add_argument_group("Phase conditioning")
    pcond.add_argument(
        "-no_phase_unwrap",
        action="store_true",
        help="Skip the temporal unwrap. By default each run is unwrapped along time "
        "about the voxel's circular mean, correcting only steps a multiple of 2pi "
        "brings under pi/2: a no-op on unwrapped input, and what makes wrapped "
        "input usable.",
    )
    pcond.add_argument(
        "-phase_global_order",
        type=int,
        default=-1,
        metavar="N",
        help="Remove a per-volume spatial phase term on the final grid: -1 off "
        "(default), 0 a brain-wide offset, N>=1 a Legendre polynomial of degree N "
        "(Bodenschatz & Rowe 2026). A fallback: ffs_phasedrift upstream does this "
        "per slice in native space and removes much more.",
    )
    pcond.add_argument(
        "-phase_filter",
        choices=["none", "sgf", "explore"],
        default="none",
        help="Smooth the phase before the fit (Barry & Gore 2014). 'sgf': "
        "Savitzky-Golay with -sgf_window/-sgf_order. 'explore': per-voxel search "
        "over the -sgf_*_max grid, keeping the unfiltered series as a candidate; "
        "maximises |r(phase, magnitude)|, the same argmax as their minimum-residual "
        "rule under OLS. Slow.",
    )
    pcond.add_argument(
        "-sgf_window",
        type=int,
        default=None,
        metavar="N",
        help="[sgf] Window in TRs, odd. Default round(20 s/TR), min 5: a placeholder "
        "close to an HRF's length, which can smooth away task-locked phase. Set it "
        "from your design, or use explore.",
    )
    pcond.add_argument(
        "-sgf_order",
        type=int,
        default=3,
        metavar="P",
        help="[sgf] Polynomial order, below the window.",
    )
    pcond.add_argument(
        "-sgf_window_max",
        type=int,
        default=None,
        metavar="N",
        help="[explore] Largest window. Default n_TRs // 2 (Barry & Gore).",
    )
    pcond.add_argument(
        "-sgf_order_max",
        type=int,
        default=None,
        metavar="P",
        help="[explore] Largest order, min 2. Default sgf_window_max // 4 (Barry & Gore).",
    )
    pcond.add_argument(
        "-sgf_step",
        type=int,
        default=4,
        metavar="N",
        help="[explore] Window step.",
    )

    sprg = parser.add_argument_group("Source-localized PR (sPR; Vu & Gallant 2015)")
    sprg.add_argument(
        "-spr",
        action="store_true",
        help="Regress each voxel's magnitude on the phase of whichever neighbour "
        "(or itself) tracks it best. For veins whose own phase is ~0: field "
        "offsets cancel inside a voxel-sized vein, or the vessel is near the magic "
        "angle, while a neighbour sees one lobe of the field. Vu also used OLS, "
        "which suppresses less; keep -regression deming if veins survive. "
        "Choose donors on a held-out run (-spr_select_run): in-sample selection "
        "over-suppresses. On 7T 0.8 mm data (split-half, held-out donors, share "
        "of the task effect kept) it took phase-blind voxels from 0.91 to 0.72 "
        "and veins from 0.71 to 0.65, at a cost of 0.97 -> 0.94 in vessel-free "
        "controls, because most voxels borrow on noise. QC: "
        "prefix_spr_donor_offset should trace vessels.",
    )
    sprg.add_argument(
        "-spr_neighborhood",
        "-spr_connectivity",
        type=int,
        choices=[6, 18, 26],
        default=6,
        dest="spr_neighborhood",
        metavar="K",
        help="Donor neighbourhood: 6 faces (Vu & Gallant), 18, or 26. Larger K overfits more.",
    )
    sprg.add_argument(
        "-spr_select_run",
        type=int,
        default=None,
        metavar="R",
        help="0-based run used only to choose donors (Vu & Gallant held out a run). "
        "Default: all runs, in-sample, which biases toward over-suppression.",
    )

    vg = parser.add_argument_group("Vein exclusion mask")
    vg.add_argument(
        "-vein_mask",
        action="store_true",
        help="Also write prefix_vein_mask: voxels whose magnitude and phase "
        "correlate beyond chance (oriented vessels; cf. Rowe 2005). For layer "
        "profiles that would rather drop those voxels than trust a subtraction. "
        "The corrected series is unchanged. With -spr the test uses the donor's "
        "phase, which catches phase-blind veins but dilates the mask by about a "
        "voxel.",
    )
    vg.add_argument(
        "-vein_fdr",
        type=float,
        default=0.05,
        metavar="Q",
        help="BH-FDR q for -vein_mask, after a Sidak correction for the -spr donor argmax.",
    )

    proc = parser.add_argument_group("Processing")
    proc.add_argument(
        "-keep_drift",
        action="store_true",
        help="Keep polynomial drift in prefix_corrected (output = input - A*phase). "
        "Use when a GLM follows, as it models drift itself. Default: drift dropped "
        "and the series re-centred on its mean (phaseprep).",
    )
    proc.add_argument(
        "-polort",
        type=str,
        default="A",
        metavar="N",
        help="Legendre drift order per run, removed from both series before the fit. "
        "'A' (default) = AFNI's rule from run length; 1 = phaseprep; -1 none.",
    )
    proc.add_argument(
        "-tr",
        type=float,
        metavar="SECONDS",
        help="Repetition time. Default: the magnitude header.",
    )
    proc.add_argument(
        "-mask",
        metavar="FILE",
        help="Analysis mask. Default: an automask of the mean magnitude.",
    )
    proc.add_argument(
        "-no_auto_mask",
        action="store_true",
        help="Without -mask, fit every voxel above -signal_thresh instead of automasking.",
    )
    proc.add_argument(
        "-signal_thresh",
        type=float,
        default=0.03,
        metavar="FRAC",
        help="Skip voxels whose mean is below this fraction of the maximum (phaseprep). "
        "0 fits everything, air included.",
    )
    proc.add_argument(
        "-concat_output",
        action="store_true",
        help="Write each 4D output as one file concatenated over runs, not per run.",
    )
    proc.add_argument(
        "-motion",
        nargs="+",
        metavar="FILE",
        help="(Deprecated: -ortvec_run FILE motion RUN.) Motion files, one per run.",
    )
    add_ortvec_arguments(proc)

    out = parser.add_argument_group("Output and hardware")
    out.add_argument(
        "-r2_mode",
        choices=["odr", "naive", "both"],
        default="odr",
        help="R^2 map(s): 'odr' (phaseprep parity), 'naive' (no shrinkage: where the "
        "correction was largest), or both.",
    )
    out.add_argument(
        "-save_macro",
        action="store_true",
        help="Also write prefix_macro, the 4D series that was subtracted. Off by "
        "default: it is exactly input - corrected and doubles the write time.",
    )
    out.add_argument(
        "-save_intermediates",
        action="store_true",
        help="Also write the detrended, filtered and task-residual series "
        "(prefix_mag_dt, _pha_dt, _pha_dt_filt, _mag_res, _pha_res_filt).",
    )
    add_device_arg(out)
    add_verbose_arg(out, default=0)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for ffs_phasereg."""
    parser = create_parser()
    args = parser.parse_args(argv)
    check_bids_event_args(args)

    # ── Imports ──────────────────────────────────────────────────────────
    try:
        from fastfuncstuff.cli_utils import (
            auto_polort,
            input_nifti_ext,
            parse_device_arg,
            parse_prefix,
        )
        from fastfuncstuff.io.afni import (
            from_voxel_major,
            load_afni_mask,
            load_and_concatenate_runs,
            save_nifti,
        )
        from fastfuncstuff.io.headers import nifti_shape, read_nifti_header
        from fastfuncstuff.phasereg.core import phase_regress
        from fastfuncstuff.utils import configure_torch_backends, get_device  # noqa: F401
    except ImportError as e:
        print(f"ERROR: Could not import fastfuncstuff: {e}", file=sys.stderr)
        return 1

    pfx = parse_prefix(args.prefix, default_ext=input_nifti_ext(args.magnitude[0]))
    args.prefix = pfx.stem
    nii_ext = pfx.nifti_ext

    device, _, _ = parse_device_arg(args.device)
    configure_torch_backends(device)

    if args.verb >= 1:
        print(f"\n{'=' * 70}")
        print("Phase Regression (ffs_phasereg)")
        print(f"{'=' * 70}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {device}")

    # ── Validate inputs ──────────────────────────────────────────────────
    n_runs = len(args.magnitude)
    if len(args.phase) != n_runs:
        print(
            f"ERROR: Number of magnitude ({n_runs}) and phase "
            f"({len(args.phase)}) files must match.",
            file=sys.stderr,
        )
        return 1

    if args.task_removal != "none" and not args.onsets and not args.events:
        print(
            f"ERROR: -task_removal {args.task_removal} requires -onsets or -events.",
            file=sys.stderr,
        )
        return 1

    if args.onsets and args.events:
        print("ERROR: -onsets and -events are mutually exclusive.", file=sys.stderr)
        return 1

    # ── Parse frequency range ────────────────────────────────────────────
    if len(args.freq_range) == 1:
        freq_range = (args.freq_range[0], None)
    elif len(args.freq_range) == 2:
        freq_range = (args.freq_range[0], args.freq_range[1])
    else:
        print("ERROR: -freq_range takes 1 or 2 values.", file=sys.stderr)
        return 1

    # ── Load data ────────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nLoading data...")

    for p in (*args.magnitude, *args.phase):
        if not Path(p).exists():
            print(f"ERROR: File not found: {p}", file=sys.stderr)
            return 1

    # Shapes and TRs from the headers alone: a .nii.zst run is gigabytes, and
    # nothing here needs its data to be checked.
    n_tp_per_run = []
    tr_values = []
    nx = ny = nz = None
    ref_img_path = args.magnitude[0]
    for mag_path, pha_path in zip(args.magnitude, args.phase, strict=True):
        mag_hdr = read_nifti_header(mag_path)
        mag_shape = nifti_shape(mag_path)
        pha_shape = nifti_shape(pha_path)
        if len(mag_shape) != 4:
            print(f"ERROR: Expected 4D data, got {len(mag_shape)}D: {mag_path}", file=sys.stderr)
            return 1
        if pha_shape != mag_shape:
            print(
                f"ERROR: Phase shape {pha_shape} != magnitude shape {mag_shape}",
                file=sys.stderr,
            )
            return 1
        if nx is None:
            nx, ny, nz = mag_shape[:3]
        elif mag_shape[:3] != (nx, ny, nz):
            print(
                f"ERROR: {mag_path} grid {mag_shape[:3]} != first run's {(nx, ny, nz)}",
                file=sys.stderr,
            )
            return 1
        n_tp_per_run.append(int(mag_shape[3]))
        tr_values.append(float(mag_hdr.get_zooms()[3]))
        if args.verb >= 1:
            print(f"  Run {len(n_tp_per_run)}: {mag_path} ({mag_shape[3]} TRs)")

    # Magnitude first: the automask is built from its mean, and the phase is
    # then loaded through that mask, which skips reordering the ~75% of the
    # grid outside it. Kept on the CPU: phase_regress streams voxel chunks to
    # the device itself.
    mask_np = None
    if args.mask:
        mask_np = load_afni_mask(args.mask)
        if mask_np.shape != (nx, ny, nz):
            print(
                f"ERROR: Mask shape {mask_np.shape} != data shape {(nx, ny, nz)}", file=sys.stderr
            )
            return 1
        mask_np = mask_np.flatten().astype(bool)
    mags, _ = load_and_concatenate_runs(args.magnitude, keep_on_cpu=True, mask_flat=mask_np)

    if args.tr is None:
        if len(set(tr_values)) > 1:
            print(f"ERROR: Inconsistent TRs: {tr_values}", file=sys.stderr)
            return 1
        tr = tr_values[0]
        if not tr > 0:
            print("ERROR: No TR in the magnitude header; pass -tr.", file=sys.stderr)
            return 1
    else:
        tr = args.tr

    if args.verb >= 1:
        print(f"  TR: {tr}s")
        print(f"  Volume: {nx} x {ny} x {nz}")

    # ── SGF window: TR-adaptive default (~20s, HRF-duration based) ──────
    if args.sgf_window is None:
        sgf_window = round(20.0 / tr)
        if sgf_window % 2 == 0:
            sgf_window += 1
        sgf_window = max(sgf_window, 5)
        if args.verb >= 1:
            print(f"  SGF window: {sgf_window} (~20s / {tr:.2f}s TR)")
    else:
        sgf_window = args.sgf_window
        if sgf_window % 2 == 0:
            print("ERROR: -sgf_window must be odd.", file=sys.stderr)
            return 1

    # ── Mask ─────────────────────────────────────────────────────────────
    n_all_voxels = nx * ny * nz
    if mask_np is not None:
        if args.verb >= 1:
            print(f"  Mask (provided): {int(mask_np.sum()):,} / {n_all_voxels:,} voxels")
    elif not args.no_auto_mask:
        from fastfuncstuff.processing.mask import automask

        mean_mag = torch.stack([m.mean(dim=1) for m in mags.split(n_tp_per_run, dim=1)]).mean(0)
        mask3d = automask(
            mean_mag.reshape(nx, ny, nz), dilate_extra=1, device=device, verbose=args.verb >= 1
        )
        mask_np = mask3d.flatten().cpu().numpy().astype(bool)
        mags = mags[torch.from_numpy(mask_np)]
        if args.verb >= 1:
            print(f"  Mask (auto): {int(mask_np.sum()):,} / {n_all_voxels:,} voxels")
    elif args.verb >= 1:
        print(f"  Mask: none (using all {n_all_voxels:,} voxels)")
    mask_flat = None if mask_np is None else torch.from_numpy(mask_np)

    phas, _ = load_and_concatenate_runs(args.phase, keep_on_cpu=True, mask_flat=mask_np)
    mag_list = list(mags.split(n_tp_per_run, dim=1))
    pha_list = list(phas.split(n_tp_per_run, dim=1))

    # ── Polort ───────────────────────────────────────────────────────────
    polort_str = str(args.polort).strip().upper()
    if polort_str == "A":
        run_dur = min(n_tp_per_run) * tr
        polort = auto_polort(run_dur, formula="afni")
        if args.verb >= 1:
            print(f"  Polort: A -> {polort} (run duration {run_dur:.1f}s)")
    else:
        polort = int(polort_str)

    # ── Parse onsets ─────────────────────────────────────────────────────
    onsets_per_condition = None
    if args.onsets:
        from fastfuncstuff.design.builder import parse_afni_timing_file

        onsets_per_condition = []
        for onset_file in args.onsets:
            onsets_by_run = parse_afni_timing_file(onset_file)
            if len(onsets_by_run) != n_runs:
                print(
                    f"ERROR: {onset_file} has {len(onsets_by_run)} runs, expected {n_runs}",
                    file=sys.stderr,
                )
                return 1
            onsets_per_condition.append(onsets_by_run)
        if args.verb >= 1:
            print(f"  Onsets: {len(onsets_per_condition)} conditions")

    elif args.events:
        from fastfuncstuff.design.bids_events import check_events_pairing, parse_bids_events

        if len(args.events) not in (1, n_runs):
            print(
                f"ERROR: -events requires one TSV per run or a single shared TSV: "
                f"got {len(args.events)} events files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        try:
            check_events_pairing(
                list(args.magnitude), args.events, n_runs=n_runs, verbose=args.verb >= 1
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        event_cols = tuple(args.event_cols) if args.event_cols else None
        try:
            bids_onsets, bids_durations, bids_labels = parse_bids_events(
                event_files=args.events,
                event_ignore=args.event_ignore,
                event_cols=event_cols,
                event_filters=resolve_event_filters(args),
                n_runs=n_runs,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        onsets_per_condition = bids_onsets
        if args.verb >= 1:
            print(f"  BIDS events: {len(bids_labels)} conditions ({', '.join(bids_labels)})")

    # ── Auto tent window from durations ──────────────────────────────────
    tent_window = args.tent_window
    if args.task_removal == "tent" and args.durations and onsets_per_condition is not None:
        from fastfuncstuff.design.builder import parse_durations
        from fastfuncstuff.design.hrf import compute_windows_from_durations

        n_conds = len(onsets_per_condition)
        cond_labels = [f"cond{i}" for i in range(n_conds)]
        condition_durations = parse_durations(args.durations, n_conds, cond_labels)
        windows = compute_windows_from_durations(condition_durations, tr)
        tent_window = max(top for _, top in windows)
        if args.verb >= 1:
            print(f"  Auto TENT window: {tent_window:.1f}s (from stimulus durations)")

    # ── Build per-run nuisance from -motion / -ortvec* flags ─────────────
    # -motion remains as a deprecated alias for N invocations of
    # -ortvec_run FILE motion RUN. Both styles funnel through the unified
    # NuisanceBlock pathway so per-run / glob / full-length inputs all work.
    if args.motion:
        if len(args.motion) != n_runs:
            print(
                f"ERROR: {len(args.motion)} motion files but {n_runs} runs.",
                file=sys.stderr,
            )
            return 1
        if args.verb >= 1:
            print(
                "WARNING: -motion is deprecated; use "
                "-ortvec_run FILE motion RUN (repeatable) instead."
            )
        args.ortvec_run = list(args.ortvec_run or []) + [
            [str(f), "motion", str(i + 1)] for i, f in enumerate(args.motion)
        ]

    nuisance_per_run = None
    if args.ortvec or args.ortvec_run or args.ortvec_glob or args.ortvec_concat:
        from fastfuncstuff.cli_utils import collect_nuisance_blocks

        run_lengths = [int(m.shape[1]) for m in mag_list]
        run_starts = [0]
        for rl in run_lengths[:-1]:
            run_starts.append(run_starts[-1] + rl)
        n_timepoints = sum(run_lengths)

        blocks = collect_nuisance_blocks(
            args,
            run_starts,
            n_timepoints,
            verbose=(args.verb >= 1),
        )
        nuisance_per_run = []
        for run_idx, run_length in enumerate(run_lengths):
            cols: list[np.ndarray] = []
            for block in blocks:
                if block.n_columns == 0:
                    continue
                m = block.get_run(run_idx, run_length).copy()
                col_mean = m.mean(axis=0, keepdims=True)
                if np.max(np.abs(col_mean)) > 1e-4:
                    m = m - col_mean
                cols.append(m)
            if cols:
                arr = np.concatenate(cols, axis=1).astype(np.float32)
                nuisance_per_run.append(torch.tensor(arr, dtype=torch.float32, device=device))
            else:
                nuisance_per_run.append(
                    torch.zeros((run_length, 0), dtype=torch.float32, device=device)
                )
        if args.verb >= 1:
            print(
                f"  Nuisance: {nuisance_per_run[0].shape[1]} regressor(s) per run "
                f"({len(blocks)} block(s))"
            )

    # ── Run phase regression ─────────────────────────────────────────────
    if args.verb >= 1:
        print()

    result = phase_regress(
        magnitude=mag_list,
        phase=pha_list,
        tr=tr,
        task_removal=args.task_removal,
        onsets_per_condition=onsets_per_condition,
        nuisance_per_run=nuisance_per_run,
        max_poly_degree=polort,
        phi=args.phi,
        phi_method=args.phi_method,
        phi_freq_range=freq_range,
        regression=args.regression,
        max_slope_ratio=args.max_slope_ratio or None,
        slope_on=args.slope_on,
        tent_window=tent_window,
        phase_filter=args.phase_filter,
        sgf_window=sgf_window,
        sgf_order=args.sgf_order,
        sgf_window_max=args.sgf_window_max,
        sgf_order_max=args.sgf_order_max,
        sgf_step=args.sgf_step,
        signal_thresh=args.signal_thresh,
        phase_unwrap=not args.no_phase_unwrap,
        phase_global_order=args.phase_global_order,
        keep_drift=args.keep_drift,
        shrink_mode=args.shrink,
        vein_mask=args.vein_mask,
        vein_fdr_q=args.vein_fdr,
        spr=args.spr,
        spr_connectivity=args.spr_neighborhood,
        spr_select_run=args.spr_select_run,
        volume_shape=(nx, ny, nz),
        mask_flat=mask_flat,
        device=str(device),
        verbose=args.verb >= 1,
        r2_mode=args.r2_mode,
        save_intermediates=args.save_intermediates,
    )

    # ── Save outputs ─────────────────────────────────────────────────────
    if args.verb >= 1:
        print("\nSaving outputs...")

    def _to_volume(data_flat, is_4d=False):
        if is_4d:
            return from_voxel_major(data_flat, (nx, ny, nz), mask_np)
        if mask_np is not None:
            vol = np.zeros(n_all_voxels, dtype=np.float32)
            vol[mask_np] = data_flat
        else:
            vol = data_flat
        return vol.reshape(nx, ny, nz)

    outputs = {}
    split_runs = n_runs > 1 and not args.concat_output
    run_edges = np.cumsum([0, *n_tp_per_run])

    def _save_4d(key, data_flat):
        """One file per run (each on its own run's header), or one concatenated."""
        if not split_runs:
            fname = f"{args.prefix}_{key}{nii_ext}"
            save_nifti(_to_volume(data_flat, is_4d=True), fname, reference_img=ref_img_path)
            outputs[key] = fname
            return
        for i in range(n_runs):
            fname = f"{args.prefix}_{key}_run-{i + 1:02d}{nii_ext}"
            vol = _to_volume(data_flat[:, run_edges[i] : run_edges[i + 1]], is_4d=True)
            save_nifti(vol, fname, reference_img=args.magnitude[i])
            outputs[f"{key}_run-{i + 1:02d}"] = fname

    with spinner("Writing outputs"):
        _save_4d("corrected", result.magnitude_corrected.numpy())
        if args.save_macro:
            _save_4d("macro", result.macrovascular_component.numpy())

        # Temporal std of what was subtracted: the recommended QC overlay for
        # localising suppression (vessel-shaped), since the ODR R² saturates.
        macro_std_np = result.macrovascular_component.std(dim=1).numpy()
        fname = f"{args.prefix}_macro_std{nii_ext}"
        save_nifti(_to_volume(macro_std_np), fname, reference_img=ref_img_path)
        outputs["macro_std"] = fname

        fname = f"{args.prefix}_slope{nii_ext}"
        save_nifti(_to_volume(result.slope.numpy()), fname, reference_img=ref_img_path)
        outputs["slope"] = fname

        if args.r2_mode in ("odr", "both"):
            fname = f"{args.prefix}_r2{nii_ext}"
            save_nifti(_to_volume(result.r2_phase.numpy()), fname, reference_img=ref_img_path)
            outputs["r2"] = fname

        if args.r2_mode in ("naive", "both") and result.r2_naive is not None:
            fname = f"{args.prefix}_r2_naive{nii_ext}"
            save_nifti(_to_volume(result.r2_naive.numpy()), fname, reference_img=ref_img_path)
            outputs["r2_naive"] = fname

        fname = f"{args.prefix}_phi{nii_ext}"
        save_nifti(_to_volume(result.phi.numpy()), fname, reference_img=ref_img_path)
        outputs["phi"] = fname

        fname = f"{args.prefix}_mask{nii_ext}"
        save_nifti(
            _to_volume(result.voxel_mask.numpy().astype(np.float32)),
            fname,
            reference_img=ref_img_path,
        )
        outputs["mask"] = fname

        # Vein exclusion mask: the voxels to DROP from a layer profile.
        if result.vein_exclude is not None and result.vein_p is not None:
            fname = f"{args.prefix}_vein_mask{nii_ext}"
            save_nifti(
                _to_volume(result.vein_exclude.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["vein_mask"] = fname

            fname = f"{args.prefix}_vein_keep{nii_ext}"
            keep = result.voxel_mask & ~result.vein_exclude
            save_nifti(
                _to_volume(keep.numpy().astype(np.float32)), fname, reference_img=ref_img_path
            )
            outputs["vein_keep"] = fname

            fname = f"{args.prefix}_vein_p{nii_ext}"
            save_nifti(_to_volume(result.vein_p.numpy()), fname, reference_img=ref_img_path)
            outputs["vein_p"] = fname

            if result.coupling_r is not None:
                fname = f"{args.prefix}_coupling_r{nii_ext}"
                save_nifti(_to_volume(result.coupling_r.numpy()), fname, reference_img=ref_img_path)
                outputs["coupling_r"] = fname

        # sPR diagnostics: where phase was borrowed from, and how well it fit.
        if result.spr_donor_corr is not None and result.spr_donor_offset is not None:
            fname = f"{args.prefix}_spr_donor_corr{nii_ext}"
            save_nifti(_to_volume(result.spr_donor_corr.numpy()), fname, reference_img=ref_img_path)
            outputs["spr_donor_corr"] = fname

            fname = f"{args.prefix}_spr_donor_offset{nii_ext}"
            save_nifti(
                _to_volume(result.spr_donor_offset.numpy()),
                fname,
                reference_img=ref_img_path,
            )
            outputs["spr_donor_offset"] = fname

        # Explore-mode diagnostic: the per-voxel window/order the search chose.
        if result.sgf_window_map is not None and result.sgf_order_map is not None:
            fname = f"{args.prefix}_sgf_window{nii_ext}"
            save_nifti(
                _to_volume(result.sgf_window_map.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["sgf_window"] = fname

            fname = f"{args.prefix}_sgf_order{nii_ext}"
            save_nifti(
                _to_volume(result.sgf_order_map.numpy().astype(np.float32)),
                fname,
                reference_img=ref_img_path,
            )
            outputs["sgf_order"] = fname

        if args.save_intermediates:
            interm_specs = [
                ("mag_dt", result.mag_detrended, "poly-detrended magnitude"),
                ("pha_dt", result.pha_detrended, "poly-detrended phase"),
                ("pha_dt_filt", result.pha_detrended_filt, "detrended + filtered phase"),
                ("mag_res", result.mag_residual, "magnitude task-residual"),
                ("pha_res_filt", result.pha_residual_filt, "phase task-residual + filtered"),
            ]
            for key, data, label in interm_specs:
                if data is None:
                    continue
                _save_4d(key, data.numpy())
                if args.verb >= 1:
                    print(f"  {key}: {label}")

    if args.verb >= 1:
        for name, path in outputs.items():
            print(f"  {name}: {path}")
        print(f"\n{'=' * 70}")
        print("Phase regression complete!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 70}")
    elif split_runs:
        print(
            f"Phase regression complete. Main output: {args.prefix}_corrected_run-NN{nii_ext}, "
            f"one per run in input order:"
        )
        for i in range(n_runs):
            print(f"  run-{i + 1:02d} <- {args.magnitude[i]}")
    else:
        print(f"Phase regression complete. Main output: {outputs['corrected']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
