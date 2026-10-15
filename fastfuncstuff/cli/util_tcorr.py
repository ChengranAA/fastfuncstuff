"""ffs_util_tcorr — voxel-to-voxel temporal correlation between datasets on one grid.

AFNI's 3dTcorrelate, for N datasets at once: each voxel's time course in one input
is correlated with the SAME voxel in another, one r-map per pair, written as a 4-D
bucket with a sub-brick per pair (labelled ``1-2``, ``1-3``, ...; tagged as
correlations so AFNI thresholds them).

Typical uses: echo 1 vs echo 2 of a run (how much of each voxel is shared signal),
two runs of the same task, or a series before vs after a correction. With
``-before``, the same pairs are also correlated on a second set of inputs and the
change ``r(input) - r(before)`` is written -- where a correction made agreement
worse is then just ``delta < 0``.

Blur raises these correlations (it averages down independent noise), and between
echoes motion is shared, so removing it can lower r. Eyeball, don't score.

Examples
--------
    ffs_util_tcorr -input e1.nii.gz e2.nii.gz e3.nii.gz -prefix echo_r.nii.gz
    ffs_util_tcorr -input e1_moco.nii.gz e2_moco.nii.gz \\
        -before e1.nii.gz e2.nii.gz -prefix e12_r.nii.gz
    ffs_util_tcorr -input run1.nii.gz run2.nii.gz run3.nii.gz -pairs ref:1 \\
        -polort 3 -mask brain.nii.gz -prefix runs_r.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, parse_prefix, setup_device, spinner


def _build_parser() -> argparse.ArgumentParser:
    p = FfsArgumentParser(
        prog="ffs_util_tcorr",
        description=__doc__,
        formatter_class=FfsHelpFormatter,
    )
    p.add_argument(
        "-input",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Two or more 4-D datasets on the same grid with the same number of frames.",
    )
    p.add_argument("-prefix", required=True, help="Output r-map (4-D, one sub-brick per pair).")
    p.add_argument(
        "-pairs",
        default="all",
        metavar="all|adjacent|ref:K",
        help="Which pairs to correlate.\n"
        "  all       every i<j (N(N-1)/2 maps).\n"
        "  adjacent  1-2, 2-3, ... (consecutive echoes).\n"
        "  ref:K     input K (1-based) against every other one.",
    )
    p.add_argument(
        "-polort",
        default="auto",
        metavar="N|auto",
        help="Legendre drift degree removed from every voxel first. Shared slow drift\n"
        "correlates two series regardless of their content, and echoes of one run share\n"
        "all of it. auto = AFNI's 1 + floor(run_seconds/150), from the header TR (or\n"
        "-tr). 0 = mean only (a plain Pearson r); -1 = no detrend at all.",
    )
    p.add_argument(
        "-tr",
        type=float,
        default=None,
        metavar="SEC",
        help="TR for -polort auto. Default: the first input's header.",
    )
    p.add_argument(
        "-before",
        nargs="+",
        default=None,
        metavar="FILE",
        help="A second set of inputs, one per -input and in the same order (e.g. the\n"
        "uncorrected echoes). The same pairs are correlated on them and written as\n"
        "{prefix}_before, plus the change r(input) - r(before) as {prefix}_delta.",
    )
    p.add_argument(
        "-mask",
        default=None,
        metavar="FILE",
        help="Correlate only inside this mask (nonzero); 0 elsewhere. Saves the time\n"
        "spent outside the head; not needed for correctness.",
    )
    p.add_argument(
        "-fisher",
        action="store_true",
        help="Write Fisher z = atanh(r) instead of r (the delta is then a z difference).",
    )
    add_device_arg(p)
    p.add_argument("-quiet", action="store_true", help="Suppress progress output.")
    return p


def _load(paths: list[str], what: str):
    from fastfuncstuff.io.afni import load_nifti

    arrays, img0 = [], None
    for path in paths:
        with spinner(f"Loading {Path(path).name}"):
            img = load_nifti(path)
            # dataobj, not get_fdata: no float64 copy, and the correlation streams the
            # F-ordered array in x-slabs without ever copying it whole.
            arr = np.asarray(img.dataobj, dtype=np.float32)
        if arr.ndim != 4:
            raise ValueError(f"{what} {path} must be 4-D, got shape {arr.shape}")
        if arrays and arr.shape != arrays[0].shape:
            raise ValueError(f"{what} {path} has shape {arr.shape}, the first {arrays[0].shape}")
        arrays.append(arr)
        img0 = img0 or img
    return arrays, img0


def _resolve_polort(spec: str, n_t: int, tr: float | None) -> int:
    from fastfuncstuff.stats.task_coupling import default_polort

    if spec != "auto":
        return int(spec)
    if tr is None or not 0.1 <= tr <= 20.0:
        raise ValueError(
            f"-polort auto needs a plausible TR; the header says {tr}. Pass -tr SEC or -polort N."
        )
    return default_polort(n_t, tr)


def main(argv: list[str] | None = None) -> int:

    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.stats.paired_correlation import pair_list, paired_voxel_correlation

    args = _build_parser().parse_args(argv)
    verbose = not args.quiet
    try:
        if len(args.input) < 2:
            raise ValueError("-input needs at least two datasets")
        if args.before is not None and len(args.before) != len(args.input):
            raise ValueError(
                f"-before takes one file per -input ({len(args.input)}), got {len(args.before)}"
            )
        pairs = pair_list(len(args.input), args.pairs)
        device = setup_device(args.device)
        data, img = _load(args.input, "-input")
        n_t = data[0].shape[3]
        zooms = img.header.get_zooms()
        tr = args.tr if args.tr is not None else (float(zooms[3]) if len(zooms) > 3 else None)
        polort = _resolve_polort(args.polort, n_t, tr)
        mask = None
        if args.mask:
            mask = np.asarray(load_nifti(args.mask).dataobj) != 0
            if mask.shape != data[0].shape[:3]:
                raise ValueError(f"-mask shape {mask.shape} != grid {data[0].shape[:3]}")
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    labels = [f"{a + 1}-{b + 1}" for a, b in pairs]
    if verbose:
        names = ", ".join(f"{k + 1}={Path(p).name}" for k, p in enumerate(args.input))
        print(f"   {len(pairs)} pair(s) over {n_t} frames, polort {polort}, device {device}")
        print(f"   inputs: {names}")

    def correlate(series):
        return paired_voxel_correlation(
            series, pairs, polort=polort, mask=mask, fisher=args.fisher, device=device,
            verbose=verbose,
        )  # fmt: skip

    pre = parse_prefix(args.prefix)
    stem, ext = pre.stem, pre.nifti_ext
    # A Pearson r on detrended series: AFNI's fico, (samples, fit params, ort params).
    stataux = None if args.fisher else {i: (2, (n_t, 1, polort + 1)) for i in range(len(pairs))}

    def write(path, maps, sa):
        with spinner(f"Writing {Path(path).name}"):
            save_nifti(maps.numpy(), path, affine=img.affine, brick_labels=labels, brick_stataux=sa)

    after = correlate(data)
    del data
    write(f"{stem}{ext}", after, stataux)
    if args.before is not None:
        try:
            prior, _ = _load(args.before, "-before")
        except (ValueError, FileNotFoundError) as exc:
            print(f"❌ {exc}", file=sys.stderr)
            return 2
        if prior[0].shape != (*after.shape[:3], n_t):
            print("❌ -before must match -input in grid and frame count.", file=sys.stderr)
            return 2
        before = correlate(prior)
        del prior
        write(f"{stem}_before{ext}", before, stataux)
        write(f"{stem}_delta{ext}", after - before, None)

    return 0


if __name__ == "__main__":
    sys.exit(main())
