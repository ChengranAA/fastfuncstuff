"""Remove per-slice dynamic field fluctuation from a native-space phase series.

Command: ffs_phasedrift (registered as entry point in pyproject.toml)

Breathing and scanner frequency drift move the B0 field of the whole head from
moment to moment. In 2-D multi-slice EPI every slice is read at a different
moment, so one volume carries as many field states as acquisition times, and
the fluctuation shows up as a per-slice phase offset (plus a gentle in-plane
gradient) that changes every TR. It has no magnitude counterpart, so to phase
regression it is pure regressor noise. Vessel phase is a local dipole field,
which a low-order in-plane fit cannot absorb.

Run it in NATIVE space, before slice-timing correction and resampling: after
those the slices are mixed and the per-slice states can no longer be separated.

Usage:
    ffs_phasedrift -input phase_unwrapped.nii.gz -magnitude mag.nii.gz \\
                   -prefix phase_clean.nii.gz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_verbose_arg, spinner

_AXES = {"i": 0, "j": 1, "k": 2}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_phasedrift",
        description="Remove per-slice (or per-volume) dynamic field fluctuation from a "
        "native-space phase time series, ahead of phase regression.",
        epilog=__doc__.split("Usage:")[0].split("\n\n", 1)[1],
    )
    parser.add_argument("-input", required=True, help="Phase series, radians (wrapped or not)")
    parser.add_argument("-prefix", required=True, help="Output phase series")
    parser.add_argument(
        "-magnitude",
        default=None,
        help="Matching magnitude series; its automask chooses the voxels the fit "
        "is estimated on. Without it (or -mask) every voxel with non-zero phase is used.",
    )
    parser.add_argument("-mask", default=None, help="Fit mask (overrides -magnitude)")
    parser.add_argument(
        "-order",
        type=int,
        default=1,
        metavar="N",
        help="In-plane Legendre polynomial degree fit per slice per volume: 0 = "
        "offset only, 1 = offset + in-plane gradients (default), 2 adds curvature. "
        "Higher orders begin to reach local structure; keep it low with partial "
        "coverage.",
    )
    parser.add_argument(
        "-by",
        choices=["slice", "volume"],
        default="slice",
        help="'slice' (default): one fit per acquired slice. 'volume': one fit per "
        "volume over the whole head (3-D polynomial) — for 3-D EPI, or data whose "
        "slices are no longer the acquired ones.",
    )
    parser.add_argument(
        "-slice_axis",
        choices=["auto", "i", "j", "k"],
        default="auto",
        help="Acquired slice axis. 'auto' reads the NIfTI slice_dim, else k.",
    )
    parser.add_argument(
        "-no_tunwrap",
        action="store_true",
        help="Skip the temporal unwrap (about each voxel's circular mean) that "
        "runs first. Harmless on unwrapped input, required on wrapped input.",
    )
    parser.add_argument(
        "-save_trace",
        default=None,
        metavar="FILE.1D",
        help="Write the removed term averaged over each slice's fit voxels, one "
        "column per slice (one column with -by volume). Breathing shows up here.",
    )
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from fastfuncstuff.phasereg.conditioning import remove_global_phase, temporal_unwrap
    from fastfuncstuff.processing.io import load_image, save_image

    with spinner("Loading phase", enabled=args.verb >= 1):
        phase, hdr = load_image(args.input)
    if phase.ndim != 4:
        print(
            f"ERROR: expected a 4-D phase series, got shape {tuple(phase.shape)}", file=sys.stderr
        )
        return 1
    n_t, sz, sy, sx = phase.shape  # load_image order: (t, k, j, i)

    if args.mask:
        fit3d = load_image(args.mask)[0] > 0
    elif args.magnitude:
        from fastfuncstuff.processing.mask import automask

        mag = load_image(args.magnitude)[0]
        if mag.shape != phase.shape:
            print("ERROR: -magnitude shape does not match -input", file=sys.stderr)
            return 1
        fit3d = automask(mag.mean(0), device=torch.device("cpu"))
        del mag
    else:
        fit3d = (phase != 0).any(0)
    if tuple(fit3d.shape) != (sz, sy, sx):
        print("ERROR: mask grid does not match -input", file=sys.stderr)
        return 1

    if args.slice_axis == "auto":
        dim = hdr["header"].get_dim_info()[2]
        slice_nib = 2 if dim is None else int(dim)
    else:
        slice_nib = _AXES[args.slice_axis]
    slice_dim = 2 - slice_nib  # nibabel axis -> index into (k, j, i)

    grid = torch.stack(
        torch.meshgrid(torch.arange(sz), torch.arange(sy), torch.arange(sx), indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    p = phase.reshape(n_t, -1)
    del phase
    if not args.no_tunwrap:
        p = temporal_unwrap(p)

    fit_mask = fit3d.reshape(-1).bool()
    if args.by == "slice":
        groups = grid[:, slice_dim]
        coords = grid[:, [d for d in range(3) if d != slice_dim]]
    else:
        groups, coords = None, grid
    with spinner(f"Fitting order-{args.order} {args.by} terms", enabled=args.verb >= 1):
        cleaned, trace = remove_global_phase(
            p, args.order, fit_mask, coords if args.order >= 1 else None, groups=groups
        )
    if args.verb >= 1:
        before = p[:, fit_mask].std(0).median().item()
        after = cleaned[:, fit_mask].std(0).median().item()
        print(f"ffs_phasedrift: median temporal phase std {before:.4f} -> {after:.4f} rad")
    del p

    with spinner("Writing", enabled=args.verb >= 1):
        save_image(cleaned.reshape(n_t, sz, sy, sx), args.prefix, header_info=hdr)
    if args.save_trace:
        tr2d = trace.reshape(n_t, -1).numpy()
        np.savetxt(args.save_trace, tr2d, fmt="%.6f")
    return 0


if __name__ == "__main__":
    sys.exit(main())
