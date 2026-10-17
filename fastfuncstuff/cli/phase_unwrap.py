"""Unwrap a phase time series (one or more echoes) against one ROMEO'd reference, on GPU.

Command: ffs_phase_unwrap (registered as entry point in pyproject.toml)

ROMEO's region growing is sequential, so unwrapping every frame spatially is what
makes MEDIC (warpkit) and ``romeo -t epi`` slow. Neighbouring frames differ by a
small field change, so this unwraps ONE reference spatially (the temporal-mean
complex image, its 2π counts and phase offset checked against a few ROMEO'd
frames) and anchors every frame to it, per voxel, on the GPU. Voxels whose change
really leaves the shortest echo's unambiguous range (motion through a steep field,
drift) are followed along time instead. Multi-echo: each later echo's 2π count
comes from the earlier echoes' fit, and a field map in Hz is written too.

Outputs are in radians. Multi-echo phase has the phase offset (MCPC-3D-S) removed,
like warpkit's; a single echo keeps it, like ROMEO's.

Usage:
    ffs_phase_unwrap -phase ph.nii.gz -magnitude mag.nii.gz -prefix ph_uw.nii.gz
    ffs_phase_unwrap -phase p1 p2 p3 -magnitude m1 m2 m3 -echo_times 7.6 21.7 35.8 \\
                     -prefix uw.nii.gz        # -> uw_e1/2/3 + uw_fieldmap
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, spinner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_phase_unwrap",
        description="GPU phase unwrapping of a whole time series against one ROMEO'd "
        "reference (single- or multi-echo).",
        epilog=__doc__.split("Usage:")[0].split("\n\n", 1)[1],
    )
    parser.add_argument(
        "-phase",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Wrapped phase series, one 4-D file per echo, shortest echo first. Raw "
        "scanner units or radians: the range is mapped onto [-pi, pi) (warpkit's rule).",
    )
    parser.add_argument(
        "-magnitude", nargs="+", required=True, metavar="FILE", help="Matching magnitudes."
    )
    parser.add_argument(
        "-echo_times",
        nargs="+",
        type=float,
        default=None,
        metavar="MS",
        help="Echo times in ms. Required with 2+ echoes; with one echo it only labels "
        "the period of the field it would correspond to.",
    )
    parser.add_argument(
        "-prefix",
        required=True,
        help="Output. One echo: the unwrapped phase itself. Several: {stem}_eN per echo "
        "plus {stem}_fieldmap (Hz, unfiltered).",
    )
    parser.add_argument(
        "-anchor_frames",
        type=int,
        default=8,
        metavar="N",
        help="Frames also ROMEO'd to vote on the reference's 2pi counts (and, multi-echo, "
        "its phase offset). 0 = the temporal mean alone; each one costs a ROMEO run.",
    )
    parser.add_argument(
        "-no_temporal_path",
        action="store_true",
        help="Never follow a voxel's change along time: every frame stays within half "
        "a period of the reference (romeo -t epi's behaviour).",
    )
    parser.add_argument(
        "-save_mask", default=None, metavar="FILE", help="Write the mask (2 core, 1 border)."
    )
    parser.add_argument(
        "-save_path_map",
        default=None,
        metavar="FILE",
        help="Write the voxels whose change was followed along time (1) -- where the "
        "field moved by more than half the shortest echo's period.",
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def _error(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ne = len(args.phase)
    if len(args.magnitude) != ne:
        return _error(f"{ne} -phase files but {len(args.magnitude)} -magnitude files.")
    if ne >= 2 and args.echo_times is None:
        return _error("-echo_times is required with more than one echo.")
    if args.echo_times is not None and len(args.echo_times) != ne:
        return _error(f"{len(args.echo_times)} -echo_times for {ne} echoes.")
    if args.echo_times is not None and any(np.diff(args.echo_times) <= 0):
        return _error("-echo_times must increase (shortest echo first).")
    if args.anchor_frames < 0:
        return _error("-anchor_frames must be >= 0.")

    import torch

    from fastfuncstuff.cli_utils import parse_prefix, setup_device
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.phase_unwrap import raw_phase_to_radians, unwrap_timeseries

    device = setup_device(args.device)
    verbose = args.verb >= 1
    hdr = None
    vols: dict[str, list[np.ndarray]] = {"phase": [], "magnitude": []}
    for kind in ("phase", "magnitude"):
        for path in getattr(args, kind):
            with spinner(f"Loading {path}", enabled=verbose):
                data, h = load_image(path)
            if data.ndim != 4:
                return _error(f"{path}: expected a 4-D series, got shape {tuple(data.shape)}")
            vols[kind].append(data.numpy())
            hdr = hdr or h
    shapes = {v.shape for v in vols["phase"] + vols["magnitude"]}
    if len(shapes) != 1:
        return _error(f"all inputs must share one grid and length, got {sorted(shapes)}")
    # load_image gives (t, z, y, x); the unwrap works on an (x, y, z, e, t) VIEW of the
    # stacked (t, e, z, y, x) block, streaming slabs along z (slowest spatial axis).
    phase = np.stack(vols["phase"], 1).transpose(4, 3, 2, 1, 0)
    mag = np.stack(vols["magnitude"], 1).transpose(4, 3, 2, 1, 0)
    del vols
    phase = raw_phase_to_radians(phase)
    tes = args.echo_times if args.echo_times is not None else [1.0]
    res = unwrap_timeseries(
        phase,
        mag,
        tes,
        device=device,
        temporal_path=not args.no_temporal_path,
        anchor_frames=args.anchor_frames,
        return_unwrapped=True,
        verbose=verbose,
    )
    del phase, mag
    assert res.unwrapped is not None

    def tkji(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a.transpose(*range(a.ndim)[::-1])))

    pfx = parse_prefix(args.prefix)
    with spinner("Writing", enabled=verbose):
        if ne == 1:
            save_image(tkji(res.unwrapped[..., 0, :]), args.prefix, header_info=hdr)
        else:
            for e in range(ne):
                out = f"{pfx.stem}_e{e + 1}{pfx.nifti_ext}"
                save_image(tkji(res.unwrapped[..., e, :]), out, header_info=hdr)
            save_image(tkji(res.field_hz), f"{pfx.stem}_fieldmap{pfx.nifti_ext}", header_info=hdr)
        if args.save_mask:
            save_image(tkji(res.mask.astype(np.float32)), args.save_mask, header_info=hdr)
        if args.save_path_map:
            save_image(
                tkji(res.temporal_path.astype(np.float32)), args.save_path_map, header_info=hdr
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
