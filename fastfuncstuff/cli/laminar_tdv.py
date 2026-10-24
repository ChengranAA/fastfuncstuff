"""ffs_laminar_tdv: continuous model-based removal of ascending-vein transport."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import (
    add_device_arg,
    add_verbose_arg,
    parse_input_files,
    parse_prefix,
    setup_device,
)


def create_parser():
    p = FfsArgumentParser(
        prog="ffs_laminar_tdv",
        description=__doc__,
        epilog="""Example:
  ffs_laminar_tdv -input ge_run*.nii.gz -depth depth.nii.gz -gmfrac gmfrac.nii.gz \\
    -depth_order pial_to_white -repeated_runs -prefix tdv

Fixed vascular physiology by default; this is a linear approximation about rest.
Multiple-strength LORO selection requires runs with identical task schedules.
For other data (including rest) supply a single -strengths value.
SE is an independent validation target; never use it to tune the correction.
Segmentation-only depth maps cannot resolve touching gyri or true subvoxel PSFs.
""",
    )
    p.add_argument("-input", nargs="+", required=True)
    p.add_argument("-depth", required=True, help="Aligned scalar depth map")
    p.add_argument("-depth_order", required=True, choices=("pial_to_white", "white_to_pial"))
    p.add_argument(
        "-gmfrac", required=True, help="Aligned GM fraction; keeps depth=0 background out"
    )
    p.add_argument("-gm_threshold", type=float, default=0.5)
    p.add_argument("-columns", help="Optional aligned integer column/patch labels (0 background)")
    p.add_argument(
        "-patch_mm", type=float, default=5.0, help="GM geodesic reach when columns are absent"
    )
    p.add_argument(
        "-depths",
        type=int,
        default=5,
        help="Vascular CBF input depths; all must be sampled in each patch",
    )
    p.add_argument(
        "-repeated_runs",
        action="store_true",
        help="Assert identical task schedules for GE LORO selection",
    )
    p.add_argument("-strengths", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0])
    p.add_argument(
        "-target",
        choices=("local", "micro"),
        default="local",
        help="Retain local response (incl. own AV) or venules only",
    )
    p.add_argument(
        "-transfer_mode",
        choices=("dynamic", "static"),
        default="dynamic",
        help="Static holds the DC mixing at every frequency, as a control",
    )
    p.add_argument(
        "-vascular_json",
        help="JSON with p0 baseline overrides and scalar log-deviations under parameters",
    )
    p.add_argument("-tr", "-TR", type=float, help="TR seconds (otherwise input header)")
    p.add_argument(
        "-te", type=float, default=0.028, help="GE echo time seconds; set from acquisition sidecar"
    )
    p.add_argument("-b0", type=float, default=7.0)
    p.add_argument("-pad_seconds", type=float, default=60.0)
    p.add_argument(
        "-cutoff_hz",
        type=float,
        default=0.1,
        help="Frequency where temporal inverse penalty starts rising",
    )
    p.add_argument("-polort", type=int, default=2)
    p.add_argument("-prefix", required=True)
    add_device_arg(p)
    add_verbose_arg(p)
    return p


def main():
    args = create_parser().parse_args()
    if args.depths < 2 or args.gm_threshold <= 0 or args.gm_threshold > 1:
        raise SystemExit("Need -depths >=2 and 0 < -gm_threshold <= 1")
    if len(args.strengths) > 1 and not args.repeated_runs:
        raise SystemExit(
            "Use -repeated_runs for identical schedules, or supply one -strengths value"
        )
    from fastfuncstuff.laminar.params import ModelSpec, zero_params
    from fastfuncstuff.laminar.pipeline import reference_p0
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.laminar_tdv import depth_patches, devein_volume_runs

    device = setup_device(args.device)
    files = parse_input_files(args.input)
    pfx = parse_prefix(args.prefix)
    Path(pfx.stem).parent.mkdir(parents=True, exist_ok=True)
    depth, header = load_image(args.depth)
    frac, fh = load_image(args.gmfrac)
    if (
        depth.ndim != 3
        or frac.shape != depth.shape
        or not np.allclose(header["affine"], fh["affine"], atol=1e-4)
    ):
        raise SystemExit("Depth and GM fraction must share a 3D grid")
    mask = (
        (frac.numpy() >= args.gm_threshold)
        & np.isfinite(depth.numpy())
        & (depth.numpy() >= 0)
        & (depth.numpy() <= 1)
    )
    data, heads = [], []
    for path in files:
        d, h = load_image(path)
        if (
            d.ndim != 4
            or d.shape[1:] != depth.shape
            or not np.allclose(header["affine"], h["affine"], atol=1e-4)
        ):
            raise SystemExit(f"{path}: input must share the depth map's grid")
        mask &= torch.isfinite(d).all(0).numpy() & (d.amin(0).numpy() > 0)
        data.append(d)
        heads.append(h)
    tr = args.tr or float(heads[0]["header"].get_zooms()[3])
    if args.tr is None and any(abs(float(h["header"].get_zooms()[3]) - tr) > 1e-5 for h in heads):
        raise SystemExit("Input TRs differ")
    if args.depth_order == "white_to_pial":
        depth = 1 - depth
    if args.columns:
        cols, ch = load_image(args.columns)
        if cols.shape != depth.shape or not np.allclose(header["affine"], ch["affine"], atol=1e-4):
            raise SystemExit("Column labels must share the depth grid")
        if not torch.isfinite(cols).all() or not torch.equal(cols, cols.round()):
            raise SystemExit("Column labels must be finite integers")
        mask &= cols.numpy() > 0
        labels = cols.numpy()[mask].astype(np.int64)
    else:
        spacing = np.sqrt(np.square(header["affine"][:3, :3]).sum(0))[::-1]
        labels = depth_patches(mask, depth.numpy(), spacing, args.patch_mm)
    if not mask.any():
        raise SystemExit("No covered GM voxels")
    baselines = reference_p0()
    overrides = json.loads(Path(args.vascular_json).read_text()) if args.vascular_json else {}
    baselines = replace(baselines, **overrides.get("p0", {}))
    spec = ModelSpec(
        N=args.depths,
        K=args.depths,
        TR=tr,
        TE=args.te,
        B0=args.b0,
        p0=baselines,
        device=device,
        dtype=torch.float32,
    )
    params = zero_params(spec)
    for name, val in overrides.get("parameters", {}).items():
        if name not in params or params[name].ndim:
            raise SystemExit(f"Only scalar vascular parameter deviations are supported: {name}")
        params[name] = torch.as_tensor(val, device=device, dtype=spec.dtype)
    runs = [d[:, mask].T.contiguous() for d in data]
    if args.verb:
        print(
            f"Continuous {args.target} BOLD: {len(runs)} runs, {len(labels):,} voxels, {len(np.unique(labels)):,} patches; fixed vascular physiology"
        )
    result = devein_volume_runs(
        runs,
        depth.numpy()[mask],
        labels,
        spec,
        params=params,
        target=args.target,
        transfer_mode=args.transfer_mode,
        strengths=tuple(args.strengths),
        pad_seconds=args.pad_seconds,
        cutoff_hz=args.cutoff_hz,
        polort=args.polort,
        verbose=args.verb > 0,
    )
    ext = pfx.nifti_ext
    for i, (d, h, corrected, removed, smooth) in enumerate(
        zip(data, heads, result.corrected, result.removed_psc, result.smooth_psc, strict=True), 1
    ):
        d[:, mask] = corrected.T
        save_image(d, f"{pfx.stem}_corrected_run-{i:02d}{ext}", h)
        for suffix, arr in (("removed_psc", removed), ("smooth_psc", smooth)):
            out = torch.zeros_like(d)
            out[:, mask] = arr.T
            save_image(out, f"{pfx.stem}_{suffix}_run-{i:02d}{ext}", h)
    diagnostic = torch.zeros(5, *depth.shape)
    for i, values in enumerate(
        (result.regularization, result.cv_r2, result.noise_gain, result.valid, labels + 1)
    ):
        diagnostic[i][mask] = torch.as_tensor(values, dtype=torch.float32)
    save_image(
        diagnostic,
        f"{pfx.stem}_diagnostics{ext}",
        header,
        brick_labels=["regularization", "ge_cv_r2", "noise_gain", "valid", "patch"],
    )
    meta = dict(
        vars(args),
        input_files=files,
        actual_tr=tr,
        p0=asdict(baselines),
        n_valid=int(result.valid.sum()),
        n_gm=int(mask.sum()),
        depth_convention="0=pial, 1=white",
        calibration="fixed physiology; GE-only LORO regularization",
        boundary_exclude_seconds=args.pad_seconds,
    )
    Path(f"{pfx.stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    if args.verb:
        print(
            f"Corrected {int(result.valid.sum()):,}/{int(mask.sum()):,} GM voxels; wrote {pfx.stem}_corrected_run-* and diagnostics"
        )


if __name__ == "__main__":
    main()
