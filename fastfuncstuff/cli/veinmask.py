"""ffs_veinmask — vein candidates from an EPI mean and anat tissue fractions.

For ffs_phasereg -spr -spr_mask when there is no GRE/SWI: see
:mod:`fastfuncstuff.processing.veinmask` for the model.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(
        prog="ffs_veinmask",
        description=(
            "Vein candidates for ffs_phasereg -spr_mask, from the data you already have. "
            "The anat's GM/WM/CSF fractions times a smooth bias field predict the EPI "
            "mean; voxels in cortex (or in CSF touching it) that come out much darker "
            "than predicted are vein candidates. A vein darkens at the EPI's echo time; "
            "white matter and tissue boundaries, which a plain darkness map also flags, "
            "are predicted by the tissue model and so are not. Outputs (PREFIX_): mask "
            "(the -spr_mask), ratio (EPI / expected; 1 = as the tissue predicts), "
            "expected (the predicted EPI image), region (where veins are looked for)."
        ),
    )
    p.add_argument(
        "-epi",
        nargs="+",
        required=True,
        help="EPI mean(s) or runs (4-D: the temporal mean is taken), averaged. Use the "
        "runs phase regression will correct: a spin-echo run has no vein darkening",
    )
    p.add_argument(
        "-tissue",
        nargs="+",
        required=True,
        help="GM, WM, CSF fractions: one 4-D file whose first three volumes are GM, WM, "
        "CSF (SPM c1/c2/c3 order, a subject TPM), or three 3-D files in that order. Any "
        "grid: resampled onto the EPI's through the headers (smoothed to the EPI voxel "
        "size first when finer, so the fractions stay partial-volume fractions)",
    )
    p.add_argument("-prefix", required=True, help="output prefix")
    p.add_argument(
        "-mask",
        help="brain mask on the EPI grid (default: EPI above 20%% of its 98th percentile)",
    )
    p.add_argument(
        "-darker",
        type=float,
        default=0.3,
        help="flag voxels this much darker than predicted (0.3 = 30%% darker). On "
        "ds003427 (7T, 0.8 mm, TE 23 ms) 0.3 marked 7-10%% of cortex+pial CSF, the share "
        "the GRE vein masks it was validated against marked",
    )
    p.add_argument(
        "-dilate",
        type=int,
        default=1,
        help="grow the mask this many voxels (6-connected): residual EPI-anat "
        "misregistration, and the vein's field reaching its neighbours",
    )
    p.add_argument(
        "-bias_fwhm",
        type=float,
        default=24.0,
        help="FWHM (mm) of the bias field. Smaller follows coil shading more closely "
        "but starts to absorb clusters of veins into the bias",
    )
    p.add_argument(
        "-gm_min",
        type=float,
        default=0.3,
        help="GM fraction from which a voxel counts as cortex (the search region also "
        "takes CSF-dominant voxels touching GM >= 0.5: the pial veins)",
    )
    add_device_arg(p)
    add_verbose_arg(p)
    return p.parse_args(argv)


def _tissue_on_grid(paths, epi_shape, epi_affine, device) -> torch.Tensor:
    from fastfuncstuff.processing.grid import resample_to_grid
    from fastfuncstuff.processing.io import load_image

    if len(paths) == 1:
        t, h = load_image(paths[0], device=device)
        if t.ndim != 4 or t.shape[0] < 3:
            raise SystemExit("ffs_veinmask: a single -tissue file needs >= 3 volumes (GM, WM, CSF)")
        t = t[:3]
    elif len(paths) == 3:
        vols = [load_image(q, device=device) for q in paths]
        t, h = torch.stack([v[0] for v in vols]), vols[0][1]
    else:
        raise SystemExit("ffs_veinmask: -tissue takes one 4-D file or three 3-D files")
    t = t.float()
    src_aff = np.asarray(h["affine"], np.float64)
    if t.shape[1:] == epi_shape and np.allclose(src_aff, epi_affine, atol=1e-4):
        return t
    # Finer tissue maps are blurred to the EPI voxel first: point sampling a 0.3 mm
    # map at 0.8 mm spacing would turn fractions back into near-binary labels.
    from fastfuncstuff.processing.veinmask import masked_blur

    d_src = np.linalg.norm(src_aff[:3, :3], axis=0)
    d_epi = float(np.cbrt(abs(np.linalg.det(epi_affine[:3, :3]))))
    sig = [math.sqrt(max(d_epi**2 - d**2, 0.0)) / 2.3548 / d for d in d_src]
    one = torch.ones(t.shape[1:], device=t.device)
    t = torch.stack([masked_blur(v, one, (sig[2], sig[1], sig[0])) for v in t])
    return resample_to_grid(t, src_aff, epi_shape, epi_affine, interp="linear")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = setup_device(args.device)
    t0 = time.time()
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.veinmask import tissue_darkness, vein_mask, vein_region

    with spinner("Reading the EPI", enabled=args.verb >= 1):
        means, header = [], None
        for path in args.epi:
            v, h = load_image(path, device=device)
            means.append(v.float().mean(0) if v.ndim == 4 else v.float())
            header = header or h
        assert header is not None
        mean = torch.stack(means).mean(0)
    epi_aff = np.asarray(header["affine"], np.float64)
    shape = tuple(mean.shape)
    with spinner("Tissue fractions onto the EPI grid", enabled=args.verb >= 1):
        tissue = _tissue_on_grid(args.tissue, shape, epi_aff, device)
    if args.mask:
        brain = load_image(args.mask, device=device)[0] > 0.5
    else:
        brain = mean > 0.2 * torch.quantile(mean.flatten()[::7], 0.98)
    brain &= tissue.sum(0) > 0.5

    zooms = header["header"].get_zooms()[:3]  # x, y, z; the volume is (z, y, x)
    sig = tuple(args.bias_fwhm / 2.3548 / float(z) for z in reversed(zooms))
    ratio, expected, levels = tissue_darkness(mean, tissue, brain, sig)
    region = vein_region(tissue, brain, gm_min=args.gm_min)
    core = region & (ratio < 1.0 - args.darker)
    mask = vein_mask(ratio, region, args.darker, args.dilate)

    if args.verb >= 1:
        lv = levels.tolist()
        print(f"  tissue levels before bias: GM {lv[0]:.0f}, WM {lv[1]:.0f}, CSF {lv[2]:.0f}")
        print(
            f"  region {int(region.sum())} voxels; {int(core.sum())} "
            f"({float(core.sum()) / max(int(region.sum()), 1):.1%}) > {args.darker:g} darker "
            f"than predicted; mask after dilation {int(mask.sum())}"
        )
    for name, vol in (
        ("mask", mask.to(torch.uint8)),
        ("ratio", ratio),
        ("expected", expected),
        ("region", region.to(torch.uint8)),
    ):
        save_image(vol, f"{args.prefix}_{name}.nii.gz", header_info=header)
    if args.verb >= 1:
        print(
            f"  wrote {args.prefix}_{{mask,ratio,expected,region}}.nii.gz ({time.time() - t0:.1f}s)"
        )


if __name__ == "__main__":
    main()
