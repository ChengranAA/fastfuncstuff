"""CLI: smooth surface time series along the cortex, BY a FWHM or TO one.

Command: ffs_util_blursurf

Usage:
    ffs_util_blursurf -input run1.lh.func.gii run2.lh.func.gii -do_blur 4 -prefix b4.lh
    ffs_util_blursurf -input run*.lh.func.gii -blur_to_fwhm 6 \
        -matrix design.xmat.1D -mask all.lh.mask.shape.gii -prefix bto6.lh
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg

_DESCRIPTION = """\
SurfSmooth's HEAT_07, the ffs way: heat diffusion ALONG the cortex on the data's own
midthickness (never across a sulcus), inside -mask (its edge is a wall). The same
smoother ffs_reml -do_blur / -blur_to_fwhm uses.

-do_blur F smooths BY a kernel of F mm. -blur_to_fwhm F smooths TO a smoothness
(SurfSmooth -target_fwhm): the heat kernel is chosen so a master measures F mm
(1-difference FWHM along mesh edges), then every run is smoothed with that one
kernel. The master, best first:
  -blurmaster   residual time series you already have, used as given;
  -matrix       the OLS residuals of the input against this design (an AFNI xmat:
                its RunStart and GoodList are honoured, -censor overrides). Blurring
                commutes with that fit, so these are exactly the residuals the
                blurred data will have;
  (neither)     the input, detrended per run (-polort). The task stays in, reads as
                extra smoothness, and the data come out under-blurred.
Only -blur_to_columns time points, spread over the runs and never censored, carry
the search.

-blur_to_local makes it spatially varying (3dBlurToFWHM's method): small heat steps,
each stopping wherever the local smoothness (pooled over -blur_to_nbhd mm) has reached
the target, until the global one has. Rough cortex gets more blur than smooth.

-blur_to_tsnr T blurs each place until its residual tSNR (temporal mean over the
master's SD, pooled over -blur_to_nbhd) reaches T: noisy cortex gets more, clean none,
nothing past -blur_tsnr_max_fwhm, and a place stops once more blur stops buying tSNR.
Cluster inference on such data wants a null that knows the smoothness varies
(ffs_reml -clustsim_null resid).

Inputs are runs of one hemisphere on one mesh. Outputs: one PREFIX.func.gii for a
single input, else PREFIX.rNN.func.gii per run; the metadata records the kernel
(blur_kernel_fwhm) and, for -blur_to_fwhm, the target and the master's measured
FWHM before and after.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(prog="ffs_util_blursurf", description=_DESCRIPTION)
    p.add_argument("-input", nargs="+", required=True, metavar="RUN.func.gii",
                   help="Surface time series, one per run, same hemisphere and mesh.")  # fmt: skip
    p.add_argument("-prefix", required=True, help="Output stem (see above).")
    how = p.add_mutually_exclusive_group(required=True)
    how.add_argument("-do_blur", type=float, metavar="FWHM", default=None,
                     help="Smooth BY a heat kernel of FWHM mm.")  # fmt: skip
    how.add_argument("-blur_to_fwhm", type=float, metavar="FWHM", default=None,
                     help="Smooth TO this FWHM (mm), measured on the master.")  # fmt: skip
    how.add_argument("-blur_to_tsnr", type=float, metavar="TSNR", default=None,
                     help="Smooth each place only until its RESIDUAL tSNR (mean over the "
                     "master's SD) reaches this, up to -blur_tsnr_max_fwhm, and not where "
                     "it stops paying (structured noise does not average down).")  # fmt: skip
    p.add_argument("-blur_tsnr_max_fwhm", type=float, default=8.0, metavar="MM",
                   help="-blur_to_tsnr: no place is smoothed past this local FWHM.")  # fmt: skip
    p.add_argument("-surf_geom", metavar="SURF.gii", default=None,
                   help="The mesh the data live on (this subject's midthickness). Default: "
                   "the 'geometry' named in the input's metadata.")  # fmt: skip
    p.add_argument("-mask", metavar="MASK.shape.gii", default=None,
                   help="Smooth and measure only inside this mask; the rest is copied.")  # fmt: skip
    master = p.add_mutually_exclusive_group()
    master.add_argument("-blurmaster", nargs="+", metavar="RESID.func.gii", default=None,
                        help="-blur_to_fwhm: residual time series to measure, as given.")  # fmt: skip
    master.add_argument("-matrix", metavar="X.xmat.1D", default=None,
                        help="-blur_to_fwhm: measure the OLS residuals against this design.")  # fmt: skip
    p.add_argument("-censor", metavar="CENSOR.1D", default=None,
                   help="With -matrix: 1=keep, 0=censor per concatenated TR (overrides "
                   "the xmat GoodList).")  # fmt: skip
    p.add_argument("-polort", type=int, default=None,
                   help="Detrended master (no -matrix/-blurmaster): Legendre order per run. "
                   "Default: 1 + floor(run seconds / 150) when the TR is known, else 3.")  # fmt: skip
    p.add_argument("-blur_to_columns", type=int, default=None, metavar="N",
                   help="Time points that carry the -blur_to_fwhm search. Default 40, or "
                   "160 with -blur_to_local (a local estimate needs more).")  # fmt: skip
    p.add_argument("-blur_to_local", action="store_true",
                   help="-blur_to_fwhm, spatially varying: diffuse in steps and stop wherever "
                   "the local smoothness has reached the target (3dBlurToFWHM's method).")  # fmt: skip
    p.add_argument("-blur_to_nbhd", type=float, default=8.0, metavar="MM",
                   help="-blur_to_local: neighbourhood (Gaussian FWHM, mm) of each local "
                   "estimate.")  # fmt: skip
    p.add_argument("-save_blur_fwhm", metavar="MAP.shape.gii", default=None,
                   help="-blur_to_fwhm: write the local FWHM of the blurred master.")  # fmt: skip
    add_device_arg(p)
    args = p.parse_args(argv)
    to_target = args.blur_to_fwhm is not None or args.blur_to_tsnr is not None
    if not to_target and (args.blurmaster or args.matrix):
        p.error("-blurmaster / -matrix choose a master for -blur_to_fwhm / -blur_to_tsnr")
    if args.censor and not args.matrix:
        p.error("-censor applies to the -matrix fit")
    if args.blur_to_fwhm is None and args.blur_to_local:
        p.error("-blur_to_local goes with -blur_to_fwhm")
    if not to_target and args.save_blur_fwhm:
        p.error("-save_blur_fwhm goes with -blur_to_fwhm / -blur_to_tsnr")
    if args.blur_to_columns is None:
        varies = args.blur_to_local or args.blur_to_tsnr is not None
        args.blur_to_columns = 160 if varies else 40
    return args


def _outputs(prefix: str, n: int) -> list[str]:
    stem = prefix.removesuffix(".gii").removesuffix(".func")
    return [f"{stem}.func.gii"] if n == 1 else [f"{stem}.r{i + 1:02d}.func.gii" for i in range(n)]


def _master(args, data, lengths, tr) -> np.ndarray:
    from fastfuncstuff.stats.blur_to import residual_master, sample_columns
    from fastfuncstuff.surface.acf import detrend

    if args.blurmaster:
        from fastfuncstuff.io.gifti import load_gifti_data

        bm = [np.atleast_2d(load_gifti_data(p)[0].T).T for p in args.blurmaster]
        cols = sample_columns([b.shape[1] for b in bm], args.blur_to_columns)
        print(f"  master: -blurmaster, {len(cols)} time points")
        return np.concatenate(bm, axis=1)[:, cols]
    if args.matrix:
        from fastfuncstuff.io.afni import read_afni_design_matrix, read_censor_1d

        info = read_afni_design_matrix(args.matrix)
        x = np.asarray(info["matrix"])
        if x.shape[0] != data.shape[1]:
            raise SystemExit(
                f"ffs_util_blursurf: -matrix has {x.shape[0]} rows, the input "
                f"{data.shape[1]} time points"
            )
        good = info.get("good_list")
        if args.censor:
            good = read_censor_1d(args.censor, n_expected=data.shape[1], run_lengths_tr=lengths)
        cols = sample_columns(lengths, args.blur_to_columns, good=good)
        n_cens = 0 if good is None else data.shape[1] - len(good)
        print(f"  master: OLS residuals ({x.shape[1]} regressors, {n_cens} TRs censored), "
              f"{len(cols)} time points")  # fmt: skip
        return residual_master(data, x, cols, good=good)
    cols = sample_columns(lengths, args.blur_to_columns)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    out = []
    for s, n in zip(starts, lengths, strict=True):
        deg = args.polort
        if deg is None:
            deg = 1 + int(n * tr // 150) if tr else 3
        mine = cols[(cols >= s) & (cols < s + n)]
        out.append(detrend(data[:, s : s + n], degree=deg)[:, mine - s])
    print(
        f"  master: the input detrended per run, {len(cols)} time points (task signal "
        "stays in: -matrix or -blurmaster blur more accurately)"
    )
    return np.concatenate(out, axis=1)


def main(argv: list[str] | None = None) -> int:
    from fastfuncstuff.cli_utils import setup_device
    from fastfuncstuff.io.afni import load_afni_mask
    from fastfuncstuff.io.gifti import data_geometry, load_gifti_data, save_gifti_data
    from fastfuncstuff.surface.acf import (
        blur_to_fwhm,
        blur_to_fwhm_local,
        blur_to_tsnr,
        local_surface_fwhm,
    )
    from fastfuncstuff.surface.smooth import HeatSmoother, surface_fwhm

    args = parse_args(argv)
    device = setup_device(args.device)
    solve_on = device if device.type == "cuda" else None  # CPU: the exact direct solve

    loaded = [load_gifti_data(p) for p in args.input]
    runs = [np.atleast_2d(d.T).T for d, _ in loaded]
    meta0 = loaded[0][1]
    if (
        len({r.shape[0] for r in runs}) > 1
        or len({m.get("mesh_fingerprint") for _, m in loaded}) > 1
    ):
        raise SystemExit("ffs_util_blursurf: the inputs are not all on one mesh")
    n_v = runs[0].shape[0]
    try:
        vertices, faces = data_geometry(args.input[0], meta0, n_v, args.surf_geom)
    except ValueError as e:
        raise SystemExit(f"ffs_util_blursurf: {e}") from None
    mask = load_afni_mask(args.mask).reshape(-1).astype(bool) if args.mask else None
    lengths = [r.shape[1] for r in runs]
    tr = float(meta0["TR_seconds"]) if meta0.get("TR_seconds") else None
    print(f"ffs_util_blursurf: {len(runs)} run(s), {n_v:,} vertices, {sum(lengths)} time points")

    record: dict[str, str] = {}
    if args.do_blur is not None:
        hs = HeatSmoother(vertices, faces, args.do_blur, mask=mask, device=solve_on)
        record["blur_kernel_fwhm"] = f"{args.do_blur:g}"
    else:
        data = np.concatenate(runs, axis=1)
        master = _master(args, data, lengths, tr)
        mean = data.mean(axis=1, dtype=np.float64)
        del data
        start = surface_fwhm(master, vertices, faces, mask)[0]
        local = None
        if args.blur_to_tsnr is not None:
            hs, t0, t1, local = blur_to_tsnr(
                master, mean, vertices, faces, args.blur_to_tsnr, args.blur_tsnr_max_fwhm,
                mask, args.blur_to_nbhd, device=solve_on,
            )  # fmt: skip
            achieved = surface_fwhm(master if hs is None else hs(master), vertices, faces, mask)[0]
            ok = np.isfinite(t1)
            reach = 100 * np.mean(t1[ok] >= args.blur_to_tsnr)
            how = (
                f"tSNR-driven diffusion (median tSNR {np.nanmedian(t0):.1f} -> "
                f"{np.nanmedian(t1):.1f}, {reach:.0f}% reach {args.blur_to_tsnr:g})"
            )
            record.update(
                blur_to_tsnr=f"{args.blur_to_tsnr:g}",
                blur_tsnr_max_fwhm=f"{args.blur_tsnr_max_fwhm:g}",
                tsnr_median_before=f"{np.nanmedian(t0):.4g}",
                tsnr_median_after=f"{np.nanmedian(t1):.4g}",
                blur_steps=str(0 if hs is None else len(hs.steps)),
            )
        elif args.blur_to_local:
            hs, _, achieved, local = blur_to_fwhm_local(
                master, vertices, faces, args.blur_to_fwhm, mask, args.blur_to_nbhd,
                device=solve_on,
            )  # fmt: skip
            how = "spatially varying diffusion"
            if hs is not None:
                record["blur_steps"] = str(len(hs.steps))
                record["blur_nbhd"] = f"{args.blur_to_nbhd:g}"
        else:
            hs, kernel, achieved = blur_to_fwhm(
                master, vertices, faces, args.blur_to_fwhm, mask,
                n_columns=master.shape[1], device=solve_on,
            )  # fmt: skip
            record["blur_kernel_fwhm"] = f"{kernel:.4g}"
            how = f"heat kernel {kernel:.2f} mm"
        record.update(master_fwhm_before=f"{start:.4g}", master_fwhm_after=f"{achieved:.4g}")
        if args.blur_to_fwhm is not None:
            record["blur_to_fwhm"] = f"{args.blur_to_fwhm:g}"
        if hs is None:
            print(f"  master already at the target ({start:.2f} mm smooth): no blur")
        else:
            print(f"  {how} takes the master {start:.2f} -> {achieved:.2f} mm")
        if args.save_blur_fwhm:
            if local is None:
                after = master if hs is None else hs(master)
                local = local_surface_fwhm(after, vertices, faces, mask, args.blur_to_nbhd)
            keep = {k: v for k, v in meta0.items() if k in ("mesh_fingerprint",)}
            save_gifti_data(
                args.save_blur_fwhm, np.nan_to_num(local).astype(np.float32), keep,
                time_series=False,
            )  # fmt: skip
            print(f"  -> {Path(args.save_blur_fwhm).name} (local FWHM after blurring)")

    for run, (_, meta), src, out in zip(
        runs, loaded, args.input, _outputs(args.prefix, len(runs)), strict=True
    ):
        smoothed = run if hs is None else hs(run)
        meta = {**meta, **record}
        if meta.get("geometry") and args.surf_geom is None:
            # 'geometry' is relative to the data file: keep it pointing at the same mesh.
            geom = Path(src).parent / meta["geometry"]
            meta["geometry"] = os.path.relpath(geom, Path(out).parent)
        save_gifti_data(out, np.asarray(smoothed, np.float32), meta)
        print(f"  -> {Path(out).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
