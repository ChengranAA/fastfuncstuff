"""ffs_megre — multi-echo GRE anatomy: T2*/R2*, S0, field, SWI and vein contrast.

Builds on ffs_t2smap's fit (one 3D volume per echo) and ROMEO's multi-echo
unwrap; see :mod:`fastfuncstuff.processing.megre` for the image-space steps.
"""

from __future__ import annotations

import argparse
import time

import torch

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = FfsArgumentParser(
        prog="ffs_megre",
        description=(
            "Multi-echo GRE anatomy. From the magnitude echoes: T2*, R2*, S0, the "
            "T2*-weighted echo combination (an alignment target with EPI-like "
            "contrast) and R2*·S0, the vein contrast of Stanley et al. 2021. With the "
            "phase echoes too: ROMEO's B0 field (Hz), the local field after a masked "
            "high-pass, and SWI with its minimum-intensity projection."
        ),
    )
    p.add_argument("-mag", nargs="+", required=True, help="magnitude echoes, ascending TE")
    p.add_argument("-phase", nargs="+", help="phase echoes, same order (optional)")
    p.add_argument(
        "-tes",
        nargs="+",
        type=float,
        help="echo times in s or ms (default: EchoTime from the BIDS sidecars)",
    )
    p.add_argument("-prefix", required=True, help="output prefix: PREFIX_T2star.nii.gz, ...")
    p.add_argument("-mask", help="brain mask (default: 3dAutomask of the first echo)")
    p.add_argument(
        "-fittype",
        choices=("loglin", "curvefit"),
        default="loglin",
        help="T2*/S0 fit: log-linear, or nonlinear least squares started from it",
    )
    p.add_argument(
        "-hp_fwhm",
        type=float,
        default=4.0,
        help="FWHM (mm) of the high-pass that removes the background field. Larger "
        "keeps more of the local field, and more background with it",
    )
    p.add_argument(
        "-swi_sign",
        type=int,
        choices=(-1, 1),
        default=-1,
        help="sign of venous phase in this data (-1: Haacke's convention). If the "
        "veins do not darken in the SWI, flip it",
    )
    p.add_argument("-swi_power", type=int, default=4, help="phase-mask exponent")
    p.add_argument(
        "-mip_mm",
        type=float,
        default=8.0,
        help="slab thickness (mm) of the sliding minimum-intensity projection",
    )
    p.add_argument("-romeo", default="romeo", help="ROMEO binary (MRItools)")
    add_device_arg(p)
    add_verbose_arg(p)
    return p.parse_args(argv)


def _echo_times_ms(paths: list[str], given: list[float] | None) -> list[float]:
    from fastfuncstuff.processing.b0fmap import read_echo_times

    if given:
        return [t * 1000.0 if max(given) < 1.0 else t for t in given]
    tes = read_echo_times(paths)
    if tes is None:
        raise SystemExit("no EchoTime in the sidecars; pass -tes")
    return tes


def _sorted_by_te(paths: list[str], tes: list[float]) -> tuple[list[str], list[float]]:
    """Echoes in ascending TE. Filenames are not trusted for the order: ds003427
    sub-05's echo-N labels follow neither TE nor each other between mag and phase."""
    order = sorted(range(len(paths)), key=lambda i: tes[i])
    return [paths[i] for i in order], [tes[i] for i in order]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from fastfuncstuff.processing import megre
    from fastfuncstuff.processing import multiecho as me
    from fastfuncstuff.processing.io import load_image, save_image
    from fastfuncstuff.processing.mask import automask

    device = setup_device(args.device)
    verb = args.verb
    t0 = time.time()
    tes_ms = _echo_times_ms(args.mag, args.tes)
    if len(tes_ms) != len(args.mag):
        raise SystemExit(f"{len(args.mag)} magnitude echoes but {len(tes_ms)} echo times")
    args.mag, tes_ms = _sorted_by_te(args.mag, tes_ms)
    if args.phase:
        if len(args.phase) != len(args.mag):
            raise SystemExit(f"{len(args.mag)} magnitude echoes but {len(args.phase)} phase echoes")
        args.phase, ph_tes = _sorted_by_te(args.phase, _echo_times_ms(args.phase, args.tes))
        if any(abs(a - b) > 1e-3 for a, b in zip(ph_tes, tes_ms, strict=True)):
            raise SystemExit(f"phase echo times {ph_tes} do not match magnitude {tes_ms}")

    with spinner(f"Loading {len(args.mag)} echoes"):
        vols, header = [], None
        for path in args.mag:
            v, h = load_image(path, device=device)
            vols.append(v)
            header = header or h
        mag = torch.stack(vols)  # (E, nz, ny, nx)
    assert header is not None
    spatial = tuple(mag.shape[1:])
    zooms = header["header"].get_zooms()[:3]
    voxel_mm = (float(zooms[2]), float(zooms[1]), float(zooms[0]))  # (nz, ny, nx) order

    if args.mask:
        m, _ = load_image(args.mask, device=device)
        mask = m > 0.5
    else:
        mask = automask(mag[0], device=device, verbose=verb >= 2) > 0
    n_vox = int(mask.sum())
    if verb >= 1:
        print(
            f"ffs_megre: {len(tes_ms)} echoes, TE = {[round(t, 2) for t in tes_ms]} ms, "
            f"{'x'.join(map(str, spatial[::-1]))} at {voxel_mm[::-1]} mm, {n_vox} voxels in mask"
        )

    def save(vol: torch.Tensor, name: str) -> None:
        save_image(vol, f"{args.prefix}_{name}.nii.gz", header_info=header)

    def unmask(values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(spatial, dtype=values.dtype, device=values.device)
        out[mask] = values
        return out

    # T2*/S0 per voxel; the fit is multiecho's (V, E, T) form with T = 1.
    tes = torch.tensor(tes_ms, dtype=torch.float32, device=device)
    data_cat = mag[:, mask].T.unsqueeze(-1).contiguous()  # (V, E, 1)
    _, adaptive = me.make_adaptive_mask(data_cat)
    fit = me.fit_decay(data_cat, tes, adaptive, fittype=args.fittype, device=device)
    t2s, s0, _, _ = me.modify_t2s_s0_maps(fit["t2s"], fit["s0"], adaptive, tes)
    combined = me.make_optcom(data_cat, tes, adaptive, t2s=t2s)[:, 0]
    r2s = 1000.0 / t2s.clamp_min(1e-3)  # 1/s

    save(unmask(t2s / 1000.0), "T2star")
    save(unmask(r2s), "R2star")
    save(unmask(s0), "S0")
    save(unmask(r2s * s0), "R2sS0")
    save(unmask(combined), "combined")
    save(mask.float(), "mask")

    if args.phase:
        from fastfuncstuff.processing.b0fmap import run_romeo

        with spinner(f"Loading {len(args.phase)} phase echoes"):
            ph = torch.stack([load_image(p, device=device)[0] for p in args.phase])
        # ROMEO wants nibabel's (nx, ny, nz, E) layout.
        out = run_romeo(
            ph.permute(3, 2, 1, 0).cpu().numpy(),
            mag.permute(3, 2, 1, 0).cpu().numpy(),
            tes_ms,
            header["affine"],
            romeo_bin=args.romeo,
            verbose=verb >= 2,
        )
        b0 = torch.from_numpy(out.b0_hz.T.copy()).to(device)
        # ROMEO leaves out what it could not unwrap (low SNR far from the coil); a
        # high-pass across that edge would put a step into the local field.
        fmask = mask & torch.from_numpy(out.mask.T.copy()).to(device)
        local = megre.masked_highpass(b0, fmask, args.hp_fwhm, voxel_mm)
        save(b0 * fmask, "B0")
        save(fmask.float(), "fieldmask")
        save(local, "localfield")
        # The mask is applied to the combined magnitude with the phase the local
        # field accrues by the echo the combination weights most.
        te_ref = float(
            (tes * tes * torch.exp(-tes / t2s.median())).sum()
            / (tes * torch.exp(-tes / t2s.median())).sum()
        )
        phase_ref = 2.0 * torch.pi * local * te_ref / 1000.0
        swi = unmask(combined) * megre.swi_phase_mask(phase_ref, args.swi_sign, args.swi_power)
        save(swi, "swi")
        n_mip = max(1, round(args.mip_mm / voxel_mm[0]))
        save(megre.minimum_intensity_projection(swi, n_mip, axis=0) * mask, "swi_mip")
        if verb >= 1:
            print(
                f"  field: B0 + local (high-pass {args.hp_fwhm:g} mm), SWI at TE {te_ref:.1f} ms, mIP over {n_mip} slices"
            )

    if verb >= 1:
        print(f"ffs_megre: done in {time.time() - t0:.1f}s -> {args.prefix}_*.nii.gz")


if __name__ == "__main__":
    main()
