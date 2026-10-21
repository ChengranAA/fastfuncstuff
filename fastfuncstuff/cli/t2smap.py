"""CLI for GPU multi-echo T2*/S0 estimation and optimal combination.

Command: ffs_t2smap (registered as entry point in pyproject.toml)

A GPU-first rebuild of tedana's ``t2smap``: estimates voxel-wise S0 and T2* from a
monoexponential decay, optimally combines echoes, and adds a leave-one-echo-out (LOEO)
QC plus an optional robust (Tukey-biweight) refit that down-weights echoes which disagree
with the decay model.

Usage:
    ffs_t2smap -input echo1.nii.gz echo2.nii.gz echo3.nii.gz \\
        -tes 0.015 0.039 0.063 -prefix sub01_ \\
        -fittype curvefit -device cuda
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fastfuncstuff.cli_help import FfsArgumentParser, FfsHelpFormatter
from fastfuncstuff.cli_utils import add_device_arg, add_verbose_arg, setup_device, spinner
from fastfuncstuff.processing import multiecho as me
from fastfuncstuff.processing.io import load_image, save_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = FfsArgumentParser(
        formatter_class=FfsHelpFormatter,
        prog="ffs_t2smap",
        description="Multi-echo T2*/S0 fitting, optimal combination, and "
        "leave-one-echo-out QC (GPU, tedana t2smap compatible).",
    )
    parser.add_argument(
        "-input",
        "-d",
        nargs="+",
        required=True,
        help="Per-echo 4D files in ascending echo order (one file per echo time). GIfTI "
        "series (.gii, vertices x time, e.g. ffs_nwarp -surf output) work too: the same "
        "fit and combination per vertex, every output written as .gii.",
    )
    parser.add_argument(
        "-tes",
        "-e",
        nargs="+",
        type=float,
        required=True,
        help="Echo times in seconds (BIDS). Millisecond values are accepted and "
        "converted automatically.",
    )
    parser.add_argument(
        "-prefix",
        required=True,
        help="Output prefix/stem. Outputs are <prefix>T2starmap.nii.gz, etc.",
    )
    parser.add_argument(
        "-optcom",
        default=None,
        metavar="FILE",
        help="Also write the optimally combined series to exactly this path (any format "
        "ffs writes, e.g. .nii.zst), for pipelines that read it under their own name.",
    )
    parser.add_argument(
        "-phase",
        nargs="+",
        default=None,
        metavar="FILE",
        help="Unwrapped phase (radians), one file per echo, same order and grid as -input. "
        "With -field_hz: the echoes' common frequency shift is fitted from them.",
    )
    parser.add_argument(
        "-field_hz",
        default=None,
        metavar="FILE",
        help="Write each voxel's frequency-shift time series (Hz) from ALL echoes' phase "
        "(needs -phase): phase is 2 pi df TE (+ an offset free of TE) at every echo, so "
        "df is estimated by inverse-variance weighting, a much cleaner regressor than any "
        "single echo's phase -- e.g. for phase regression of the optimal combination.",
    )
    parser.add_argument("-mask", default=None, help="Binary brain mask (strongly recommended).")
    parser.add_argument(
        "-automask",
        action="store_true",
        help="Derive a brain mask from the first echo instead of -mask.",
    )
    parser.add_argument(
        "-fittype",
        default="loglin",
        choices=["loglin", "curvefit", "robust"],
        help="loglin (fast log-linear), curvefit (batched nonlinear LM), or robust "
        "(curvefit with LOEO-driven echo down-weighting).",
    )
    parser.add_argument(
        "-fitmode",
        default="all",
        choices=["all", "ts"],
        help="all = one T2*/S0 per voxel; ts = per voxel and per timepoint (4D maps).",
    )
    parser.add_argument(
        "-fit_all_timepoints",
        "-fit-all-timepoints",
        action="store_true",
        help="With -fitmode all, fit jointly across every timepoint (tedana-exact) "
        "instead of the per-echo temporal mean (default, faster).",
    )
    parser.add_argument(
        "-combmode",
        default="t2s",
        choices=["t2s", "paid"],
        help="Echo combination: t2s (Posse 1999) or paid (Poser 2006).",
    )
    parser.add_argument(
        "-masktype",
        nargs="+",
        default=["dropout"],
        choices=["dropout", "decay", "none"],
        help="Adaptive-mask method(s) for counting good echoes per voxel.",
    )
    parser.add_argument(
        "-no_qc",
        "-no-qc",
        action="store_true",
        help="Skip the leave-one-echo-out QC maps.",
    )
    parser.add_argument(
        "-verbose_out",
        "-verbose-out",
        action="store_true",
        help="Also write curvefit variance/covariance maps.",
    )
    add_device_arg(parser)
    add_verbose_arg(parser, default=1)
    return parser.parse_args(argv)


def _tes_to_ms(tes: list[float]) -> torch.Tensor:
    """Convert echo times to milliseconds, mirroring tedana's seconds-vs-ms heuristic."""
    t = torch.tensor(tes, dtype=torch.float32)
    if torch.all((t > 0) & (t < 1)):  # all in (0, 1) => seconds per BIDS
        t = t * 1000.0
    return t


def _load_surface(path, device):
    """A GIfTI (V,) or (V, T) file as a (T, 1, 1, V) / (1, 1, V) tensor + its metadata."""
    from fastfuncstuff.io.gifti import load_gifti_data

    data, meta = load_gifti_data(path)
    t = torch.from_numpy(data).to(device)
    if t.ndim == 1:
        return t.reshape(1, 1, -1), meta
    return t.T.reshape(t.shape[1], 1, 1, t.shape[0]).contiguous(), meta


def _save_surface(grid: torch.Tensor, path: str, meta) -> None:
    from fastfuncstuff.io.gifti import save_gifti_data

    a = grid.detach().cpu().numpy()
    a = a.reshape(-1) if a.ndim == 3 else a.reshape(a.shape[0], -1).T
    save_gifti_data(path, a, meta=meta)


def _field_in_chunks(
    phases: torch.Tensor, mask: torch.Tensor, data_cat: torch.Tensor, tes: torch.Tensor, device
) -> torch.Tensor:
    """:func:`me.phase_field_hz` over voxel chunks sized from the free device memory.

    ``phases`` is ``(E, T, nz, ny, nx)`` on the host; each chunk is moved over, fitted
    and dropped. A chunk holds the phase, its validity, the weights and the products
    (~6 copies of ``(E, T)`` floats per voxel), taken to 30% of what is free.
    """
    from fastfuncstuff.memory import get_available_memory

    n_echo, n_t = phases.shape[0], phases.shape[1]
    host = phases[:, :, mask.cpu()].permute(2, 0, 1)  # (V, E, T), a host copy
    n_vox = host.shape[0]
    chunk = max(1, int(0.3 * get_available_memory(device) // (6 * n_echo * n_t * 4)))
    out = torch.empty((n_vox, n_t), dtype=torch.float32, device=device)
    for s in range(0, n_vox, chunk):
        sl = slice(s, min(n_vox, s + chunk))
        out[sl] = me.phase_field_hz(host[sl].to(device), data_cat[sl], tes)
    return out


def _unmask(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Scatter masked samples back to the spatial grid.

    Args:
        values: (V,) for a 3D map, (V, K) for a K-brick stack, or (V, T) timeseries.
        mask: (nz, ny, nx) boolean.

    Returns:
        (nz, ny, nx) for 3D or (K, nz, ny, nx) for a stack/timeseries.
    """
    spatial = mask.shape
    if values.ndim == 1:
        out = torch.zeros(spatial, dtype=values.dtype, device=values.device)
        out[mask] = values
        return out
    k = values.shape[1]
    out = torch.zeros((k, *spatial), dtype=values.dtype, device=values.device)
    out[:, mask] = values.T
    return out


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if len(args.input) != len(args.tes):
        raise SystemExit(
            f"Got {len(args.input)} echo file(s) but {len(args.tes)} echo time(s); "
            "provide one file per echo time."
        )

    if (args.phase is None) != (args.field_hz is None):
        raise SystemExit("-phase and -field_hz go together.")
    if args.phase is not None and len(args.phase) != len(args.input):
        raise SystemExit(f"Got {len(args.phase)} -phase file(s) for {len(args.input)} echoes.")
    device = setup_device(args.device)
    tes = _tes_to_ms(args.tes).to(device)
    n_echos = len(tes)
    verb = args.verb
    t0 = time.time()

    # Load echoes -> (E, T, nz, ny, nx); a lone 3D volume reads as T=1. A surface series
    # rides the same path as a (T, 1, 1, V) "volume": every step below is per sample.
    surface = [str(p).endswith(".gii") for p in args.input]
    if any(surface) and not all(surface):
        raise SystemExit("-input mixes GIfTI and volume files; give one kind.")
    surface = all(surface)
    if surface and args.automask:
        raise SystemExit("-automask is a volume method; surface data are masked by default.")
    #
    # The full-grid stack is staged in host memory and only the masked (V, E, T) series
    # goes to the device. Loading every echo there and then stacking them held the
    # whole multi-echo grid on the card twice, for data the fit never looks at outside
    # the mask. Each echo still passes through `device` on its way in, so the default
    # mask's temporal mean is computed exactly where it always was.
    default_mask = args.mask is None and not args.automask
    data = None
    positive = None
    header = None
    for e, path in enumerate(args.input):
        if surface:
            vol, hdr = _load_surface(path, device)
        else:
            vol, hdr = load_image(path, device=device)
        if vol.ndim == 3:
            vol = vol.unsqueeze(0)
        if data is None:
            data = torch.empty((n_echos, *vol.shape), dtype=vol.dtype)  # (E, T, nz, ny, nx)
        elif tuple(vol.shape) != tuple(data.shape[1:]):
            raise SystemExit(
                f"All echo files must share a shape; got {tuple(data.shape[1:])} and "
                f"{tuple(vol.shape)}"
            )
        if default_mask:
            # Voxels positive in every echo's temporal mean.
            pos = vol.mean(dim=0) > 0
            positive = pos if positive is None else positive & pos
        data[e].copy_(vol)
        del vol
        header = header or hdr
    assert data is not None
    spatial = data.shape[2:]
    phases = None
    if args.phase is not None:
        if surface:
            raise SystemExit("-field_hz is a volume output.")
        # Held on the host: the fit below streams voxel chunks to the device, so the phase
        # does not double the (already large) echo stack there.
        phases = []
        for path in args.phase:
            ph, _ = load_image(path, device=torch.device("cpu"))
            phases.append(ph.unsqueeze(0) if ph.ndim == 3 else ph)
        phases = torch.stack(phases, dim=0)
        if tuple(phases.shape) != tuple(data.shape):
            raise SystemExit(f"-phase shape {tuple(phases.shape)} != -input {tuple(data.shape)}")

    # Mask -> boolean (nz, ny, nx).
    if args.mask is not None:
        with spinner(f"Loading {Path(args.mask).name}"):
            mvol, _ = (_load_surface if surface else load_image)(args.mask, device=device)
        if mvol.ndim == 4:
            mvol = mvol[0]
        if tuple(mvol.shape) != tuple(spatial):
            raise SystemExit("-mask grid does not match input grid")
        mask = mvol > 0.5
    elif args.automask:
        from fastfuncstuff.processing.mask import automask

        mask = automask(data[0, 0].to(device), dilate_extra=1, device=device, verbose=verb >= 1) > 0
    else:
        assert positive is not None
        mask = positive
    mask = mask.to(device)

    n_vox = int(mask.sum())
    if n_vox == 0:
        raise SystemExit("Mask is empty; nothing to fit.")

    from fastfuncstuff.memory import estimate_keep_on_cpu

    n_t = data.shape[1]
    if device.type == "cuda" and estimate_keep_on_cpu(n_vox, n_echos * n_t, device):
        gb = n_vox * n_echos * n_t * 4 / 1024**3
        raise SystemExit(
            f"The masked {n_echos}-echo series ({gb:.1f} GB) does not fit in the free "
            f"memory of {device}, and the decay fits need several working copies of it. "
            "Free the card, tighten -mask, or run with -device cpu."
        )

    # (E, T, nz, ny, nx) -> masked (V, E, T), gathered on the host, then moved over.
    data_cat = data[:, :, mask.cpu()].permute(2, 0, 1).contiguous().to(device)  # (V, E, T)
    field_hz = None
    if phases is not None:
        with spinner("Fitting the frequency shift from every echo's phase"):
            field_hz = _field_in_chunks(phases, mask, data_cat, tes, device)
        del phases
    del data

    if verb >= 1:
        print(f"ffs_t2smap: {n_echos} echoes, {data_cat.shape[2]} TRs, {n_vox} voxels")
        print(f"  TEs (ms) = {[round(float(x), 3) for x in tes]}  device={device}")
        print(f"  fittype={args.fittype} fitmode={args.fitmode} combmode={args.combmode}")

    _, adaptive = me.make_adaptive_mask(data_cat, methods=tuple(args.masktype))

    # --- Fit ---
    if args.fittype == "robust":
        fit = me.fit_robust(
            data_cat,
            tes,
            adaptive,
            fitmode=args.fitmode,
            fit_all_timepoints=args.fit_all_timepoints,
            device=device,
            verbose=verb >= 1,
        )
    else:
        fit = me.fit_decay(
            data_cat,
            tes,
            adaptive,
            fittype=args.fittype,
            fitmode=args.fitmode,
            fit_all_timepoints=args.fit_all_timepoints,
            device=device,
            verbose=verb >= 1,
        )
    t2s, s0 = fit["t2s"], fit["s0"]

    # Floors/ceilings + full vs limited maps (per-voxel "all" maps only).
    if args.fitmode == "all":
        t2s_full, s0_full, t2s_lim, s0_lim = me.modify_t2s_s0_maps(t2s, s0, adaptive, tes)
    else:
        t2s_full, s0_full, t2s_lim, s0_lim = t2s, s0, t2s, s0

    # --- Save core maps ---
    def write(grid, path, labels=None):
        if surface:
            _save_surface(grid, path, header)
        else:
            save_image(grid, path, header_info=header, brick_labels=labels)

    if field_hz is not None:
        with spinner("Writing the frequency-shift series"):
            write(_unmask(field_hz, mask), args.field_hz)
        del field_hz

    def save(values, name, labels=None):
        ext = ".func.gii" if surface else ".nii.gz"
        write(_unmask(values, mask), f"{args.prefix}{name}{ext}", labels)

    with spinner("Writing core maps"):
        save(t2s_full / 1000.0, "T2starmap")  # ms -> seconds for storage
        save(s0_full, "S0map")
        save(t2s_lim / 1000.0, "desc-limited_T2starmap")
        save(s0_lim, "desc-limited_S0map")
        save(adaptive.to(torch.float32), "adaptive_mask")

        rmse = me.rmse_of_fit(data_cat, tes, adaptive, t2s_full, s0_full)
        rmse = torch.nan_to_num(rmse, nan=0.0)
        save(rmse, "rmse")

        optcom = me.make_optcom(data_cat, tes, adaptive, t2s=t2s_full, combmode=args.combmode)
        save(optcom, "desc-optcom_bold")
        if args.optcom:
            write(_unmask(optcom, mask), args.optcom)

        echo_labels = [f"echo-{i + 1}" for i in range(n_echos)]
        if args.fittype == "curvefit" or args.fittype == "robust":
            save(fit["failures"].to(torch.float32), "fit_failures")
            if args.verbose_out:
                save(fit["t2s_var"], "T2starvar")
                save(fit["s0_var"], "S0var")
                save(fit["t2s_s0_covar"], "T2star_S0_covar")
        if args.fittype == "robust":
            save(fit["echo_weight"], "echo_weight", labels=echo_labels)

    # --- Leave-one-echo-out QC ---
    if not args.no_qc:
        mean_echo = data_cat.mean(dim=2)
        avail = me.availability_weights(adaptive, n_echos)
        loeo_fittype = "loglin" if args.fittype == "loglin" else "curvefit"
        resid, resid_frac = me.leave_one_echo_out(
            mean_echo, tes, avail, fittype=loeo_fittype, device=device
        )
        save(torch.nan_to_num(resid, nan=0.0), "loeo_resid", labels=echo_labels)
        save(torch.nan_to_num(resid_frac, nan=0.0), "loeo_resid_frac", labels=echo_labels)

    if verb >= 1:
        print(f"  limited T2* median (s): {(t2s_lim[t2s_lim > 0] / 1000.0).median():.4f}")
        print(f"Saved maps with prefix: {args.prefix}")
        print(f"Time: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
