"""Depth sampling and voxel mapping must preserve residuals and data units."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.processing.io import load_image, save_image
from fastfuncstuff.processing.laminar_tdv import (
    depth_interpolation,
    depth_patches,
    devein_volume_runs,
)

CPU = torch.device("cpu")


def test_patches_never_jump_between_disconnected_banks():
    mask = np.zeros((9, 9, 9), bool)
    mask[:, :, 2:4] = True
    mask[:, :, 5:7] = True
    depth = np.indices(mask.shape)[0] / 8
    labels = depth_patches(mask, depth, np.ones(3), 5)
    z = np.indices(mask.shape)[2][mask]
    assert not set(labels[z < 4]) & set(labels[z > 4])
    assert (labels >= 0).all()


def _runs():
    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer

    T, K = 90, 3
    spec = ModelSpec(N=K, K=K, TR=2, device=CPU)
    freq = torch.fft.rfftfreq(T, spec.TR)
    h, _ = vascular_transfer(linearized_vasculature(spec), freq)
    torch.manual_seed(20)
    a = torch.randn(T, K, dtype=torch.float64) * 0.002
    a = torch.fft.irfft(torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0)
    y = torch.fft.irfft(torch.einsum("fkn,fn->fk", h, torch.fft.rfft(a, dim=0)), n=T, dim=0).float()
    depth = np.tile(np.array([1 / 6, 0.5, 5 / 6]), 20)
    labels = np.zeros(len(depth), int)
    read = torch.from_numpy(depth_interpolation(depth, K))
    mean = torch.linspace(100, 500, len(depth))[:, None]
    return (
        [mean * (1 + ((read @ y.T) + 0.005 * torch.randn(len(depth), T)) / 100) for _ in range(3)],
        depth,
        labels,
        spec,
    )


@pytest.mark.parametrize("transfer_mode", ["dynamic", "static"])
def test_voxel_mapping_preserves_baseline_units_and_deepest_series(transfer_mode):
    runs, depth, labels, spec = _runs()
    res = devein_volume_runs(
        runs,
        depth,
        labels,
        spec,
        strengths=(0.01,),
        pad_seconds=0,
        transfer_mode=transfer_mode,
        verbose=False,
    )
    assert res.valid.all()
    for raw, out, rem in zip(runs, res.corrected, res.removed_psc, strict=True):
        torch.testing.assert_close(out, raw - rem * raw.mean(-1, keepdim=True) / 100)
        torch.testing.assert_close(out[depth > 0.8], raw[depth > 0.8])
        torch.testing.assert_close(out.mean(-1), raw.mean(-1), rtol=1e-6, atol=1e-5)
        assert rem[depth < 0.3].abs().max() > 0.001
    assert np.isfinite(res.noise_gain).all()


def test_cli_writes_full_grid_and_keeps_outside_voxels(tmp_path):
    runs, depth, labels, spec = _runs()
    shape = (5, 4, 3)
    info = {"affine": np.diag([1.0, 1.0, 1.0, 1.0])}
    # All 60 voxels belong to one supplied patch; one outside it passes through.
    depth = depth.reshape(shape)
    gm = np.ones(shape, np.float32)
    gm[0, 0, 0] = 0
    for i, r in enumerate(runs[:2]):
        save_image(r.T.reshape(r.shape[1], *shape), tmp_path / f"r{i}.nii.gz", info)
    for name, arr in [("depth", depth), ("gm", gm), ("columns", np.ones(shape))]:
        save_image(torch.as_tensor(arr), tmp_path / f"{name}.nii.gz", info)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.laminar_tdv",
            "-input",
            "r0.nii.gz",
            "r1.nii.gz",
            "-depth",
            "depth.nii.gz",
            "-gmfrac",
            "gm.nii.gz",
            "-depth_order",
            "pial_to_white",
            "-columns",
            "columns.nii.gz",
            "-depths",
            "3",
            "-tr",
            "2",
            "-ridge",
            ".01",
            "-te",
            ".025",
            "-b0",
            "7",
            "-device",
            "cpu",
            "-prefix",
            "out",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    raw, _ = load_image(tmp_path / "r0.nii.gz")
    out, _ = load_image(tmp_path / "out_corrected_run-01.nii.gz")
    torch.testing.assert_close(out[:, 0, 0, 0], raw[:, 0, 0, 0])
    assert out.shape == raw.shape
    meta = json.loads((tmp_path / "out_metadata.json").read_text())
    assert meta["n_valid"] == 59


def test_cli_refuses_to_guess_echo_time(tmp_path):
    runs, *_ = _runs()
    info = {"affine": np.eye(4)}
    save_image(runs[0].T.reshape(-1, 5, 4, 3), tmp_path / "r0.nii.gz", info)
    (tmp_path / "r0.json").write_text(json.dumps({"MagneticFieldStrength": 7}))
    from fastfuncstuff.cli.laminar_tdv import main

    with pytest.raises(SystemExit, match="-te"):
        main(["-input", str(tmp_path / "r0.nii.gz"), "-fs_subj", str(tmp_path), "-ridge",
              "0.1", "-prefix", str(tmp_path / "o")])  # fmt: skip


@pytest.mark.slow  # 17s
def test_cli_surfaces_remove_deep_drainage_and_leave_the_rest(tmp_path):
    """Sphere-shell ribbon, one deep source: removal grows toward pial, off-ribbon is untouched."""
    import nibabel as nib

    from fastfuncstuff.laminar.timeseries import linearized_vasculature, vascular_transfer
    from tests.viewer.test_surfaces import RADIUS, _subject

    vox, n, T, K, tr = 1.0, 60, 90, 3, 2.0
    aff = np.diag([vox, vox, vox, 1.0])
    aff[:3, 3] = -(n - 1) * vox / 2
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * vox + aff[:3, 3], axis=-1)
    frac = np.clip((r - RADIUS) / (0.2 * RADIUS), 0, 1)  # 0 white .. 1 pial
    ribbon = (r > RADIUS) & (r < 1.2 * RADIUS)
    spec = ModelSpec(N=K, K=K, TR=tr, device=CPU)
    freq = torch.fft.rfftfreq(T, tr)
    h, _ = vascular_transfer(linearized_vasculature(spec), freq)
    torch.manual_seed(3)
    a = torch.zeros(T, K, dtype=torch.float64)
    a[:, -1] = torch.randn(T, dtype=torch.float64) * 0.2  # deepest model depth only
    a = torch.fft.irfft(torch.fft.rfft(a, dim=0) / (1 + (freq[:, None] / 0.04) ** 4), n=T, dim=0)
    y = torch.fft.irfft(torch.einsum("fkn,fn->fk", h, torch.fft.rfft(a, dim=0)), n=T, dim=0)
    read = depth_interpolation(1 - frac[ribbon], K)  # model order: 0 = pial
    signal = (read @ y.numpy().T).astype(np.float32)
    rng = np.random.default_rng(0)
    for run in (1, 2):
        data = 100 + rng.normal(0, 0.05, (n, n, n, T)).astype(np.float32)
        data[ribbon] += signal
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"run{run}.nii.gz"))
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.laminar_tdv", "-input", "run1.nii.gz",
         "run2.nii.gz", "-fs_subj", str(_subject(tmp_path)), "-hemi", "lh", "-depths", str(K),
         "-ridge", "0.001", "-te", "0.025", "-b0", "7", "-tr", str(tr), "-pad_seconds", "0",
         "-device", "cpu", "-prefix", "ltdv"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp_path,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    img = nib.load(tmp_path / "ltdv_ltdv.nii.gz")
    names, b = _read_brick_labels(img), np.asanyarray(img.dataobj)
    valid = b[..., names.index("valid")] > 0
    assert valid[ribbon].mean() > 0.8 and not valid[r < 0.8 * RADIUS].any()
    depth = b[..., names.index("depth")]
    assert np.corrcoef(depth[valid], frac[valid])[0, 1] > 0.8
    rms = b[..., names.index("removed_rms")]
    deep, sup = valid & (frac < 0.25), valid & (frac > 0.75)
    assert np.median(rms[sup]) > 5 * np.median(rms[deep])
    raw = np.asanyarray(nib.load(tmp_path / "run1.nii.gz").dataobj)
    out = np.asanyarray(nib.load(tmp_path / "ltdv_corrected_run-01.nii.gz").dataobj)
    np.testing.assert_array_equal(out[~valid], raw[~valid])
    leak = signal[(frac[ribbon] > 0.75)].std(1).mean()
    assert np.abs(out[sup] - 100).std(-1).mean() < 0.5 * leak
