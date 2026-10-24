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
            "-strengths",
            ".01",
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
