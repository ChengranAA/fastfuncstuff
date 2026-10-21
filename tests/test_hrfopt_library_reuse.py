"""Consumers of -hrfopt_prefix fit with the library the indices came from.

Every tool but ffs_reml used to conjure the stock 20-curve library instead of
reading the one ffs_hrfopt saved: a 40-curve library (PIGHS, a multi-group
ffs_librarian set) died with `IndexError: index 20 is out of bounds`, and a
same-sized different one silently fit every voxel with the wrong shape.
"""

import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import load_hrfopt_library


def _save(tmp_path, n_hrfs, *, duration_convolved=False):
    lib = torch.arange(n_hrfs, dtype=torch.float32)[:, None] * torch.ones(320)
    torch.save(
        {"hrf_library": lib, "duration_convolved": duration_convolved},
        tmp_path / "hrf_hrf_library.pt",
    )
    return str(tmp_path / "hrf")


def test_saved_library_is_used_not_the_stock_twenty(tmp_path):
    prefix = _save(tmp_path, 40)
    lib, durations = load_hrfopt_library(
        prefix, torch.tensor([0, 20, 39]), microtime_dt=0.1, model_durations=[2.0]
    )
    assert len(lib) == 40
    assert durations == [2.0]


def test_duration_convolved_library_zeroes_the_model_durations(tmp_path):
    prefix = _save(tmp_path, 20, duration_convolved=True)
    _, durations = load_hrfopt_library(
        prefix, torch.tensor([0, 19]), microtime_dt=0.1, model_durations=[2.0, 3.0]
    )
    assert durations == [0.0, 0.0]


def test_indices_past_the_library_end_are_an_explained_error(tmp_path):
    prefix = _save(tmp_path, 20)
    with pytest.raises(SystemExit, match="reach index 25"):
        load_hrfopt_library(prefix, torch.tensor([0, 25]), microtime_dt=0.1)


def test_missing_library_falls_back_to_the_stock_one(tmp_path):
    lib, _ = load_hrfopt_library(str(tmp_path / "nothing"), torch.tensor([0, 19]), microtime_dt=0.1)
    assert len(lib) == 20


def test_ridge_cli_fits_a_forty_curve_library(tmp_path, monkeypatch):
    """The reported crash, end to end: indices 20..39 against a stock-20 guess."""
    from fastfuncstuff.cli.ridge import main
    from fastfuncstuff.design.hrf import get_hrf_library

    rng = np.random.default_rng(5)
    inputs = []
    for i in range(2):
        path = tmp_path / f"vol{i}.nii.gz"
        img = nib.Nifti1Image(100 + rng.normal(size=(2, 2, 2, 45)).astype(np.float32), np.eye(4))
        img.header.set_zooms((1, 1, 1, 2))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, path)
        inputs.append(str(path))

    stock = get_hrf_library(mode="library", microtime_dt=0.1, device=torch.device("cpu"))
    torch.save(
        {"hrf_library": torch.cat([stock, stock]), "duration_convolved": False},
        tmp_path / "hrf_hrf_library.pt",
    )
    # 1-indexed in the NIfTI, so 21..28 lands on 20..27 — past a stock library's end.
    nib.save(
        nib.Nifti1Image(np.arange(21, 29, dtype=np.float32).reshape(2, 2, 2), np.eye(4)),
        tmp_path / "hrf_hrf_index.nii.gz",
    )
    timing = tmp_path / "onsets.1D"
    timing.write_text("4 22 44\n4 22 44\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ffs_ridge",
            "-input",
            *inputs,
            "-onsets",
            str(timing),
            "-durations",
            "1",
            "-tr",
            "2",
            "-device",
            "cpu",
            "-polort",
            "1",
            # Two fracs: the per-voxel-HRF path selects among all but the last.
            "-ridge_fracs",
            "0.5",
            "1",
            "0.5",
            "-noise_ceiling",
            "off",
            # The ffs_reml spelling, which these tools used to reject outright.
            "-hrfopt_prefix",
            str(tmp_path / "hrf"),
            "-prefix",
            str(tmp_path / "ridge"),
        ],
    )
    main()
    assert nib.load(str(tmp_path / "ridge_single_trial_full_r2.nii.gz")).shape == (2, 2, 2)
