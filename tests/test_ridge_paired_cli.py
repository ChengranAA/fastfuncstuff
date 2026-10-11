"""End-to-end CPU paired ridge fits write separate, correctly shaped domains."""

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.ridge import main
from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data


@pytest.mark.parametrize("single_trials", [False, True])
def test_paired_ridge_cli(tmp_path, monkeypatch, single_trials):
    rng = np.random.default_rng(2)
    vols, surfs = [], []
    for run in range(2):
        vol = tmp_path / f"vol{run}.nii.gz"
        surf = tmp_path / f"surf{run}.func.gii"
        img = nib.Nifti1Image(rng.normal(100, 1, (2, 2, 2, 24)).astype(np.float32), np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((1, 1, 1, 2))
        nib.save(img, vol)
        save_gifti_data(
            surf,
            rng.normal(100, 1, (4, 24)),
            meta={"mesh_fingerprint": "V4_F2_test", "TR_seconds": "2"},
        )
        vols.append(str(vol))
        surfs.append(str(surf))
    mask = tmp_path / "surface_mask.shape.gii"
    save_gifti_data(
        mask, np.array([1, 0, 1, 1], np.float32), meta={"mesh_fingerprint": "V4_F2_test"}
    )
    onset = tmp_path / "onsets.1D"
    onset.write_text("2 8 14 20\n2 8 14 20\n")
    prefix = str(tmp_path / "ridge")
    argv = [
        "ffs_ridge",
        "-input",
        *vols,
        "-surface_input",
        *surfs,
        "-surface_mask",
        str(mask),
        "-onsets",
        str(onset),
        "-durations",
        "1",
        "-device",
        "cpu",
        "-polort",
        "0",
        "-prefix",
        prefix,
    ]
    if single_trials:
        argv += ["-single_trials", "-ridge_fracs", "0.1", "1.0", "0.9"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    tag = "single_trial_betas" if single_trials else "betas_single_trial"
    vol_path = (
        tmp_path / f"ridge_volume_{tag}.nii.gz"
        if single_trials
        else tmp_path / "ridge_betas_single_trial_volume.nii.gz"
    )
    surf_path = (
        tmp_path / f"ridge_surface_{tag}.shape.gii"
        if single_trials
        else tmp_path / "ridge_betas_single_trial_surface.shape.gii"
    )
    assert nib.load(vol_path).shape[:3] == (2, 2, 2)
    values, meta = load_gifti_data(surf_path)
    assert values.shape[0] == 4
    assert np.all(values[1] == 0)
    assert meta["mesh_fingerprint"] == "V4_F2_test"
