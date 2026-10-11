"""Paired ridge CLI emits independent volume and surface maps."""

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.ridge import main
from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data


def _inputs(tmp_path, *, surface_lengths=(45, 45)):
    rng = np.random.default_rng(5)
    volumes, surfaces = [], []
    for i, length in enumerate(surface_lengths):
        volume = tmp_path / f"vol{i}.nii.gz"
        surface = tmp_path / f"surf{i}.func.gii"
        data = 100 + rng.normal(size=(2, 2, 2, 45)).astype(np.float32)
        img = nib.Nifti1Image(data, np.eye(4))
        img.header.set_zooms((1, 1, 1, 2))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, volume)
        save_gifti_data(
            surface,
            100 + rng.normal(size=(4, length)).astype(np.float32),
            {"mesh_fingerprint": "V4_F2_test", "TR_seconds": "2"},
        )
        volumes.append(str(volume))
        surfaces.append(str(surface))
    return volumes, surfaces


def _args(tmp_path, volumes, surfaces):
    timing = tmp_path / "onsets.1D"
    timing.write_text("4 22 44\n4 22 44\n")
    return [
        "ffs_ridge",
        "-input",
        *volumes,
        "-surface_input",
        *surfaces,
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
        "-ridge_fracs",
        "1",
        "1",
        "1",
        "-noise_ceiling",
        "off",
        "-prefix",
        str(tmp_path / "ridge"),
    ]


def test_paired_ridge_outputs_and_masks(tmp_path, monkeypatch):
    volumes, surfaces = _inputs(tmp_path)
    vm = tmp_path / "volume_mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.array([1, 0, 1, 1, 1, 1, 1, 1], np.float32).reshape(2, 2, 2), np.eye(4)),
        vm,
    )
    sm = tmp_path / "surface_mask.shape.gii"
    save_gifti_data(sm, np.array([1, 1, 0, 1], np.float32), {"mesh_fingerprint": "V4_F2_test"})
    monkeypatch.setattr(
        sys,
        "argv",
        _args(tmp_path, volumes, surfaces) + ["-mask", str(vm), "-surface_mask", str(sm)],
    )
    main()
    for stem, n_frames in (
        ("r2", None),
        ("optimal_frac", None),
        ("betas_single_trial", 6),
        ("betas_condition", 1),
        ("r2_by_frac", 1),
    ):
        vol = nib.load(str(tmp_path / f"ridge_{stem}_volume.nii.gz"))
        surf, meta = load_gifti_data(tmp_path / f"ridge_{stem}_surface.shape.gii")
        assert vol.shape == ((2, 2, 2) if n_frames is None else (2, 2, 2, n_frames))
        assert surf.shape == ((4,) if n_frames is None or n_frames == 1 else (4, n_frames))
        assert meta["mesh_fingerprint"] == "V4_F2_test"
        assert np.all(np.asanyarray(vol.dataobj)[0, 0, 1] == 0)
        assert np.all(surf[2] == 0)


def test_paired_single_trials_outputs(tmp_path, monkeypatch):
    vols, surfs = _inputs(tmp_path)
    args = _args(tmp_path, vols, surfs)
    frac_arg = args.index("-ridge_fracs")
    args[frac_arg + 1 : frac_arg + 4] = ["0.5", "1", "0.5"]
    monkeypatch.setattr(sys, "argv", args + ["-single_trials"])
    main()
    for domain, ext, spatial in (("volume", ".nii.gz", (2, 2, 2)), ("surface", ".shape.gii", (4,))):
        path = tmp_path / f"ridge_{domain}_single_trial_betas{ext}"
        if domain == "volume":
            assert nib.load(str(path)).shape == (*spatial, 6)
        else:
            data, meta = load_gifti_data(path)
            assert data.shape == (*spatial, 6)
            assert meta["mesh_fingerprint"] == "V4_F2_test"
        assert (tmp_path / f"ridge_{domain}_single_trial_xval_sse{ext}").exists()
    for stem in ("optimal_frac", "single_trial_full_r2", "sse_by_frac"):
        assert (tmp_path / f"ridge_{stem}_volume.nii.gz").exists()
        assert (tmp_path / f"ridge_{stem}_surface.shape.gii").exists()


def test_paired_hrf_opt_rejected_before_loading(tmp_path, monkeypatch, capsys):
    vols, surfs = _inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _args(tmp_path, vols, surfs) + ["-hrf_opt", "not_a_prefix"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "paired -surface_input does not support -hrf_opt" in capsys.readouterr().err
