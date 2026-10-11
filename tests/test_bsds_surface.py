"""BSDS keeps one labelled surface parcel space across runs."""

import argparse

import numpy as np
import pytest

from fastfuncstuff.cli.bsds import _load_sessions
from fastfuncstuff.io.gifti import save_gifti_data


def _args(paths, atlas, mask=None):
    return argparse.Namespace(
        input=[str(p) for p in paths],
        parcellation="atlas",
        atlas=str(atlas),
        mask=mask,
        aggregate="mean",
    )


def test_surface_atlas_stable_parcels_and_mesh(tmp_path):
    atlas = tmp_path / "labels.shape.gii"
    meta = {"mesh_fingerprint": "V6_F4_abcd", "geometry": "lh.surf.gii"}
    save_gifti_data(atlas, np.array([1, 1, 2, 2, 3, 3]), meta=meta)
    paths = [tmp_path / f"run{i}.func.gii" for i in range(2)]
    for path in paths:
        save_gifti_data(path, np.arange(6 * 8).reshape(6, 8), meta=meta)
    sessions, ids, outmeta = _load_sessions(_args(paths, atlas))
    assert ids.tolist() == [1, 2, 3]
    assert outmeta["mesh_fingerprint"] == meta["mesh_fingerprint"]
    assert len(sessions) == 2
    np.testing.assert_array_equal(sessions[0], sessions[1])

    other = tmp_path / "other.func.gii"
    save_gifti_data(other, np.ones((6, 8)), meta={"mesh_fingerprint": "different"})
    with pytest.raises(SystemExit, match="mesh"):
        _load_sessions(_args([paths[0], other], atlas))


def test_surface_atlas_rejects_volume_run(tmp_path):
    import nibabel as nib

    atlas = tmp_path / "labels.shape.gii"
    save_gifti_data(atlas, np.array([1, 1, 2, 2]), meta={"mesh_fingerprint": "same"})
    surface = tmp_path / "run.func.gii"
    save_gifti_data(surface, np.ones((4, 5)), meta={"mesh_fingerprint": "same"})
    volume = tmp_path / "run.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 1, 1, 5), dtype=np.float32), np.eye(4)), volume)
    with pytest.raises(SystemExit, match="mix volumes and surfaces"):
        _load_sessions(_args([surface, volume], atlas))


def test_surface_model_writes_label_map_with_mesh(tmp_path):
    import json

    from fastfuncstuff.cli.bsds import main
    from fastfuncstuff.io.gifti import load_gifti_data

    meta = {"mesh_fingerprint": "V6_F4_abcd", "geometry": "lh.surf.gii"}
    atlas = tmp_path / "atlas.shape.gii"
    save_gifti_data(atlas, np.array([1, 1, 2, 2, 3, 3]), meta=meta)
    rng = np.random.default_rng(4)
    run = tmp_path / "run.func.gii"
    save_gifti_data(run, rng.standard_normal((6, 60)), meta=meta)
    stem = str(tmp_path / "out" / "fit")
    assert (
        main(
            [
                "-input",
                str(run),
                "-atlas",
                str(atlas),
                "-parcellation",
                "atlas",
                "-prefix",
                stem,
                "-n_states",
                "2",
                "-max_ldim",
                "1",
                "-n_init",
                "1",
                "-n_iter",
                "3",
                "-device",
                "cpu",
                "-plots",
                "none",
            ]
        )
        == 0
    )
    label_map, saved_meta = load_gifti_data(f"{stem}_parcel_labels.shape.gii")
    np.testing.assert_array_equal(label_map, [1, 1, 2, 2, 3, 3])
    assert saved_meta["mesh_fingerprint"] == meta["mesh_fingerprint"]
    saved = np.load(f"{stem}_model.npz")
    assert saved["parcel_map"].item() == f"{stem}_parcel_labels.shape.gii"
    assert (
        json.loads((tmp_path / "out" / "fit_summary.json").read_text())["parcel_mesh_fingerprint"]
        == meta["mesh_fingerprint"]
    )
