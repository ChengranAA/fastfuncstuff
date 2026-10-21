"""Paired volume/surface loading keeps time aligned and geometries separate."""

import numpy as np
import pytest
import torch

from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.io.gifti import save_gifti_data, surface_meta
from fastfuncstuff.io.paired import load_paired_runs


def _inputs(tmp_path, lengths=(5, 6)):
    import nibabel as nib

    vols, surfs = [], []
    for i, length in enumerate(lengths):
        vol = tmp_path / f"vol{i}.nii.gz"
        surf = tmp_path / f"surf{i}.func.gii"
        arr = np.arange(8 * length, dtype=np.float32).reshape(2, 2, 2, length) + i * 100
        img = nib.Nifti1Image(arr, np.eye(4))
        img.header.set_zooms((1, 1, 1, 2))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, vol)
        save_gifti_data(
            surf,
            np.arange(3 * length, dtype=np.float32).reshape(3, length) + i * 1000,
            meta={"mesh_fingerprint": "V3_F1_test", "TR_seconds": "2"},
        )
        vols.append(str(vol))
        surfs.append(str(surf))
    return vols, surfs


def test_paired_run_split_and_save(tmp_path):
    vols, surfs = _inputs(tmp_path)
    paired = load_paired_runs(vols, surfs, device=torch.device("cpu"), drop_first=1)
    assert paired.data.shape == (11, 9)
    assert paired.run_starts == [0, 4]
    assert paired.tr == 2
    vm, sm = paired.split_map(torch.arange(22).reshape(11, 2))
    assert vm.shape == (8, 2) and sm.shape == (3, 2)
    vp, sp = tmp_path / "map.nii.gz", tmp_path / "map.shape.gii"
    paired.save_map(torch.arange(11, dtype=torch.float32), vp, sp)
    assert np.array_equal(np.asanyarray(load_nifti(vp).dataobj).reshape(-1), np.arange(8))
    out = load_nifti(sp)
    assert out.shape == (3, 1, 1)
    assert np.array_equal(np.asanyarray(out.dataobj).reshape(-1), np.arange(8, 11))
    assert surface_meta(out.header)["mesh_fingerprint"] == "V3_F1_test"


def test_reject_unpaired_lengths_and_tr(tmp_path):
    vols, surfs = _inputs(tmp_path)
    with pytest.raises(ValueError, match="one volume and one surface"):
        load_paired_runs(vols, surfs[:1])
    save_gifti_data(
        surfs[1],
        np.ones((3, 7), np.float32),
        meta={"mesh_fingerprint": "V3_F1_test", "TR_seconds": "2"},
    )
    with pytest.raises(ValueError, match="equal time lengths"):
        load_paired_runs(vols, surfs)
    with pytest.raises(ValueError, match="inconsistent TR"):
        load_paired_runs(vols[:1], surfs[:1], tr=1)


def test_surface_mask_kept_in_surface_output(tmp_path):
    vols, surfs = _inputs(tmp_path, lengths=(5,))
    mask = tmp_path / "mask.shape.gii"
    save_gifti_data(mask, np.array([1, 0, 1], np.float32), meta={"mesh_fingerprint": "V3_F1_test"})
    paired = load_paired_runs(vols, surfs, surface_mask=str(mask))
    assert paired.surface.n_voxels == 2
    paired.save_map(
        np.arange(10, dtype=np.float32), tmp_path / "v.nii.gz", tmp_path / "s.shape.gii"
    )
    assert np.array_equal(
        np.asanyarray(load_nifti(tmp_path / "s.shape.gii").dataobj).reshape(-1), [8, 0, 9]
    )


def test_paired_header_splits_saved_maps(tmp_path):
    """Anything saved on the pseudo-grid lands in one file per domain."""
    from fastfuncstuff.io.afni import save_nifti

    vols, surfs = _inputs(tmp_path, lengths=(5,))
    paired = load_paired_runs(vols, surfs)
    assert paired.volume.data.numel() == 0 and paired.surface.data.numel() == 0
    assert paired.mask.shape == paired.header.shape == (11, 1, 1)
    values = np.arange(22, dtype=np.float32).reshape(11, 1, 1, 2)
    save_nifti(values, tmp_path / "map.nii.gz", header=paired.header, brick_labels=["a", "b"])
    vol = np.asanyarray(load_nifti(tmp_path / "map_volume.nii.gz").dataobj)
    surf = load_nifti(tmp_path / "map_surface.shape.gii")
    assert vol.shape == (2, 2, 2, 2)
    assert np.array_equal(vol.reshape(8, 2), values[:8, 0, 0])
    assert np.array_equal(np.asanyarray(surf.dataobj).reshape(3, 2), values[8:, 0, 0])
    assert surface_meta(surf.header)["mesh_fingerprint"] == "V3_F1_test"
