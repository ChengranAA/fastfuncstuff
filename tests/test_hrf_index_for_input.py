"""-hrf_opt maps are read in the analysed data's own domain."""

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli_utils import load_hrf_index_for_input, preflight_check
from fastfuncstuff.io.afni import load_nifti
from fastfuncstuff.io.gifti import save_gifti_data
from fastfuncstuff.io.paired import PairedHeader

MESH = {"mesh_fingerprint": "V3_F1_test"}


def _maps(tmp_path):
    """One ffs_hrfopt prefix run once per domain: hrf_hrf_index.{nii.gz,shape.gii}."""
    vol = tmp_path / "hrf_hrf_index.nii.gz"
    nib.save(nib.Nifti1Image(np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2), np.eye(4)), vol)
    surf = tmp_path / "hrf_hrf_index.shape.gii"
    save_gifti_data(surf, np.array([11, 12, 13], np.float32), MESH, time_series=False)
    return str(tmp_path / "hrf"), load_nifti(vol).header, load_nifti(surf).header


def test_each_domain_reads_its_own_map(tmp_path):
    prefix, vol_header, surf_header = _maps(tmp_path)
    assert load_hrf_index_for_input(prefix, vol_header).tolist() == list(range(8))
    assert load_hrf_index_for_input(prefix, surf_header).tolist() == [10, 11, 12]


def test_surface_map_from_another_mesh_is_refused(tmp_path):
    prefix, _, _ = _maps(tmp_path)
    other = tmp_path / "other.shape.gii"
    save_gifti_data(other, np.zeros(3, np.float32), {"mesh_fingerprint": "V3_F1_other"})
    with pytest.raises(ValueError, match="different mesh"):
        load_hrf_index_for_input(prefix, load_nifti(other).header)


def test_paired_input_concatenates_masked_domains(tmp_path):
    prefix, vol_header, surf_header = _maps(tmp_path)
    header = PairedHeader((2, 2, 2), np.eye(4), vol_header, (3, 1, 1), surf_header)
    mask = np.ones(11, bool)
    mask[[0, 9]] = False
    got = load_hrf_index_for_input(prefix, header, mask.reshape(11, 1, 1))
    assert got.tolist() == [1, 2, 3, 4, 5, 6, 7, 10, 12]


def test_preflight_looks_for_the_surface_map(tmp_path, capsys):
    prefix, _, _ = _maps(tmp_path)
    run = tmp_path / "run.func.gii"
    save_gifti_data(run, np.zeros((3, 4), np.float32), MESH)
    preflight_check([str(run)], hrf_opt_prefix=prefix)
    (tmp_path / "hrf_hrf_index.shape.gii").unlink()
    with pytest.raises(SystemExit):
        preflight_check([str(run)], hrf_opt_prefix=prefix)
    assert "hrf_hrf_index.shape.gii" in capsys.readouterr().out
