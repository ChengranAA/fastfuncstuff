"""Surface ridge maps and HRF index reuse retain the input mesh."""

import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import save_4d_nifti, save_volume_nifti
from fastfuncstuff.glm.ridge import load_hrf_indices
from fastfuncstuff.io.gifti import gifti_as_nifti, load_gifti_data, save_gifti_data


def test_masked_surface_ridge_maps_and_hrf_handoff(tmp_path):
    source = tmp_path / "run.func.gii"
    save_gifti_data(source, np.ones((4, 6)), {"mesh_fingerprint": "V4_F2_test"})
    header = gifti_as_nifti(source).header
    mask = np.array([True, False, True, True])
    path = tmp_path / "ridge_optimal_frac.shape.gii"
    save_volume_nifti(
        torch.tensor([0.2, 0.3, 0.4]), str(path), (4, 1, 1), np.eye(4), mask, header=header
    )
    values, meta = load_gifti_data(path)
    np.testing.assert_allclose(values, [0.2, 0, 0.3, 0.4])
    assert meta["mesh_fingerprint"] == "V4_F2_test"

    betas_path = tmp_path / "ridge_betas.shape.gii"
    save_4d_nifti(
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        str(betas_path),
        (4, 1, 1),
        np.eye(4),
        mask,
        header=header,
    )
    betas, meta = load_gifti_data(betas_path)
    np.testing.assert_array_equal(betas, [[1, 2], [0, 0], [3, 4], [5, 6]])
    assert meta["mesh_fingerprint"] == "V4_F2_test"

    index_path = tmp_path / "hrf_index.shape.gii"
    save_4d_nifti(
        torch.tensor([[1.0, 0.6], [2.0, 0.7], [3.0, 0.8]]),
        str(index_path),
        (4, 1, 1),
        np.eye(4),
        mask,
        header=header,
    )
    torch.testing.assert_close(load_hrf_indices(str(index_path), mask), torch.tensor([0, 1, 2]))


def test_surface_hrf_index_mask_length_rejected(tmp_path):
    source = tmp_path / "index.shape.gii"
    save_gifti_data(source, np.ones((4, 2)))
    with pytest.raises(IndexError):
        load_hrf_indices(str(source), np.ones(3, bool))
