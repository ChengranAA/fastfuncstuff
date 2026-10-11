"""Surface CLI loader keeps mesh identity and uses mesh heat smoothing."""

import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import load_and_preprocess_runs
from fastfuncstuff.io.gifti import (
    mesh_fingerprint,
    save_gifti_data,
    save_gifti_surface,
    surface_meta,
)
from fastfuncstuff.surface.smooth import HeatSmoother


def _fixture(tmp_path):
    ii, jj = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    vertices = np.c_[ii.ravel(), jj.ravel(), np.zeros(ii.size)]
    index = (ii * 5 + jj)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    faces = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    save_gifti_surface(tmp_path / "mid.surf.gii", vertices, faces)
    meta = {"mesh_fingerprint": mesh_fingerprint(faces, 25), "geometry": "mid.surf.gii", "TR_seconds": "2"}
    data = np.random.default_rng(4).normal(size=(25, 8)).astype(np.float32)
    runs = []
    for n in range(2):
        path = tmp_path / f"run{n}.func.gii"
        save_gifti_data(path, data + n, meta)
        runs.append(path)
    mask = tmp_path / "mask.shape.gii"
    save_gifti_data(mask, np.ones(25, np.float32), meta, time_series=False)
    return runs, mask, data, vertices, faces, meta


def test_loader_surface_blur_and_header(tmp_path):
    runs, mask, data, vertices, faces, meta = _fixture(tmp_path)
    result = load_and_preprocess_runs([str(p) for p in runs], mask_file=str(mask),
                                      blur_fwhm=2, device=torch.device("cpu"), verbose=False)
    expected = HeatSmoother(vertices, faces, 2, mask=np.ones(25, bool))(data)
    np.testing.assert_allclose(result.data.numpy()[:, :8], expected, rtol=2e-6, atol=2e-6)
    assert result.volume_shape == (25, 1, 1)
    assert result.run_starts == [0, 8]
    assert surface_meta(result.nifti_header)["mesh_fingerprint"] == meta["mesh_fingerprint"]


def test_loader_refuses_wrong_mesh_even_with_equal_vertex_count(tmp_path):
    runs, mask, *_ = _fixture(tmp_path)
    save_gifti_data(runs[1], np.ones((25, 8), np.float32),
                    {"mesh_fingerprint": "other", "TR_seconds": "2"})
    with pytest.raises(ValueError, match="mesh"):
        load_and_preprocess_runs([str(p) for p in runs], device=torch.device("cpu"), verbose=False)
    with pytest.raises(ValueError, match="mesh"):
        load_and_preprocess_runs([str(runs[0])], mask_file=str(runs[1]),
                                 device=torch.device("cpu"), verbose=False)
