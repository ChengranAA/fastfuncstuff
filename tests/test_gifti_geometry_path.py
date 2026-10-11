"""A surface output's mesh path stays valid wherever the output is written."""

import os

import numpy as np
import pytest

from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data


@pytest.mark.parametrize(
    ("out_dir", "expected"),
    [
        ("func", os.path.join("..", "surf", "lh.midthickness.surf.gii")),
        ("out", os.path.join("..", "surf", "lh.midthickness.surf.gii")),
        (os.path.join("out", "deep"), os.path.join("..", "..", "surf", "lh.midthickness.surf.gii")),
    ],
)
def test_geometry_is_rebased_on_the_output(tmp_path, out_dir, expected):
    src = tmp_path / "func" / "run.func.gii"
    src.parent.mkdir()
    meta = {"mesh_fingerprint": "V3_F1_test", "geometry": "../surf/lh.midthickness.surf.gii"}
    save_gifti_data(src, np.zeros((3, 4), np.float32), meta)

    out = tmp_path / out_dir / "map.shape.gii"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_nifti(np.ones((3, 1, 1), np.float32), out, header=load_nifti(src).header)
    _, written = load_gifti_data(out)
    assert written["geometry"] == expected
