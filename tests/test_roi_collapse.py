"""RoiSet: ROI files resolved against a grid and a mask (processing/roi_collapse.py)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.roi_collapse import build_roi_set

CPU = torch.device("cpu")
SHAPE = (4, 3, 2)


def _save(tmp_path, name, arr):
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(np.asarray(arr), np.eye(4)), path)
    return path


@pytest.fixture
def labels(tmp_path):
    lab = np.zeros(SHAPE, dtype=np.int16)
    lab[:2, :, 0] = 3
    lab[2:, :, 0] = 7
    return _save(tmp_path, "atlas.nii.gz", lab)


def test_rows_are_roi_cover_and_collapse_is_the_mean(labels):
    roi = build_roi_set(labels, SHAPE)
    assert roi.ids == [3, 7]
    assert roi.rows.sum() == 12  # only z=0 is covered: nothing else would be loaded
    data = torch.arange(roi.n_rows * 2, dtype=torch.float32).reshape(roi.n_rows, 2)
    out = roi.collapse(data, device=CPU)
    flat = np.zeros(SHAPE, dtype=bool)
    flat[:2, :, 0] = True
    expected = data[torch.from_numpy(flat[roi.rows])].mean(dim=0)
    assert torch.allclose(out[0], expected)


def test_mask_intersection_is_reported_and_applied(labels):
    mask = np.ones(SHAPE, dtype=bool)
    mask[0] = False
    roi = build_roi_set(labels, SHAPE, mask=mask)
    assert roi.rows.sum() == 9
    assert not roi.rows[0].any()
    assert any("intersected with -mask" in n for n in roi.notes)


def test_mask_that_empties_every_roi_raises(labels):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[:, :, 1] = True
    with pytest.raises(ValueError, match="no ROI voxel"):
        build_roi_set(labels, SHAPE, mask=mask)


def test_binary_masks_stack_and_are_named_after_the_file(tmp_path):
    a = np.zeros(SHAPE)
    a[0, 0, 0] = 1
    b = np.zeros(SHAPE)
    b[:2, 0, 0] = 1  # overlaps a
    roi = build_roi_set(
        [_save(tmp_path, "lh_v1.nii.gz", a), _save(tmp_path, "rh_v1.nii.gz", b)], SHAPE
    )
    assert roi.names == ["lh_v1", "rh_v1"]
    assert not roi.disjoint
    assert roi.sizes.tolist() == [1.0, 2.0]
    data = torch.tensor([[2.0], [4.0]])  # rows (0,0,0), (1,0,0)
    assert roi.collapse(data, device=CPU).flatten().tolist() == [2.0, 3.0]


def test_mask_emptying_one_roi_drops_it_and_keeps_the_others_index(tmp_path):
    a = np.zeros(SHAPE)
    a[0, 0, 0] = 1
    b = np.zeros(SHAPE)
    b[3, 2, 1] = 1
    mask = np.ones(SHAPE, dtype=bool)
    mask[3, 2, 1] = False
    roi = build_roi_set(
        [_save(tmp_path, "a.nii.gz", a), _save(tmp_path, "b.nii.gz", b)], SHAPE, mask=mask
    )
    assert roi.names == ["a"]
    assert roi.ids == [1]
    assert any("emptied" in n for n in roi.notes)


def test_paint_round_trips_the_roi_mean_and_leaves_uncovered_zero(labels):
    roi = build_roi_set(labels, SHAPE)
    vol = roi.to_volume(np.array([[1.0, 10.0], [2.0, 20.0]]))
    assert vol.shape == (*SHAPE, 2)
    assert vol[0, 0, 0].tolist() == [1.0, 10.0]
    assert vol[3, 2, 0].tolist() == [2.0, 20.0]
    assert not vol[..., 1, :].any()


def test_grid_mismatch_and_non_integer_labels_are_refused(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        build_roi_set(_save(tmp_path, "x.nii.gz", np.ones((2, 2, 2))), SHAPE)
    with pytest.raises(ValueError, match="integer"):
        build_roi_set(_save(tmp_path, "p.nii.gz", np.full(SHAPE, 0.5)), SHAPE)


def test_write_table_and_timecourses(labels, tmp_path):
    roi = build_roi_set(labels, SHAPE)
    tsv = roi.write_table(tmp_path / "t.tsv", {"r2": np.array([0.1, 0.2]), "b": np.ones((2, 2))})
    lines = open(tsv).read().splitlines()
    assert lines[0].split("\t") == ["roi", "name", "n_voxels", "r2", "b_0", "b_1"]
    assert lines[1].split("\t")[:3] == ["3", "3", "6"]
    txt = roi.write_timecourses(
        tmp_path / "c.tsv", np.array([0.0, 2.0]), np.arange(4.0).reshape(2, 2)
    )
    rows = open(txt).read().splitlines()
    assert rows[0].split("\t") == ["time", "3", "7"]
    assert rows[2].split("\t") == ["2", "1", "3"]
