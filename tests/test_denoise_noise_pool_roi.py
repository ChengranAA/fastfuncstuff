"""-noise_pool_exclude / -noise_pool_mask keep an ROI out of (or alone in) the pool.

The voxels stay in the fit and in every output map; only where the noise
components may be drawn FROM changes.
"""

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli_utils import roi_mask_on_data_rows


def _runs(tmp_path):
    rng = np.random.default_rng(1)
    paths = []
    for i in range(2):
        path = tmp_path / f"run{i}.nii.gz"
        img = nib.Nifti1Image(100 + rng.normal(size=(8, 8, 8, 60)).astype(np.float32), np.eye(4))
        img.header.set_zooms((1, 1, 1, 2))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, path)
        paths.append(str(path))
    (tmp_path / "onsets.1D").write_text("4 22 44\n4 22 44\n")
    roi = np.zeros((8, 8, 8), np.float32)
    roi[:4] = 1  # the x < 4 half
    nib.save(nib.Nifti1Image(roi, np.eye(4)), tmp_path / "roi.nii.gz")
    return paths


def _argv(tmp_path, paths, prefix, *extra):
    return [
        "ffs_denoise",
        "-input",
        *paths,
        "-onsets",
        str(tmp_path / "onsets.1D"),
        "-durations",
        "1",
        "-tr",
        "2",
        "-device",
        "cpu",
        "-polort",
        "3",
        "-brainthresh",
        "0",
        "0",
        "-min_noise_voxels",
        "10",
        "-max_comps",
        "2",
        "-plots",
        "no",
        "-noise_ceiling",
        "off",
        *extra,
        "-prefix",
        str(tmp_path / prefix),
    ]


def _pool(tmp_path, prefix):
    path = tmp_path / f"{prefix}_noise_pool_mask.nii.gz"
    return np.asanyarray(nib.load(str(path)).dataobj)


def test_excluded_roi_is_absent_from_the_noise_pool(tmp_path, monkeypatch):
    from fastfuncstuff.cli.denoise import main

    paths = _runs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, paths, "ex", "-noise_pool_exclude", str(tmp_path / "roi.nii.gz")),
    )
    main()
    pool = _pool(tmp_path, "ex")
    assert pool[:4].sum() == 0  # the excluded half
    assert pool[4:].sum() > 0  # the rest still supplies a pool


def test_noise_pool_mask_keeps_only_the_roi(tmp_path, monkeypatch):
    from fastfuncstuff.cli.denoise import main

    paths = _runs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(tmp_path, paths, "ms", "-noise_pool_mask", str(tmp_path / "roi.nii.gz")),
    )
    main()
    pool = _pool(tmp_path, "ms")
    assert pool[4:].sum() == 0
    assert pool[:4].sum() > 0


def test_roi_on_another_grid_is_resampled_onto_the_data(tmp_path):
    _runs(tmp_path)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    coarse = np.zeros((4, 4, 4), np.float32)
    coarse[:2] = 1
    nib.save(nib.Nifti1Image(coarse, affine), tmp_path / "roi_2mm.nii.gz")
    got = roi_mask_on_data_rows(
        str(tmp_path / "roi_2mm.nii.gz"),
        input_file=str(tmp_path / "run0.nii.gz"),
        mask_flat=None,
        n_voxels=512,
        flag="-noise_pool_mask",
    )
    assert got.numel() == 512
    inside = got.numpy().reshape(8, 8, 8)
    assert inside[:3].all() and not inside[5:].any()


def test_roi_is_read_in_the_masked_voxel_domain(tmp_path):
    _runs(tmp_path)
    mask_flat = np.zeros((8, 8, 8), bool)
    mask_flat[:, :, :6] = True  # the analysed rows
    got = roi_mask_on_data_rows(
        str(tmp_path / "roi.nii.gz"),
        input_file=str(tmp_path / "run0.nii.gz"),
        mask_flat=mask_flat.reshape(-1),
        n_voxels=int(mask_flat.sum()),
        flag="-noise_pool_exclude",
    )
    assert int(got.sum()) == 4 * 8 * 6  # x < 4 AND z < 6


def test_missing_roi_file_is_an_explained_exit(tmp_path):
    with pytest.raises(SystemExit, match="file not found"):
        roi_mask_on_data_rows(
            str(tmp_path / "nope.nii.gz"),
            input_file=str(tmp_path / "run0.nii.gz"),
            mask_flat=None,
            n_voxels=512,
            flag="-noise_pool_exclude",
        )
