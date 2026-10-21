"""``ffs_util_surf2layers -roi``: layers for one patch, measured as if for all of it."""

import numpy as np
import pytest
import torch

from fastfuncstuff.surface.mesh import MeshTopology, close_patch, dilate_patch
from fastfuncstuff.surface.volume_depth import (
    RIM_CSF,
    RIM_GM,
    RIM_WM,
    cortical_depth_volumes,
    export_roi_depth_volumes,
    grow_columns,
)
from tests.test_surface_volume_depth import CENTRE, R2, radius, shell

CPU = torch.device("cpu")


class _Roi:
    """What ``processing.roi_depth`` hands the export."""

    def __init__(self, name, members, value=1):
        self.name, self.members, self.value, self.rid = name, members, value, value


def cap(v, cos=0.8):
    """Vertices of a cap around +x, as a mask on the unit sphere's vertices."""
    return v[:, 0] / np.linalg.norm(v, axis=1) > cos


def test_patch_closing_fills_holes_without_moving_the_outline():
    s, v, _, _ = shell()
    mid = 0.5 * (s.white + s.pial)
    topo = MeshTopology.from_faces(s.faces, len(s.white))
    patch = cap(v, 0.8)
    # Punch a few one-vertex holes well inside it: what a threshold leaves behind.
    # (A hole on the patch's own rim is a notch in the outline, not a hole, and a
    # closing is meant to leave the outline alone.)
    holes = np.flatnonzero(cap(v, 0.9))[::7]
    speckled = patch.copy()
    speckled[holes] = False
    closed = close_patch(mid, topo, speckled, 1.5)
    assert closed[holes].all(), "closing did not fill the holes"
    # The outline stays put: closing adds nothing far outside the original patch.
    assert not (closed & ~patch).any()
    # A dilation of the same radius would have grown it instead.
    assert (dilate_patch(mid, topo, speckled, 1.5) & ~patch).sum() > 100


def test_grow_columns_fills_then_grows():
    s, v, _, _ = shell()
    patch = cap(v, 0.9)
    own, padded = grow_columns([s], {0: patch}, fill_mm=1.0, surround_mm=4.0)
    assert own[0].sum() >= patch.sum()
    assert padded[0][own[0]].all() and padded[0].sum() > own[0].sum()
    # The surround reaches about as far as asked, along the midthickness.
    mid = 0.5 * (s.white + s.pial)
    grown = mid[padded[0] & ~own[0]]
    from scipy.spatial import KDTree

    d = KDTree(mid[own[0]]).query(grown)[0]
    assert d.max() <= 4.0 + 1e-6


def test_restriction_keeps_exactly_what_the_full_run_computed():
    """The depths of a restricted run are the unrestricted ones, voxel for voxel.

    Only the voxels are dropped -- the distances and the column CDF are measured
    first -- so with the pooling pinned the two runs must agree bit for bit.
    """
    s, v, affine, shape = shell(step=0.5)
    full = cortical_depth_volumes([s], affine, shape, device=CPU, owners=True, column_rounds=3)
    part = cortical_depth_volumes(
        [s], affine, shape, device=CPU, columns=[cap(v, 0.8)], column_rounds=3
    )
    assert 0 < part.n_gm < full.n_gm and part.n_outside == full.n_gm - part.n_gm
    keep = np.isin(full.gm_idx, part.gm_idx)
    np.testing.assert_array_equal(full.gm_idx[keep], part.gm_idx)
    np.testing.assert_array_equal(full.rho[keep], part.rho)
    np.testing.assert_array_equal(full.equivol[keep], part.equivol)
    np.testing.assert_array_equal(full.thick[keep], part.thick)
    # A dropped voxel is not a border: the patch's lateral edge is open, not CSF.
    assert not np.isin(part.border_idx, full.gm_idx[~keep]).any()


def test_a_hemisphere_the_region_misses_is_dropped_unmeasured():
    """A hemisphere with no column of the region costs nothing but its fill.

    Its voxels are all dropped, and "0 voxels per column" would otherwise ask the
    equivolume CDF for its 500-round maximum over a whole hemisphere.
    """
    import time

    from fastfuncstuff.surface.volume_depth import RibbonSurfaces

    s, v, affine, shape = shell(step=0.5)
    # A second hemisphere on the same grid, clear of the first (the shells must not
    # intersect, or one's white surface turns the other's GM into WM), which the
    # region will not touch. The grid is widened along x to hold both.
    off = np.array([19.0, 0.0, 0.0])
    other = RibbonSurfaces("rh", s.white + off, s.pial + off, s.faces)
    shape = (shape[0] + int(2 * off[0]), shape[1], shape[2])
    empty = np.zeros(len(s.white), bool)
    t0 = time.time()
    out = cortical_depth_volumes(
        [s, other], affine, shape, device=CPU, columns=[cap(v, 0.8), empty]
    )
    assert [h for h, _ in out.smooth_rounds] == [0], out.smooth_rounds
    assert out.n_gm > 0 and all(r < 500 for _, r in out.smooth_rounds)
    assert time.time() - t0 < 60


def test_a_partial_depth_volume_roi_becomes_whole_columns(tmp_path):
    """A blob covering only the outer half of the ribbon gives full-depth columns."""
    from fastfuncstuff.io.afni import load_nifti, save_nifti
    from fastfuncstuff.processing.roi_depth import CLAIM_ANY, CLAIM_MID, read_depth_rois

    s, v, affine, shape = shell(step=0.5)
    r = radius(affine, shape)
    x = (np.arange(shape[0]) * 0.5 - CENTRE[0])[:, None, None]
    blob = ((r > 8.0) & (r < R2 - 0.2) & (x / np.maximum(r, 1e-9) > 0.8)).astype(np.int16)
    assert blob.sum() > 100
    path = str(tmp_path / "blob.nii.gz")
    save_nifti(blob, path, affine=affine)

    notes: list[str] = []
    rois = read_depth_rois([path], [s], notes, CLAIM_ANY)
    assert [r_.name for r_ in rois] == ["blob"]
    # This blob sits in the outer third of the ribbon, so reading at mid-depth alone
    # finds nothing at all -- which is why 'any' is the default here.
    with pytest.raises(ValueError, match="no ROI covers"):
        read_depth_rois([path], [s], notes, CLAIM_MID)

    res = export_roi_depth_volumes(
        [s], affine, shape, str(tmp_path / "L"), ".nii.gz", rois,
        fill_mm=1.0, pad_mm=1.0, device=CPU,
    )  # fmt: skip
    assert len(res) == 1
    out = res[0].out
    assert res[0].enough_surround, res[0].pooled_mm
    # Cropped, and most of what was computed was surround that got thrown away.
    # (The box is only about half the master's here: the auto surround is ~14 mm at
    # this coarse step and the whole test sphere is 25 mm across. On real cortex at
    # 0.2 mm the surround is a couple of millimetres and the box is tiny.)
    assert np.prod(out.shape) < np.prod(shape)
    assert out.n_outside > out.n_gm
    # The columns span the depth, although the blob only touched the outer half.
    assert out.rho.min() < 0.1 and out.rho.max() > 0.9
    rim = out.rim
    assert set(np.unique(rim)) == {0, RIM_CSF, RIM_WM, RIM_GM}
    # Both borders are there, so LN2_LAYERS could rebuild the depths from the rim.
    assert (rim == RIM_WM).sum() > 50 and (rim == RIM_CSF).sum() > 50
    # Written files: every LN2 output plus the region as a mask, on the same grid.
    head = f"{tmp_path}/L_blob_"
    tags = {p.removeprefix(head).removesuffix(".nii.gz") for p in res[0].written}
    assert {"rim", "metric_equivol", "thickness", "roi"} <= tags
    mask = np.asanyarray(load_nifti(f"{tmp_path}/L_blob_roi.nii.gz").dataobj)
    np.testing.assert_array_equal(mask > 0, out.metric_equidist.astype(bool) | (rim == RIM_GM))


def test_too_small_a_surround_is_reported(tmp_path):
    from fastfuncstuff.surface.volume_depth import roi_report

    s, v, affine, shape = shell(step=0.5)
    rois = [_Roi("cap", {0: cap(v, 0.9)})]
    res = export_roi_depth_volumes(
        [s], affine, shape, str(tmp_path / "L"), ".nii.gz", rois,
        fill_mm=0.0, surround_mm=0.01, device=CPU,
    )  # fmt: skip
    assert not res[0].enough_surround
    assert "-roi_surround" in roi_report(res[0])[-1]
