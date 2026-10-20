"""ROI x depth sampling along the ribbon, refereed by concentric spheres.

Between white (radius RW) and pial (RP) spheres every depth question has a closed
form: an equidistant bin is a shell of equal thickness, an equivolume bin a shell of
equal volume, and the volume-weighted mean of the radius over a shell [a, b] is
3/4 (b^4 - a^4) / (b^3 - a^3). Data whose value is the radius therefore says
exactly where each bin sits and whether it was weighted by volume.
"""

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.roi_depth import DepthOptions, build_depth_roi_set
from fastfuncstuff.surface.volume_depth import RibbonSurfaces
from tests.test_surface_voxelize import icosphere

RW, RP = 15.0, 21.0
CPU = torch.device("cpu")


def _shell_mean(a, b):
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


def _edges(n, method):
    t = np.linspace(0, 1, n + 1)
    if method == "equidistant":
        return RW + t * (RP - RW)
    return (RW**3 + t * (RP**3 - RW**3)) ** (1 / 3)


@pytest.fixture(scope="module")
def sphere():
    v, f = icosphere(5)
    surf = RibbonSurfaces("lh", v * RW, v * RP, f.astype(np.int64))
    # A 2 mm "EPI" grid whose value is the distance from the centre, plus a second
    # time point at twice that, to see the collapse keep columns apart.
    n = 26
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(ijk @ affine[:3, :3].T + affine[:3, 3], axis=1)
    data = np.stack([r, 2 * r], 1).astype(np.float32)
    return surf, affine, (n, n, n), data, v


def _all_label(tmp_path, n_vertices, ids=None):
    ids = np.arange(n_vertices) if ids is None else ids
    path = tmp_path / "lh.all.label"
    rows = "\n".join(f"{i}  0 0 0 1" for i in ids)
    path.write_text(f"#!ascii label\n{len(ids)}\n{rows}\n")
    return str(path)


def _means(roi_set, data):
    sel = torch.from_numpy(data[roi_set.rows.reshape(-1)])
    return roi_set.collapse(sel, device=CPU).numpy()


@pytest.mark.parametrize("method", ["equidistant", "equivolume"])
def test_bins_land_on_the_analytic_shells(sphere, tmp_path, method):
    surf, affine, shape, data, v = sphere
    label = _all_label(tmp_path, len(v))
    opts = DepthOptions(n_depths=3, method=method, dxyz=1.0)
    roi_set = build_depth_roi_set(label, [surf], affine, shape, opts=opts, device=CPU)
    assert roi_set.index["depth"] == ["1", "2", "3"] and roi_set.names == ["lh.all"] * 3
    got = _means(roi_set, data)
    e = _edges(3, method)
    want = np.array([_shell_mean(a, b) for a, b in zip(e[:-1], e[1:], strict=True)])
    # Cubic reads of the 2 mm grid carry no curvature bias (trilinear's is h^2/(6r), up
    # to 0.046 mm here). What is left is the cell tiling of a faceted sphere: at 1 mm
    # spacing up to 0.022 mm in the outer bin, halving with the spacing (0.008 at 0.5).
    np.testing.assert_allclose(got[:, 0], want, atol=0.025)
    np.testing.assert_allclose(got[:, 1], 2 * want, atol=0.05)
    # Volume per bin: equal shells for equivolume; equidistant grows outward.
    vol = roi_set.sizes
    want_vol = 4 / 3 * np.pi * np.diff(e**3)
    np.testing.assert_allclose(vol, want_vol, rtol=0.005)


def test_the_two_methods_differ_where_they_should(sphere, tmp_path):
    surf, affine, shape, data, v = sphere
    label = _all_label(tmp_path, len(v))
    out = {}
    for method in ("equidistant", "equivolume"):
        opts = DepthOptions(n_depths=3, method=method, dxyz=1.0)
        out[method] = _means(build_depth_roi_set(label, [surf], affine, shape, opts=opts), data)
    # On a sphere the outer shells hold more volume, so equal-volume bins move out.
    assert (out["equivolume"][:, 0] - out["equidistant"][:, 0])[1] > 0.3


def test_one_depth_is_the_whole_ribbon_of_a_cap(sphere, tmp_path):
    surf, affine, shape, data, v = sphere
    cap = np.flatnonzero(v[:, 2] > 0.5)
    label = _all_label(tmp_path, len(v), cap)
    roi_set = build_depth_roi_set(label, [surf], affine, shape, opts=DepthOptions(dxyz=1.0))
    assert roi_set.n_rois == 1
    # z of every grid voxel the cap reads: none of them sits below the cap's base.
    ijk = np.argwhere(roi_set.rows)
    z = (ijk @ affine[:3, :3].T + affine[:3, 3])[:, 2]
    assert z.min() > 0.5 * RW - 4.0
    want = _shell_mean(RW, RP)
    np.testing.assert_allclose(_means(roi_set, data)[0, 0], want, atol=0.02)


def test_volume_roi_maps_to_columns_and_mask_limits_reads(sphere, tmp_path):
    import nibabel as nib

    surf, affine, shape, data, v = sphere
    # A coarse half-space ROI on its own 3 mm grid: x > 0.
    n = 20
    aff3 = np.diag([3.0, 3.0, 3.0, 1.0])
    aff3[:3, 3] = -28.5
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    x = (ijk @ aff3[:3, :3].T + aff3[:3, 3])[..., 0]
    nib.save(nib.Nifti1Image((x > 0).astype(np.int16), aff3), tmp_path / "half.nii.gz")
    mask = np.ones(shape, bool)
    mask[:, :, : shape[2] // 2] = False  # z < 0 excluded
    opts = DepthOptions(n_depths=2, dxyz=1.0)
    roi_set = build_depth_roi_set(str(tmp_path / "half.nii.gz"), [surf], affine, shape, mask, opts)
    assert roi_set.n_rois == 2
    xyz = np.argwhere(roi_set.rows) @ affine[:3, :3].T + affine[:3, 3]
    # Cubic reads reach one voxel past trilinear's: taps -1..+2 of a 2 mm grid.
    assert (xyz[:, 0] > -4.5).all() and (xyz[:, 2] > -2.5).all()
    assert mask.reshape(-1)[roi_set.rows.reshape(-1)].all()
    assert any("-mask" in n for n in roi_set.notes)
    # Same columns at every depth: column counts agree to the grid's rounding.
    cols = [int(c) for c in roi_set.index["n_columns"]]
    assert abs(cols[0] - cols[1]) < 0.05 * max(cols)


def test_long_rows_and_table(sphere, tmp_path):
    surf, affine, shape, _, v = sphere
    label = _all_label(tmp_path, len(v))
    roi_set = build_depth_roi_set(label, [surf], affine, shape, opts=DepthOptions(2, dxyz=1.0))
    table = roi_set.write_table(tmp_path / "t.tsv", {"beta": np.array([1.0, 2.0])})
    rows = [ln.split("\t") for ln in open(table).read().splitlines()]
    assert rows[0] == [
        "roi",
        "name",
        "hemi",
        "depth",
        "depth_frac",
        "n_columns",
        "volume_mm3",
        "beta",
    ]
    assert [r[3] for r in rows[1:]] == ["1", "2"] and rows[1][2] == "lh"
    curves = roi_set.write_timecourses(tmp_path / "c.tsv", np.array([0.0, 1.5]), np.ones((2, 2)))
    rows = [ln.split("\t") for ln in open(curves).read().splitlines()]
    assert rows[0] == ["time", "roi", "name", "hemi", "depth", "depth_frac", "n_columns", "beta"]
    assert len(rows) == 1 + 2 * 2


def test_label_coordinates_pick_and_check_the_hemisphere(tmp_path):
    """A viewer .label carries white tkregister coordinates: they must sit on the
    hemisphere it is used with, which catches an lh label on rh (same vertex count)."""
    from fastfuncstuff.io.freesurfer import read_surface, write_label
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces
    from tests.test_util_surf2layers import _subject

    subj, master = _subject(tmp_path)
    surfaces = resolve_ribbon_surfaces(subj)
    assert [s.name for s in surfaces] == ["lh", "rh"] and surfaces[0].tkr_to_scanner is not None
    ids = np.arange(0, 600, 3)
    for hemi in ("lh", "rh"):
        tkr = read_surface(subj / "surf" / f"{hemi}.white").vertices[ids]
        write_label(tmp_path / f"roi.{hemi}.label", ids, tkr, np.ones(ids.size))
    # rh coordinates under an lh name: same vertex count, the wrong surface.
    (tmp_path / "bad.lh.label").write_bytes((tmp_path / "roi.rh.label").read_bytes())
    import nibabel as nib

    img = nib.load(master)
    opts = DepthOptions(dxyz=1.0)
    good = build_depth_roi_set(
        [str(tmp_path / "roi.lh.label"), str(tmp_path / "roi.rh.label")],
        surfaces, img.affine, img.shape, opts=opts, device=CPU,
    )  # fmt: skip
    assert [h for h in good.index["hemi"]] == ["lh", "rh"]
    assert not any("off this white" in n for n in good.notes)
    with pytest.raises(ValueError, match="from the white surface"):
        build_depth_roi_set(
            str(tmp_path / "bad.lh.label"), surfaces, img.affine, img.shape, opts=opts
        )
