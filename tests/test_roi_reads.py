"""ROI means read through the transform chain (ffs_nwarp -atlas/-rois), refereed twice.

* A volume ROI is read at its voxels' centres, so through any chain the result must be
  the ROI mean of nwarp's own volume output -- the same reads, averaged afterwards.
* On concentric white/pial spheres every depth bin has a closed-form mean, and reading
  in passes under a small budget must not change a number.
"""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.nwarpforge import nwarpforge
from fastfuncstuff.processing.roi_collapse import build_roi_set
from fastfuncstuff.processing.roi_depth import DepthOptions, pass_groups, tile_costs
from fastfuncstuff.processing.roi_reads import (
    default_max_reads,
    plan_depth_reads,
    plan_volume_reads,
    read_through_chain,
    write_roi_outputs,
)
from fastfuncstuff.surface.volume_depth import RibbonSurfaces
from tests.test_nwarp_points import _chain, _oblique_affine, _write_aff1d
from tests.test_surface_voxelize import icosphere

DEV = torch.device("cpu")


def test_volume_roi_read_through_a_chain_is_the_mean_of_the_volume_output(tmp_path):
    n, nt = 10, 4
    aff = _oblique_affine(2.0, 12.0)
    src_path, chain = _chain(tmp_path, n, nt, aff)
    lab = np.zeros((n, n, n), np.int16)
    lab[1:5, 2:8, 2:8] = 1
    lab[6:9, 1:9, 3:7] = 2
    nib.Nifti1Image(lab, aff).to_filename(str(tmp_path / "atlas.nii"))

    common = dict(
        source_path=str(src_path), nwarp_specs=chain, master_path=str(src_path),
        interp="wsinc5", device=DEV, verb=0, auto_pad=False,
    )  # fmt: skip
    nwarpforge(prefix=str(tmp_path / "vol.nii"), **common)
    vol = np.asarray(nib.load(str(tmp_path / "vol.nii")).dataobj)  # (x, y, z, t)
    want = np.stack([vol[lab == k].mean(axis=0) for k in (1, 2)])  # (roi, t)

    roi_set = build_roi_set(str(tmp_path / "atlas.nii"), (n, n, n))
    plan = plan_volume_reads(roi_set, aff)
    got = read_through_chain(
        plan,
        str(src_path),
        chain,
        str(src_path),
        interp="wsinc5",
        device=DEV,
        verb=0,
        auto_pad=False,
    )
    np.testing.assert_allclose(got.values.T, want, atol=2e-4 * np.abs(want).max())
    assert (got.cover > 0.9999).all()


RW, RP, N = 15.0, 21.0, 26


@pytest.fixture(scope="module")
def sphere_world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sphere")
    v, f = icosphere(4)
    surf = RibbonSurfaces("lh", v * RW, v * RP, f.astype(np.int64))
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -(N - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(ijk @ aff[:3, :3].T + aff[:3, 3], axis=1).reshape(N, N, N)
    nt = 3
    data = np.stack([r * (1 + 0.5 * t) for t in range(nt)], -1).astype(np.float32)
    path = tmp / "r.nii"
    nib.Nifti1Image(data, aff).to_filename(str(path))
    ident = tmp / "ident.aff12.1D"
    _write_aff1d(ident, [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]] * nt)
    label = tmp / "lh.all.label"
    label.write_text(
        "#!ascii label\n" + f"{len(v)}\n" + "\n".join(f"{i}  0 0 0 1" for i in range(len(v))) + "\n"
    )
    return surf, str(path), str(ident), str(label), len(v)


def _shell_mean(a, b):
    return 0.75 * (b**4 - a**4) / (b**3 - a**3)


def test_ribbon_reads_land_on_the_analytic_shells_in_passes(sphere_world):
    surf, path, ident, label, n_v = sphere_world
    opts = DepthOptions(n_depths=3, dxyz=1.0)
    kw = dict(interp="cubic", device=DEV, verb=0, auto_pad=False)

    one = plan_depth_reads([label], [surf], 2.0, opts, max_reads=10_000_000, device=DEV)
    a = read_through_chain(one, path, [ident], path, **kw)
    assert len(one.passes) == 1 and a.values.shape == (3, 3)
    e = RW + np.linspace(0, 1, 4) * (RP - RW)
    want = np.array([_shell_mean(x, y) for x, y in zip(e[:-1], e[1:], strict=True)])
    np.testing.assert_allclose(a.values[0], want, atol=0.03)
    np.testing.assert_allclose(a.values[2], 2 * want, atol=0.06)  # frame 2 is 2x frame 0
    assert (a.cover > 0.999).all()
    np.testing.assert_allclose(one.sizes, 4 / 3 * np.pi * np.diff(e**3), rtol=0.01)


def test_two_rois_in_passes_equal_one_pass(sphere_world, tmp_path):
    surf, path, ident, label, n_v = sphere_world
    v, _ = icosphere(4)
    up = np.flatnonzero(v[:, 2] > 0)
    down = np.flatnonzero(v[:, 2] <= 0)
    files = []
    for name, ids in (("lh.up", up), ("lh.down", down)):
        p = tmp_path / f"{name}.label"
        p.write_text(
            "#!ascii label\n" + f"{len(ids)}\n" + "\n".join(f"{i}  0 0 0 1" for i in ids) + "\n"
        )
        files.append(str(p))
    opts = DepthOptions(n_depths=2, dxyz=1.0)
    kw = dict(interp="cubic", device=DEV, verb=0, auto_pad=False)
    big = plan_depth_reads(files, [surf], 2.0, opts, max_reads=10_000_000, device=DEV)
    # A cap over either ROI's own (halo-inclusive) tiling but under both together.
    cap = int(tile_costs(big.build["rois"], [surf], 1.0, 2, 0.5).max()) + 1
    small = plan_depth_reads(files, [surf], 2.0, opts, max_reads=cap, device=DEV)
    assert len(big.passes) == 1 and len(small.passes) == 2
    ra = read_through_chain(big, path, [ident], path, **kw)
    rb = read_through_chain(small, path, [ident], path, **kw)
    np.testing.assert_allclose(rb.values, ra.values, atol=1e-5)
    np.testing.assert_allclose(rb.cover, ra.cover, atol=1e-6)
    assert any("2 pass" in n for n in small.notes)


def test_a_roi_over_the_budget_is_refused_with_advice(sphere_world):
    surf, path, ident, label, n_v = sphere_world
    with pytest.raises(ValueError, match="raise -roi_max_reads"):
        plan_depth_reads([label], [surf], 2.0, DepthOptions(n_depths=3, dxyz=1.0), max_reads=1000)


def test_pass_groups_fit_first_largest_first():
    costs = np.array([5, 4, 3, 3, 1])
    groups = pass_groups(costs, 7)
    assert sorted(i for g in groups for i in g) == [0, 1, 2, 3, 4]
    assert all(costs[g].sum() <= 7 for g in groups)
    assert pass_groups(costs, None) == [[0, 1, 2, 3, 4]]
    assert pass_groups(costs, 100) == [[0, 1, 2, 3, 4]]


def test_outputs_are_roi_by_one_by_depth_by_time_with_a_table(sphere_world, tmp_path):
    surf, path, ident, label, n_v = sphere_world
    plan = plan_depth_reads(
        [label], [surf], 2.0, DepthOptions(n_depths=3, dxyz=1.0), max_reads=10_000_000, device=DEV
    )
    res = read_through_chain(
        plan, path, [ident], path, interp="cubic", device=DEV, verb=0, auto_pad=False
    )
    paths = write_roi_outputs(str(tmp_path / "out"), plan, res, tr=2.0)
    img = nib.load(paths[0])
    assert img.shape == (1, 1, 3, 3) and img.header["intent_name"].tobytes().startswith(b"ffs_roi")
    assert float(img.header.get_zooms()[3]) == 2.0
    data = np.asarray(img.dataobj)
    np.testing.assert_allclose(data[0, 0, :, 0], res.values[0], rtol=1e-6)  # depth = z
    rows = [ln.split("\t") for ln in open(paths[1]).read().splitlines()]
    assert rows[0] == [
        "x", "z", "roi", "name", "hemi", "depth", "depth_frac", "n_columns", "volume_mm3", "coverage",
    ]  # fmt: skip
    assert [r[1] for r in rows[1:]] == ["0", "1", "2"] and [r[5] for r in rows[1:]] == [
        "1",
        "2",
        "3",
    ]
    doc = json.load(open(paths[2]))
    assert doc["axes"] == ["roi", "1", "depth", "time"] and doc["n_depths"] == 3
    assert doc["depth_order"].startswith("depth 1 = deepest") and doc["n_passes"] == 1
    assert len(np.load(paths[3]).files) == 1


def test_default_budget_is_bounded_and_positive():
    n = default_max_reads(DEV)
    assert 200_000 <= n <= 60_000_000
