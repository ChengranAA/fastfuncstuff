"""ffs_nwarp -atlas/-rois: ROI means (and ROI x depth) read through the chain, end to end."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli import nwarp
from fastfuncstuff.io.gifti import save_gifti_surface
from tests.test_nwarp_points import _chain, _oblique_affine, _write_aff1d
from tests.test_surface_voxelize import icosphere

RW, RP, N, NT = 15.0, 21.0, 26, 3


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("nwroi")
    v, f = icosphere(4)
    save_gifti_surface(tmp / "lh.white.surf.gii", v * RW, f)
    save_gifti_surface(tmp / "lh.pial.surf.gii", v * RP, f)
    cap = np.flatnonzero(v[:, 2] > 0.3)
    rows = "\n".join(f"{i}  0 0 0 1" for i in cap)
    (tmp / "cap.lh.label").write_text(f"#!ascii label\n{cap.size}\n{rows}\n")
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = -(N - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1).reshape(-1, 3)
    r = np.linalg.norm(ijk @ aff[:3, :3].T + aff[:3, 3], axis=1).reshape(N, N, N)
    data = np.stack([r * (1 + 0.5 * t) for t in range(NT)], -1).astype(np.float32)
    img = nib.Nifti1Image(data, aff)
    img.header.set_zooms((2, 2, 2, 2.0))
    nib.save(img, tmp / "r.nii")
    _write_aff1d(tmp / "ident.aff12.1D", [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]] * NT)
    return tmp


def _run(*argv):
    nwarp.main([*map(str, argv)])


def _depth_args(w, prefix, *extra):
    return [
        "-source", w / "r.nii", "-nwarp", w / "ident.aff12.1D", "-master", w / "r.nii",
        "-prefix", prefix, "-atlas", w / "cap.lh.label",
        "-white_mesh", w / "lh.white.surf.gii", "-pial_mesh", w / "lh.pial.surf.gii",
        "-depths", "3", "-interp", "cubic", "-device", "cpu", "-verb", "0", *extra,
    ]  # fmt: skip


def test_depth_rois_through_the_cli_write_the_three_files(world):
    prefix = str(world / "out")
    _run(*_depth_args(world, prefix))
    img = nib.load(f"{prefix}.roi.nii.gz")
    assert img.shape == (1, 1, 3, NT)
    d = np.asarray(img.dataobj)[0, 0]  # (depth, time)
    assert (np.diff(d[:, 0]) > 1.5).all()  # radius grows outward: depth 1 is deepest
    e = RW + np.linspace(0, 1, 4) * (RP - RW)
    shell = 0.75 * np.diff(e**4) / np.diff(e**3)
    # Default spacing is one 2 mm voxel along the surface: 0.054 mm worst here, 0.025 at 1 mm.
    np.testing.assert_allclose(d[:, 0], shell, atol=0.08)
    np.testing.assert_allclose(d[:, 2], 2 * shell, atol=0.16)
    rows = [ln.split("\t") for ln in open(f"{prefix}.roi.tsv").read().splitlines()]
    assert rows[0][:8] == ["x", "z", "roi", "name", "hemi", "depth", "depth_frac", "n_columns"]
    assert {r[3] for r in rows[1:]} == {"cap.lh"}
    doc = json.load(open(f"{prefix}.roi.json"))
    assert doc["axes"] == ["roi", "1", "depth", "time"] and doc["interp"] == "cubic"
    assert doc["master"].endswith("r.nii") and doc["TR"] == 2.0


def test_a_small_read_cap_runs_passes_and_changes_nothing(world):
    one = str(world / "one")
    _run(*_depth_args(world, one, "-roi_max_reads", "10000000"))
    two = str(world / "two")
    # Two ROIs (the cap's halves) that cannot share a pass under the small cap.
    v, _ = icosphere(4)
    for name, sel in (("up", v[:, 2] > 0.3), ("dn", v[:, 2] < -0.3)):
        ids = np.flatnonzero(sel)
        (world / f"{name}.lh.label").write_text(
            "#!ascii label\n" + f"{ids.size}\n" + "\n".join(f"{i}  0 0 0 1" for i in ids) + "\n"
        )
    args = _depth_args(world, two)
    args[args.index("-atlas") + 1] = world / "up.lh.label"
    args.insert(args.index("-atlas") + 2, world / "dn.lh.label")
    _run(*args, "-roi_max_reads", "10000000")
    big = json.load(open(f"{two}.roi.json"))
    assert big["n_passes"] == 1
    small = str(world / "small")
    args2 = _depth_args(world, small)
    args2[args2.index("-atlas") + 1] = world / "up.lh.label"
    args2.insert(args2.index("-atlas") + 2, world / "dn.lh.label")
    # A cap over either ROI's own (halo-inclusive) tiling but under both together.
    from fastfuncstuff.processing.roi_depth import read_depth_rois, tile_costs
    from fastfuncstuff.surface.volume_depth import RibbonSurfaces

    v4, f4 = icosphere(4)
    surf = RibbonSurfaces("lh", (v4 * RW).astype(np.float32), (v4 * RP).astype(np.float32), f4)
    rois = read_depth_rois(
        [str(world / "up.lh.label"), str(world / "dn.lh.label")], [surf], []
    )  # fmt: skip
    cap = int(tile_costs(rois, [surf], 2.0, 3, 0.5).max()) + 1
    _run(*args2, "-roi_max_reads", str(cap))
    doc = json.load(open(f"{small}.roi.json"))
    assert doc["n_passes"] == 2
    a = np.asarray(nib.load(f"{two}.roi.nii.gz").dataobj)
    b = np.asarray(nib.load(f"{small}.roi.nii.gz").dataobj)
    np.testing.assert_allclose(b, a, atol=1e-5)
    assert a.shape == (2, 1, 3, NT)


def test_volume_atlas_through_a_real_chain_is_the_mean_of_the_volume_output(tmp_path):
    n, nt = 10, 4
    aff = _oblique_affine(2.0, 12.0)
    src, chain = _chain(tmp_path, n, nt, aff)
    lab = np.zeros((n, n, n), np.int16)
    lab[1:5, 2:8, 2:8] = 1
    lab[6:9, 1:9, 3:7] = 2
    nib.Nifti1Image(lab, aff).to_filename(str(tmp_path / "atlas.nii"))
    common = ["-source", src, "-nwarp", " ".join(map(str, chain)), "-master", src]
    common += ["-interp", "wsinc5", "-device", "cpu", "-verb", "0", "-no_autopad"]
    _run(*common, "-prefix", tmp_path / "vol.nii")
    vol = np.asarray(nib.load(str(tmp_path / "vol.nii")).dataobj)
    want = np.stack([vol[lab == k].mean(axis=0) for k in (1, 2)])
    _run(*common, "-prefix", tmp_path / "roi", "-atlas", tmp_path / "atlas.nii")
    got = np.asarray(nib.load(str(tmp_path / "roi.roi.nii.gz")).dataobj)  # (roi, 1, 1, t)
    assert got.shape == (2, 1, 1, nt)
    np.testing.assert_allclose(got[:, 0, 0, :], want, atol=2e-4 * np.abs(want).max())
    rows = [ln.split("\t") for ln in open(tmp_path / "roi.roi.tsv").read().splitlines()]
    assert rows[0][:4] == ["x", "z", "roi", "name"] and rows[0][4] == "n_voxels"
    assert [r[4] for r in rows[1:]] == [str(int((lab == 1).sum())), str(int((lab == 2).sum()))]


def test_refusals(world):
    base = [
        "-source", world / "r.nii", "-nwarp", world / "ident.aff12.1D", "-master", world / "r.nii",
        "-prefix", world / "bad", "-device", "cpu", "-verb", "0",
    ]  # fmt: skip
    with pytest.raises(SystemExit, match="-depths"):  # depth flags without surfaces
        _run(*base, "-atlas", world / "cap.lh.label", "-depths", "3")
    with pytest.raises(SystemExit, match="volume options"):
        _run(*_depth_args(world, world / "bad"), "-dxyz", "1")


def _psf_table(path, k):
    rows = [ln.split("\t") for ln in open(path).read().splitlines()]
    assert rows[0] == ["roi", "name", "hemi", "depth", "true_depth", "weight"]
    m = np.zeros((k, k))
    for r in rows[1:]:
        m[int(r[3]) - 1, int(r[4]) - 1] = float(r[5])
    return m


@pytest.mark.slow  # 22s
def test_the_depth_psf_through_an_identity_chain_is_the_grid_one(world):
    """With no transform the native EPI grid is the master grid, so the matrix nwarp
    computes from exact chain coordinates must equal the one computed for data on that
    grid -- the same voxels, the same reads, the same kernel."""
    from fastfuncstuff.processing.roi_depth import DepthOptions, depth_sampling_matrices
    from fastfuncstuff.surface.volume_depth import RibbonSurfaces

    prefix = str(world / "psf")
    _run(*_depth_args(world, prefix, "-depth_dxyz", "1.0"))
    got = _psf_table(f"{prefix}.roi.depth_psf.tsv", 3)
    v, f = icosphere(4)
    surf = RibbonSurfaces("lh", v * RW, v * RP, f.astype(np.int64))
    img = nib.load(world / "r.nii")
    ((_, _, want),) = depth_sampling_matrices(
        str(world / "cap.lh.label"), [surf], img.affine, img.shape[:3],
        DepthOptions(n_depths=3, dxyz=1.0, interp="cubic"), normalise=False,
    )  # fmt: skip
    # Float32 meshes (the GIfTI) against float64 here: a few cells on voxel boundaries
    # fall the other way (1e-4 at worst).
    np.testing.assert_allclose(got, want, rtol=0, atol=3e-4)
    assert (np.diag(got) > 0.4).all()


@pytest.mark.slow  # 14s
def test_several_depth_counts_read_the_data_once_and_write_a_set_per_count(world):
    """-depths 3 5: one pass over the data, PREFIX.K3.roi.* and PREFIX.K5.roi.*; each
    count's profile is the analytic shell means (so the shared tiling serves both) and the
    two counts agree where their bins coincide in volume (the whole ribbon)."""
    prefix = str(world / "kk")
    args = _depth_args(world, prefix, "-depth_dxyz", "1.0")
    args[args.index("-depths") + 1 : args.index("-depths") + 2] = ["3", "5"]
    _run(*args)
    for k in (3, 5):
        img = nib.load(f"{prefix}.K{k}.roi.nii.gz")
        assert img.shape == (1, 1, k, NT)
        d = np.asarray(img.dataobj)[0, 0, :, 0]
        e = RW + np.linspace(0, 1, k + 1) * (RP - RW)
        shell = 0.75 * np.diff(e**4) / np.diff(e**3)
        np.testing.assert_allclose(d, shell, atol=0.06)
        psf = _psf_table(f"{prefix}.K{k}.roi.depth_psf.tsv", k)
        assert (np.argmax(psf, axis=1) == np.arange(k)).all()
        doc = json.load(open(f"{prefix}.K{k}.roi.json"))
        assert doc["K"] == k and doc["depth_counts"] == [3, 5] and doc["n_passes"] == 1
    # The same reads: both counts' volume-weighted mean over the whole ribbon agree.
    a = np.asarray(nib.load(f"{prefix}.K3.roi.nii.gz").dataobj)[0, 0, :, 0]
    b = np.asarray(nib.load(f"{prefix}.K5.roi.nii.gz").dataobj)[0, 0, :, 0]
    vol3 = np.asarray(
        [float(r.split("\t")[8]) for r in open(f"{prefix}.K3.roi.tsv").read().splitlines()[1:]]
    )
    vol5 = np.asarray(
        [float(r.split("\t")[8]) for r in open(f"{prefix}.K5.roi.tsv").read().splitlines()[1:]]
    )
    assert (a * vol3).sum() / vol3.sum() == pytest.approx((b * vol5).sum() / vol5.sum(), abs=0.01)
