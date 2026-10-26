"""ffs_pyrf on surface data: GIfTI runs in, a GIfTI bucket on the same mesh out."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.io.gifti import (
    mesh_fingerprint,
    save_gifti_data,
    save_gifti_surface,
)


def _sheet(nx: int, ny: int):
    """A triangulated rectangular sheet: a mesh with real topology, cheap to fit on."""
    vertices = np.array([[x, y, 0.0] for y in range(ny) for x in range(nx)], np.float32)
    faces = []
    for y in range(ny - 1):
        for x in range(nx - 1):
            a = y * nx + x
            faces += [[a, a + 1, a + nx], [a + 1, a + nx + 1, a + nx]]
    return vertices, np.asarray(faces, np.int32)


@pytest.fixture
def surface_runs(tmp_path):
    """Two GIfTI runs of pRF responses to a sweeping bar, on a 10x6 sheet."""
    rng = np.random.default_rng(0)
    n_tps, pixels = 40, 10
    aperture = np.zeros((pixels, pixels, n_tps), np.float32)
    for t in range(n_tps):
        aperture[:, t % pixels, t] = 1.0
    np.save(tmp_path / "aperture.npy", aperture)

    vertices, faces = _sheet(10, 6)
    save_gifti_surface(tmp_path / "mesh.surf.gii", vertices, faces)
    meta = {
        "TR_seconds": "1.0",
        "geometry": "mesh.surf.gii",
        "mesh_fingerprint": mesh_fingerprint(faces, len(vertices)),
    }
    grid_y, grid_x = np.mgrid[0:pixels, 0:pixels]
    runs = []
    for run in range(2):
        series = np.zeros((len(vertices), n_tps), np.float32)
        for v in range(len(vertices)):
            cx, cy = rng.uniform(2, pixels - 2, 2)
            weights = np.exp(-(((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / 8.0))
            series[v] = 100 + 5 * (aperture * weights[:, :, None]).sum((0, 1))
        series += rng.normal(0, 1, series.shape).astype(np.float32)
        path = tmp_path / f"run{run}.func.gii"
        save_gifti_data(path, series, meta=meta, time_series=True)
        runs.append(str(path))
    return runs, str(tmp_path / "aperture.npy"), meta


def _argv(runs, aperture, prefix, *extra):
    return [
        "-input", *runs,
        "-stimulus", *([aperture] * len(runs)),
        "-prefix", prefix,
        "-device", "cpu",
        "-hrf", "canonical",
        "-grid_angles", "4",
        "-grid_ecc_steps", "3",
        "-grid_sigma_steps", "2",
        "-screen_extent", "10",
        *extra,
    ]  # fmt: skip


def test_gifti_runs_fit_and_write_a_bucket_on_the_same_mesh(surface_runs, tmp_path):
    """A surface run goes in and a labeled per-vertex bucket comes out, mesh identity kept."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, meta = surface_runs
    prefix = str(tmp_path / "prf.func.gii")

    assert main(_argv(runs, aperture, prefix)) == 0

    # Through load_nifti, the route every ffs tool reads a .gii by: a (V, 1, 1, K)
    # image whose labels and mesh identity ride in the surface extension.
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import surface_meta

    image = load_nifti(prefix)
    out_meta = surface_meta(image.header) or {}
    assert image.shape[:3] == (60, 1, 1)
    assert out_meta["mesh_fingerprint"] == meta["mesh_fingerprint"]
    labels = out_meta["brick_labels"].split("\t")
    assert {"x", "y", "sigma", "r2"} <= set(labels)
    assert np.isfinite(np.asarray(image.dataobj)[:, 0, 0, labels.index("r2")]).all()


def test_surface_blur_needs_a_mesh_mask(surface_runs, tmp_path):
    """Heat blur reads the mask to know which vertices carry data; say so, don't crash."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, _ = surface_runs
    argv = _argv(runs, aperture, str(tmp_path / "prf.func.gii"), "-do_blur", "4")

    with pytest.raises(SystemExit):
        main(argv)


def test_noise_pool_mask_is_refused_on_surface_input(surface_runs, tmp_path):
    """The pool reaches the loader as a flat array, which has no mesh identity."""
    from fastfuncstuff.cli.pyrf import main

    runs, aperture, _ = surface_runs
    argv = _argv(
        runs, aperture, str(tmp_path / "prf.func.gii"), "-denoise", "-noise_pool_mask", "all"
    )

    with pytest.raises(SystemExit):
        main(argv)
