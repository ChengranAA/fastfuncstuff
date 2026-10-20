"""The ribbon tiling behind thick-depth reads, refereed by concentric spheres.

Between a white sphere (radius RW) and a pial one (RP) the cells must add up to the
exact volume between the two polyhedra, equidistant bins are shells of equal
thickness and equivolume bins shells of equal volume -- also when every pial vertex
slides sideways of its white partner, as FreeSurfer's do, because the pairing only
tiles the ribbon and never decides depth.
"""

import numpy as np
import pytest

from fastfuncstuff.surface.slab import patch_faces, ribbon_cells
from fastfuncstuff.surface.volume_depth import RibbonSurfaces
from tests.test_surface_voxelize import icosphere

RW, RP = 15.0, 21.0


def _polyhedron_volume(x, f):
    t = x[f]
    return np.einsum("fx,fx->f", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6


def _sphere(slide=0.0):
    v, f = icosphere(4)
    c, s = np.cos(slide), np.sin(slide)
    turn = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return RibbonSurfaces("lh", v * RW, (v @ turn.T) * RP, f.astype(np.int64)), v, f


def _bins(cells, k):
    cell, b, share = cells.bin_shares(k)
    vol = np.bincount(b, cells.volume[cell] * share, k)
    r = np.linalg.norm(cells.centre, axis=1)[cell]
    return vol, np.bincount(b, cells.volume[cell] * share * r, k) / vol


@pytest.mark.parametrize("slide", [0.0, 0.08])  # 0.08 rad: 1.7 mm at the pial sphere
@pytest.mark.parametrize("method", ["equidistant", "equivolume"])
def test_cells_tile_the_ribbon_and_bins_are_the_shells(method, slide):
    surf, v, f = _sphere(slide)
    k = 4
    cells = ribbon_cells([surf], {0: np.arange(len(f))}, step=1.0, n_depths=k, method=method)
    exact = _polyhedron_volume(v * RP, f) - _polyhedron_volume(v * RW, f)
    np.testing.assert_allclose(cells.volume.sum(), exact, rtol=1e-9)
    assert cells.inverted_mm3 == 0
    t = np.linspace(0, 1, k + 1)
    edges = (
        RW + t * (RP - RW) if method == "equidistant" else (RW**3 + t * (RP**3 - RW**3)) ** (1 / 3)
    )
    vol, mean_r = _bins(cells, k)
    # The polyhedra sit a little inside the spheres, so compare shares, not mm^3.
    np.testing.assert_allclose(
        vol / vol.sum(), np.diff(edges**3) / np.diff(edges**3).sum(), atol=0.004
    )
    np.testing.assert_allclose(mean_r, 0.75 * np.diff(edges**4) / np.diff(edges**3), atol=0.06)


def test_a_cell_across_a_boundary_splits_its_volume():
    surf, v, f = _sphere()
    cells = ribbon_cells([surf], {0: np.arange(len(f))}, step=2.0, n_depths=3)
    cell, b, share = cells.bin_shares(3)
    np.testing.assert_allclose(np.bincount(cell, share, len(cells)), 1.0)
    # 7 sheets through the depth cannot all stop on thirds: some cells are cut.
    assert (np.bincount(cell, minlength=len(cells)) > 1).any()


def test_a_patch_holds_every_cell_its_columns_own():
    surf, v, f = _sphere(slide=0.08)
    cap = np.flatnonzero(v[:, 2] > 0.6)
    whole = ribbon_cells([surf], {0: np.arange(len(f))}, step=1.0, n_depths=2)
    margin = float(np.linalg.norm(surf.pial - surf.white, axis=1).max()) + 3.0
    patch = ribbon_cells([surf], {0: patch_faces(surf, cap, margin)}, step=1.0, n_depths=2)
    in_cap = np.isin(whole.vertex, cap)
    np.testing.assert_allclose(
        patch.volume[np.isin(patch.vertex, cap)].sum(), whole.volume[in_cap].sum(), rtol=1e-9
    )
