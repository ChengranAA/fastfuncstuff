"""The flattened vertex patch the surface graph draws its time courses on."""

from __future__ import annotations

import numpy as np

from fastfuncstuff.viewer.vertexpatch import vertex_patch
from tests.viewer.test_surface_render import _sheet


def _flat(rings=3):
    h = _sheet()
    w = h.states["white"].astype(np.float64)
    centre = int(np.argmin(np.linalg.norm(w, axis=1)))  # the middle of the sheet
    # Seen from +z with +y up: the page should be the sheet itself.
    return h, w, centre, vertex_patch("lh", h.faces, w, w, centre, rings, np.array([0, 1.0, 0]))


def test_rings_pick_the_vertices_and_the_centre_comes_first():
    _, w, centre, patch = _flat(rings=2)
    assert patch.vertices[0] == centre and patch.ring[0] == 0
    # A triangulated grid: two rings are the centre's 5 x 5 block less two corners.
    d = np.abs(w[patch.vertices] - w[centre])[:, :2] / 1.0
    assert d.max() <= 2.0 + 1e-6 and len(patch.vertices) == 19
    assert set(np.unique(patch.ring)) == {0, 1, 2}


def test_a_flat_sheet_lays_out_as_itself_turned_to_the_camera():
    _, w, centre, patch = _flat()
    true = (w[patch.vertices] - w[centre])[:, :2]
    # Directions exactly; distances are edge paths, never shorter than straight.
    ang = np.arctan2(patch.xy[1:, 1], patch.xy[1:, 0])
    np.testing.assert_allclose(
        np.cos(ang), true[1:, 0] / np.linalg.norm(true[1:], axis=1), atol=1e-9
    )
    r, straight = np.linalg.norm(patch.xy, axis=1), np.linalg.norm(true, axis=1)
    assert (r >= straight - 1e-9).all() and (r <= 1.42 * straight + 1e-9).all()

    # The same sheet seen with +x up: the page turns with the camera.
    h = _sheet()
    turned = vertex_patch("lh", h.faces, w, w, centre, 3, np.array([1.0, 0, 0]))
    np.testing.assert_allclose(turned.xy[:, 1], patch.xy[:, 0], atol=1e-9)

    # Wound the other way, the page must not mirror once told where the eye is.
    flipped = vertex_patch(
        "lh", h.faces[:, ::-1], w, w, centre, 3, np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
    )
    np.testing.assert_allclose(flipped.xy, patch.xy, atol=1e-9)


def test_the_outline_is_the_patch_boundary():
    _, _, _, patch = _flat(rings=2)
    on_outline = np.unique(patch.outline)
    outer = patch.vertices[patch.ring == 2]
    assert set(on_outline) == set(outer)
    # Edges are row pairs into the patch, each once.
    assert patch.edges.max() < len(patch.vertices)
    assert len(np.unique(np.sort(patch.edges, 1), axis=0)) == len(patch.edges)


def test_hex_layout_keeps_every_vertex_on_its_own_site_near_where_it_was():
    from scipy.spatial import KDTree

    from fastfuncstuff.viewer.vertexpatch import hex_layout

    _, _, _, patch = _flat(rings=4)
    hx = hex_layout(patch.xy)
    assert hx.shape == patch.xy.shape
    # One vertex per site: no two cells on the same spot.
    d, _ = KDTree(hx).query(hx, k=2)
    spacing = np.median(d[:, 1])
    assert d[:, 1].min() > 0.99 * spacing
    # On a lattice: every neighbour distance is the spacing or more.
    assert np.allclose(np.sort(d[:, 1])[: len(hx) // 2], spacing)
    # The centre stays put, and nothing wanders far from where it was.
    np.testing.assert_allclose(hx[0], 0.0, atol=1e-12)
    assert np.linalg.norm(hx - patch.xy, axis=1).max() < 2.5 * spacing
