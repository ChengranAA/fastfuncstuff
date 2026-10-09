"""The vertex graph: each vertex's own time course, drawn where it sits."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

from tests.viewer.test_surface_window import window  # noqa: E402, F401  (fixture)


@pytest.fixture
def graphed(window, tmp_path):  # noqa: F811
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.vocab import SetXYZ

    session, win = window
    # The fixture's grid: 30 x 30 x 10 at 2 mm, origin (-29, -29, -9). Each voxel's
    # course says where it is, so a cell can be checked against its vertex.
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-29.0, -29.0, -9.0]
    i, j = np.meshgrid(np.arange(30), np.arange(30), indexing="ij")
    run = (1000.0 * i + 10.0 * j)[:, :, None, None] + np.arange(6.0)[None, None, None, :]
    run = np.broadcast_to(run, (30, 30, 10, 6)).astype(np.float32)
    img = nib.Nifti1Image(run, aff)
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 2.0
    nib.save(img, tmp_path / "run.nii.gz")
    session.load(str(tmp_path / "run.nii.gz"))
    # A 4-D load finishes in the background; without this the graph can read
    # it before it is resident and draw nothing.
    session.store.ensure_ram(session.graph_layers()[0].key)
    session.do(SetXYZ(1.0, 1.0, 1.5))
    win.refresh(Aspect.ALL)
    win._open_graph()
    graph = win.vertex_graph
    graph.detrend = -1  # the raw values, to compare against the volume
    graph.refresh()
    yield session, win, graph
    if win.vertex_graph is not None:
        win.vertex_graph.close()


def test_every_cell_is_its_own_vertex_read_where_the_3d_window_samples(graphed):
    session, win, graph = graphed
    run = next(ly for ly in session.graph_layers())
    patch = graph.patch
    assert patch is not None and len(graph.graph._cells) == len(patch.vertices)
    voxels = session.vertex_voxels(patch.hemi, patch.vertices, graph._depth())
    for cell, ijk in zip(graph.graph._cells, voxels, strict=True):
        (key, values), *_ = cell.traces
        assert key == run.key
        np.testing.assert_allclose(values, session.timeseries(run.key, ijk))
    # Neighbours on the sheet are different voxels, so their lines differ.
    assert len({tuple(c.traces[0][1]) for c in graph.graph._cells}) > 1
    assert graph.graph._cells[0].is_centre


def test_rings_grow_the_patch_and_the_surface_outlines_it(graphed):
    _, win, graph = graphed
    n = len(graph.patch.vertices)
    graph.step_rings(1)
    assert len(graph.patch.vertices) > n
    hemi, pairs = win.canvas.outline
    assert (
        hemi == graph.patch.hemi
        and len(pairs)
        and set(np.unique(pairs)) <= set(graph.patch.vertices)
    )
    assert win.canvas._marker_geometry(320) is not None
    graph.close()
    assert win.canvas.outline is None and win.vertex_graph is None


def test_double_click_moves_the_crosshair_to_that_vertex(graphed):
    _, win, graph = graphed
    got = []
    win.located.connect(lambda x, y, z: got.append((x, y, z)))
    row = len(graph.patch.vertices) - 1
    graph._choose(row)
    vertex = graph.patch.vertices[row]
    h = win.session.surfaces.hemis[graph.patch.hemi]
    d = graph._depth()
    want = (1 - d) * h.states["white"][vertex] + d * h.states["pial"][vertex]
    np.testing.assert_allclose(got[-1], want, atol=1e-5)


def test_cells_sit_at_their_page_positions(graphed):
    _, _, graph = graphed
    g = graph.graph
    g.resize(400, 400)
    rects = g._cell_rects()
    centres = np.array([[r.center().x(), r.center().y()] for r in rects])
    xy = graph.patch.xy
    # Page +y is screen up: the same picture, scaled and flipped in y only.
    a = np.c_[xy[:, 0], -xy[:, 1], np.ones(len(xy))]
    fit, *_ = np.linalg.lstsq(a, centres, rcond=None)
    np.testing.assert_allclose(a @ fit, centres, atol=1e-6)
    assert fit[0, 0] > 0 and fit[1, 1] > 0 and abs(fit[0, 0] - fit[1, 1]) < 1e-9


def test_detrend_is_a_dropdown_and_the_arrows_step_time(graphed):
    session, _, graph = graphed
    box = graph.detrend_box
    box.setCurrentIndex(box.findData(1))
    box.activated.emit(box.currentIndex())
    assert graph.detrend == 1
    # Linear out of a ramp leaves nothing.
    (_, values), *_ = graph.graph._cells[0].traces
    assert np.abs(values).max() < 1e-3
    graph.cycle_detrend()
    assert box.currentData() == graph.detrend == 2
    start = session.state.time_index
    graph.step_time(1)
    assert session.state.time_index == start + 1
    graph.step_time(-2)
    assert session.state.time_index == 5  # wraps to the last of six


def test_hex_puts_the_same_vertices_on_a_lattice(graphed):
    _, _, graph = graphed
    exact = graph.graph._xy.copy()
    cells = len(graph.graph._cells)
    graph.hex_check.setChecked(True)
    assert len(graph.graph._cells) == cells  # every vertex still drawn
    assert not np.allclose(graph.graph._xy, exact)
    graph.hex_check.setChecked(False)
    np.testing.assert_allclose(graph.graph._xy, exact)
