"""The graph grid's painting: what a volume step redraws, and how a trace is built."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtGui  # noqa: E402

from fastfuncstuff.viewer.ui.gridgraph import Cell, GridGraph, _polygon  # noqa: E402


@pytest.fixture
def graph():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    g = GridGraph()
    g.resize(300, 200)
    yield g
    g.close()
    app.processEvents()


def _cells(seed: int) -> list[Cell]:
    rng = np.random.default_rng(seed)
    return [
        Cell(ijk=(i, 0, 0), traces=[("run", rng.normal(size=120))], is_centre=i == 1)
        for i in range(4)
    ]


def _paint(g: GridGraph) -> QtGui.QImage:
    img = QtGui.QImage(g.size(), QtGui.QImage.Format.Format_ARGB32)
    g.render(img)
    return img


def test_a_polygon_holds_the_points_it_was_given():
    xy = np.array([[0.5, 1.0], [2.0, -3.25], [7.0, 8.0]])
    poly = _polygon(xy)
    assert [(p.x(), p.y()) for p in poly] == [tuple(r) for r in xy]
    assert _polygon(np.zeros((0, 2))).size() == 0


def test_a_volume_step_moves_the_cursor_without_repainting_the_traces(graph):
    """A held arrow key redrew every trace for each step; only the cursor moves."""
    cells = _cells(0)
    graph.set_cells(cells, 2, 10)
    first = _paint(graph)
    static = graph._static
    # Fresh arrays with the same values, as GraphWindow.refresh hands over.
    graph.set_cells(
        [Cell(c.ijk, [(k, v.copy()) for k, v in c.traces], c.is_centre) for c in cells], 2, 90
    )
    second = _paint(graph)
    assert graph._static is static
    assert first != second, "the cursor moved"


def test_new_traces_repaint_them(graph):
    graph.set_cells(_cells(0), 2, 10)
    _paint(graph)
    static = graph._static
    graph.set_cells(_cells(1), 2, 10)
    _paint(graph)
    assert graph._static is not static
