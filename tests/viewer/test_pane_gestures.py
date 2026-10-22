"""Mouse gestures on a slice pane: right-drag zooms, middle or shift+drag pans."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore, QtGui  # noqa: E402

from fastfuncstuff.viewer.slicing import Plane  # noqa: E402

B = QtCore.Qt.MouseButton
M = QtCore.Qt.KeyboardModifier


@pytest.fixture
def pane():
    from fastfuncstuff.viewer.ui.panes import ImagePane

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    p = ImagePane(Plane.AXIAL)
    p.resize(200, 200)
    got: dict[str, list] = {"zoomed": [], "panned": [], "picked": []}
    p.zoomed.connect(lambda f: got["zoomed"].append(f))
    p.panned.connect(lambda r, c: got["panned"].append((r, c)))
    p.picked.connect(lambda r, c: got["picked"].append((r, c)))
    yield p, got
    p.close()
    app.processEvents()


def _drag(widget, button, mods, start, end):
    def send(kind, pos, buttons):
        ev = QtGui.QMouseEvent(
            kind, QtCore.QPointF(*pos), QtCore.QPointF(*pos), button, buttons, mods
        )
        QtWidgets.QApplication.sendEvent(widget, ev)

    none = QtCore.Qt.MouseButton.NoButton
    send(QtCore.QEvent.Type.MouseButtonPress, start, button)
    send(QtCore.QEvent.Type.MouseMove, end, button)
    send(QtCore.QEvent.Type.MouseButtonRelease, end, none)


def test_a_drag_picks_once_per_frame_and_ends_where_released(pane):
    """A mouse reports faster than a pick draws; each extra pick redrew every
    window for a place never shown. The newest place waits for the frame."""
    import torch

    from fastfuncstuff.viewer.compose import PaneImage

    p, got = pane
    p.set_pane(PaneImage(torch.zeros(20, 20, 4, dtype=torch.uint8), Plane.AXIAL, 0))
    p.show()
    QtWidgets.QApplication.processEvents()

    def send(kind, pos, button, buttons):
        ev = QtGui.QMouseEvent(
            kind, QtCore.QPointF(*pos), QtCore.QPointF(*pos), button, buttons, M.NoModifier
        )
        QtWidgets.QApplication.sendEvent(p, ev)

    def move(pos):
        send(QtCore.QEvent.Type.MouseMove, pos, B.NoButton, B.LeftButton)

    send(QtCore.QEvent.Type.MouseButtonPress, (10, 10), B.LeftButton, B.LeftButton)
    move((50, 50))
    move((100, 100))
    move((150, 150))
    assert got["picked"] == [(1, 1), (5, 5)], "the press, then the first move; the rest wait"
    p.repaint()  # the frame for (5, 5) is shown
    QtWidgets.QApplication.processEvents()
    assert got["picked"][-1] == (15, 15), "the newest place, not the one in between"
    move((60, 60))
    send(QtCore.QEvent.Type.MouseButtonRelease, (60, 60), B.LeftButton, B.NoButton)
    assert got["picked"][-1] == (6, 6), "the drag ends where the button came up"


def test_right_drag_up_zooms_in_and_does_not_pan(pane):
    p, got = pane
    _drag(p, B.RightButton, M.NoModifier, (100, 120), (100, 60))
    assert got["zoomed"] and got["zoomed"][0] > 1.0
    assert not got["panned"]


@pytest.mark.parametrize(
    ("button", "mods"), [(B.MiddleButton, M.NoModifier), (B.LeftButton, M.ShiftModifier)]
)
def test_middle_or_shift_drag_pans_and_does_not_move_the_crosshair(pane, button, mods):
    p, got = pane
    _drag(p, button, mods, (100, 100), (130, 100))
    assert got["panned"]
    assert not got["zoomed"]
    assert not got["picked"]


def test_a_toast_paints_a_solid_box_and_fades_away(pane):
    from fastfuncstuff.viewer.ui import theme

    p, _ = pane
    p.show_toast("end the stroke on the white outline it began on")
    img = p.grab().toImage()
    warn = QtGui.QColor(theme.palette().warn)
    # Somewhere along the bottom band is the toast's fill.
    row = p.height() - 14
    assert any(img.pixelColor(x, row) == warn for x in range(p.width()))
    p._on_toast(0.0)
    assert p._toast == ""
