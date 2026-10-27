"""The image window's header: a view row, and the surface tools under it."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")


@pytest.fixture
def window(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.imagewindow import ImageWindow
    from fastfuncstuff.viewer.vocab import OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    nib.save(nib.Nifti1Image(np.zeros((20, 20, 20), np.float32), np.eye(4)), tmp_path / "a.nii")
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "a.nii"))
    session.do(OpenView("A1", "image", "axial"))
    win = ImageWindow("A1", session, session.do)
    win.apply(session.state.viewports.get("A1"))
    win.show()
    app.processEvents()
    yield session, win, app
    win.close()
    session.close()
    app.processEvents()


def test_surface_tools_have_their_own_row_and_only_with_surfaces(window):
    from tests.viewer.test_surface_render import _sheet

    session, win, _ = window
    assert win.edit_row.isHidden()  # nothing to edit
    session.surfaces.hemis = {"lh": _sheet()}
    win.apply(session.state.viewports.get("A1"))
    assert not win.edit_row.isHidden()
    row = win._tools_row
    tools = {row.itemAt(i).widget() for i in range(row.count())}
    assert {win.edit_button, win.draw_button, win.point_button, win.nudge_button} <= tools
    assert win.solo_button not in tools and win.lock_button not in tools


def test_each_row_keeps_its_full_labels_while_they_fit(window):
    session, win, app = window
    win.resize(900, 400)
    app.processEvents()
    assert win.solo_button.text() != "[S]" and win.edit_button.text() != "[g]"
    win.resize(250, 400)
    app.processEvents()
    assert win.solo_button.text() == "[S]" and win.edit_button.text() == "[g]"
