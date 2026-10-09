"""The controller is as tall as what it shows: no empty column in View mode."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")


def test_the_controller_fits_its_mode_and_shrinks_back(tmp_path):
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    run = np.random.default_rng(0).normal(100, 1, (12, 12, 8, 40)).astype(np.float32)
    nib.save(nib.Nifti1Image(run, np.eye(4)), tmp_path / "run.nii")
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "run.nii"))
    win = ViewerWindow(session)
    try:
        win.refresh(Aspect.ALL)
        win.dock_left()
        win.show()

        def settle() -> int:
            for _ in range(3):
                app.processEvents()
            return win.height()

        plain = settle()
        cap = win.screen().availableGeometry().height()
        assert plain < cap  # not full height with nothing below the layer form
        win._switch_mode("instacorr")
        assert settle() > plain  # its parameters get room
        win._switch_mode("plain")
        assert settle() == plain
    finally:
        win.close()
        session.close()
        app.processEvents()


def test_the_time_row_shows_for_a_series_and_re_ranges_the_view(tmp_path):
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SelectLayer

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    rng = np.random.default_rng(0)
    run = (1000 + rng.normal(0, 5, (12, 12, 8, 40))).astype(np.float32)
    img = nib.Nifti1Image(run, np.eye(4))
    img.header.set_xyzt_units("mm", "sec")
    img.header["pixdim"][4] = 2.0
    nib.save(img, tmp_path / "run.nii.gz")
    nib.save(nib.Nifti1Image(run[..., 0], np.eye(4)), tmp_path / "anat.nii.gz")
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "anat.nii.gz"))
    session.load(str(tmp_path / "run.nii.gz"))
    run_key = session.graph_layers()[0].key
    session.store.ensure_ram(run_key)
    win = ViewerWindow(session)
    try:
        win.refresh(Aspect.ALL)
        win.show()
        win._dispatch(SelectLayer(run_key))
        app.processEvents()
        assert not win.time_row.isHidden()
        box = win.time_detrend_box
        box.setCurrentIndex(box.findData(0))
        box.activated.emit(box.currentIndex())
        layer = session.state.layers.get(run_key)
        assert layer.time_detrend == 0
        # Around zero now, not around 1000.
        assert layer.range_lo is not None and layer.range_lo < 0 < layer.range_hi
        assert layer.range_hi < 100
        win._dispatch(SelectLayer(session.state.layers.base.key))
        app.processEvents()
        assert win.time_row.isHidden()
    finally:
        win.close()
        session.close()
        app.processEvents()
