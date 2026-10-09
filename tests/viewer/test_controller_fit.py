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
