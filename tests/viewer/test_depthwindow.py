"""Depth window: a region's profile against depth, and its export."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
nib = pytest.importorskip("nibabel")

from tests.viewer.test_surfaces import _anat, _shell_anat, _subject  # noqa: E402


def test_profile_of_a_disc_reads_wm_gm_csf_and_exports(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.depthwindow import DepthWindow, export_table
    from fastfuncstuff.viewer.vocab import LoadSurfaces, OpenView, SetDepthView, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(_anat(tmp_path)))
        # Shell phantom (WM < 21, GM < 24, CSF) under a white=20/pial=24 pair.
        session.load(str(_shell_anat(tmp_path)))
        session.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
        session.do(SetXYZ(0.0, 0.0, 22.0))
        session.do(OpenView("D1", "depth", "axial"))
        session.do(SetDepthView("D1", "disc", 4.0, "", ""))
        win = DepthWindow("D1", session, session.do)
        win.apply(session.state.viewports.get("D1"))
        r = win.result
        assert r is not None and r.n_vertices > 3 and r.timecourses is None
        mean = np.nanmean(r.profiles, axis=0)
        f = r.fractions
        assert mean[np.argmin(abs(f + 0.3))] > 100  # WM below white
        assert 60 < mean[np.argmin(abs(f - 0.6))] < 80  # GM mid-ribbon
        assert mean[-1] < 30  # CSF past pial
        assert session.surfaces.depth_roi_vertices["lh"].size == r.n_vertices
        out = tmp_path / "p.tsv"
        export_table(r, out)
        rows = out.read_text().splitlines()
        assert rows[0].startswith("row\t") and rows[1].startswith("mean\t")
        assert len(rows[1].split("\t")) == f.size + 1
        assert "SET_DEPTH_VIEW D1 disc 4.0" in session.to_script()
        win.close()
        assert session.surfaces.depth_roi_vertices == {}
    finally:
        session.close()
        app.processEvents()


def test_depth_time_view_is_percent_of_each_depths_own_mean_after_detrend():
    """A deep and a superficial depth with the same 2% response but a 3x baseline
    and a drift read the same once scaled by their own means; the old
    mean-subtracted panel drew the superficial one 3x larger."""
    from fastfuncstuff.viewer.ui.depthwindow import depth_time_view

    t = np.arange(120, dtype=np.float64)
    bump = (np.sin(2 * np.pi * t / 20) > 0).astype(np.float64)
    bump -= bump.mean()
    drift = np.linspace(-1, 1, t.size)
    data = np.stack([100 * (1 + 0.02 * bump) + 5 * drift, 300 * (1 + 0.02 * bump) + 15 * drift], 1)
    view, is_psc = depth_time_view(data, 1)
    assert is_psc
    np.testing.assert_allclose(view[:, 0], view[:, 1], atol=1e-9)
    # The drift is gone: what is left is the 2% square wave, less its own linear part.
    wave = 2.0 * bump
    expected = wave - np.polyval(np.polyfit(t, wave, 1), t)
    np.testing.assert_allclose(view[:, 0], expected, atol=1e-9)
    # Not detrended, the drift survives (mean removed only); -1 and 0 coincide.
    raw, _ = depth_time_view(data, -1)
    np.testing.assert_allclose(raw, depth_time_view(data, 0)[0], atol=1e-9)
    assert np.ptp(raw[:, 0]) > 8
    # A series with a non-positive mean has no percent: own units, still detrended.
    resid, is_psc = depth_time_view(data - data.mean(0), 1)
    assert not is_psc and abs(resid.mean()) < 1e-9


def test_depth_detrend_is_the_viewports_and_steps_with_d(tmp_path):
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.depthwindow import DepthWindow
    from fastfuncstuff.viewer.vocab import OpenView, SetViewDetrend

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), tmp_path / "a.nii")
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(tmp_path / "a.nii"))
        session.do(OpenView("D1", "depth", "axial"))
        win = DepthWindow("D1", session, session.do)
        win.apply(session.state.viewports.get("D1"))
        assert win.detrend_box.currentData() == 1  # linear by default, as a carpet
        session.do(SetViewDetrend("D1", 3))
        win.refresh(Aspect.VIEWPORTS)
        assert win.detrend_box.currentData() == 3
        win._cycle_detrend()
        assert session.state.viewports.get("D1").detrend == 4
        assert "SET_VIEW_DETREND D1 4" in session.to_script()
        win.close()
    finally:
        session.close()
        app.processEvents()


def test_a_closing_depth_window_does_not_dispatch_for_its_gone_viewport(tmp_path):
    """Closing moves focus off the radius box, whose editingFinished then fired
    SET_DEPTH_VIEW for a viewport the close had already removed: KeyError 'D1'."""
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.depthwindow import DepthWindow
    from fastfuncstuff.viewer.vocab import CloseView, OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), tmp_path / "a.nii")
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(tmp_path / "a.nii"))
        session.do(OpenView("D1", "depth", "axial"))
        win = DepthWindow("D1", session, session.do)
        session.do(CloseView("D1"))
        win.radius_spin.editingFinished.emit()  # what losing focus does
        win._changed()
        app.processEvents()
    finally:
        session.close()


def test_a_pinned_dataset_follows_the_region_beside_another_and_exports(tmp_path):
    """PIN keeps the dataset, not the curve: after switching to another dataset
    and moving the region, the pinned trace is the first dataset over the *new*
    region, so two processing stages compare spot by spot."""
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.depthwindow import DepthWindow, export_table
    from fastfuncstuff.viewer.vocab import LoadSurfaces, OpenView, SetDepthView, SetXYZ

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    session = ViewerSession(device=torch.device("cpu"))
    try:
        session.load(str(_anat(tmp_path)))
        session.load(str(_shell_anat(tmp_path)))
        session.do(LoadSurfaces(str(_subject(tmp_path)), "lh"))
        session.do(SetXYZ(0.0, 0.0, 22.0))
        session.do(OpenView("D1", "depth", "axial"))
        session.do(SetDepthView("D1", "disc", 4.0, "", ""))
        win = DepthWindow("D1", session, session.do)
        win.apply(session.state.viewports.get("D1"))
        first = win.result
        assert first is not None
        shell = first.layer_name
        win.pin()
        win.pin()  # the same dataset twice is one pin
        assert len(win.plot.pins) == 1
        base = session.state.layers.base
        assert base is not None
        session.do(SetDepthView("D1", "disc", 8.0, base.key, ""))
        win.refresh(SetDepthView.aspects)
        assert win.result is not None and win.result.layer_name == base.name
        assert len(win.plot.pins) == 1
        pin = win.plot.pins[0]
        assert pin.label == shell
        assert pin.n_vertices == win.result.n_vertices > first.n_vertices
        # The pin is the shell over the 8 mm disc: what profiling it live there gives.
        win.plot.resize(400, 300)
        win.plot.grab()  # paints the pin and the live trace without raising
        out = tmp_path / "p.tsv"
        export_table(win.result, out, win.plot.pins)
        lines = out.read_text().splitlines()
        assert lines[0] == f"# pin1: {shell}"
        assert any(ln.startswith("pin1 mean\t") for ln in lines)
        session.do(SetDepthView("D1", "disc", 8.0, "", ""))
        win.refresh(SetDepthView.aspects)
        assert win.result.layer_name == shell
        np.testing.assert_allclose(win.plot.pins[0].mean, np.nanmean(win.result.profiles, axis=0))
        win.clear_pins()
        assert win.plot.pins == [] and not win.unpin_button.isEnabled()
        win.close()
    finally:
        session.close()
        app.processEvents()
