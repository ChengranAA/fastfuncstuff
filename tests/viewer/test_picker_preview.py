"""Looking through the DATA picker before choosing.

A click in the picker shows a dataset's first volume as a layer; only a
double-click or Enter loads it. The failures worth pinning: a glance that
starts the full background inflate (the cost the preview exists to avoid), a
glance that lands in the session script, a chosen preview opened a second time,
and a picker closed without choosing that leaves its last glance behind.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel")

CPU = torch.device("cpu")
AFF = np.diag([3.0, 3.0, 3.0, 1.0])


def _write(path, data):
    from fastfuncstuff.io.afni import save_nifti

    save_nifti(np.asarray(data, np.float32), path, affine=AFF)
    return path


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    _write(tmp_path / "a.anat.nii.gz", rng.random((6, 7, 5)) * 100)
    _write(tmp_path / "b.run.nii.gz", rng.random((6, 7, 5, 12)) * 100)
    _write(tmp_path / "c.run.nii.gz", rng.random((6, 7, 5, 9)) * 100)
    return tmp_path


@pytest.fixture
def session(folder):
    from fastfuncstuff.viewer.session import ViewerSession

    s = ViewerSession(device=CPU)
    s.load(folder / "a.anat.nii.gz")
    s.bus.clear_log()
    yield s
    s.close()


def _keys(session):
    return [ly.key for ly in session.state.layers]


def _loading(session, key):
    res = session.store.get(key)
    return res.pending or res.array is not None


def test_a_preview_is_volume_zero_only_and_unrecorded(session, folder):
    session.preview(folder / "b.run.nii.gz")
    key = _keys(session)[-1]
    assert len(_keys(session)) == 2
    assert not _loading(session, key), "a glance started the full inflate"
    assert session.bus.log == ()


def test_the_next_preview_replaces_the_last(session, folder):
    session.preview(folder / "b.run.nii.gz")
    session.preview(folder / "c.run.nii.gz")
    names = [ly.name for ly in session.state.layers]
    assert names == ["a.anat.nii.gz", "c.run.nii.gz"]


def test_withdrawing_restores_the_stack_and_selection(session, folder):
    before = (_keys(session), session.state.selected)
    session.preview(folder / "b.run.nii.gz")
    assert session.state.selected != before[1]
    session.preview(None)
    assert (_keys(session), session.state.selected) == before
    assert session.preview_path is None


def test_committing_records_a_load_that_replays_to_the_same_key(session, folder):
    from fastfuncstuff.viewer.session import ViewerSession

    session.preview(folder / "c.run.nii.gz")
    session.preview(folder / "b.run.nii.gz")
    key = session.commit_preview()
    assert key == _keys(session)[-1]
    assert _loading(session, key), "the chosen dataset never started its full load"
    script = session.to_script()
    assert "c.run" not in script

    replay = ViewerSession(device=CPU)
    try:
        replay.load(folder / "a.anat.nii.gz")
        replay.run_script(script)
        assert _keys(replay) == _keys(session)
    finally:
        replay.close()


# ---------------------------------------------------------------------------
# through the picker
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def win(qapp, folder):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    w = ViewerWindow(ViewerSession(device=CPU))
    w.read_directory(folder)
    w.open_path(folder / "a.anat.nii.gz")
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _settle(qapp, seconds=0.3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)


def _row(picker, name):
    return next(r for r in range(picker.list.count()) if name in picker.list.item(r).text())


def _picker(win):
    from fastfuncstuff.viewer.ui.widgets import open_combo_picker

    return open_combo_picker(win.data_box)


def test_highlighting_previews_and_enter_loads_once(win, qapp, folder):
    import shiboken6
    from PySide6 import QtCore, QtTest

    picker = _picker(win)
    picker.list.setCurrentRow(_row(picker, "b.run"))
    _settle(qapp)
    assert [ly.name for ly in win.session.state.layers] == ["a.anat.nii.gz", "b.run.nii.gz"]
    QtTest.QTest.keyClick(picker.list, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert not shiboken6.isValid(picker) or not picker.isVisible()
    assert [ly.name for ly in win.session.state.layers] == ["a.anat.nii.gz", "b.run.nii.gz"]
    assert win.session.preview_path is None
    assert "b.run" in win.session.to_script()


def test_closing_the_picker_takes_the_preview_away(win, qapp):
    picker = _picker(win)
    picker.list.setCurrentRow(_row(picker, "c.run"))
    _settle(qapp)
    assert len(win.session.state.layers) == 2
    picker.reject()
    qapp.processEvents()
    assert [ly.name for ly in win.session.state.layers] == ["a.anat.nii.gz"]
