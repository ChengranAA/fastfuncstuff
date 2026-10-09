"""Playing time one frame after the last has been drawn."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")


class FakeCanvas(QtCore.QObject):
    frameSubmitted = QtCore.Signal()  # noqa: N815 (Qt)

    def __init__(self, visible: bool = True) -> None:
        super().__init__()
        self.visible = visible

    def isVisible(self) -> bool:  # noqa: N802 (Qt)
        return self.visible


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _spin(app, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 5)


def test_a_step_waits_for_every_visible_canvas_to_draw(app, monkeypatch):
    from fastfuncstuff.viewer.ui.player import Player

    monkeypatch.setattr(Player, "MIN_MS", 0)
    a, b, hidden = FakeCanvas(), FakeCanvas(), FakeCanvas(visible=False)
    steps = []
    player = Player(lambda: steps.append(1), lambda: [a, b, hidden])
    player.start()
    _spin(app, 0.05)
    assert len(steps) == 1  # nothing drawn yet: no second step
    a.frameSubmitted.emit()
    _spin(app, 0.05)
    assert len(steps) == 1  # b still drawing; the hidden one is not waited for
    b.frameSubmitted.emit()
    _spin(app, 0.05)
    assert len(steps) == 2
    player.stop()
    a.frameSubmitted.emit()
    b.frameSubmitted.emit()
    _spin(app, 0.05)
    assert len(steps) == 2 and not player.isActive()


def test_a_canvas_that_never_draws_does_not_stall_playback(app, monkeypatch):
    from fastfuncstuff.viewer.ui.player import Player

    monkeypatch.setattr(Player, "MIN_MS", 0)
    monkeypatch.setattr(Player, "STALL_MS", 150)
    steps = []
    mute = FakeCanvas()
    player = Player(lambda: steps.append(time.monotonic()), lambda: [mute])
    player.start()
    _spin(app, 0.6)
    player.stop()
    # One stall, then it stops waiting for that canvas and plays at its cap.
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    assert gaps[0] >= 0.14 and len(steps) >= 6 and max(gaps[1:]) < 0.1


def test_with_nothing_to_wait_for_it_plays_at_its_cap(app):
    from fastfuncstuff.viewer.ui.player import Player

    steps = []
    rates = []
    player = Player(lambda: steps.append(time.monotonic()), lambda: [])
    player.rate.connect(rates.append)
    player.start()
    _spin(app, 0.5)
    player.stop()
    gaps = [b - a for a, b in zip(steps, steps[1:], strict=False)]
    assert len(steps) >= 5
    assert min(gaps) >= Player.MIN_MS / 1000 * 0.8  # capped, not racing
    assert rates and rates[-1] <= 1000 / Player.MIN_MS * 1.2


def test_every_time_step_draws_the_3d_window_and_its_button_shows_play(tmp_path, app):
    """The player paces itself on the 3-D window's frames, so a time step must
    reach it at once rather than being held back like a crosshair scroll."""
    import nibabel as nib
    import numpy as np
    import torch

    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.manager import WindowManager
    from fastfuncstuff.viewer.viewports import ViewKind
    from fastfuncstuff.viewer.vocab import SetIndex
    from tests.viewer.test_surface_render import _sheet

    nib.save(nib.Nifti1Image(np.zeros((8, 8, 4, 5), np.float32), np.eye(4)), tmp_path / "r.nii")
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "r.nii"))
    session.surfaces.hemis = {"lh": _sheet()}
    session.surfaces.version = {"lh": 1}
    manager = WindowManager(session, lambda cmd: manager.redraw(session.do(cmd)))
    try:
        vid = manager.open(ViewKind.SURFACE)
        manager.sync()
        win = manager.windows[vid]
        drawn = []
        win.refresh = lambda dirty: drawn.append(dirty)
        for t in range(1, 4):
            manager.redraw(session.do(SetIndex(t)))
        assert len(drawn) == 3 and not manager._held
        manager.set_playing(True)
        assert win.animate_button.isChecked()
        assert manager.surface_canvases() == [win.canvas]
        manager.set_playing(False)
        assert not win.animate_button.isChecked()
    finally:
        for w in list(manager.windows.values()):
            w.close()
        session.close()
