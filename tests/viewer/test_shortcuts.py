"""Held keys: auto-repeats that waited in the queue are dropped, fresh presses never."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
from PySide6 import QtCore, QtGui  # noqa: E402

from fastfuncstuff.viewer.ui.shortcuts import _StaleRepeats  # noqa: E402

E = QtCore.QEvent.Type
RIGHT = QtCore.Qt.Key.Key_Right


@pytest.fixture
def filt():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    now = [10_000.0]
    f = _StaleRepeats(app, clock=lambda: now[0])
    yield f, now
    f.deleteLater()


def _key(ts: int, *, repeat: bool, kind=E.KeyPress) -> QtGui.QKeyEvent:
    ev = QtGui.QKeyEvent(kind, RIGHT, QtCore.Qt.KeyboardModifier.NoModifier, "", repeat)
    ev.setTimestamp(ts)
    return ev


def test_a_repeat_that_waited_behind_a_slow_step_is_dropped(filt):
    """Measured on X: with 60 ms steps, 31 queued repeats ran on for 2.5 s
    after the key came up."""
    f, now = filt
    # Server time 5000 arrives at local 10000: that is the clocks' offset.
    assert not f.stale(_key(5000, repeat=False))
    # A step took 100 ms; the repeats at 5033 and 5066 waited 67 and 34 ms.
    now[0] = 10_100.0
    assert f.stale(_key(5033, repeat=True))
    assert f.stale(_key(5066, repeat=True))
    # The one at 5099 has waited 1 ms: the viewer is free, and it lands.
    assert not f.stale(_key(5099, repeat=True))


def test_a_fresh_press_is_never_dropped(filt):
    f, now = filt
    f.stale(_key(5000, repeat=False))
    now[0] = 10_500.0
    assert not f.stale(_key(5010, repeat=False))


def test_one_event_gets_one_verdict_through_override_and_press(filt):
    """Qt filters a key as a ShortcutOverride and then as a KeyPress (and again
    on each parent): judging each afresh would let half of a stale repeat through."""
    f, now = filt
    f.stale(_key(5000, repeat=False))
    now[0] = 10_080.0
    assert f.stale(_key(5033, repeat=True, kind=E.ShortcutOverride))
    now[0] = 10_000.0 + 33  # even if the clock read now would say fresh
    assert f.stale(_key(5033, repeat=True))


def test_a_synthetic_event_has_no_clock_and_passes(filt):
    f, now = filt
    f.stale(_key(5000, repeat=False))
    now[0] = 99_999.0
    assert not f.stale(_key(0, repeat=True))


def test_a_restarted_clock_re_anchors_rather_than_dropping_everything(filt):
    f, now = filt
    f.stale(_key(5000, repeat=False))
    now[0] = 20_000.0  # server time wrapped / restarted: gap jumps by ~10 s
    assert not f.stale(_key(1, repeat=True))
    now[0] = 20_033.0
    assert not f.stale(_key(34, repeat=True))


def test_a_stale_override_is_taken_from_the_shortcut_map(filt):
    f, now = filt
    f.stale(_key(5000, repeat=False))
    now[0] = 10_080.0
    ev = _key(5033, repeat=True, kind=E.ShortcutOverride)
    ev.ignore()
    assert f.eventFilter(None, ev)
    assert ev.isAccepted(), "accepted, so no shortcut fires for it"
    assert f.eventFilter(None, _key(5033, repeat=True))
