"""Playing time: the next frame once the last one has been drawn.

A fixed timer (or a held key) steps whether or not anything kept up, so the
slow windows -- the 3-D surface above all -- skip frames: the index moves on
while a frame is still being drawn. The player steps, then waits for every
visible 3-D canvas to submit its frame before stepping again. No frame is
skipped, and the rate is whatever the open windows can draw (closing the
slices speeds it up), capped so a light scene does not race.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from PySide6 import QtCore, QtWidgets

__all__ = ["Player"]


class Player(QtCore.QObject):
    #: Playing (True) or stopped.
    toggled = QtCore.Signal(bool)
    #: Frames per second over the last second of playback.
    rate = QtCore.Signal(float)

    #: The fastest it plays, ms a frame: ~30 fps.
    MIN_MS = 33
    #: How long to wait for a canvas that never reports its frame (hidden,
    #: minimised, or its window closed mid-play) before stepping anyway.
    STALL_MS = 1000

    def __init__(
        self,
        step: Callable[[], None],
        canvases: Callable[[], list[QtWidgets.QRhiWidget]],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._step = step
        self._canvases = canvases
        self._on = False
        self._waiting: set[int] = set()
        self._started = 0.0
        self._times: list[float] = []
        self._next = QtCore.QTimer(self)
        self._next.setSingleShot(True)
        self._next.timeout.connect(self._tick)
        self._stall = QtCore.QTimer(self)
        self._stall.setSingleShot(True)
        self._stall.setInterval(self.STALL_MS)
        self._stall.timeout.connect(self._stalled)
        self._hooked: set[int] = set()
        #: Canvases that let a stall run out: not waited for again this playback,
        #: so a canvas that never reports costs one hitch, not every frame.
        self._mute: set[int] = set()

    def isActive(self) -> bool:  # noqa: N802 (the QTimer spelling it replaces)
        return self._on

    def start(self) -> None:
        if self._on:
            return
        self._on = True
        self._times.clear()
        self._mute.clear()
        self.toggled.emit(True)
        self._tick()

    def stop(self) -> None:
        if not self._on:
            return
        self._on = False
        self._next.stop()
        self._stall.stop()
        self._waiting.clear()
        self.toggled.emit(False)

    def toggle(self) -> None:
        self.stop() if self._on else self.start()

    def _tick(self) -> None:
        if not self._on:
            return
        self._started = time.monotonic()
        self._step()
        if not self._on:  # the step stopped it (nothing left to play)
            return
        self._waiting = set()
        for canvas in self._canvases():
            if not canvas.isVisible():
                continue
            ident = id(canvas)
            if ident in self._mute:
                continue
            if ident not in self._hooked:
                # Connected once per canvas for the player's life; the id is
                # matched against what this frame is waiting for.
                canvas.frameSubmitted.connect(lambda c=canvas: self._drawn(id(c)))
                canvas.destroyed.connect(lambda _=None, i=ident: self._forget(i))
                self._hooked.add(ident)
            self._waiting.add(ident)
        if self._waiting:
            self._stall.start()
        else:
            self._drawn_all()

    def _stalled(self) -> None:
        self._mute |= self._waiting
        self._drawn_all()

    def _forget(self, ident: int) -> None:
        self._hooked.discard(ident)
        self._mute.discard(ident)
        self._drawn(ident)

    def _drawn(self, ident: int) -> None:
        if ident in self._waiting:
            self._waiting.discard(ident)
            if not self._waiting:
                self._drawn_all()

    def _drawn_all(self) -> None:
        if not self._on:
            return
        self._stall.stop()
        self._waiting.clear()
        now = time.monotonic()
        self._times = [t for t in self._times if now - t < 1.0] + [now]
        if len(self._times) > 1:
            self.rate.emit((len(self._times) - 1) / (self._times[-1] - self._times[0]))
        spent = (now - self._started) * 1000.0
        self._next.start(int(max(0.0, self.MIN_MS - spent)))
