"""Keyboard bindings, declared once and used twice.

The table here is both what gets installed and what the help lists, so a key
cannot exist without being documented or be documented without existing. That
is the whole reason it is data rather than a run of ``addAction`` calls
alongside a hand-written help string that drifts from them.

Each window declares its own table, and ``h`` shows the table for the window
that has focus -- a graph window's keys are not the main window's.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.ui import theme

if TYPE_CHECKING:
    from fastfuncstuff.viewer.commands import Command


@dataclass(frozen=True)
class Binding:
    keys: str
    description: str
    action: Callable[[], None] | None = None
    group: str = ""
    #: Extra spellings that do the same thing but are not worth listing twice.
    aliases: tuple[str, ...] = field(default=())


def double_tap(
    once: Callable[[], None], twice: Callable[[], None], clock: Callable[[], float] | None = None
) -> Callable[[], None]:
    """One action for a key, another when it is struck again within a double-click.

    The first tap always runs ``once`` -- waiting to see whether a second is
    coming would make the common single tap feel laggy -- so ``twice`` should
    be a superset of it (centre this window, then centre every window).
    """
    if clock is None:
        timer = QtCore.QElapsedTimer()
        timer.start()

        def clock() -> float:
            return timer.elapsed() / 1000.0

    last = [float("-inf")]

    def fire() -> None:
        now = clock()
        window = QtWidgets.QApplication.doubleClickInterval() / 1000.0
        if now - last[0] <= window:
            last[0] = float("-inf")  # a third tap starts over rather than re-firing
            twice()
        else:
            last[0] = now
            once()

    return fire


def jump_back_key(dispatch: Callable[[Command], object]) -> Binding:
    """`j`, in every window: the crosshair is shared, so its way back is too."""
    from fastfuncstuff.viewer.vocab import JumpBack

    return Binding(
        "j",
        "jump back to the previous location (again: return)",
        lambda: dispatch(JumpBack()),
        group="navigate",
    )


_STEP_KEYS = frozenset(
    {
        QtCore.Qt.Key.Key_Up,
        QtCore.Qt.Key.Key_Down,
        QtCore.Qt.Key.Key_PageUp,
        QtCore.Qt.Key.Key_PageDown,
    }
)


class _StepKeys(QtCore.QObject):
    """Let a focused spin box or slider keep Up / Down.

    Qt offers a key to the focused widget before a window shortcut gets it
    (``ShortcutOverride``), and a spin box's line edit declines the arrows, so
    the crosshair nudge took them: click into a number, press Up, and the slice
    moved instead of the number.
    """

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        if event.type() == QtCore.QEvent.Type.ShortcutOverride and event.key() in _STEP_KEYS:
            # The focus widget is a spin box's inner line edit, one level down.
            widget = obj if isinstance(obj, QtWidgets.QWidget) else None
            for _ in range(2):
                if isinstance(widget, QtWidgets.QAbstractSpinBox | QtWidgets.QAbstractSlider):
                    event.accept()
                    break
                widget = widget.parentWidget() if widget is not None else None
        return False


class _StaleRepeats(QtCore.QObject):
    """Drop key auto-repeats that waited in the queue, so a held key never runs ahead.

    A held key repeats every ~33 ms whatever the viewer is doing. When a step
    takes longer than that, the repeats queue up behind it and are then all
    run back to back with no paint between them: the picture freezes, jumps,
    and keeps going after the key is let go. A repeat that sat in the queue
    is one the person has already pressed past, so it is dropped, and a held
    key steps exactly as fast as the viewer can draw.

    Queue delay is the event's own timestamp against the local clock, with the
    clocks' offset taken as the smallest gap seen -- the delivery that waited
    least. Only repeats are judged; a fresh press always lands.
    """

    #: A repeat older than this when it arrives is stale. Under one repeat
    #: interval, so a viewer that keeps up loses nothing.
    STALE_MS = 20.0
    #: A gap this far off is a clock that restarted, not a queue; re-anchor.
    RESYNC_MS = 2000.0

    def __init__(self, parent: QtCore.QObject | None = None, clock=None) -> None:
        super().__init__(parent)
        self._clock = clock or (lambda: time.monotonic() * 1000.0)
        self._offset: float | None = None
        #: One verdict per physical event: it is filtered again as a
        #: ShortcutOverride, a KeyPress, and on every parent it propagates to.
        self._last: tuple[int, int, bool] = (-1, -1, False)

    def stale(self, event: QtGui.QKeyEvent) -> bool:
        ts = int(event.timestamp())
        if ts <= 0:  # synthetic (scripts, tests): no clock to judge by
            return False
        if self._last[:2] == (ts, event.key()):
            return self._last[2]
        gap = self._clock() - ts
        if self._offset is None or gap < self._offset or gap - self._offset > self.RESYNC_MS:
            self._offset = gap
        drop = event.isAutoRepeat() and gap - self._offset > self.STALE_MS
        self._last = (ts, event.key(), drop)
        return drop

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt)
        kind = event.type()
        if kind not in (QtCore.QEvent.Type.ShortcutOverride, QtCore.QEvent.Type.KeyPress):
            return False
        if not self.stale(event):
            return False
        if kind == QtCore.QEvent.Type.ShortcutOverride:
            # Accepted, the shortcut map stands aside; the KeyPress that
            # follows is then swallowed below.
            event.accept()
        return True


_step_keys: _StepKeys | None = None
_stale_repeats: _StaleRepeats | None = None


def _keep_step_keys() -> None:
    global _step_keys, _stale_repeats
    app = QtWidgets.QApplication.instance()
    if app is not None and _step_keys is None:
        _step_keys = _StepKeys(app)
        app.installEventFilter(_step_keys)
        _stale_repeats = _StaleRepeats(app)
        app.installEventFilter(_stale_repeats)


def install(widget: QtWidgets.QWidget, bindings: Sequence[Binding]) -> None:
    """Register every binding on ``widget`` as a window-level shortcut.

    Refuses a table with two bindings on one key. Qt does **not** distinguish
    case in a key sequence -- ``QKeySequence("d") == QKeySequence("D")`` -- so a
    table offering `d` for one thing and `D` for another registers two actions
    on the same key, and Qt then fires *neither*, reporting an ambiguous
    overload to stderr where nobody reads it. Five pairs were written that way
    before anyone tried them. Uppercase means ``shift+d``, spelled out.
    """
    _keep_step_keys()
    seen: dict[str, str] = {}
    for binding in bindings:
        for spelling in (binding.keys, *binding.aliases):
            if binding.action is None:
                continue
            key = QtGui.QKeySequence(spelling).toString()
            # An unknown name parses to an empty sequence, which binds nothing
            # and says nothing: "PgDn" (Qt spells it PgDown) sat dead that way.
            if not key:
                raise ValueError(f"{spelling!r} is not a key Qt knows ({binding.description!r})")
            if key in seen:
                raise ValueError(
                    f"two shortcuts on {key!r}: {seen[key]!r} and {binding.description!r}. "
                    "Qt ignores case, so 'd' and 'D' are one key; write 'shift+d'."
                )
            seen[key] = binding.description

    for binding in bindings:
        if binding.action is None:
            continue
        for spelling in (binding.keys, *binding.aliases):
            action = QtGui.QAction(widget)
            action.setShortcut(QtGui.QKeySequence(spelling))
            action.setShortcutContext(QtCore.Qt.ShortcutContext.WindowShortcut)
            # Swallow QAction.triggered's `checked` argument. PySide6 passes it
            # to any slot that will accept one, so a binding written as
            # `lambda p=plane: ...` -- the natural way to capture a loop
            # variable -- silently receives False as `p` instead of the plane.
            # Every binding here is a nullary gesture, so none of them want it.
            action.triggered.connect(lambda *_, fn=binding.action: fn())
            widget.addAction(action)


#: Widget types that eat plain keystrokes when focused. Lists and combos both
#: implement type-to-search, so a focused one consumes every letter; a button
#: takes Space. None of them need the keyboard here -- each has a shortcut of
#: its own -- and a control that silently disables the shortcut table the
#: moment you click it is worse than one you cannot tab to.
_KEY_EATERS = (QtWidgets.QAbstractItemView, QtWidgets.QComboBox, QtWidgets.QAbstractButton)


def keep_keys_for_shortcuts(root: QtWidgets.QWidget) -> None:
    """Stop ``root``'s controls from swallowing single-key shortcuts.

    Clicking a layer in the list is the most common gesture in the viewer, and
    it moved focus into a QListWidget -- after which `d`, `c`, `o` and the rest
    went to its type-to-search instead of to the action table. Reported as "the
    keyboard shortcuts don't work", which is exactly what it looked like.

    Text entry is deliberately untouched: while you are typing a path into a
    line edit, the letters belong to the line edit.
    """
    for widget in root.findChildren(QtWidgets.QWidget):
        if isinstance(widget, _KEY_EATERS):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)


#: Rows (a key, or a group heading) a single column of the help holds before
#: the table is split, and a second column before it takes a third.
ONE_COLUMN_ROWS = 20
TWO_COLUMN_ROWS = 60


def split_columns(groups: list[tuple[str, list]]) -> list[list[tuple[str, list]]]:
    """Whole groups, in order, into 1-3 columns of about equal height.

    A group heading counts as a row, and a group is never split: a key cut off
    from its heading reads as belonging to the group above it.
    """
    heights = [len(items) + 1 for _, items in groups]
    total = sum(heights)
    n = 1 if total <= ONE_COLUMN_ROWS else 2 if total <= TWO_COLUMN_ROWS else 3
    out: list[list[tuple[str, list]]] = [[]]
    filled = 0
    for group, height in zip(groups, heights, strict=True):
        # Start the next column when this group would carry the current one
        # further past its share than stopping short of it would.
        share = total * len(out) / n
        if out[-1] and len(out) < n and filled + height - share > share - filled:
            out.append([])
        out[-1].append(group)
        filled += height
    return out


class ShortcutsDialog(QtWidgets.QDialog):
    """The `h` panel: what this window's keys do."""

    def __init__(
        self, title: str, bindings: Sequence[Binding], parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"keys · {title}")
        self.setStyleSheet(theme.stylesheet())
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(4)

        # Whole groups per column, so a long table (CHEDI's, the image
        # window's) fits on a screen instead of running off the bottom.
        # A table may visit a group more than once (the image window's VIEW
        # keys are in three places); each group is listed once, where it
        # first appears.
        by_name: dict[str, list[Binding]] = {}
        for binding in bindings:
            by_name.setdefault(binding.group, []).append(binding)
        groups = list(by_name.items())
        columns = QtWidgets.QHBoxLayout()
        columns.setSpacing(28)
        for column in split_columns(groups):
            grid = QtWidgets.QGridLayout()
            grid.setHorizontalSpacing(18)
            grid.setVerticalSpacing(3)
            row = 0
            for n, (group, items) in enumerate(column):
                if n:
                    row += 1
                head = QtWidgets.QLabel(group.upper())
                head.setObjectName("group")
                grid.addWidget(head, row, 0, 1, 2)
                row += 1
                for binding in items:
                    key = QtWidgets.QLabel(binding.keys)
                    key.setObjectName("key")
                    key.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
                    grid.addWidget(key, row, 0)
                    grid.addWidget(QtWidgets.QLabel(binding.description), row, 1)
                    row += 1
            grid.setRowStretch(row, 1)
            columns.addLayout(grid)
        outer.addLayout(columns)

        hint = QtWidgets.QLabel("esc or h to close")
        hint.setObjectName("group")
        hint.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        outer.addSpacing(6)
        outer.addWidget(hint)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 (Qt)
        # h toggles, so the key that opened it also closes it.
        if event.key() in (QtCore.Qt.Key.Key_Escape, QtCore.Qt.Key.Key_H):
            self.close()
            return
        super().keyPressEvent(event)


class ShortcutHelp:
    """Mixin-ish helper: give a window `h` and a dialog built from its table."""

    def __init__(self, widget: QtWidgets.QWidget, title: str) -> None:
        self._widget = widget
        self._title = title
        self._bindings: list[Binding] = []
        self._dialog: ShortcutsDialog | None = None

    def apply(self, bindings: Sequence[Binding]) -> None:
        self._bindings = list(bindings)
        install(self._widget, self._bindings)

    def toggle(self) -> None:
        if self._dialog is not None and self._dialog.isVisible():
            self._dialog.close()
            return
        self._dialog = ShortcutsDialog(self._title, self._bindings, self._widget)
        self._dialog.show()
