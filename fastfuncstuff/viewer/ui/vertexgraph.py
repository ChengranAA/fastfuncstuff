"""The vertex graph: time courses drawn where their vertices sit on a flattened patch.

The voxel graph's grid has no counterpart on a mesh, and a regular grid sampled
off the surface would hide the very thing this window is for: seeing why one
vertex and not its neighbour. So every vertex within a few mesh rings gets a
cell at its own place on the patch (:mod:`viewer.vertexpatch`), with the mesh
edges drawn faintly between them, and the 3-D window outlines the patch on the
surface. Crowded at many rings by design -- a bigger window spreads it out.

Cells are drawn by the voxel graph's own painter, so the shared y-scale, the
overlay wash, decimation and the time cursor mean the same thing in both.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from fastfuncstuff.viewer.ui import theme
from fastfuncstuff.viewer.ui.gridgraph import DETREND_LABELS, Cell, FlowLayout, GridGraph
from fastfuncstuff.viewer.ui.shortcuts import Binding, ShortcutHelp, keep_keys_for_shortcuts

#: Cell side as a fraction of the typical spacing between neighbours on the page.
CELL_FILL = 0.88
MAX_RINGS = 12


class PatchGraph(GridGraph):
    """Cells at page positions instead of on a square grid."""

    #: A double-click on a cell: its row in the patch.
    chosen = QtCore.Signal(int)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._xy = np.zeros((0, 2))
        self._edges = np.zeros((0, 2), np.int64)
        # The grid's own label (voxel ijk at small sizes) means nothing here.
        self._n = 99

    def set_patch(self, cells, xy, edges, index, colors=None, time_keys=None) -> None:
        self._xy = np.asarray(xy, np.float64)
        self._edges = np.asarray(edges, np.int64)
        self.set_cells(cells, 99, index, colors, time_keys)

    def _layout(self) -> tuple[np.ndarray, float]:
        """Cell centres in pixels and the cell side."""
        xy = self._xy
        if len(xy) == 0:
            return np.zeros((0, 2)), 0.0
        if len(xy) == 1:
            side = 0.8 * min(self.width(), self.height())
            return np.array([[self.width() / 2, self.height() / 2]]), side
        from scipy.spatial import KDTree

        d, _ = KDTree(xy).query(xy, k=2)
        spacing = float(np.median(d[:, 1])) or 1.0
        lo, hi = xy.min(0), xy.max(0)
        span = hi - lo
        # Room for half a cell beyond the outermost centres on every side.
        pad = CELL_FILL * spacing
        scale = min(self.width() / (span[0] + pad), self.height() / (span[1] + pad))
        mid = (lo + hi) / 2
        px = np.c_[
            self.width() / 2 + (xy[:, 0] - mid[0]) * scale,
            self.height() / 2 - (xy[:, 1] - mid[1]) * scale,
        ]
        return px, CELL_FILL * spacing * scale

    def _cell_rects(self) -> list[QtCore.QRectF]:
        px, side = self._layout()
        return [QtCore.QRectF(x - side / 2, y - side / 2, side, side) for x, y in px]

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 (Qt)
        # Last drawn is on top, so search from the end.
        rects = self._cell_rects()
        for row in range(len(rects) - 1, -1, -1):
            if rects[row].contains(event.position()):
                self.chosen.emit(row)
                return

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 (Qt)
        if not self._cells:
            super().paintEvent(event)
            return
        p = QtGui.QPainter(self)
        c = theme.palette()
        p.fillRect(self.rect(), QtGui.QColor(c.bg))
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        px, _ = self._layout()
        # The mesh under the cells, faint: who neighbours whom.
        edge = QtGui.QColor(c.faint)
        edge.setAlphaF(0.6)
        p.setPen(QtGui.QPen(edge, 1.0))
        for a, b in self._edges:
            p.drawLine(QtCore.QPointF(*px[a]), QtCore.QPointF(*px[b]))
        shared = self._bounds(self._cells) if self._shared_scale else None
        rects = self._cell_rects()
        bg = QtGui.QColor(c.bg)
        # Outer rings first and the centre last, so overlaps favour the middle.
        for row in range(len(self._cells) - 1, -1, -1):
            p.fillRect(rects[row], bg)
            self._paint_cell(p, rects[row], self._cells[row], shared)
        p.end()


class VertexGraphWindow(QtWidgets.QWidget):
    """Time courses of the vertices around the crosshair, for one 3-D window."""

    closed = QtCore.Signal()

    def __init__(self, surface_window, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.surface = surface_window
        self.session = surface_window.session
        self.setWindowFlag(QtCore.Qt.WindowType.Window, True)
        self.setWindowTitle(f"vertex graph · {surface_window.vid}")
        self.setStyleSheet(theme.stylesheet())
        self.resize(560, 520)
        self.setMinimumSize(120, 100)
        #: View state, kept here rather than in a recorded viewport: what is
        #: plotted around a vertex, not something a script needs to replay.
        self.rings = 3
        self.detrend = 0
        self.hidden: set[str] = set()
        self.patch = None
        self._patch_key: tuple | None = None
        #: The patch's hex positions, made once per patch.
        self._hex: tuple[object, np.ndarray] | None = None
        self._legend: tuple | None = None

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(7, 6, 7, 7)
        v.setSpacing(6)
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(6)
        for text, tip, slot in (
            ("[-]", "One ring fewer (-)", lambda: self.step_rings(-1)),
            ("[+]", "One ring more (+)", lambda: self.step_rings(1)),
        ):
            b = QtWidgets.QPushButton(text)
            b.setToolTip(tip)
            b.setMaximumWidth(42)
            b.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            bar.addWidget(b)
            if text == "[-]":
                self.count_label = QtWidgets.QLabel("")
                self.count_label.setObjectName("value")
                bar.addWidget(self.count_label)
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel(theme.key_label("detrend", "d")))
        self.detrend_box = QtWidgets.QComboBox()
        self.detrend_box.setToolTip("Polynomial taken out of every line before drawing (d steps)")
        self.detrend_box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for order, name in DETREND_LABELS:
            self.detrend_box.addItem(name, userData=order)
        self.detrend_box.setCurrentIndex(self.detrend_box.findData(self.detrend))
        self.detrend_box.activated.connect(
            lambda i: self.set_detrend(int(self.detrend_box.itemData(i)))
        )
        bar.addWidget(self.detrend_box)
        self.shared_check = QtWidgets.QCheckBox(theme.key_label("shared scale", "s"))
        self.shared_check.setChecked(True)
        self.shared_check.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.shared_check.toggled.connect(lambda on: self.graph.set_shared_scale(on))
        bar.addWidget(self.shared_check)
        self.tint_check = QtWidgets.QCheckBox(theme.key_label("olay colour", "o"))
        self.tint_check.setToolTip(
            "Wash each cell in the overlay's colour at its vertex, grey where it is cut"
        )
        self.tint_check.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.tint_check.toggled.connect(lambda _: self.refresh())
        bar.addWidget(self.tint_check)
        self.hex_check = QtWidgets.QCheckBox(theme.key_label("hex", "x"))
        self.hex_check.setToolTip(
            "Cells on a hexagonal lattice, each vertex on the free site nearest its\n"
            "own place: tidier, every vertex still shown. Off: exact positions."
        )
        self.hex_check.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.hex_check.toggled.connect(lambda _: self.refresh())
        bar.addWidget(self.hex_check)
        bar.addStretch(1)
        v.addLayout(bar)

        self.trace_host = QtWidgets.QWidget()
        self.trace_bar = FlowLayout(self.trace_host, spacing=8)
        v.addWidget(self.trace_host)

        self.graph = PatchGraph()
        self.graph.scrubbed.connect(self._scrub)
        self.graph.chosen.connect(self._choose)
        v.addWidget(self.graph, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setObjectName("value")
        v.addWidget(self.info)

        self.help = ShortcutHelp(self, f"vertex graph · {surface_window.vid}")
        self.help.apply(
            [
                Binding("+", "one mesh ring more", lambda: self.step_rings(1), group="patch"),
                Binding("=", "one mesh ring more", lambda: self.step_rings(1), group="patch"),
                Binding("-", "one mesh ring fewer", lambda: self.step_rings(-1), group="patch"),
                Binding("d", "next detrend order", self.cycle_detrend, group="lines"),
                Binding("Right", "next time point", lambda: self.step_time(1), group="time"),
                Binding("Left", "previous time point", lambda: self.step_time(-1), group="time"),
                Binding(".", "next time point", lambda: self.step_time(1), group="time"),
                Binding(",", "previous time point", lambda: self.step_time(-1), group="time"),
                Binding("s", "shared scale on / off", self.shared_check.toggle, group="lines"),
                Binding("o", "overlay colour on / off", self.tint_check.toggle, group="lines"),
                Binding(
                    "x", "hexagonal lattice / exact positions", self.hex_check.toggle, group="patch"
                ),
                Binding("click", "go to that time point", None, group="lines"),
                Binding("double-click", "move the crosshair to that vertex", None, group="patch"),
                Binding("h", "this list", self.help.toggle, group="window"),
                Binding("ctrl+w", "close this window", self.close, group="window"),
            ]
        )
        keep_keys_for_shortcuts(self)

    # -- controls ------------------------------------------------------------
    def step_rings(self, delta: int) -> None:
        self.rings = int(np.clip(self.rings + delta, 1, MAX_RINGS))
        self.refresh()

    def set_detrend(self, order: int) -> None:
        self.detrend = int(order)
        self.refresh()

    def cycle_detrend(self) -> None:
        orders = [k for k, _ in DETREND_LABELS]
        self.set_detrend(orders[(orders.index(self.detrend) + 1) % len(orders)])

    def step_time(self, delta: int) -> None:
        """Step the shared volume index, wrapping, as , and . do everywhere."""
        from fastfuncstuff.viewer.vocab import SetIndex

        st = self.session.state
        hi = st.max_time_index()
        if hi > 0:
            self.surface._dispatch(SetIndex((st.time_index + delta) % (hi + 1)))

    def _scrub(self, index: int) -> None:
        from fastfuncstuff.viewer.vocab import SetIndex

        self.surface._dispatch(SetIndex(int(index)))

    def _choose(self, row: int) -> None:
        """Double-click: the crosshair to that vertex, at the sampled depth."""
        if self.patch is None:
            return
        hemi, vertex = self.patch.hemi, int(self.patch.vertices[row])
        h = self.session.surfaces.hemis[hemi]
        d = self._depth()
        mm = (1 - d) * np.asarray(h.states["white"][vertex]) + d * np.asarray(
            h.states.get("pial", h.states["white"])[vertex]
        )
        self.surface.located.emit(float(mm[0]), float(mm[1]), float(mm[2]))

    def _depth(self) -> float:
        vp = self.surface._viewport()
        return 0.5 if vp is None else float(np.mean(vp.depth))

    # -- drawing -------------------------------------------------------------
    def _legend_sync(self, layers, colours) -> None:
        signature = tuple((ly.key, ly.name, ly.key in self.hidden) for ly in layers)
        if signature == self._legend:
            return
        self._legend = signature
        while self.trace_bar.count():
            item = self.trace_bar.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()
        faint = theme.palette().faint
        for ly in layers:
            colour = colours[ly.key]
            on = ly.key not in self.hidden
            box = QtWidgets.QCheckBox(ly.name)
            box.setToolTip(ly.path)
            box.setChecked(on)
            box.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            box.setStyleSheet(
                f"QCheckBox {{ color: {colour.name() if on else faint}; }}"
                f"QCheckBox::indicator:checked {{ background: {colour.name()};"
                f" border: 1px solid {colour.name()}; }}"
            )
            box.toggled.connect(lambda checked, k=ly.key: self._toggle(k, checked))
            self.trace_bar.addWidget(box)
        if not layers:
            self.trace_bar.addWidget(QtWidgets.QLabel("no time series loaded"))
        self.trace_host.updateGeometry()

    def _toggle(self, key: str, on: bool) -> None:
        (self.hidden.discard if on else self.hidden.add)(key)
        self.refresh()

    def refresh(self) -> None:
        from fastfuncstuff.viewer.derive import detrend_trace
        from fastfuncstuff.viewer.vertexpatch import vertex_patch

        self.detrend_box.setCurrentIndex(self.detrend_box.findData(self.detrend))
        session, canvas = self.session, self.surface.canvas
        st = session.state
        mm = st.crosshair_mm
        found = None if mm is None else canvas.nearest_vertex(mm)
        if found is None:
            self.patch = None
            canvas.outline = None
            self.graph.set_patch([], [], [], st.time_index)
            self.info.setText("no surface shown")
            return
        hemi, centre = found
        h = session.surfaces.hemis[hemi]
        shown = canvas.drawn_positions(hemi)
        vp = self.surface._viewport()
        key = (
            hemi,
            centre,
            self.rings,
            session.surfaces.version.get(hemi, 0),
            None if vp is None else vp.shape,
            np.round(canvas.camera.rotation, 2).tobytes(),
        )
        if key != self._patch_key or self.patch is None:
            white = np.asarray(h.states["white"], np.float64)
            pial = np.asarray(h.states.get("pial", h.states["white"]), np.float64)
            self.patch = vertex_patch(
                hemi,
                h.faces,
                0.5 * (white + pial),
                white if shown is None else shown,
                centre,
                self.rings,
                canvas.camera.rotation[1],
                canvas.camera.rotation[2],
            )
            self._patch_key = key
        patch = self.patch
        canvas.outline = (hemi, patch.outline)
        canvas.update()

        layers = session.graph_layers()
        ink = QtGui.QColor(theme.palette().text)
        series = ((ink.redF(), ink.greenF(), ink.blueF()), *theme.palette().series)
        colours = {
            ly.key: QtGui.QColor.fromRgbF(*series[i % len(series)]) for i, ly in enumerate(layers)
        }
        self._legend_sync(layers, colours)
        depth = self._depth()
        courses = {
            ly.key: session.vertex_timeseries(ly, hemi, patch.vertices, depth)
            for ly in layers
            if ly.key not in self.hidden
        }
        tints = (
            session.vertex_tints(hemi, patch.vertices, depth)
            if self.tint_check.isChecked()
            else None
        )
        cells = [
            Cell(
                ijk=(int(v), 0, 0),
                traces=[
                    (k, detrend_trace(arr[row], self.detrend))
                    for k, arr in courses.items()
                    if arr.shape[1]
                ],
                is_centre=row == 0,
                tint=tints[row] if tints else None,
            )
            for row, v in enumerate(patch.vertices)
        ]
        xy = patch.xy
        if self.hex_check.isChecked():
            from fastfuncstuff.viewer.vertexpatch import hex_layout

            if self._hex is None or self._hex[0] is not patch:
                self._hex = (patch, hex_layout(patch.xy))
            xy = self._hex[1]
        self.graph.set_tinting(tints is not None)
        self.graph.set_patch(
            cells,
            xy,
            patch.edges,
            st.time_index,
            colors=colours,
            time_keys=set(courses),
        )
        n = len(patch.vertices)
        self.count_label.setText(f"{self.rings} ring{'s' * (self.rings > 1)} · {n} vertices")
        self.info.setText(
            f"{hemi} vertex {centre}  ·  depth {depth:.2f}  ·  "
            f"{np.ptp(patch.xy[:, 0]):.1f} x {np.ptp(patch.xy[:, 1]):.1f} mm along the cortex"
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt)
        self.surface.canvas.outline = None
        self.surface.canvas.update()
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["PatchGraph", "VertexGraphWindow"]
