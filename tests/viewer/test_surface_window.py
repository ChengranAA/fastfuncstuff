"""Surface window logic that needs Qt but not a GPU: commands, uploads, picking."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import torch

QtWidgets = pytest.importorskip("PySide6.QtWidgets")
QtGui = pytest.importorskip("PySide6.QtGui")
QtCore = pytest.importorskip("PySide6.QtCore")
nib = pytest.importorskip("nibabel")

from fastfuncstuff.viewer.commands import Aspect  # noqa: E402
from tests.viewer.test_surface_render import _sheet  # noqa: E402


@pytest.fixture
def window(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.surfacewindow import SurfaceWindow
    from fastfuncstuff.viewer.vocab import OpenView

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-29.0, -29.0, -9.0]
    nib.save(nib.Nifti1Image(np.zeros((30, 30, 10), np.float32), aff), str(tmp_path / "a.nii.gz"))
    session = ViewerSession(device=torch.device("cpu"))
    session.load(str(tmp_path / "a.nii.gz"))
    session.surfaces.hemis = {"lh": _sheet()}
    session.surfaces.version = {"lh": 1}
    session.do(OpenView("S1", "surface", "axial"))
    win = SurfaceWindow("S1", session, session.do)
    win.canvas.resize(320, 320)
    win.apply(session.state.viewports.get("S1"))
    yield session, win
    win.close()
    session.close()
    app.processEvents()


def test_shape_command_morphs_from_what_is_drawn(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    c = win.canvas
    before = c.current("lh")
    assert before is not None
    session.do(SetSurfaceShape("S1", "flat"))
    win.apply(session.state.viewports.get("S1"))
    # Shape A is where it was drawn, shape B the new one; the morph runs between.
    np.testing.assert_allclose(c._cpu["lh"]["posA"], before[0], atol=1e-5)
    assert c._anim.state() == c._anim.State.Running
    with pytest.raises(ValueError, match="unknown surface shape"):
        session.do(SetSurfaceShape("S1", "pancake"))


def test_an_edit_under_a_laid_out_shape_uploads_only_where_data_is_sampled(window):
    session, win = window
    c = win.canvas
    c._pending.clear()
    hemi = session.surfaces.hemis["lh"]
    hemi.states["pial"][10] += np.float32([0, 0, 1.0])
    session.surfaces.version["lh"] += 1
    from fastfuncstuff.viewer.commands import Aspect

    win.refresh(Aspect.SLICES)
    # Areas too: equivolume depth depends on them, and an edit changes them.
    assert set(c._pending["lh"]) == {"white", "pial", "areas"}
    np.testing.assert_array_equal(c._cpu["lh"]["pial"], hemi.states["pial"])


def test_click_on_the_surface_gives_the_mid_depth_point_under_it(window):
    from PySide6.QtCore import QPointF

    from fastfuncstuff.viewer.vocab import SetSurfaceDepth, SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    session.do(SetSurfaceDepth("S1", 0.5, 0.5, 1))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    mm = c.pick_mm(QPointF(160.0, 160.0))
    assert mm is not None
    # Seen from straight above the sheet's centre: x, y near the target, and z
    # half-way between white (0) and pial (3).
    target = c.camera.target
    assert abs(mm[0] - target[0]) < 1.0 and abs(mm[1] - target[1]) < 1.0
    assert mm[2] == pytest.approx(1.5, abs=1e-4)


def test_depth_and_hemisphere_commands_round_trip_through_a_script(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceDepth, SetSurfaceHemis

    session, _ = window
    session.do(SetSurfaceDepth("S1", 0.0, 1.0, 5))
    session.do(SetSurfaceHemis("S1", "rh", 20.0))
    vp = session.state.viewports.get("S1")
    assert vp.depth == (0.0, 1.0) and vp.samples == 5
    assert vp.hemis == "rh" and vp.split == 20.0
    script = session.to_script()
    assert "SET_SURFACE_DEPTH S1 0.0 1.0 5" in script
    assert "SET_SURFACE_HEMIS S1 rh 20.0" in script


def test_a_turned_hemisphere_pivots_on_its_centre_and_still_picks_true_mm(window):
    from PySide6.QtCore import QPointF

    from fastfuncstuff.viewer import surface3d as s3
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    before = c.drawn_positions("lh")
    assert before is not None
    c.hemi_rotation["lh"] = s3._rotation([0, 0, 1], np.pi / 3)
    after = c.drawn_positions("lh")
    assert after is not None
    np.testing.assert_allclose(after.mean(0), before.mean(0), atol=1e-3)
    assert np.abs(after - before).max() > 5.0
    # Turned on screen, but the point under the cursor is still a point of the
    # sheet in scanner space: picking goes through barycentrics, not display xyz.
    mm = c.pick_mm(QPointF(160.0, 160.0))
    assert mm is not None and mm[2] == pytest.approx(1.5, abs=1e-4)
    win._reset_camera()
    assert c.hemi_rotation == {}


def test_a_recomputed_mode_overlay_reaches_the_surface(window):
    """The bug: instacorr re-adopts its map under the same key on every seed
    click, and the texture cache keyed on the layer alone kept drawing the first
    map while the slices moved on."""
    from fastfuncstuff.viewer.commands import Aspect
    from fastfuncstuff.viewer.modes.base import ComputedOverlay

    session, win = window
    source = session.state.layers.base.key
    aff = session.state.layers.base.affine

    def install(fill: float) -> np.ndarray:
        values = np.full((30, 30, 10), fill, np.float32)
        session.install_computed_overlay(source, ComputedOverlay(values, aff, "icorr"))
        win.refresh(Aspect.LAYERS)
        return win.canvas.overlays[-1].value

    first = install(0.2)
    second = install(0.7)
    assert not np.array_equal(first, second), "the surface still draws the first map"
    assert np.allclose(second, 0.7)


def test_ctrl_click_on_the_surface_seeds_and_a_plain_click_only_locates(window):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    located, seeded = [], []
    win.located.connect(lambda *mm: located.append(mm))
    win.seeded.connect(lambda *mm: seeded.append(mm))

    def click(mods: Qt.KeyboardModifier) -> None:
        at = QPointF(160.0, 160.0)
        for kind in (QMouseEvent.Type.MouseButtonPress, QMouseEvent.Type.MouseButtonRelease):
            buttons = (
                Qt.MouseButton.LeftButton
                if kind == QMouseEvent.Type.MouseButtonPress
                else Qt.MouseButton.NoButton
            )
            ev = QMouseEvent(kind, at, at, Qt.MouseButton.LeftButton, buttons, mods)
            (
                c.mousePressEvent
                if kind == QMouseEvent.Type.MouseButtonPress
                else c.mouseReleaseEvent
            )(ev)

    click(Qt.KeyboardModifier.ControlModifier)
    assert len(seeded) == 1 and not located
    assert seeded[0] == pytest.approx(c.pick_mm(QPointF(160.0, 160.0)))
    click(Qt.KeyboardModifier.NoModifier)
    assert len(seeded) == 1 and len(located) == 1


def test_a_surface_seed_lands_on_the_voxel_the_crosshair_moved_to(window):
    from fastfuncstuff.viewer.ui.manager import WindowManager
    from fastfuncstuff.viewer.vocab import SetSeed, SetXYZ

    session, _ = window
    sent = []

    def dispatch(cmd):
        sent.append(cmd)
        session.do(cmd)

    WindowManager(session, dispatch)._on_seeded(-9.0, -19.0, 1.0)
    assert [type(c) for c in sent] == [SetXYZ, SetSeed]
    # The fixture's grid: origin (-29, -29, -9) mm, 2 mm voxels.
    assert session.state.crosshair == (10, 5, 5)
    assert session.state.seed == (10, 5, 5)


def test_the_crosshair_marks_its_nearest_vertex_and_c_aims_the_camera_there(window):
    """The crosshair is rarely at mid-depth; a mark only within 2 mm of it drew
    nothing. And on an inflated surface its mm is not where it is drawn."""
    from fastfuncstuff.viewer.vocab import SetSurfaceShape, SetXYZ

    session, win = window
    session.do(SetSurfaceShape("S1", "flat"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    # 9 mm above the sheet (mid-depth is z = 1.5): far off the surface.
    session.do(SetXYZ(5.0, -7.0, 9.0))
    win.refresh(Aspect.ALL)
    cx, cy, cz, r = c.cross
    assert r > 0 and cz == pytest.approx(1.5, abs=1e-4)
    assert abs(cx - 5.0) < 1.5 and abs(cy + 7.0) < 1.5

    hemi, k = c.nearest_vertex((5.0, -7.0, 9.0))
    win._centre_view()
    assert np.allclose(c.camera.target, c.drawn_positions(hemi)[k], atol=1e-4)

    # On a laid-out shape the overlay marks the nearest vertex as drawn.
    centre, _ = c.cross_marker()
    np.testing.assert_allclose(centre, c.drawn_positions(hemi)[k], atol=1e-4)

    assert c.cross_style == "lines"
    win._cycle_cross()
    assert c.cross_style == "off" and c.cross[3] == 0.0 and c.cross_marker() is None
    win._cycle_cross()
    assert c.cross_style == "dot" and c.cross[3] > 0 and c.cross_marker() is not None
    win._cycle_cross()
    assert c.cross_style == "lines"


def _overlay_pixels(c, rgb, tol=40):
    """How many of the overlay's pixels are near ``rgb`` (0-255)."""
    # Onto a clear image: grab() lays the overlay over an opaque background.
    img = QtGui.QImage(c._overlay.size(), QtGui.QImage.Format.Format_RGBA8888)
    img.fill(QtCore.Qt.GlobalColor.transparent)
    c._overlay.render(img)
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine() // 4, 4)
    near = (np.abs(a[..., :3].astype(int) - np.asarray(rgb)) < tol).all(-1) & (a[..., 3] > 0)
    return near


def test_the_crosshair_lines_go_through_the_crosshair_itself_and_are_drawn_on_top(window):
    """Off the surface (in white matter, say) the lines still meet at the crosshair
    on an anatomical shape, and the overlay draws them whatever the mesh hides."""
    from fastfuncstuff.viewer.ui import theme
    from fastfuncstuff.viewer.vocab import SetSurfaceShape, SetXYZ

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    c._overlay.resize(c.size())
    session.do(SetXYZ(5.0, -7.0, 9.0))
    win.refresh(Aspect.ALL)
    centre, axes = c.cross_marker()
    m = c.model("lh")
    np.testing.assert_allclose(centre, m[:3, :3] @ [5.0, -7.0, 9.0] + m[:3, 3], atol=1e-6)
    np.testing.assert_allclose(axes, m[:3, :3])

    rgb = np.round(np.asarray(theme.palette().crosshair[:3]) * 255)
    lines = _overlay_pixels(c, rgb).sum()
    win._cycle_cross()  # off
    assert _overlay_pixels(c, rgb).sum() == 0
    win._cycle_cross()  # dot: the ring alone
    ring = _overlay_pixels(c, rgb).sum()
    assert 0 < ring < lines
    # The ring sits where the crosshair projects.
    x, y = c.camera.to_widget(centre, c._overlay.width(), c._overlay.height())[0][0]
    ys, xs = np.nonzero(_overlay_pixels(c, rgb))
    assert abs(xs.mean() - x) < 2 and abs(ys.mean() - y) < 2


def test_camera_puts_its_target_at_the_middle_and_px_per_mm_matches_projection():
    from fastfuncstuff.viewer.surface3d import Camera

    cam = Camera()
    cam.target = np.array([3.0, -4.0, 5.0])
    xy, front = cam.to_widget(cam.target, 400, 300)
    x, y = xy[0]
    assert front[0] and x == pytest.approx(200) and y == pytest.approx(150)
    # 10 mm along the screen's up axis at the target: 10 * px_per_mm pixels.
    up = cam.rotation[1]
    y2 = cam.to_widget(cam.target + 10 * up, 400, 300)[0][0, 1]
    assert y - y2 == pytest.approx(10 * cam.px_per_mm(300), rel=1e-6)
    behind = cam.target + cam.rotation[2] * (cam.distance + 10)
    assert not cam.to_widget(behind, 400, 300)[1][0]


def test_a_shape_change_keeps_the_zoom_relative_to_the_brain(window):
    """Pial to inflated filled the window: the camera stayed put while the
    shape grew. It now backs off in step with the morph."""
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.surfaces.hemis = {"lh": _sheet(pial_scale=2.0)}
    session.surfaces.version = {"lh": 2}
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c._on_morph(1.0)
    win._reset_camera()
    before = c.camera.distance
    session.do(SetSurfaceShape("S1", "pial"))
    win.apply(session.state.viewports.get("S1"))
    c._anim.stop()
    c._on_morph(0.0)
    assert c.camera.distance == pytest.approx(before)
    c._on_morph(1.0)
    # The pial sheet is twice the white's width (its diagonal near twice too).
    assert 1.8 < c.camera.distance / before < 2.1


def test_a_continuous_map_shows_its_scale_and_a_parcellation_does_not(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceMap

    session, win = window
    session.surfaces.hemis["lh"].morph["curv"] = np.linspace(-1, 1, 41 * 41).astype(np.float32)
    session.do(SetSurfaceMap("S1", "curv"))
    win.apply(session.state.viewports.get("S1"))
    assert not win.legend.isHidden()
    lo, hi, name = win.legend._scale
    assert lo == pytest.approx(-hi) and lo > 0 and name == "RdBu"
    win.legend.grab()  # paints
    session.do(SetSurfaceMap("S1", ""))
    win.apply(session.state.viewports.get("S1"))
    assert win.legend.isHidden()


def test_d_cycles_the_depth_statistic_and_it_reaches_the_canvas(window):
    from fastfuncstuff.viewer.vocab import SetSurfaceDepth, SetSurfaceDepthStat

    session, win = window
    session.do(SetSurfaceDepth("S1", 0.0, 1.0, 6))
    win._cycle_depth_stat()
    vp = session.state.viewports.get("S1")
    assert vp.depth_stat == "median"
    win.refresh(Aspect.ALL)
    assert win.canvas.depth_stat == "median"
    assert "median" in win.depth_label.text()
    assert "SET_SURFACE_DEPTH_STAT" in session.to_script()
    with pytest.raises(ValueError, match="unknown depth statistic"):
        session.do(SetSurfaceDepthStat("S1", "mode"))


def test_ctrl_drag_swings_the_hemispheres_open_and_o_cycles_the_presets(window):
    from PySide6 import QtCore, QtGui, QtWidgets

    session, win = window
    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    B, M = QtCore.Qt.MouseButton, QtCore.Qt.KeyboardModifier

    def send(kind, x, buttons, mods):
        pos = QtCore.QPointF(x, 160.0)
        QtWidgets.QApplication.sendEvent(
            c, QtGui.QMouseEvent(kind, pos, pos, B.LeftButton, buttons, mods)
        )

    E = QtCore.QEvent.Type
    send(E.MouseButtonPress, 100.0, B.LeftButton, M.ControlModifier)
    send(E.MouseMove, 110.0, B.LeftButton, M.ControlModifier)
    send(E.MouseMove, 180.0, B.LeftButton, M.ControlModifier)
    send(E.MouseButtonRelease, 180.0, B.NoButton, M.ControlModifier)
    vp = session.state.viewports.get("S1")
    assert vp.hinge > 0
    win.refresh(Aspect.ALL)  # the manager does this after every command
    # Turned about the vertical: z is untouched, x/y rotated.
    m = c._model["lh"]
    assert m[2, 2] == pytest.approx(1.0) and m[0, 0] < 1.0
    assert session.state.seed is None  # a drag, not a ctrl+click

    send(E.MouseButtonPress, 100.0, B.LeftButton, M.ControlModifier | M.ShiftModifier)
    send(E.MouseMove, 140.0, B.LeftButton, M.ControlModifier | M.ShiftModifier)
    assert session.state.viewports.get("S1").split > 0
    send(E.MouseButtonRelease, 140.0, B.NoButton, M.NoModifier)

    from fastfuncstuff.viewer.vocab import SetSurfaceHinge

    session.do(SetSurfaceHinge("S1", 0.0))
    win._cycle_hinge()
    assert session.state.viewports.get("S1").hinge == 180.0
    win._cycle_hinge()
    assert session.state.viewports.get("S1").hinge == -180.0
    win._cycle_hinge()
    assert session.state.viewports.get("S1").hinge == 0.0


def test_paint_mode_paints_the_vertex_under_the_cursor_instead_of_turning(window):
    from PySide6 import QtCore, QtGui, QtWidgets

    from fastfuncstuff.viewer.vocab import SetSurfaceShape

    session, win = window
    session.do(SetSurfaceShape("S1", "white"))
    win.apply(session.state.viewports.get("S1"))
    c = win.canvas
    c._anim.stop()
    c.morph = 1.0
    win._reset_camera()
    got = []
    c.painted.connect(lambda h, v, e: got.append((h, v, e)))
    win._toggle_paint()
    turned = c.camera.rotation.copy()
    pos = QtCore.QPointF(160.0, 160.0)
    B = QtCore.Qt.MouseButton
    for kind, buttons in (
        (QtCore.QEvent.Type.MouseButtonPress, B.LeftButton),
        (QtCore.QEvent.Type.MouseMove, B.LeftButton),
    ):
        QtWidgets.QApplication.sendEvent(
            c,
            QtGui.QMouseEvent(
                kind, pos, pos, B.LeftButton, buttons, QtCore.Qt.KeyboardModifier.NoModifier
            ),
        )
    assert got and got[0][0] == "lh" and got[0][2] is False
    np.testing.assert_allclose(c.camera.rotation, turned)
    assert session.surfaces.highlight["lh"].sum() > 0


def test_a_surface_layer_draws_on_its_vertices_with_the_layer_controls(window, tmp_path):
    # A reml bucket on this mesh, loaded like any dataset: the surface window draws
    # it on the vertices, over the anatomical map, and the layer's own threshold
    # command decides which vertices show -- no surface-only controls.
    from fastfuncstuff.io.afni import save_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, set_surface_meta
    from fastfuncstuff.viewer.vocab import SetThreshold, SetThresholdIndex

    session, win = window
    h = session.surfaces.hemis["lh"]
    hdr = nib.Nifti2Header()
    set_surface_meta(hdr, {"mesh_fingerprint": mesh_fingerprint(h.faces, h.n_vertices)})
    xy = h.states["white"][:, :2]
    t = (8.0 * np.exp(-(xy**2).sum(1) / 40.0)).astype(np.float32)
    vals = np.stack([t, t], 1)[:, None, None, :]
    save_nifti(vals, tmp_path / "s.lh.func.gii", header=hdr,
               brick_labels=["c#0_Coef", "c#0_Tstat"], brick_stataux={1: (3, (60.0,))})  # fmt: skip
    key = session.load(str(tmp_path / "s.lh.func.gii"))
    session.do(SetThresholdIndex(key, 1))
    session.do(SetThreshold(key, 3.5))
    win.refresh(Aspect.ALL)
    shown = win.canvas._cpu["lh"]["vcolor"][:, 3] > 0
    np.testing.assert_array_equal(shown, t > 3.5)
    session.do(SetThreshold(key, 6.0))
    win.refresh(Aspect.ALL)
    np.testing.assert_array_equal(win.canvas._cpu["lh"]["vcolor"][:, 3] > 0, t > 6.0)
    # drawn per vertex, so never also sampled from the ribbon it is painted into
    assert key not in [ly.key for ly in win._overlay_layers()]


def test_paint_mode_outlines_the_brush_at_the_cursor_at_its_size(window):
    session, win = window
    c = win.canvas
    c._overlay.resize(c.size())
    win._cycle_cross()  # crosshair off: only the brush is drawn
    win._toggle_paint()
    pos = QtCore.QPointF(150.0, 170.0)
    hover = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        pos,
        pos,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(c, hover)

    def radius():
        ys, xs = np.nonzero(_overlay_pixels(c, (255, 255, 255)))
        assert xs.size, "no brush outline drawn"
        return float(np.median(np.hypot(xs - pos.x(), ys - pos.y())))

    def expected():
        # At the overlay's height as painted: it follows the canvas's.
        return session.state.surface_brush[0] * c.camera.px_per_mm(c._overlay.height())

    assert radius() == pytest.approx(expected(), abs=2.0)
    r0 = expected()
    win._brush_by(2.0)
    assert radius() == pytest.approx(2 * r0, abs=2.0)
    win._toggle_paint()  # off: no outline
    assert not _overlay_pixels(c, (255, 255, 255)).any()
