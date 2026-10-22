"""PAINT mode: drawing integer-label ROIs on the slices (viewer.paint, paintlayer, modes.paint)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.compose import render_plane
from fastfuncstuff.viewer.modes.paint import PaintMode
from fastfuncstuff.viewer.paint import brush_voxels, decode_points, encode_points, stroke_centres
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import ModeAction, PaintUndo, SetMode, SetModeParam

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
SHAPE = (30, 32, 20)


@pytest.fixture
def session(tmp_path):
    aff = np.diag([2.0, 2.0, 2.5, 1.0])
    aff[:3, 3] = [-30.0, -32.0, -25.0]
    rng = np.random.default_rng(0)
    path = tmp_path / "anat.nii.gz"
    nib.save(nib.Nifti1Image(rng.random(SHAPE).astype(np.float32) * 100, aff), str(path))
    s = ViewerSession(device=CPU)
    s.load(path)
    s.do(SetMode("paint"))
    yield s
    s.close()


def _mm(s, ijk):
    return np.asarray(s.state.grid.ijk_to_mm(ijk), dtype=float)


def _labels(s, key):
    return s.store.get(key).array[..., 0]


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_a_disk_stays_on_its_slice_and_is_round_in_mm():
    aff = np.diag([1.0, 2.0, 3.0, 1.0])
    pts = brush_voxels((10.0, 10.0, 9.0), 4.0, affine=aff, shape=(40, 40, 40), plane=Plane.AXIAL)
    assert set(pts[:, 2]) == {3}  # z = 9 mm is voxel 3; axial disk keeps it
    mm = pts * np.array([1.0, 2.0, 3.0])
    assert np.all(np.linalg.norm(mm[:, :2] - [10.0, 10.0], axis=1) <= 4.0)
    # Anisotropic voxels: wider in i (1 mm) than in j (2 mm), as a circle in mm is.
    assert np.ptp(pts[:, 0]) == 8 and np.ptp(pts[:, 1]) == 4


def test_a_sphere_reaches_neighbouring_slices_and_a_tiny_brush_paints_one_voxel():
    aff = np.eye(4)
    sphere = brush_voxels(
        (5, 5, 5), 1.5, affine=aff, shape=(10, 10, 10), plane=Plane.AXIAL, brush="sphere"
    )
    assert set(sphere[:, 2]) == {4, 5, 6}
    one = brush_voxels((5.2, 5.1, 4.9), 0.1, affine=aff, shape=(10, 10, 10), plane=Plane.CORONAL)
    assert one.tolist() == [[5, 5, 5]]
    edge = brush_voxels((0, 0, 0), 3.0, affine=aff, shape=(10, 10, 10), plane=Plane.AXIAL)
    assert (edge >= 0).all()


def test_a_fast_drag_is_filled_in_and_points_round_trip():
    c = stroke_centres((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), 2.0)
    assert np.allclose(np.diff(c[:, 0]), 1.0)  # half the radius apart
    assert np.allclose(decode_points(encode_points(c)), c)


# ---------------------------------------------------------------------------
# painting in a session
# ---------------------------------------------------------------------------


def _stroke(s, at, *, erase=False):
    mode = s.mode
    assert isinstance(mode, PaintMode)
    mode.begin(_mm(s, at[0]), Plane.AXIAL, erase=erase)
    for p in at[1:]:
        mode.extend(_mm(s, p))
    mode.end()
    return mode.target()


def test_the_first_stroke_makes_an_roi_layer_on_the_display_grid(session):
    key = _stroke(session, [(10, 10, 8), (20, 10, 8)])
    layer = session.state.layers.get(key)
    assert layer.roi and layer.shape == session.state.grid.shape
    lab = _labels(session, key)
    assert lab[10:21, 10, 8].min() == 1 and lab[:, :, 7].max() == 0
    assert render_plane(session, Plane.AXIAL) is not None


def test_right_drag_erases_and_overwrite_off_respects_other_labels(session):
    key = _stroke(session, [(15, 15, 8)])
    session.do(SetModeParam("value", "2"))
    session.do(SetModeParam("overwrite", "0"))
    _stroke(session, [(13, 15, 8), (17, 15, 8)])
    lab = _labels(session, key)
    assert lab[15, 15, 8] == 1 and lab[13, 15, 8] == 2  # 2 stopped at label 1
    _stroke(session, [(15, 15, 8)], erase=True)
    assert lab[15, 15, 8] == 0


def test_undo_takes_back_one_whole_stroke(session):
    key = _stroke(session, [(10, 10, 8), (20, 20, 8)])
    first = _labels(session, key).copy()
    session.do(SetModeParam("value", "3"))
    _stroke(session, [(5, 25, 8), (25, 5, 8)])
    session.do(PaintUndo(key))
    np.testing.assert_array_equal(_labels(session, key), first)


def test_labels_keep_their_colour_whatever_else_is_painted(session):
    key = _stroke(session, [(5, 5, 8)])
    session.do(SetModeParam("value", "2"))
    _stroke(session, [(10, 10, 8)])
    two = session.roi_palette(key, CPU)[2].clone()
    session.do(SetModeParam("value", "7"))
    _stroke(session, [(20, 20, 8)])
    assert torch.equal(session.roi_palette(key, CPU)[2], two)
    assert session.roi_palette(key, CPU).shape[0] == 8


def test_label_names_reach_the_readout(session):
    from fastfuncstuff.viewer.vocab import SetIJK

    session.do(SetModeParam("label_name", "putamen"))
    _stroke(session, [(12, 12, 8)])
    session.do(SetIJK(12, 12, 8))
    assert any("putamen" in str(line) for line in session.overlay_readout())


def test_a_recorded_session_replays_to_the_same_voxels(session):
    key = _stroke(session, [(10, 10, 8), (20, 12, 8)])
    session.do(SetModeParam("value", "4"))
    session.do(SetModeParam("brush", "sphere"))
    _stroke(session, [(15, 25, 10)])
    _stroke(session, [(15, 25, 10)], erase=True)
    replay = ViewerSession(device=CPU)
    try:
        replay.run_script(session.to_script())
        np.testing.assert_array_equal(_labels(replay, key), _labels(session, key))
    finally:
        replay.close()


def test_save_writes_integer_labels_on_its_grid_or_resampled(session, tmp_path):
    key = _stroke(session, [(10, 10, 8), (20, 10, 8)])
    session.do(SetModeParam("save_to", str(tmp_path / "rois.nii.gz")))
    session.do(ModeAction("save"))
    img = nib.load(tmp_path / "rois.nii.gz")
    assert img.get_data_dtype() == np.int16 and img.shape == SHAPE
    np.testing.assert_array_equal(np.asarray(img.dataobj), _labels(session, key).astype(np.int16))
    # Resampled onto a coarser layer's grid: nearest, so still only 0 and 1.
    coarse = tmp_path / "coarse.nii.gz"
    aff = np.diag([4.0, 4.0, 5.0, 1.0])
    aff[:3, 3] = [-30.0, -32.0, -25.0]
    nib.save(nib.Nifti1Image(np.zeros((15, 16, 10), np.float32), aff), str(coarse))
    ck = session.load(coarse)
    out = session.painter.save(key, tmp_path / "rois_coarse.nii.gz", like=ck)
    vals = np.unique(np.asarray(nib.load(out).dataobj))
    assert nib.load(out).shape == (15, 16, 10) and set(vals) == {0, 1}


# ---------------------------------------------------------------------------
# in the window
# ---------------------------------------------------------------------------


def test_left_drag_paints_right_drag_erases_ctrl_wheel_sizes(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SetUnderlay

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    path = tmp_path / "anat.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.float32), aff), str(path))
    s = ViewerSession(device=CPU)
    win = ViewerWindow(s)
    try:
        win.refresh(s.do(SetUnderlay(str(path))))
        win.show()
        app.processEvents()
        win._switch_mode("paint")
        app.processEvents()
        vid = next(vp.id for vp in s.state.viewports.images if vp.plane is Plane.AXIAL)
        image = win.manager.windows[vid]
        image.redraw()
        assert image.pane.painting
        image._paint_press(10.0, 10.0, False)
        image._paint_drag(10.0, 14.0)
        image._paint_release()
        key = s.mode.target()
        assert key is not None and _labels(s, key).sum() > 0
        painted = _labels(s, key).sum()
        image._paint_press(10.0, 12.0, True)
        image._paint_release()
        assert _labels(s, key).sum() < painted
        r = s.mode.params["radius"]
        image._paint_resize(1)
        assert s.mode.params["radius"] > r
        # Leaving the mode gives the mouse back to the crosshair.
        win._switch_mode("plain")
        app.processEvents()
        image.redraw()
        assert not image.pane.painting
    finally:
        win.close()
        s.close()
        app.processEvents()


def test_a_stroke_redraws_once_a_pass_and_only_the_slices_it_can_reach(tmp_path, monkeypatch):
    """Every dab redrew all three windows, each on its own, whether or not
    the stroke could appear in them; the stroke's end still redraws them all."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.ui.window import ViewerWindow
    from fastfuncstuff.viewer.vocab import SetIJK, SetUnderlay

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    path = tmp_path / "anat.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, np.float32), np.diag([2.0, 2.0, 2.0, 1.0])), str(path))
    s = ViewerSession(device=CPU)
    win = ViewerWindow(s)
    try:
        win.refresh(s.do(SetUnderlay(str(path))))
        win.show()
        win.refresh(s.do(SetIJK(2, 2, 10)))  # coronal and sagittal slices far from the stroke
        win._switch_mode("paint")
        app.processEvents()
        windows = {vp.plane: win.manager.windows[vp.id] for vp in s.state.viewports.images}
        counts = dict.fromkeys(windows, 0)
        for plane, w in windows.items():

            def counted(orig=w.redraw, plane=plane):
                counts[plane] += 1
                orig()

            monkeypatch.setattr(w, "redraw", counted)
        image = windows[Plane.AXIAL]
        image._paint_press(15.0, 15.0, False)
        app.processEvents()
        counts.update(dict.fromkeys(counts, 0))
        for col in (16.0, 17.0, 18.0):
            image._paint_drag(15.0, col)
        app.processEvents()
        assert counts == {Plane.AXIAL: 1, Plane.CORONAL: 0, Plane.SAGITTAL: 0}
        image._paint_release()
        assert counts[Plane.CORONAL] == counts[Plane.SAGITTAL] == 1
    finally:
        win.close()
        s.close()
        app.processEvents()


def test_label_counts_track_painting_erasing_and_undo(session):
    key = _stroke(session, [(10, 10, 8), (20, 10, 8)])
    session.do(SetModeParam("value", "2"))
    _stroke(session, [(15, 10, 8)])
    _stroke(session, [(12, 10, 8)], erase=True)

    def truth():
        lab = _labels(session, key)
        v, n = np.unique(lab[lab > 0].astype(int), return_counts=True)
        return dict(zip(v.tolist(), n.tolist(), strict=True))

    assert session.painter.counts(key) == truth()
    session.do(PaintUndo(key))
    assert session.painter.counts(key) == truth()


# ---------------------------------------------------------------------------
# painted layers wherever ROIs are taken
# ---------------------------------------------------------------------------


def test_a_painted_mask_follows_later_strokes(session):
    key = _stroke(session, [(10, 10, 8)])
    like = session.state.layers.base
    first = session.mask_from_layer(key, like).sum()
    _stroke(session, [(20, 20, 8)])
    assert session.mask_from_layer(key, like).sum() > first


def test_strokes_and_undo_give_the_layer_a_new_generation(session):
    key = _stroke(session, [(10, 10, 8)])
    g1 = session.store.get(key).generation
    _stroke(session, [(12, 12, 8)])
    g2 = session.store.get(key).generation
    session.do(PaintUndo(key))
    assert len({g1, g2, session.store.get(key).generation}) == 3


def test_region_lists_show_the_painted_colour_and_name(session):
    session.do(SetModeParam("value", "3"))
    session.do(SetModeParam("label_name", "caudate"))
    key = _stroke(session, [(10, 10, 8)])
    rois = session.roi_set(key)
    roi = next(r for r in rois if r.index == 3)
    assert roi.name == "caudate"
    painted = (session.roi_palette(key, CPU)[3] * 255).round().to(torch.uint8).tolist()
    assert list(roi.color) == painted


def test_instapca_decomposes_inside_a_painted_mask(tmp_path):
    from fastfuncstuff.viewer.vocab import SetMode

    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(1)
    run = tmp_path / "run.nii.gz"
    img = nib.Nifti1Image(100 + rng.normal(0, 1, (20, 20, 12, 40)).astype(np.float32), aff)
    img.header["pixdim"][4] = 2.0
    nib.save(img, str(run))
    s = ViewerSession(device=CPU)
    try:
        s.load(run)
        s.do(SetMode("paint"))
        s.do(SetModeParam("radius", "4"))
        key = _stroke(s, [(10, 10, 6)])
        painted = int((_labels(s, key) > 0).sum())
        s.do(SetMode("instapca"))
        s.do(SetModeParam("live", "1"))
        s.do(SetModeParam("mask", key))
        s.do(SetModeParam("n_components", "3"))
        assert int(s.mode._result.mask.sum()) == painted
    finally:
        s.close()


def test_a_painted_layer_is_a_matrix_window_roi_source(session):
    from fastfuncstuff.viewer.viewports import ViewKind

    _stroke(session, [(8, 8, 8)])
    session.do(SetModeParam("value", "2"))
    key = _stroke(session, [(20, 20, 8)])
    vid = session.state.viewports.open(ViewKind.MATRIX, Plane.AXIAL).id
    rois = session.matrix_rois(session.state.viewports.get(vid))
    assert rois is not None and rois.indices == (1, 2)
    assert session.state.layers.get(key) in session.roi_layers()
