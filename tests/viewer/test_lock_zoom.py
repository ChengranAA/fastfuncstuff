"""LOCK: image windows that share zoom and one centre point.

The failures worth pinning: a pan copied as numbers between planes (it is
counted along each window's own rows and columns, so it moves the other window
along the wrong axis), a flipped axis that hands the point back mirrored, a
plain Lock window dragged along, and a script that replays to different views.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer import vocab
from fastfuncstuff.viewer.commands import CommandBus, parse_script
from fastfuncstuff.viewer.compose import plane_view
from fastfuncstuff.viewer.layers import Layer
from fastfuncstuff.viewer.slicing import PlaneView, plane_layout
from fastfuncstuff.viewer.state import Plane, ViewerState
from fastfuncstuff.viewer.vocab import OpenView, SetPan, SetViewLocked, SetZoom


def _affine(flip_x: bool) -> np.ndarray:
    a = np.diag([-2.0 if flip_x else 2.0, 2.5, 3.0, 1.0])
    a[:3, 3] = [30.0 if flip_x else -30.0, -40.0, -20.0]
    return a


def _bus(flip_x: bool = False) -> CommandBus:
    st = ViewerState()
    bus = vocab.install(CommandBus(st))
    layer = Layer(
        key="A",
        name="A",
        path="/tmp/A.nii.gz",
        shape=(40, 48, 32),
        n_volumes=1,
        affine=_affine(flip_x),
    )
    st.layers.add(layer)
    st.adopt_grid(layer.shape, layer.affine)
    return bus


def _centre(st, vid):
    """The window's centre as a full grid point, the fixed axis left at nan."""
    vp = st.viewports.get(vid)
    view = plane_view(st, vp)
    point = [np.nan] * 3
    point[view.layout.row], point[view.layout.col] = view.centre()
    return np.array(point)


@pytest.mark.parametrize("plane", list(Plane))
@pytest.mark.parametrize("flip_x", [False, True])
def test_centre_and_pan_for_are_inverses(plane, flip_x):
    layout = plane_layout(_affine(flip_x), plane)
    view = PlaneView(layout=layout, shape=(40, 48, 32), zoom=3.0)
    point = (11.3, 30.7, 5.2)
    view = PlaneView(layout=layout, shape=view.shape, zoom=3.0, pan=view.pan_for(point))
    got = view.centre()
    assert got == pytest.approx((point[layout.row], point[layout.col]))


def test_pan_for_a_voxel_centre_is_pan_centring():
    layout = plane_layout(_affine(True), Plane.AXIAL)
    view = PlaneView(layout=layout, shape=(40, 48, 32), zoom=2.0)
    assert view.pan_for((7, 9, 4)) == pytest.approx(view.pan_centring((7, 9, 4)))


@pytest.mark.parametrize("flip_x", [False, True])
def test_lock_windows_share_zoom_and_the_point_across_planes(flip_x):
    bus = _bus(flip_x)
    st = bus.state
    for vid, plane in (("V1", "axial"), ("V2", "axial"), ("V3", "sagittal"), ("V4", "coronal")):
        bus.dispatch(OpenView(vid, "image", plane))
    for vid in ("V1", "V2", "V3"):
        bus.dispatch(SetViewLocked(vid, True, zoom=True))

    bus.dispatch(SetZoom("V1", 4.0))
    bus.dispatch(SetPan("V1", 3.0, -5.0))

    assert [st.viewports.get(v).zoom for v in ("V1", "V2", "V3")] == [4.0, 4.0, 4.0]
    assert st.viewports.get("V2").pan == pytest.approx(st.viewports.get("V1").pan)
    axial, sagittal = _centre(st, "V1"), _centre(st, "V3")
    # Axial and sagittal share the front-back axis -- rows in one, columns in
    # the other, which is why copying the pan numbers would be wrong.
    shared = ~np.isnan(axial) & ~np.isnan(sagittal)
    assert shared.any()
    assert sagittal[shared] == pytest.approx(axial[shared])
    # Plain Lock (V4) follows the crosshair but keeps its own zoom.
    assert st.viewports.get("V4").zoom == 1.0


def test_joining_takes_the_groups_view_and_unlocking_leaves_it():
    bus = _bus()
    st = bus.state
    bus.dispatch(OpenView("V1", "image", "axial"))
    bus.dispatch(OpenView("V2", "image", "axial"))
    bus.dispatch(SetViewLocked("V1", True, zoom=True))
    bus.dispatch(SetZoom("V1", 3.0))
    bus.dispatch(SetPan("V1", 2.0, 4.0))

    bus.dispatch(SetViewLocked("V2", True, zoom=True))
    assert st.viewports.get("V2").zoom == 3.0
    assert st.viewports.get("V2").pan == pytest.approx((2.0, 4.0))

    bus.dispatch(SetViewLocked("V2", False))
    assert not st.viewports.get("V2").zoom_linked
    bus.dispatch(SetZoom("V1", 6.0))
    assert st.viewports.get("V2").zoom == 3.0


def test_old_scripts_still_mean_lock_and_a_recording_replays():
    assert parse_script("SET_VIEW_LOCKED V1 1\n")[0] == SetViewLocked("V1", True)

    bus = _bus()
    for vid, plane in (("V1", "axial"), ("V2", "coronal")):
        bus.dispatch(OpenView(vid, "image", plane))
        bus.dispatch(SetViewLocked(vid, True, zoom=True))
    bus.dispatch(SetZoom("V2", 2.5))
    bus.dispatch(SetPan("V2", -3.0, 6.0))

    replay = _bus()
    replay.run_script(bus.to_script())
    for vid in ("V1", "V2"):
        a, b = bus.state.viewports.get(vid), replay.state.viewports.get(vid)
        assert (a.zoom, a.pan, a.zoom_linked) == (b.zoom, pytest.approx(b.pan), b.zoom_linked)
