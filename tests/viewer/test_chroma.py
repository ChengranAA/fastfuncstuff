"""Colour layers: N channels per voxel drawn as one colour (viewer.chroma)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.chroma import (
    Channel,
    ChromaSpec,
    echo_basis,
    fit_map,
    oklab_to_srgb,
    parse_channels,
    sibling_order,
)
from fastfuncstuff.viewer.compose import render_plane
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import sample_volume
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import MakeRgb, SetIJK, SetIndex, SetRgb

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
TES = (10.0, 26.0, 42.0)
SHAPE = (24, 24, 16)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------


def test_channel_grammar_round_trips_and_rejects_nonsense():
    chans = parse_channels("L1:0+3, L1:1+3,L1:2+3")
    assert chans == (Channel("L1", 0, 3), Channel("L1", 1, 3), Channel("L1", 2, 3))
    assert parse_channels(",".join(c.text() for c in chans)) == chans
    # An offset alone is one fixed sub-brick, not a series starting there.
    assert parse_channels("L4:2,L4:5")[0] == Channel("L4", 2, 0)
    with pytest.raises(ValueError):
        parse_channels("L1")
    with pytest.raises(ValueError):
        parse_channels("L1:x,L2")


def test_channel_frames_for_interleaved_and_stacked_layouts():
    # 3 echoes x 10 frames interleaved: echo 2 reads bricks 2, 5, ..., 29.
    c = Channel("L1", 2, 3)
    assert c.n_frames(30) == 10
    assert c.brick(9, 30) == 29
    assert Channel("L1", 0, 0).n_frames(30) == 1


def test_echo_basis_reads_decay_rate_with_faster_decay_positive():
    """The te basis's second coordinate is R2*, signed so that iron is warm."""
    W = echo_basis(TES)
    r2 = np.array([0.02, 0.04])  # per ms
    logs = np.log(1000.0) - np.outer(r2, TES)
    coords = logs @ W.T
    assert coords[1, 1] > coords[0, 1] > 0
    # A pure monoexponential has no curvature.
    np.testing.assert_allclose(coords[:, 2], 0.0, atol=1e-9)
    # The level does not depend on the decay's sign convention.
    np.testing.assert_allclose(coords[:, 0], logs.mean(1), rtol=1e-9)


def test_dstretch_decorrelates_correlated_channels():
    """What makes pick-3 grey: correlation. Whitening must remove it."""
    rng = np.random.default_rng(0)
    shared = rng.normal(size=(20000, 1))
    x = shared * np.array([[3.0, 2.5, 2.0]]) + 0.2 * rng.normal(size=(20000, 3))
    assert np.corrcoef(x.T)[0, 2] > 0.95
    spec = ChromaSpec(parse_channels("A,B,C"), basis="dstretch", signal="raw", floor=0.0)
    cmap = fit_map(x, spec)
    y = cmap.coords(torch.as_tensor(x.T, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(np.cov(y), np.eye(3), atol=0.05)


def test_floor_stops_the_noise_component_being_amplified():
    rng = np.random.default_rng(1)
    shared = rng.normal(size=(20000, 1))
    x = shared * np.array([[3.0, 2.5, 2.0]]) + 0.05 * rng.normal(size=(20000, 3))
    full = fit_map(
        x, ChromaSpec(parse_channels("A,B,C"), basis="dstretch", signal="raw", floor=0.0)
    )
    capped = fit_map(
        x, ChromaSpec(parse_channels("A,B,C"), basis="dstretch", signal="raw", floor=0.1)
    )
    assert torch.linalg.matrix_norm(capped.W) < 0.5 * torch.linalg.matrix_norm(full.W)


def test_oklab_neutral_axis_is_grey():
    L = torch.tensor([0.2, 0.5, 0.8])
    rgb = oklab_to_srgb(L, torch.zeros(3), torch.zeros(3))
    assert torch.allclose(rgb[:, 0], rgb[:, 1], atol=1e-4)
    assert torch.allclose(rgb[:, 1], rgb[:, 2], atol=1e-4)
    assert bool((rgb[1:, 0] > rgb[:-1, 0]).all())


def test_spec_refuses_what_it_could_not_draw():
    chans = parse_channels("A,B,C")
    with pytest.raises(ValueError):
        ChromaSpec(chans, pick="0,1")
    with pytest.raises(ValueError):
        ChromaSpec(chans, te="10,20")
    with pytest.raises(ValueError):
        ChromaSpec(chans, basis="nope")
    spec = ChromaSpec(chans)
    assert spec.with_param("spread", "1.5").spread == 1.5
    assert spec.with_param("masked", "0").masked is False
    assert spec.with_param("window", "7").window == 7
    with pytest.raises(KeyError):
        spec.with_param("colour", "red")


def test_siblings_differ_in_one_number_only():
    names = [
        "042_ep2d_5E_R4MB3_2mm_SBRef_e2.nii.gz",
        "042_ep2d_5E_R4MB3_2mm_SBRef_e1.nii.gz",
        "043_ep2d_5E_R4MB3_2mm_e1.nii.gz",
        "042_ep2d_5E_R4MB3_2mm_SBRef_e3.nii.gz",
        "042_ep2d_5E_R4MB3_2mm_SBRef_e3_ph.nii.gz",
    ]
    assert sibling_order(names, 0) == [1, 0, 3]
    assert sibling_order(["anat.nii.gz", "func.nii.gz"], 0) == [0]


def test_sample_volume_channel_axis_matches_per_channel():
    rng = np.random.default_rng(2)
    vol = torch.as_tensor(rng.random((3, 6, 7, 5)), dtype=torch.float32)
    ijk = torch.as_tensor(rng.random((4, 9, 3)) * 5, dtype=torch.float32)
    for mode in ("nearest", "bilinear", "cubic"):
        batched = sample_volume(vol, ijk, mode=mode)
        each = torch.stack([sample_volume(v, ijk, mode=mode) for v in vol])
        torch.testing.assert_close(batched, each)


# ---------------------------------------------------------------------------
# in a session
# ---------------------------------------------------------------------------


def _head(n_time: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A head-like ellipsoid of S0 with a fast-decaying blob (iron) on one side."""
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in SHAPE[::-1]), indexing="ij")
    xx, yy, zz = (a.transpose(2, 1, 0) for a in (xx, yy, zz))
    c = np.array(SHAPE) / 2
    r = ((xx - c[0]) / 9) ** 2 + ((yy - c[1]) / 9) ** 2 + ((zz - c[2]) / 6) ** 2
    s0 = np.where(r < 1, 1000.0, 5.0)
    r2 = np.where(r < 1, 0.025, 0.0)
    blob = ((xx - 8) ** 2 + (yy - 12) ** 2 + (zz - 8) ** 2) < 9
    r2[blob] = 0.06
    rng = np.random.default_rng(seed)
    echoes = []
    for te in TES:
        e = s0 * np.exp(-r2 * te)
        e = e[..., None] * np.ones(n_time) + rng.normal(scale=2.0, size=(*SHAPE, n_time))
        echoes.append(e.astype(np.float32))
    return np.stack(echoes), blob


def _write_echoes(tmp_path, echoes, stem="run_e{}"):
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = []
    for i, (data, te) in enumerate(zip(echoes, TES, strict=True), start=1):
        img = nib.Nifti1Image(data if data.shape[-1] > 1 else data[..., 0], aff)
        if data.shape[-1] > 1:
            img.header["pixdim"][4] = 1.5
            img.header.set_xyzt_units("mm", "sec")
        p = tmp_path / f"{stem.format(i)}.nii.gz"
        nib.save(img, str(p))
        (tmp_path / f"{stem.format(i)}.json").write_text(json.dumps({"EchoTime": te / 1000.0}))
        paths.append(p)
    return paths


@pytest.fixture
def session():
    s = ViewerSession(device=CPU)
    yield s
    s.close()


def _rgb_layer(s):
    return next(ly for ly in s.state.layers if ly.chroma is not None)


def test_make_rgb_finds_echo_siblings_and_their_echo_times(tmp_path, session):
    echoes, blob = _head()
    keys = [session.load(p) for p in _write_echoes(tmp_path, echoes)]
    session.do(MakeRgb(keys[1]))
    layer = _rgb_layer(session)
    assert [c.key for c in layer.chroma.channels] == keys
    assert layer.chroma.basis == "te"
    fit = session.chroma.fit(layer.key)
    assert fit.te_source == "sidecar"
    assert fit.te_ms == pytest.approx(TES)
    # The channels are under it now; the bottom one stays for anatomy.
    assert [session.state.layers.get(k).visible for k in keys] == [True, False, False]


def test_fast_decay_draws_warm(tmp_path, session):
    echoes, blob = _head()
    keys = [session.load(p) for p in _write_echoes(tmp_path, echoes)]
    session.do(MakeRgb(keys[0]))
    layer = _rgb_layer(session)
    session.do(SetRgb(layer.key, "lightness", "anatomy"))
    rgb = session.chroma.rgb_volume(layer.key, 0).astype(float)
    # Inside the head, away from the blob: slow decay leans blue.
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in SHAPE[::-1]), indexing="ij")
    tissue = (np.abs(xx.transpose(2, 1, 0) - 16) < 3) & (np.abs(yy.transpose(2, 1, 0) - 12) < 3)
    tissue &= np.abs(zz.transpose(2, 1, 0) - 8) < 2
    warmth = rgb[..., 0] - rgb[..., 2]
    assert warmth[blob].mean() > warmth[tissue].mean() + 20


def test_every_basis_draws_and_a_bad_value_is_refused(tmp_path, session):
    keys = [session.load(p) for p in _write_echoes(tmp_path, _head()[0])]
    session.do(MakeRgb(keys[0]))
    key = _rgb_layer(session).key
    for basis in ("pick", "dstretch", "pca", "te"):
        session.do(SetRgb(key, "basis", basis))
        pane = render_plane(session, Plane.AXIAL)
        assert pane is not None
    before = session.state.layers.get(key).chroma
    with pytest.raises(ValueError):
        session.do(SetRgb(key, "pick", "0,1"))
    assert session.state.layers.get(key).chroma == before


def test_readout_gives_channels_and_coordinates(tmp_path, session):
    keys = [session.load(p) for p in _write_echoes(tmp_path, _head()[0])]
    session.do(MakeRgb(keys[0]))
    session.do(SetIJK(12, 12, 8))
    line = next(str(v) for v in session.overlay_readout() if "R2*" in str(v))
    assert keys[0] in line and "level" in line


def test_time_series_play_and_window_averages(tmp_path, session):
    echoes, _ = _head(n_time=12)
    # A step in the second half that grows with TE: BOLD-like.
    for i, te in enumerate(TES):
        echoes[i][..., 6:] *= 1.0 + 0.0015 * te
    keys = [session.load(p) for p in _write_echoes(tmp_path, echoes)]
    for k in keys:
        session.store.ensure_ram(k)
    session.do(MakeRgb(keys[0]))
    layer = _rgb_layer(session)
    assert layer.time_linked and layer.n_volumes == 12
    assert layer.chroma.signal == "psc"
    session.do(SetRgb(layer.key, "window", "5"))
    early = session.chroma.rgb_volume(layer.key, 1).astype(float)
    late = session.chroma.rgb_volume(layer.key, 10).astype(float)
    brain = session.chroma.fit(layer.key).brain.numpy()
    # Signal up with TE is R2* down: the late frames lean blue.
    lean = (early[..., 0] - early[..., 2])[brain].mean() - (late[..., 0] - late[..., 2])[
        brain
    ].mean()
    assert lean > 10
    session.do(SetIndex(10))
    assert render_plane(session, Plane.CORONAL) is not None


def test_echoes_interleaved_in_one_file(tmp_path, session):
    echoes, _ = _head(n_time=4)
    inter = np.stack([echoes[e][..., t] for t in range(4) for e in range(3)], axis=-1)
    img = nib.Nifti1Image(inter, np.diag([2.0, 2.0, 2.0, 1.0]))
    p = tmp_path / "interleaved.nii.gz"
    nib.save(img, str(p))
    key = session.load(p)
    session.do(MakeRgb(key, f"{key}:0+3,{key}:1+3,{key}:2+3"))
    layer = _rgb_layer(session)
    assert layer.n_volumes == 4
    session.do(SetRgb(layer.key, "signal", "log"))
    session.do(SetRgb(layer.key, "te", "10,26,42"))
    session.do(SetRgb(layer.key, "basis", "te"))
    assert session.chroma.fit(layer.key).te_source == "typed"
    assert render_plane(session, Plane.SAGITTAL) is not None


def test_saved_rgb_is_rgb24_and_script_replays(tmp_path, session):
    keys = [session.load(p) for p in _write_echoes(tmp_path, _head()[0])]
    session.do(MakeRgb(keys[0]))
    key = _rgb_layer(session).key
    session.do(SetRgb(key, "basis", "pca"))
    out = session.save_layer(key, tmp_path / "rgb.nii.gz")
    img = nib.load(out)
    assert img.shape == SHAPE
    assert img.get_data_dtype().names == ("R", "G", "B")

    replay = ViewerSession(device=CPU)
    try:
        replay.run_script(session.to_script())
        assert _rgb_layer(replay).chroma == _rgb_layer(session).chroma
    finally:
        replay.close()
