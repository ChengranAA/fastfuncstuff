"""RGB mode: N images of one brain as one colour (viewer.chroma, viewer.modes.rgb)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.chroma import (
    ChromaSpec,
    Look,
    echo_basis,
    fit_map,
    oklab_to_srgb,
    sibling_order,
)
from fastfuncstuff.viewer.compose import render_plane
from fastfuncstuff.viewer.modes.base import PathListControl
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.slicing import sample_volume
from fastfuncstuff.viewer.state import Plane
from fastfuncstuff.viewer.vocab import ModeAction, SetIJK, SetIndex, SetMode, SetModeParam

nib = pytest.importorskip("nibabel")
CPU = torch.device("cpu")
TES = (10.0, 26.0, 42.0)
SHAPE = (24, 24, 16)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------


def test_echo_basis_reads_decay_rate_with_faster_decay_positive():
    """The te basis's second coordinate is R2*, signed so that iron is warm."""
    W = echo_basis(TES)
    r2 = np.array([0.02, 0.04])  # per ms
    logs = np.log(1000.0) - np.outer(r2, TES)
    coords = logs @ W.T
    assert coords[1, 1] > coords[0, 1] > 0
    # A pure monoexponential has no curvature.
    np.testing.assert_allclose(coords[:, 2], 0.0, atol=1e-9)
    np.testing.assert_allclose(coords[:, 0], logs.mean(1), rtol=1e-9)


def _correlated(noise: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(20000, 1))
    return shared * np.array([[3.0, 2.5, 2.0]]) + noise * rng.normal(size=(20000, 3))


def test_dstretch_decorrelates_correlated_channels():
    """What makes pick-3 grey: correlation. Whitening must remove it."""
    x = _correlated(0.2)
    assert np.corrcoef(x.T)[0, 2] > 0.95
    cmap = fit_map(x, ChromaSpec(basis="dstretch", signal="raw", floor=0.0))
    y = cmap.coords(torch.as_tensor(x.T, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(np.cov(y), np.eye(3), atol=0.05)


def test_floor_stops_the_noise_component_being_amplified():
    x = _correlated(0.05, seed=1)
    full = fit_map(x, ChromaSpec(basis="dstretch", signal="raw", floor=0.0))
    capped = fit_map(x, ChromaSpec(basis="dstretch", signal="raw", floor=0.1))
    assert torch.linalg.matrix_norm(capped.W) < 0.5 * torch.linalg.matrix_norm(full.W)


def test_custom_weights_are_the_map():
    x = _correlated(0.5)
    spec = ChromaSpec(basis="custom", signal="raw", weights="1,-1,0; 0,1,-1; 1,1,1")
    W = fit_map(x, spec).W.numpy()
    np.testing.assert_allclose(W, [[1, -1, 0], [0, 1, -1], [1, 1, 1]])
    with pytest.raises(ValueError):
        fit_map(x, ChromaSpec(basis="custom", signal="raw", weights="1,0; 0,1; 1,1"))


def test_look_stretch_and_shift_act_per_axis_without_refit():
    x = _correlated(0.5)
    cmap = fit_map(x, ChromaSpec(basis="pca", signal="raw"))
    y = cmap.coords(torch.as_tensor(x[:100].T, dtype=torch.float32))
    plain = cmap.unit(y, Look())
    moved = cmap.unit(y, Look(shift=(0.0, 0.5, 0.0), stretch=(1.0, 1.0, -2.0)))
    torch.testing.assert_close(moved[0], plain[0])
    torch.testing.assert_close(moved[1], plain[1] - 0.5)
    torch.testing.assert_close(moved[2], -2.0 * plain[2])


def test_hue_rotation_moves_colour_not_lightness_in_oklab():
    x = torch.as_tensor(_correlated(0.5)[:500].T, dtype=torch.float32)
    cmap = fit_map(x.T, ChromaSpec(basis="pca", signal="raw"))
    a = cmap.colors(x, Look(space="oklab"))
    b = cmap.colors(x, Look(space="oklab", hue=90.0))
    assert not torch.allclose(a, b, atol=1e-3)
    # Same lightness either way: luma barely moves while hue does.
    luma = torch.tensor([0.2126, 0.7152, 0.0722])
    assert float((a @ luma - b @ luma).abs().mean()) < 0.05


def test_oklab_neutral_axis_is_grey():
    L = torch.tensor([0.2, 0.5, 0.8])
    rgb = oklab_to_srgb(L, torch.zeros(3), torch.zeros(3))
    assert torch.allclose(rgb[:, 0], rgb[:, 1], atol=1e-4)
    assert torch.allclose(rgb[:, 1], rgb[:, 2], atol=1e-4)
    assert bool((rgb[1:, 0] > rgb[:-1, 0]).all())


def test_spec_refuses_what_it_could_not_draw():
    with pytest.raises(ValueError):
        ChromaSpec(basis="nope")
    with pytest.raises(ValueError):
        ChromaSpec(pick=(0, 1, 5)).picked(3)
    with pytest.raises(ValueError):
        ChromaSpec(te="10,20").te_ms(3)
    assert ChromaSpec().picked(5) == (0, 2, 4)
    assert ChromaSpec(pick=(2, -1, 0)).picked(3) == (2, 1, 0)


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
# the mode
# ---------------------------------------------------------------------------


def _grids():
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in SHAPE[::-1]), indexing="ij")
    return (a.transpose(2, 1, 0) for a in (xx, yy, zz))


def _head(n_time: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A head-like ellipsoid of S0 with a fast-decaying blob (iron) on one side."""
    xx, yy, zz = _grids()
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


def _files(paths, ticked=None) -> str:
    ticked = ticked or [True] * len(paths)
    return PathListControl.encode([(str(p), on) for p, on in zip(paths, ticked, strict=True)])


def _rgb(session):
    return next(ly for ly in session.state.layers if ly.colour)


def _enter(session, paths, **params):
    session.do(SetMode("rgb"))
    session.do(SetModeParam("files", _files(paths)))
    for k, v in params.items():
        session.do(SetModeParam(k, str(v)))
    return _rgb(session)


def test_entering_prefills_the_inputs_siblings_as_an_editable_list(tmp_path, session):
    paths = _write_echoes(tmp_path, _head()[0])
    (tmp_path / "unrelated_t1.nii.gz").write_bytes(paths[0].read_bytes())
    session.load(paths[1])
    session.do(SetMode("rgb"))
    files = PathListControl.parse(session.mode.params["files"])
    assert [p for p, _ in files] == [str(p) for p in paths]
    assert all(on for _, on in files)
    assert _rgb(session).n_volumes == 1


def test_any_files_any_names_and_a_skipped_channel(tmp_path, session):
    """MP2RAGE-like: three oddly named files, one unticked."""
    echoes, _ = _head()
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    names = ["INV1_weird.nii.gz", "uni-den.nii.gz", "t1map.nii.gz"]
    paths = []
    for name, data in zip(names, echoes, strict=True):
        nib.save(nib.Nifti1Image(data[..., 0], aff), str(tmp_path / name))
        paths.append(tmp_path / name)
    session.do(SetMode("rgb"))
    session.do(SetModeParam("files", _files(paths, [True, False, True])))
    session.do(SetModeParam("basis", "pick"))
    layer = _rgb(session)
    assert session.mode._chans.names == ("INV1_weird", "t1map")
    assert render_plane(session, Plane.AXIAL) is not None
    assert session.colour_sources[layer.key].chans.n == 2


def test_echo_times_from_sidecars_and_fast_decay_draws_warm(tmp_path, session):
    echoes, blob = _head()
    layer = _enter(session, _write_echoes(tmp_path, echoes), basis="te", lightness="channel 1")
    source = session.colour_sources[layer.key]
    assert source.te_source == "sidecar"
    assert source.te_ms == pytest.approx(TES)
    rgb = source.rgb_volume(0).astype(float)
    xx, yy, zz = _grids()
    tissue = (np.abs(xx - 16) < 3) & (np.abs(yy - 12) < 3) & (np.abs(zz - 8) < 2)
    warmth = rgb[..., 0] - rgb[..., 2]
    assert warmth[blob].mean() > warmth[tissue].mean() + 20


def test_look_changes_do_not_refit(tmp_path, session):
    layer = _enter(session, _write_echoes(tmp_path, _head()[0]), basis="pca")
    fitted = session.colour_sources[layer.key].map
    before = session.colour_sources[layer.key].rgb_volume(0)
    session.do(SetModeParam("stretch1", "-2.0"))
    session.do(SetModeParam("hue", "60"))
    after = session.colour_sources[layer.key]
    assert after.map is fitted
    assert not np.array_equal(after.rgb_volume(0), before)
    session.do(SetModeParam("basis", "dstretch"))
    assert session.colour_sources[layer.key].map is not fitted


def test_every_basis_draws_and_a_bad_value_says_why(tmp_path, session):
    _enter(session, _write_echoes(tmp_path, _head()[0]))
    for basis in ("pick", "dstretch", "pca", "te"):
        session.do(SetModeParam("basis", basis))
        assert render_plane(session, Plane.AXIAL) is not None
    session.do(SetModeParam("basis", "custom"))
    session.do(SetModeParam("weights", "1,0; 0,1"))
    assert "weights" in session.mode.status()


def test_stats_from_crosshair_refit_as_it_moves(tmp_path, session):
    layer = _enter(session, _write_echoes(tmp_path, _head()[0]), stats="crosshair", radius=6)
    session.do(SetIJK(8, 12, 8))
    here = session.colour_sources[layer.key].map
    session.do(SetIJK(16, 12, 8))
    assert session.colour_sources[layer.key].map is not here


def test_stats_and_lightness_from_another_layer(tmp_path, session):
    paths = _write_echoes(tmp_path, _head()[0])
    xx, yy, zz = _grids()
    roi = ((np.abs(xx - 12) < 5) & (np.abs(yy - 12) < 5) & (np.abs(zz - 8) < 4)).astype(np.float32)
    roi_path = tmp_path / "roi.nii.gz"
    nib.save(nib.Nifti1Image(roi, np.diag([2.0, 2.0, 2.0, 1.0])), str(roi_path))
    rk = session.load(roi_path)
    session.do(SetMode("rgb"))
    session.do(SetModeParam("files", _files(paths)))
    choice = next(c for c in session.mode.controls() if c.name == "stats").choices
    pick = next(c for c in choice if c.startswith(rk))
    session.do(SetModeParam("stats", pick))
    session.do(SetModeParam("lightness", pick))
    layer = _rgb(session)
    source = session.colour_sources[layer.key]
    # Lightness from the ROI: bright inside it, dark out.
    assert float(source.anatomy[12, 12, 8]) > 0.9 > float(source.anatomy[3, 3, 3])


def test_readout_and_crosshair_panel(tmp_path, session):
    _enter(session, _write_echoes(tmp_path, _head()[0]), basis="te")
    session.do(SetIJK(12, 12, 8))
    line = next(str(v) for v in session.overlay_readout() if "R2*" in str(v))
    assert line.startswith("e1 ") and "level" in line
    panel = session.mode.panels()["channels"]
    assert panel[0].x_label == "TE ms"
    assert "T2*" in panel[1].label


def test_select_by_colour_takes_the_blob(tmp_path, session):
    """On a clean monoexponential the curvature axis is noise; stretch 0 drops it."""
    echoes, blob = _head()
    _enter(session, _write_echoes(tmp_path, echoes), basis="te", tolerance=0.3, stretch2=0.0)
    session.do(SetIJK(8, 12, 8))
    session.do(ModeAction("select"))
    sel = next(ly for ly in session.state.layers if ly.source.startswith("select:"))
    mask = session.volume(sel.key) > 0.5
    assert mask[blob].mean() > 0.8
    assert mask[~blob].mean() < 0.05
    assert _rgb(session).visible


def test_time_series_play_and_window_averages(tmp_path, session):
    echoes, _ = _head(n_time=12)
    # A step in the second half that grows with TE: BOLD-like.
    for i, te in enumerate(TES):
        echoes[i][..., 6:] *= 1.0 + 0.0015 * te
    paths = _write_echoes(tmp_path, echoes)
    layer = _enter(session, paths, basis="te", signal="psc", window=5)
    assert layer.time_linked and layer.n_volumes == 12
    source = session.colour_sources[layer.key]
    early, late = source.rgb_volume(1).astype(float), source.rgb_volume(10).astype(float)
    brain = source.chans.brain.numpy()
    # Signal up with TE is R2* down: the late frames lean blue.
    lean = (early[..., 0] - early[..., 2])[brain].mean() - (late[..., 0] - late[..., 2])[
        brain
    ].mean()
    assert lean > 10
    session.do(SetIndex(10))
    assert render_plane(session, Plane.CORONAL) is not None
    assert len(session.mode.series((12, 12, 8))) == 6


def test_echoes_interleaved_in_one_file_by_selector(tmp_path, session):
    echoes, _ = _head(n_time=4)
    inter = np.stack([echoes[e][..., t] for t in range(4) for e in range(3)], axis=-1)
    p = tmp_path / "interleaved.nii.gz"
    nib.save(nib.Nifti1Image(inter, np.diag([2.0, 2.0, 2.0, 1.0])), str(p))
    entries = [f"{p}[{e}..$(3)]" for e in range(3)]
    layer = _enter(session, entries, signal="log", te="10,26,42", basis="te")
    assert layer.n_volumes == 4
    assert session.colour_sources[layer.key].te_source == "typed"


def test_saved_rgb_is_rgb24_and_kept_copy_survives_a_new_look(tmp_path, session):
    layer = _enter(session, _write_echoes(tmp_path, _head()[0]), basis="pca")
    out = session.save_layer(layer.key, tmp_path / "rgb.nii.gz")
    img = nib.load(out)
    assert img.shape == SHAPE
    assert img.get_data_dtype().names == ("R", "G", "B")
    session.do(ModeAction("keep"))
    kept = next(ly for ly in session.state.layers if ly.source.startswith("kept:"))
    frozen = session.colour_sources[kept.key].rgb_volume(0)
    session.do(SetModeParam("hue", "90"))
    np.testing.assert_array_equal(session.colour_sources[kept.key].rgb_volume(0), frozen)


def test_script_replays(tmp_path, session):
    _enter(session, _write_echoes(tmp_path, _head()[0]), basis="pca", stretch2="-1.5")
    replay = ViewerSession(device=CPU)
    try:
        replay.run_script(session.to_script())
        a = session.colour_sources[_rgb(session).key].rgb_volume(0)
        b = replay.colour_sources[_rgb(replay).key].rgb_volume(0)
        np.testing.assert_array_equal(a, b)
    finally:
        replay.close()
