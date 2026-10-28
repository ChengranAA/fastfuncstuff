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
    session.do(SetModeParam("live", "1"))
    # The contrast view: what these tests set is pinned in the echoes view.
    session.do(SetModeParam("view", str(params.pop("view", "contrast"))))
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
    # A starting point, not a read: nothing happens until GO.
    assert not any(ly.colour for ly in session.state.layers)
    assert "GO" in session.mode.status()
    session.do(ModeAction("go"))
    assert _rgb(session).n_volumes == 1


def test_building_the_list_reads_nothing_until_go(tmp_path, session):
    """Adding one file, then another, must not start a read in between."""
    paths = _write_echoes(tmp_path, _head()[0])
    session.do(SetMode("rgb"))
    session.do(SetModeParam("files", _files(paths[:2])))
    session.do(SetModeParam("files", _files(paths)))
    session.do(SetModeParam("basis", "te"))
    assert session.mode._chans is None
    assert not any(ly.colour for ly in session.state.layers)
    session.do(ModeAction("go"))
    assert session.colour_sources[_rgb(session).key].spec.basis == "te"
    # The look is never held: it is a repaint.
    before = session.colour_sources[_rgb(session).key]
    session.do(SetModeParam("hue", "45"))
    assert session.colour_sources[_rgb(session).key] is not before


def test_entering_with_nothing_to_read_does_not_fail(session):
    session.do(SetMode("rgb"))
    assert "add files" in session.mode.status() or "GO" in session.mode.status()


def test_a_one_volume_channel_on_another_grid_is_resampled(tmp_path, session):
    """A T1 beside EPI-space maps: different grid, still a channel."""
    echoes, _ = _head()
    paths = _write_echoes(tmp_path, echoes)[:2]
    fine = np.kron(echoes[2][..., 0], np.ones((2, 2, 2), np.float32))
    t1 = tmp_path / "t1_fine.nii.gz"
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    aff[:3, 3] = -0.5
    nib.save(nib.Nifti1Image(fine, aff), str(t1))
    layer = _enter(session, [*paths, t1], basis="pick", signal="raw")
    chans = session.colour_sources[layer.key].chans
    assert 2 in chans.fixed and chans.fixed[2].shape == SHAPE
    np.testing.assert_allclose(chans.fixed[2][12, 12, 8].item(), echoes[2][12, 12, 8, 0], rtol=0.05)


def test_a_time_series_on_another_grid_says_which(tmp_path, session):
    echoes, _ = _head(n_time=3)
    paths = _write_echoes(tmp_path, echoes)
    odd = tmp_path / "odd_run.nii.gz"
    nib.save(nib.Nifti1Image(echoes[0][:20], np.diag([2.0, 2.0, 2.0, 1.0])), str(odd))
    session.do(SetMode("rgb"))
    session.do(SetModeParam("files", _files([paths[0], odd])))
    with pytest.raises(ValueError, match="odd_run"):
        session.do(ModeAction("go"))
    assert "\u26a0" in session.mode.status()


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
    session.do(SetModeParam("live", "1"))
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
    session.do(SetModeParam("live", "1"))
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
    # Channels in every graph cell; the colour axes over time in a panel of their own.
    assert len(session.mode.series((12, 12, 8))) == 3
    assert len(session.mode.panels()["colour axes"]) == 3


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


def test_auto_mask_skips_a_channel_whose_background_is_noise(tmp_path, session):
    """MP2RAGE: UNI's background is salt-and-pepper, INV1/INV2's is dark."""
    echoes, _ = _head()
    rng = np.random.default_rng(5)
    xx, yy, zz = _grids()
    c = np.array(SHAPE) / 2
    head = (((xx - c[0]) / 9) ** 2 + ((yy - c[1]) / 9) ** 2 + ((zz - c[2]) / 6) ** 2) < 1
    uni = np.where(head, 2000.0, rng.uniform(0, 4000, SHAPE)).astype(np.float32)
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = []
    for name, data in (("uni", uni), ("inv1", echoes[0][..., 0]), ("inv2", echoes[1][..., 0])):
        nib.save(nib.Nifti1Image(data, aff), str(tmp_path / f"{name}.nii.gz"))
        paths.append(tmp_path / f"{name}.nii.gz")
    layer = _enter(session, paths, basis="pick", signal="raw")
    source = session.colour_sources[layer.key]
    assert float(source.chans.automask_of(0).float().mean()) > 0.6
    assert abs(float(source.brain.float().mean()) - float(head.mean())) < 0.1
    # Asked for by name, the noisy one is what you get.
    session.do(SetModeParam("mask_from", "ch 1 uni"))
    assert float(session.colour_sources[layer.key].brain.float().mean()) > 0.6


def test_quantiles_have_no_size_ceiling():
    """torch.quantile refuses > 2**24 elements; a whole-head MP2RAGE is more."""
    from fastfuncstuff.utils import quantiles

    x = torch.arange(17_000_000, dtype=torch.float32)
    lo, mid = quantiles(x, (0.0, 0.5))
    assert lo == 0.0 and abs(mid - 8_500_000) < 2


# ---------------------------------------------------------------------------
# regions A and B, and the contrast optimiser
# ---------------------------------------------------------------------------


def _two_tissues(tmp_path):
    """Two blobs that differ only in channel 1 minus channel 2 -- a direction
    pick-3 shows as a barely different grey -- inside a head."""
    xx, yy, zz = _grids()
    c = np.array(SHAPE) / 2
    head = (((xx - c[0]) / 10) ** 2 + ((yy - c[1]) / 10) ** 2 + ((zz - c[2]) / 7) ** 2) < 1
    a = ((xx - 7) ** 2 + (yy - 12) ** 2 + (zz - 8) ** 2) < 9
    b = ((xx - 17) ** 2 + (yy - 12) ** 2 + (zz - 8) ** 2) < 9
    rng = np.random.default_rng(3)
    tissue = rng.normal(0, 1, SHAPE)
    vols = []
    for ch, sign in enumerate((1.0, -1.0, 0.0)):
        v = 1000 + 150 * tissue * (1 - 0.1 * ch) + rng.normal(0, 8, SHAPE)
        v = v + 40 * sign * (a.astype(float) - b.astype(float))
        vols.append(np.where(head, v, 5.0).astype(np.float32))
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    paths = []
    for ch, v in enumerate(vols):
        p = tmp_path / f"t{ch}.nii.gz"
        nib.save(nib.Nifti1Image(v, aff), str(p))
        paths.append(p)
    return paths, a, b


def _lab_gap(source, a, b) -> float:
    from fastfuncstuff.viewer.chroma import srgb_to_oklab

    rgb = torch.as_tensor(source.rgb_volume(0).astype(np.float32) / 255.0)
    lab = srgb_to_oklab(rgb)
    return float((lab[torch.as_tensor(a)].mean(0) - lab[torch.as_tensor(b)].mean(0)).norm())


def test_move_b_places_b_and_gives_the_crosshair_back_to_a(tmp_path, session):
    paths, _, _ = _two_tissues(tmp_path)
    _enter(session, paths, basis="pick", signal="raw")
    session.do(SetIJK(7, 12, 8))
    a_mm = session.state.crosshair_mm
    session.do(SetModeParam("move_b", "1"))
    session.do(SetIJK(17, 12, 8))
    b_mm = session.state.crosshair_mm
    session.do(SetModeParam("move_b", "0"))
    assert session.state.crosshair == (7, 12, 8)
    marks = {m.label: m for m in session.mode.marks()}
    assert np.allclose(marks["A"].mm, a_mm) and np.allclose(marks["B"].mm, b_mm, atol=0.06)


def test_optimise_pulls_a_and_b_apart_and_lands_as_a_custom_basis(tmp_path, session):
    paths, a, b = _two_tissues(tmp_path)
    layer = _enter(session, paths, basis="pick", signal="raw", radius_a=4, radius_b=4)
    session.do(SetIJK(7, 12, 8))
    session.do(SetModeParam("move_b", "1"))
    session.do(SetIJK(17, 12, 8))
    session.do(SetModeParam("move_b", "0"))
    before = _lab_gap(session.colour_sources[layer.key], a, b)
    session.do(ModeAction("optimise"))
    params = session.mode.params
    assert params["basis"] == "custom" and len(params["weights"].split(";")) == 3
    after = _lab_gap(session.colour_sources[layer.key], a, b)
    assert after > 2 * before, (before, after)
    assert "ΔE" in session.mode.status()


def test_optimise_says_what_is_missing(tmp_path, session):
    paths, _, _ = _two_tissues(tmp_path)
    _enter(session, paths, basis="pick", signal="raw")
    session.do(ModeAction("optimise"))
    assert "B" in session.mode.status()


def test_image_windows_draw_both_regions_where_the_slice_cuts_them(tmp_path):
    """A at the crosshair, B elsewhere on the same slice; a sphere of radius r
    whose centre is on the slice is a circle of radius r on it."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    paths, _, _ = _two_tissues(tmp_path)
    s = ViewerSession(device=CPU)
    win = ViewerWindow(s)
    try:
        from fastfuncstuff.viewer.vocab import SetUnderlay

        win.refresh(s.do(SetUnderlay(str(paths[0]))))
        win.show()
        app.processEvents()
        # With a window, preparation is the window worker's job; run it here.
        s.do(SetMode("rgb"))
        for k, v in (("live", "1"), ("view", "contrast"), ("files", _files(paths))):
            s.do(SetModeParam(k, v))
        for k, v in (("basis", "pick"),):
            s.do(SetModeParam(k, v))
        for k, v in (("signal", "raw"), ("radius_a", "6"), ("radius_b", "4")):
            s.do(SetModeParam(k, v))
        s.mode.prepare()
        s.refresh_mode()
        s.do(SetIJK(7, 12, 8))
        s.do(SetModeParam("move_b", "1"))
        s.do(SetIJK(17, 12, 8))
        s.do(SetModeParam("move_b", "0"))
        app.processEvents()
        from fastfuncstuff.viewer.state import Plane

        vid = next(vp.id for vp in s.state.viewports.images if vp.plane is Plane.AXIAL)
        axial = win.manager.windows[vid]
        axial.redraw()
        regions = {r[5]: r for r in axial._mode_regions()}
        assert set(regions) == {"A", "B"}
        # 2 mm voxels: a 6 mm sphere on its own slice is a 3-voxel circle.
        assert regions["A"][2] == pytest.approx(3.0) and regions["B"][2] == pytest.approx(2.0)
    finally:
        win.close()
        s.close()
        app.processEvents()


def test_optimise_separates_a_target_label_from_each_other_label(tmp_path, session):
    """A from an ROI layer: label 1 against labels 2 and 3, each its own pair."""
    paths, a, b = _two_tissues(tmp_path)
    xx, yy, zz = _grids()
    c3 = ((xx - 12) ** 2 + (yy - 5) ** 2 + (zz - 8) ** 2) < 6
    labels = (a * 1 + b * 2 + c3 * 3).astype(np.float32)
    roi_path = tmp_path / "regions.nii.gz"
    nib.save(nib.Nifti1Image(labels, np.diag([2.0, 2.0, 2.0, 1.0])), str(roi_path))
    rk = session.load(roi_path)
    assert session.state.layers.get(rk).roi
    layer = _enter(session, paths, basis="pick", signal="raw")
    choice = next(
        c
        for c in next(x for x in session.mode.controls() if x.name == "regions").choices
        if c.startswith(rk)
    )
    session.do(SetModeParam("regions", choice))
    session.do(SetModeParam("a_label", "1"))
    before = _lab_gap(session.colour_sources[layer.key], a, b)
    session.do(ModeAction("optimise"))
    status = session.mode.status()
    assert "closest" in status and "[" in status, status  # one distance per B
    assert _lab_gap(session.colour_sources[layer.key], a, b) > 2 * before
    assert session.mode.marks() == []  # the ROI layer shows the regions


def test_painted_regions_drive_the_optimiser(tmp_path, session):
    """The workflow: paint A and B in PAINT mode, then separate them in RGB."""
    from fastfuncstuff.viewer.state import Plane

    paths, a, b = _two_tissues(tmp_path)
    session.load(paths[0])
    session.do(SetMode("paint"))
    paint = session.mode
    for value, ijk in ((1, (7, 12, 8)), (2, (17, 12, 8))):
        session.do(SetModeParam("value", str(value)))
        session.do(SetModeParam("radius", "4"))
        paint.begin(np.asarray(session.state.grid.ijk_to_mm(ijk)), Plane.AXIAL, erase=False)
        paint.end()
    painted = paint.target()
    layer = _enter(session, paths, basis="pick", signal="raw")
    choice = next(
        c
        for c in next(x for x in session.mode.controls() if x.name == "regions").choices
        if c.startswith(painted)
    )
    session.do(SetModeParam("regions", choice))
    before = _lab_gap(session.colour_sources[layer.key], a, b)
    session.do(ModeAction("optimise"))
    assert session.mode.params["basis"] == "custom"
    assert _lab_gap(session.colour_sources[layer.key], a, b) > 2 * before


@pytest.mark.gpu
def test_the_graphed_optimiser_step_is_the_eager_one():
    """The CUDA graph replays the same arithmetic: agree to float noise over a
    few steps. (Over a full run the millionths compound in a non-convex
    landscape, so whole results are not compared.)"""
    from fastfuncstuff.viewer import contrast

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    gen = torch.Generator().manual_seed(0)
    ctx = torch.randn(3000, 3, generator=gen).cuda()
    za = (torch.randn(800, 3, generator=gen) * 0.5 + 0.3).cuda()
    zb = (torch.randn(800, 3, generator=gen) * 0.5 - 0.3).cuda()
    runs = []
    try:
        for graphs in (True, False):
            contrast.USE_CUDA_GRAPHS = graphs
            runs.append(
                contrast.optimise_contrast(
                    za, zb, ctx, torch.eye(3, device="cuda"), Look(space="oklab"), steps=5
                )
            )
    finally:
        contrast.USE_CUDA_GRAPHS = True
    np.testing.assert_allclose(runs[0].weights, runs[1].weights, atol=1e-4)
    np.testing.assert_allclose(runs[0].stretch, runs[1].stretch, atol=1e-4)


# ---------------------------------------------------------------------------
# the echoes view and adding open files
# ---------------------------------------------------------------------------


def test_echoes_view_pins_s0_and_r2_and_hides_the_contrast_tools(tmp_path, session):
    echoes, _ = _head(n_time=6)
    paths = _write_echoes(tmp_path, echoes)
    layer = _enter(session, paths, view="echoes", basis="dstretch", signal="raw")
    source = session.colour_sources[layer.key]
    # Whatever the contrast view was left at: te on percent change, for a run.
    assert (source.spec.basis, source.spec.signal) == ("te", "psc")
    assert source.look.space == "oklab" and source.coordinate_names()[:2] == ("S0", "R2*")
    shown = {c.name for c in session.mode.panel_controls()}
    assert {"files", "te", "window", "stretch1"} <= shown
    assert not shown & {"basis", "signal", "stats", "radius_a", "tolerance"}
    te = next(c for c in session.mode.panel_controls() if c.name == "te")
    assert te.visible_when is None, "TE is the point of this view, not a te-basis extra"
    assert "optimise" not in {a.name for a in session.mode.panel_actions()}
    assert session.mode.marks() == [], "no A vs B to compare in this view"
    # Hidden is not gone: the contrast view comes back with what was set there.
    session.do(SetModeParam("view", "contrast"))
    source = session.colour_sources[_rgb(session).key]
    assert (source.spec.basis, source.spec.signal) == ("dstretch", "raw")
    assert "basis" in {c.name for c in session.mode.panel_controls()}


def test_echoes_view_on_single_volumes_fits_log_signal(tmp_path, session):
    layer = _enter(session, _write_echoes(tmp_path, _head()[0]), view="echoes")
    source = session.colour_sources[layer.key]
    assert (source.spec.basis, source.spec.signal) == ("te", "log")


def test_open_files_are_offered_to_the_channel_list(tmp_path, session):
    paths = _write_echoes(tmp_path, _head()[0])
    for p in paths[:2]:
        session.load(p)
    session.do(SetMode("rgb"))
    files = next(c for c in session.mode.controls() if c.name == "files")
    assert [path for _, path in files.offer] == [str(p) for p in paths[:2]]


def test_add_menu_offers_open_files_then_the_picker():
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from fastfuncstuff.viewer.ui.controls import add_menu

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    added, browsed = [], []
    offer = (("e1", "/d/e1.nii"), ("e2", "/d/e2.nii"), ("e3", "/d/e3.nii"))
    menu = add_menu(None, offer, {"/d/e1.nii"}, added.append, lambda: browsed.append(1))
    acts = {a.text(): a for a in menu.actions() if a.text()}
    assert not acts["e1"].isEnabled(), "already listed: shown, greyed"
    acts["e2"].trigger()
    acts["all open files"].trigger()
    assert added == [["/d/e2.nii"], ["/d/e2.nii", "/d/e3.nii"]]
    assert [a.text() for a in menu.actions()][-1].startswith("from disk")
    menu.actions()[-1].trigger()
    assert browsed == [1]
    del app
