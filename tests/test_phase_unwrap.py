"""processing/phase_unwrap: reference-anchored unwrap of a whole series."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("warpkit")

from fastfuncstuff.processing.phase_unwrap import (  # noqa: E402
    raw_phase_to_radians,
    svd_filter_fields,
    unwrap_timeseries,
)

CPU = torch.device("cpu")
SHAPE = (40, 40, 28)
NT = 24


def _head():
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij")
    r = np.sqrt((x / 0.8) ** 2 + (y / 0.9) ** 2 + (z / 0.8) ** 2)
    return x, y, z, r


def _scene(tes, *, ramp_cycles=0.0, seed=0):
    """Wrapped phase of a field spanning several wraps, plus a blob whose field ramps
    by ``ramp_cycles`` shortest-echo periods over the run (a time-varying voxel)."""
    rng = np.random.default_rng(seed)
    x, y, z, r = _head()
    tes = np.asarray(tes, np.float32)
    head = (r < 1).astype(np.float32)
    mag0 = head * (600 + 200 * np.cos(3 * x)) + 2.0
    static = (
        120 * np.exp(-((x - 0.1) ** 2 + (y - 0.5) ** 2 + (z + 0.3) ** 2) / 0.15) + 25 * x - 15 * z
    )
    offset = 0.8 * np.cos(1.5 * y + 0.5) * head
    blob = np.exp(-((x + 0.35) ** 2 + (y + 0.2) ** 2 + z**2) / 0.02)
    ramp = np.linspace(-0.5, 0.5, NT) * ramp_cycles * 1000.0 / tes[0]
    drift = 3.0 * np.sin(np.linspace(0, 3, NT))
    field = static[..., None] + drift + blob[..., None] * ramp  # (x, y, z, t) Hz
    t2s = 30.0
    mag = np.stack([mag0[..., None] * np.exp(-te / t2s) * np.ones(NT) for te in tes], 3)
    phase = 2 * np.pi * field[..., None, :] * tes[:, None] * 1e-3 + offset[..., None, None]
    phase = phase + rng.normal(0, 0.03, phase.shape) * head[..., None, None]
    wrapped = (np.angle(np.exp(1j * phase))).astype(np.float32)
    return wrapped, mag.astype(np.float32), field.astype(np.float32), head > 0, blob > 0.5


def test_multiecho_recovers_field_and_time_varying_blob():
    tes = [5.0, 12.0, 19.0]
    ph, mag, field, head, blob = _scene(tes, ramp_cycles=1.6)
    res = unwrap_timeseries(ph, mag, tes, device=CPU, verbose=False)
    core = head & (res.mask == 2)
    err = np.abs(res.field_hz - field)[core]
    assert np.mean(err < 5.0) > 0.99
    # the blob's change leaves +-1000/(2 TE1) Hz of the reference at the ends of its
    # ramp (|ramp| > 0.5 of 0.8 periods: ~37% of frames); only the temporal path
    # keeps those frames right
    bad = np.abs(res.field_hz - field)[blob & core] > 50.0
    assert bad.mean() < 0.01
    off = unwrap_timeseries(ph, mag, tes, device=CPU, temporal_path=False, verbose=False)
    wrong_off = (np.abs(off.field_hz - field) > 50.0).any(-1) & core
    assert wrong_off[blob].mean() > 0.5
    # exactly those voxels are followed in time, and almost nothing else
    assert res.temporal_path[wrong_off].mean() > 0.95
    assert res.temporal_path[core & ~blob].mean() < 0.01


def test_single_echo_unwrapped_up_to_a_global_offset():
    tes = [25.0]
    ph, mag, field, head, blob = _scene(tes, ramp_cycles=1.6, seed=1)
    res = unwrap_timeseries(ph, mag, tes, device=CPU, return_unwrapped=True, verbose=False)
    core = head & (res.mask == 2)
    _x, y, _z, _r = _head()
    truth = 2 * np.pi * field * tes[0] * 1e-3 + (0.8 * np.cos(1.5 * y + 0.5))[..., None]
    d = (res.unwrapped[..., 0, :] - truth)[core]
    d = d - 2 * np.pi * np.round(np.median(d) / (2 * np.pi))
    assert np.mean(np.abs(d) < 0.5) > 0.99


def test_rough_voxels_are_never_followed_in_time():
    # Pure noise outside the head: a temporal path there would random-walk.
    tes = [5.0, 12.0, 19.0]
    ph, mag, _field, head, _blob = _scene(tes)
    rng = np.random.default_rng(3)
    ph = np.where(head[..., None, None], ph, rng.uniform(-np.pi, np.pi, ph.shape)).astype(
        np.float32
    )
    res = unwrap_timeseries(ph, mag, tes, device=CPU, verbose=False)
    assert not res.temporal_path[(res.mask > 0) & ~head].any()


def test_raw_phase_rescale_matches_warpkit_convention():
    from warpkit.utilities import rescale_phase

    rng = np.random.default_rng(0)
    raw = rng.integers(-4096, 4095, (6, 5, 4, 3, 2)).astype(np.float32)
    raw[0, 0, 0, :, 0], raw[1, 0, 0, :, 0] = -4096, 4094  # every echo spans the range
    ours = raw_phase_to_radians(raw)
    assert np.allclose(ours, rescale_phase(raw, min=-4096, max=4094), atol=1e-5)


def test_svd_filter_matches_warpkit():
    from warpkit.unwrap import svd_filtering

    rng = np.random.default_rng(0)
    _x, _y, _z, r = _head()
    mask = (r < 0.8).astype(np.int8) + (r < 1.0).astype(np.int8)
    t = np.linspace(0, 1, 30)
    field = (
        rng.normal(0, 1, (*SHAPE, 4)) @ np.stack([np.sin(3 * t), t, t**2, np.cos(5 * t)])
    ).astype(np.float32) + rng.normal(0, 0.3, (*SHAPE, 30)).astype(np.float32)
    field *= (mask > 0)[..., None]
    ref = field.copy()
    svd_filtering(ref, np.repeat(mask[..., None], 30, -1), 2.0, 30, (1, 5), 10)
    ours = svd_filter_fields(field, mask, 2.0, device=CPU)
    assert np.allclose(ours, ref, atol=1e-3 * np.abs(ref).max())


def _write_series(path, arr_xyzt):
    from fastfuncstuff.processing.io import save_image

    save_image(
        torch.from_numpy(np.ascontiguousarray(arr_xyzt.transpose(3, 2, 1, 0))),
        str(path),
        header_info=None,
    )


def test_cli_multiecho_writes_echoes_and_field(tmp_path):
    from fastfuncstuff.cli.phase_unwrap import main
    from fastfuncstuff.processing.io import load_image

    tes = [5.0, 12.0, 19.0]
    ph, mag, field, head, _blob = _scene(tes, ramp_cycles=1.6)
    args = ["-phase"]
    for e in range(3):
        # raw scanner units: the CLI must map them back onto [-pi, pi)
        _write_series(tmp_path / f"p{e}.nii.gz", ph[..., e, :] / np.pi * 4096)
        _write_series(tmp_path / f"m{e}.nii.gz", mag[..., e, :])
    args += [str(tmp_path / f"p{e}.nii.gz") for e in range(3)]
    args += ["-magnitude", *[str(tmp_path / f"m{e}.nii.gz") for e in range(3)]]
    args += ["-echo_times", *map(str, tes), "-prefix", str(tmp_path / "uw.nii.gz")]
    args += ["-save_mask", str(tmp_path / "mask.nii.gz"), "-device", "cpu", "-verb", "0"]
    assert main(args) == 0
    f, _ = load_image(str(tmp_path / "uw_fieldmap.nii.gz"))
    m, _ = load_image(str(tmp_path / "mask.nii.gz"))
    got = f.numpy().transpose(3, 2, 1, 0)
    core = head & (m.numpy().transpose(2, 1, 0) == 2)
    assert np.mean(np.abs(got - field)[core] < 5.0) > 0.99
    for e in range(3):
        assert (tmp_path / f"uw_e{e + 1}.nii.gz").exists()


def test_cli_refuses_multiecho_without_echo_times(tmp_path):
    from fastfuncstuff.cli.phase_unwrap import main

    assert main(["-phase", "a", "b", "-magnitude", "c", "d", "-prefix", str(tmp_path / "o")]) == 1


def test_breath_hold_excursion_does_not_wrap_noisy_voxels():
    # A whole-head field step of 1.5 half-periods (a breath-hold) in the second half of
    # the run, and a region too noisy to follow along time: those voxels can only stay
    # right by wrapping their residual about the frame-wide field fit.
    tes = [25.0]
    ph, mag, field, head, _blob = _scene(tes, seed=4)
    x, y, z, r = _head()
    step = np.clip(np.linspace(-6, 6, NT), 0, 1) * 1.5 * 1000.0 / (2 * tes[0])
    excursion = (1 + 0.3 * z)[..., None] * step  # smooth in space, like breathing
    noisy = head & (x > 0.3)
    rng = np.random.default_rng(5)
    phase = (
        ph
        + (2 * np.pi * excursion * tes[0] * 1e-3)[..., None, :]
        + np.where(noisy[..., None], rng.normal(0, 1.0, ph.shape[:3] + (NT,)), 0)[..., None, :]
    )
    phase = np.angle(np.exp(1j * phase)).astype(np.float32)
    truth = field + excursion

    def wrong(res):
        d = (res.field_hz - truth) * tes[0] * 1e-3  # cycles
        d = d - np.round(np.median(d, -1, keepdims=True))
        return (np.abs(d) > 0.5)[noisy & (res.mask == 2)].mean()

    on = unwrap_timeseries(phase, mag, tes, device=CPU, verbose=False)
    off = unwrap_timeseries(phase, mag, tes, device=CPU, frame_field_order=None, verbose=False)
    assert wrong(off) > 0.2
    assert wrong(on) < 0.03


def test_cli_partition_shift_needs_axis_and_echoes(tmp_path):
    from fastfuncstuff.cli.phase_unwrap import main

    out = str(tmp_path / "s.1D")
    one = ["-phase", "a", "-magnitude", "b", "-prefix", str(tmp_path / "o")]
    assert main([*one, "-partition_axis", "IS", "-save_partition_shift", out]) == 1
    two = ["-phase", "a", "c", "-magnitude", "b", "d", "-echo_times", "5", "10"]
    assert main([*two, "-prefix", str(tmp_path / "o"), "-save_partition_shift", out]) == 1


def test_remove_offset_drift_restores_the_field():
    from fastfuncstuff.processing.phase_unwrap import field_from_unwrapped, remove_offset_drift

    tes = [5.0, 12.0, 19.0]
    ph, mag, field, head, _blob = _scene(tes)
    drift = 0.4 * np.sin(np.linspace(0, 4, NT))  # rad, the same at every echo
    x, _y, _z, _r = _head()
    ph = np.angle(
        np.exp(1j * (ph + (drift[None, None, None, :] * (1 + 0.3 * x[..., None]))[..., None, :]))
    )
    res = unwrap_timeseries(
        ph.astype(np.float32), mag, tes, device=CPU, return_unwrapped=True, verbose=False
    )
    core = head & (res.mask == 2)
    before = np.abs(res.field_hz - field)[core]
    coef = remove_offset_drift(res.unwrapped, mag, tes, res.mask == 2, device=CPU)
    after = np.abs(field_from_unwrapped(res.unwrapped, mag, tes, device=CPU) - field)[core]
    assert np.corrcoef(coef[0], drift - drift.mean())[0, 1] > 0.99
    assert np.median(after) < 0.3 * np.median(before)
