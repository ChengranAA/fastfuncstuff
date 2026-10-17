"""Joint magnitude + phase partition-flow estimator (locomoco -phase, experimental)."""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from fastfuncstuff.processing.locomoco import _shift3d_axis
from fastfuncstuff.processing.locomoco_magphase import (
    estimate_partition_flow_magphase,
    nuisance_columns,
)
from fastfuncstuff.processing.phase_input import (
    phase_input_problem,
    phase_to_hz,
    phase_wrap_fraction,
    wrap_period,
)

CPU = torch.device("cpu")
TE = [10.0, 22.0, 34.0, 46.0]
SHAPE = (28, 28, 24)
# Enough frames that the temporal-mean template is not dominated by any one frame; at 8
# the self-inclusion of each frame in its own reference shrinks the slope to ~0.8.
NT = 24
PE = 2


def _smooth(rng, sigma, scale):
    v = gaussian_filter(rng.standard_normal(SHAPE), sigma)
    return (v / v.std() * scale).astype(np.float32)


def _scene(seed=0, field_noise=0.0, field_noise_sigma=1.5, mag_noise=2.0):
    """Head blob with texture; known smooth TE-scaled shift w_true in both channels.

    ``field_noise`` adds an echo-constant (in Hz) field change per voxel and frame,
    zero-mean over time -- what the phase's flat nuisance must ignore.
    """
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in SHAPE], indexing="ij"))
    head = (np.sqrt((g**2).sum(0)) < 0.8).astype(np.float32)
    s0 = (1000 + _smooth(rng, 1.2, 150)) * gaussian_filter(head, 1.0)
    r2 = 0.03 + _smooth(rng, 1.5, 0.005)  # 1/ms
    field = _smooth(rng, 2.0, 15.0)  # Hz, static background + tissue texture
    w_true = np.stack([_smooth(rng, 4.0, 0.04) for _ in range(NT)], -1)  # x alpha_4 ~ 0.18 vox
    alpha = np.array(TE) / TE[0]
    fn = (
        np.stack([_smooth(rng, field_noise_sigma, field_noise) for _ in range(NT)], -1)
        if field_noise
        else 0.0
    )
    if field_noise:
        fn = fn - fn.mean(-1, keepdims=True)
    mags, phs = [], []
    for te, a in zip(TE, alpha, strict=True):
        m_tpl = torch.from_numpy(s0 * np.exp(-te * r2))
        p_tpl = torch.from_numpy(np.where(head > 0, field, 0.0).astype(np.float32))
        m4 = np.empty(SHAPE + (NT,), np.float32)
        p4 = np.empty(SHAPE + (NT,), np.float32)
        for t in range(NT):
            d = torch.from_numpy((-a * w_true[..., t]).astype(np.float32))[None]
            m4[..., t] = _shift3d_axis(m_tpl[None], d, PE, mode="bicubic")[0].numpy()
            p4[..., t] = _shift3d_axis(p_tpl[None], d, PE, mode="bicubic")[0].numpy()
        m4 += rng.standard_normal(m4.shape).astype(np.float32) * mag_noise
        p4 += (rng.standard_normal(p4.shape).astype(np.float32) * 0.3 + fn) * (head[..., None] > 0)
        mags.append(m4)
        phs.append(p4)
    return mags, phs, w_true, head > 0


def _interior(head):
    from scipy.ndimage import binary_erosion

    return binary_erosion(head, iterations=5)


def _run(mags, phs, **kw):
    return estimate_partition_flow_magphase(
        mags,
        phs,
        TE,
        PE,
        window_sigma=2.0,
        n_iters=4,
        warp_interp="bicubic",
        device=CPU,
        verbose=False,
        **kw,
    )


def test_recovers_te_scaled_shift():
    mags, phs, w_true, head = _scene()
    flow = _run(mags, phs, refine_rounds=1)
    m = _interior(head)
    # A temporal-mean template can only see displacement relative to the run's mean.
    est = flow.w_field.numpy()
    est = (est - est.mean(-1, keepdims=True))[m]
    tru = (w_true - w_true.mean(-1, keepdims=True))[m]
    r = np.corrcoef(est.ravel(), tru.ravel())[0, 1]
    slope = float((est * tru).sum() / (tru * tru).sum())
    assert r > 0.95, r
    assert 0.85 < slope < 1.1, slope


def _recovery_r(flow, w_true, head):
    m = _interior(head)
    est = flow.w_field.numpy()
    est = (est - est.mean(-1, keepdims=True))[m]
    tru = (w_true - w_true.mean(-1, keepdims=True))[m]
    return np.corrcoef(est.ravel(), tru.ravel())[0, 1]


def test_blind_to_echo_constant_field_change(monkeypatch):
    # A field fluctuation (Hz, identical at every echo) ~6x the displacement's phase
    # footprint. The phase's per-voxel flat nuisance must shrug it off; without that
    # term the same solve reads the field change as shift.
    mags, phs, w_true, head = _scene(field_noise=6.0)
    blind = _recovery_r(_run(mags, phs, refine_rounds=1), w_true, head)
    import fastfuncstuff.processing.locomoco_magphase as lm

    orig = lm.nuisance_columns
    monkeypatch.setattr(
        lm, "nuisance_columns", lambda k, te: orig("none" if k == "flat" else k, te)
    )
    naive = _recovery_r(_run(mags, phs, refine_rounds=1), w_true, head)
    assert blind > 0.93, blind
    assert naive < blind - 0.2, (naive, blind)


def test_s0r2_needs_three_echoes():
    mags, phs, _, _ = _scene()
    with pytest.raises(ValueError, match="3 echoes"):
        estimate_partition_flow_magphase(
            mags[:2], phs[:2], TE[:2], PE, mag_nuisance="s0r2", device=CPU
        )


def test_wrap_guard_flags_wrapped_phase():
    rng = np.random.default_rng(1)
    # A 600 Hz ramp (21 Hz/voxel) plus texture: spans several wraps at TE 10 ms (100 Hz
    # period) while every unwrapped neighbour step stays far below half a period.
    ramp = np.broadcast_to(np.linspace(0, 600, SHAPE[0])[:, None, None], SHAPE)
    field = (ramp + gaussian_filter(rng.standard_normal(SHAPE), 1.0) * 5).astype(np.float32)
    mask = np.ones(SHAPE, bool)
    te = 10.0
    period = 1000.0 / te
    wrapped = (field + period / 2) % period - period / 2
    assert phase_wrap_fraction(field, period, mask) < 1e-3
    assert phase_wrap_fraction(wrapped, period, mask) > 1e-2
    # The run-level guard: the longest echo's wrapped series lives inside one period.
    periods = [wrap_period("hz", te), wrap_period("hz", 3 * te)]
    late = field * 1.0
    late_wrapped = (late + period / 6) % (period / 3) - period / 6
    assert phase_input_problem([field, late], periods, mask) is None
    assert "WRAPPED" in phase_input_problem([field, late_wrapped], periods, mask)
    raw = np.random.default_rng(2).uniform(-4096, 4096, SHAPE).astype(np.float32)
    assert "smooth" in phase_input_problem([raw, late], periods, mask)
    # Single echo: both checks on the one echo; radians are TE-free (period 2*pi).
    assert phase_input_problem([field], periods[:1], mask) is None
    rad = (field * 2 * np.pi * te * 1e-3).astype(np.float32)
    assert phase_input_problem([rad], [wrap_period("rad", None)], mask) is None
    rad_wrapped = np.angle(np.exp(1j * rad)).astype(np.float32)
    assert "WRAPPED" in phase_input_problem([rad_wrapped], [wrap_period("rad", None)], mask)
    with pytest.raises(ValueError, match="echo time"):
        wrap_period("hz", None)


def test_phase_units_and_nuisance_columns():
    te = 20.0
    hz = phase_to_hz(np.array([2 * np.pi * 0.02 * 5.0], np.float32), te, "rad")
    assert np.allclose(hz, 5.0)
    t = torch.tensor(TE)
    assert nuisance_columns("none", t).shape == (4, 0)
    assert nuisance_columns("s0", t).shape == (4, 1)
    assert nuisance_columns("s0r2", t).shape == (4, 2)


def _cli_args(*extra):
    from fastfuncstuff.cli.locomoco import _parse

    base = ["-input", "a.nii", "b.nii", "c.nii", "-echo_times", "10", "20", "30", "-me_3depi"]
    args = _parse(base + ["-pe_dir2", "IS", *extra])
    # main() derives the -detask mode fields after parsing; no -detask here.
    for name in ("detask_widen", "detask_fit"):
        if not hasattr(args, name):
            setattr(args, name, None)
    return args


def test_cli_phase_validation_single_echo():
    from fastfuncstuff.cli.locomoco import _magphase_arg_error, _magphase_mag_nuisance, _parse

    args = _parse(["-input", "a.nii", "-pe_dir2", "IS", "-phase", "p1", "-echo_times", "30"])
    for name in ("detask_widen", "detask_fit"):
        if not hasattr(args, name):
            setattr(args, name, None)
    assert _magphase_mag_nuisance(args) == "none"
    assert _magphase_arg_error(args, dual=False, slicewise=False) is None
    args.echo_times = None
    assert "-echo_times" in _magphase_arg_error(args, dual=False, slicewise=False)


def test_cli_phase_validation():
    from fastfuncstuff.cli.locomoco import _magphase_arg_error, _magphase_mag_nuisance

    ok = _cli_args("-phase", "p1", "p2", "p3")
    assert _magphase_arg_error(ok, dual=False, slicewise=False) is None
    assert _magphase_mag_nuisance(ok) == "s0r2"
    short = _cli_args("-phase", "p1", "p2")
    assert "2 phase series for 3" in _magphase_arg_error(short, dual=False, slicewise=False)
    flat = _cli_args("-phase", "p1", "p2", "p3", "-me_flat_scaling")
    assert "-me_flat_scaling" in _magphase_arg_error(flat, dual=False, slicewise=False)
    assert "ONE encode axis" in _magphase_arg_error(ok, dual=True, slicewise=False)


def test_single_echo_highpass_recovers_shift_under_smooth_field_drift():
    # One echo: no TE law, so the phase gets the masked high-pass instead. A large smooth
    # field drift (σ 6 vox, ~6x the shift's phase footprint) must not read as shift.
    mags, phs, w_true, head = _scene(field_noise=6.0, field_noise_sigma=6.0)
    run = dict(window_sigma=2.0, n_iters=4, warp_interp="bicubic", device=CPU, verbose=False)
    hp = estimate_partition_flow_magphase(
        mags[-1:],
        phs[-1:],
        TE[-1:],
        PE,
        mag_nuisance="none",
        phase_hpf_sigma=3.0,
        refine_rounds=1,
        **run,
    )
    # Recovery is of echo 4's own shift, alpha 1 when it is the only echo.
    alpha = TE[-1] / TE[0]
    assert _recovery_r(hp, w_true * alpha, head) > 0.85
    with pytest.raises(ValueError, match="phase_hpf_sigma"):
        estimate_partition_flow_magphase(
            mags[-1:], phs[-1:], TE[-1:], PE, mag_nuisance="none", **run
        )
    with pytest.raises(ValueError, match="nuisance"):
        estimate_partition_flow_magphase(
            mags[-1:], phs[-1:], TE[-1:], PE, mag_nuisance="s0", phase_hpf_sigma=3.0, **run
        )


def test_single_echo_highpass_rescues_weak_magnitude(monkeypatch):
    # Where the magnitude is weak the phase must carry the estimate, and a smooth field
    # drift then has to be high-passed out of it. (With strong magnitude the noise weights
    # already down-weight a drifting phase, so the high-pass matters little there.)
    import fastfuncstuff.processing.locomoco_magphase as lm

    mags, phs, w_true, head = _scene(field_noise=6.0, field_noise_sigma=6.0, mag_noise=30.0)
    run = dict(
        window_sigma=2.0,
        n_iters=4,
        warp_interp="bicubic",
        device=CPU,
        verbose=False,
        mag_nuisance="none",
        phase_hpf_sigma=3.0,
        refine_rounds=1,
    )
    alpha = TE[-1] / TE[0]
    with_hp = _recovery_r(
        estimate_partition_flow_magphase(mags[-1:], phs[-1:], TE[-1:], PE, **run),
        w_true * alpha,
        head,
    )
    monkeypatch.setattr(lm, "_masked_highpass", lambda vols, *a, **k: vols)
    no_hp = _recovery_r(
        estimate_partition_flow_magphase(mags[-1:], phs[-1:], TE[-1:], PE, **run),
        w_true * alpha,
        head,
    )
    assert with_hp > 0.6, with_hp
    assert no_hp < with_hp - 0.2, (no_hp, with_hp)
