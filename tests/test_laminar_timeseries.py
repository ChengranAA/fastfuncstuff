"""Continuous inversion must remove upward leakage without removing local activity."""

import numpy as np
import pytest
import torch

from fastfuncstuff.laminar.params import ModelSpec
from fastfuncstuff.laminar.timeseries import (
    dynamic_inverse,
    linearized_vasculature,
    vascular_transfer,
)

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def system():
    return linearized_vasculature(ModelSpec(N=3, K=3, device=CPU))


def test_transfer_respects_direction_and_microvascular_locality(system):
    freq = torch.tensor([0, 0.02, 0.1], dtype=torch.float64)
    full, local = vascular_transfer(system, freq)
    _, micro = vascular_transfer(system, freq, "micro")
    assert torch.linalg.eigvals(system.A).real.max() < 0
    assert full[:, 0, 2].abs().max() > 0.1
    torch.testing.assert_close(full[:, 2, 0], torch.zeros(3, dtype=full.dtype))
    torch.testing.assert_close(micro, torch.diag_embed(micro.diagonal(dim1=-2, dim2=-1)))
    torch.testing.assert_close(local[:, 2, 2], full[:, 2, 2])
    assert (full - local).abs().max() > 0.1


def test_transfer_matches_direct_small_signal_nonlinear_ode(system):
    """DC response checked against nonlinear vascular equilibrium, not its own inverse."""
    from dataclasses import replace

    from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
    from fastfuncstuff.laminar.params import zero_params

    spec = ModelSpec(N=3, K=3, device=CPU)
    p = zero_params(spec)
    c = replace(build_cache(spec, p), n2k=torch.eye(3, dtype=spec.dtype))
    # Simulate a small sustained local CBF increase with the actual nonlinear RHS.
    x = torch.zeros(spec.n_states, dtype=spec.dtype)
    x[4 * spec.N - 1] = torch.log(torch.tensor(1.0001, dtype=spec.dtype))
    u = torch.zeros(spec.n_inputs, dtype=spec.dtype)
    for _ in range(6000):
        dx = rhs(x, u, c, spec.N, spec.K)
        x[4 * spec.N :] += 0.02 * dx[4 * spec.N :]
    y = g_obs(x, p, spec, cache=c) / 0.0001
    full, _ = vascular_transfer(system, torch.zeros(1))
    torch.testing.assert_close(y, full[0, :, -1].real, rtol=0.002, atol=0.002)


def test_inverse_recovers_local_bold_and_preserves_deepest_depth(system):
    T, tr = 180, 2.5
    freq = torch.fft.rfftfreq(T, tr)
    h, g = vascular_transfer(system, freq)
    t = torch.arange(T) * tr
    source = torch.stack(
        [
            0.01 * torch.sin(2 * torch.pi * t / 60),
            0.008 * torch.cos(2 * torch.pi * t / 90),
            0.02 * torch.sin(2 * torch.pi * t / 45),
        ],
        -1,
    ).double()
    af = torch.fft.rfft(source, dim=0)
    yf = torch.einsum("fkn,fn->fk", h, af)
    y = torch.fft.irfft(yf, n=T, dim=0)
    truth = torch.fft.irfft(torch.einsum("fkn,fn->fk", g, af), n=T, dim=0)
    profiles = y[None, None].repeat(2, 1, 1, 1)
    result = dynamic_inverse(
        profiles, h, g, tr=tr, pad=0, strengths=(1e-7,), device=CPU, verbose=False
    )
    corrected = profiles - result.correction
    assert torch.mean((corrected[0, 0] - truth) ** 2) < 1e-4 * torch.mean((y - truth) ** 2)
    torch.testing.assert_close(corrected[..., -1], profiles[..., -1], atol=1e-10, rtol=0)
    assert torch.isfinite(result.noise_gain).all()


def test_no_transport_is_identity_even_when_regularized(system):
    T = 60
    h, _ = vascular_transfer(system, torch.fft.rfftfreq(T + 20, 2.5))
    torch.manual_seed(4)
    y = torch.randn(2, 3, T, 3)
    result = dynamic_inverse(
        y, h, h, tr=2.5, pad=10, strengths=(0.01, 0.1), device=CPU, verbose=False
    )
    assert result.correction.abs().max() == 0
    assert result.regularization.shape == (3,)
    torch.testing.assert_close(result.noise_gain, torch.ones(3))


def test_independent_noise_is_not_rewarded_by_cross_run_selection(system):
    T, pad = 120, 12
    h, g = vascular_transfer(system, torch.fft.rfftfreq(T + 2 * pad, 2.5))
    torch.manual_seed(10)
    y = torch.randn(3, 8, T, 3)
    result = dynamic_inverse(
        y, h, g, tr=2.5, pad=pad, strengths=(0.0001, 0.1, 10), device=CPU, verbose=False
    )
    assert (result.regularization == 10).float().mean() >= 0.75
    assert result.cv_r2.median() <= 0.05


def test_bad_shapes_and_single_run_selection_are_refused(system):
    h, g = vascular_transfer(system, torch.fft.rfftfreq(60, 2.5))
    with pytest.raises(ValueError, match="repeated runs"):
        dynamic_inverse(torch.zeros(1, 1, 60, 3), h, g, tr=2.5, pad=0, device=CPU)


def test_dynamic_micro_recovery_against_nonlinear_time_varying_forward(system):
    """Time-varying nonlinear ODE supplies truth independently of the FFT inverse."""
    from dataclasses import replace

    from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
    from fastfuncstuff.laminar.params import zero_params

    spec = ModelSpec(N=3, K=3, TR=2, device=CPU)
    p = zero_params(spec)
    c = replace(build_cache(spec, p), n2k=torch.eye(3, dtype=spec.dtype))
    micro = replace(
        c, k1d=torch.zeros_like(c.k1d), k2d=torch.zeros_like(c.k2d), k3d=torch.zeros_like(c.k3d)
    )
    x = torch.zeros(spec.n_states, dtype=spec.dtype)
    u = torch.zeros(spec.n_inputs, dtype=spec.dtype)
    dt, T = 0.05, 120
    observed, truth = [], []
    for step in range(int(T * spec.TR / dt)):
        t = step * dt
        cbf = 0.003 * torch.sin(
            2
            * torch.pi
            * torch.tensor(t, dtype=spec.dtype)
            / torch.tensor([40.0, 60.0, 80.0], dtype=spec.dtype)
        )
        x[3 * spec.N : 4 * spec.N] = torch.log1p(cbf)
        if step % 40 == 0:
            observed.append(g_obs(x, p, spec, cache=c))
            truth.append(g_obs(x, p, spec, cache=micro))
        dx = rhs(x, u, c, spec.N, spec.K)
        x[4 * spec.N :] += dt * dx[4 * spec.N :]
    y, clean = torch.stack(observed), torch.stack(truth)
    h, g = vascular_transfer(system, torch.fft.rfftfreq(T + 60, spec.TR), "micro")
    res = dynamic_inverse(
        y[None, None], h, g, tr=spec.TR, pad=30, strengths=(1e-6,), device=CPU, verbose=False
    )
    corrected = y - res.correction[0, 0]
    interior = slice(30, -30)
    assert ((corrected[interior] - clean[interior]) ** 2).mean() < 0.05 * (
        (y[interior] - clean[interior]) ** 2
    ).mean()


def test_cbf_kernel_is_a_positive_delayed_response():
    from fastfuncstuff.laminar.timeseries import cbf_kernel

    h = cbf_kernel(ModelSpec(N=3, K=3, device=CPU), dt=0.1, seconds=40)
    peak = int(h.argmax())
    # Measured: starts at 0, peaks at 2.0 s (CBF leads BOLD), undershoot ~5% by 10 s.
    assert h[0] == 0 and h.max() == 1 and 1.0 < peak * 0.1 < 5.0
    assert h[-20:].abs().max() < 0.1


def _design_runs(spec, n_runs=3, T=200, seed=0):
    """Two conditions, a different random event schedule per run, CBF-shaped regressors."""
    from fastfuncstuff.laminar.timeseries import cbf_kernel

    h = cbf_kernel(spec, dt=0.1).double()
    rng = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_runs):
        stim = torch.zeros(2, int(T * spec.TR / 0.1))
        t = 5.0
        while t < T * spec.TR - 20:
            stim[int(torch.randint(2, (1,), generator=rng)), int(t / 0.1) : int((t + 2) / 0.1)] = 1
            t += 4 + 8 * float(torch.rand(1, generator=rng))
        x = torch.stack(
            [torch.from_numpy(np.convolve(s.numpy(), h.numpy())[: s.numel()]) for s in stim]
        )
        out.append(x[:, :: int(round(spec.TR / 0.1))][:, :T].T * 0.1)
    return out


@pytest.mark.parametrize(
    ("name", "grid", "truth"), [("t0v", (0.5, 1.0, 2.0), 1.0), ("s_d", (0.1, 0.4, 1.6), 0.4)]
)
def test_design_crossval_prefers_the_generating_physiology(name, grid, truth):
    """Different schedules per run; pooled over patches the truth must predict best."""
    from dataclasses import replace

    from fastfuncstuff.glm.core import construct_polynomial_matrix
    from fastfuncstuff.laminar.params import P0
    from fastfuncstuff.laminar.timeseries import design_crossval

    K, T, U, tr = 3, 200, 64, 2.0
    n_fft = T + 30
    base = ModelSpec(N=K, K=K, TR=tr, device=CPU)
    designs = _design_runs(base, T=T)
    nuis = [construct_polynomial_matrix(T, 1, CPU, torch.float64) for _ in designs]

    def transfer(value):
        spec = replace(base, p0=replace(P0(), **{name: value}))
        return vascular_transfer(linearized_vasculature(spec), torch.fft.rfftfreq(n_fft, tr))[0]

    torch.manual_seed(1)
    beta = 0.02 * (1 + 0.3 * torch.randn(U, K * 2, dtype=torch.float64))
    h = transfer(truth)
    prof = []
    for X, q in zip(designs, nuis, strict=True):
        Xf = torch.fft.rfft(X, n=n_fft, dim=0)
        Z = torch.fft.irfft(h[..., None] * Xf[:, None, None, :], n=n_fft, dim=0)[:T].reshape(
            T, K, -1
        )
        y = torch.einsum("tkp,up->utk", Z, beta)
        y = y + 0.5 * y.std() * torch.randn_like(y)
        qq, _ = torch.linalg.qr(q)
        prof.append(
            y
            - torch.einsum("ts,us k->utk".replace(" ", ""), qq, torch.einsum("ts,utk->usk", qq, y))
        )
    profiles = torch.stack(prof)
    scores = [
        design_crossval(profiles, designs, transfer(v), n_fft, nuisance=nuis, device=CPU)[0].sum()
        for v in grid
    ]
    assert int(np.argmin(scores)) == grid.index(truth)
