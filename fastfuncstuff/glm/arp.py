"""AR(p) REML from lagged sufficient statistics: one pass over the data, then
every candidate noise model costs nothing that depends on the series length.

Why it works
------------
REML's quadratic form is ``y'Py`` with ``P = V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹``. ``PX = 0``,
so ``y'Py = r'Pr`` for the OLS residual ``r``:

    RSS(θ) = r'V⁻¹r − g'M⁻¹g,     g = X'V⁻¹r,     M = X'V⁻¹X.

For stationary AR(p) noise with unit marginal variance the inverse Cholesky factor
of V is the Durbin–Levinson innovations filter: row t of a run is the order
``min(t, p)`` one-step predictor, scaled by its error variance. Every whitened
sample is a short filter of at most P+1 inputs, so ``r'V⁻¹r``, ``g`` and ``M`` are
quadratic forms in the filter taps against lagged cross-products

    C_rr(k,l) = Σ_{t≥P} r_{t−k} r_{t−l},   S_Xr(k,l) = Σ_{t≥P} X_{t−k} r_{t−l},
    S_XX(k,l) = Σ_{t≥P} X_{t−k} X_{t−l}',

plus the first P samples of each run, where the filter is still changing order.
That is exact for any order p ≤ P (not a Whittle approximation), and
``log|V| = Σ_t log v_{min(t,p)}`` comes from the same recursion.

The statistics add across runs. So one pass gives the whole-data fit, a fit per
run (is the noise the same in every run?), and held-out scoring (does a noise
model fitted on some runs whiten the others?), without touching the series again.

The criterion is the one ``glm.arma`` minimises —
``log|V| + log|X'V⁻¹X| + (N−K)·log RSS`` — so AR(p) and the ARMA(1,1) grid values
are directly comparable, and AR(1) at κ = a reproduces the grid's (a, 0) point.

Parametrisation: partial autocorrelations κ_j = tanh(θ_j). Every θ ∈ ℝ^p is a
stationary AR(p), and Levinson turns κ into the predictors and variances directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class LagStats:
    """Lagged cross-products of OLS residuals and design, per run.

    ``r`` is indexed (V, R, ...) — voxels, runs. Design terms are shared across
    voxels. ``support`` marks the design columns that are nonzero in each run,
    so a subset of runs can drop the columns it has no data for."""

    P: int
    n: Tensor  # (R,) run lengths
    Crr: Tensor  # (V, R, P+1, P+1)
    SXr: Tensor  # (V, R, P+1, P+1, K)
    SXX: Tensor  # (R, P+1, P+1, K, K)
    r_head: Tensor  # (V, R, P)
    X_head: Tensor  # (R, P, K)
    support: Tensor  # (R, K) bool
    # Run-summed stats on the supported columns, built once per instance: every
    # criterion evaluation needs them, and S_Xr is the largest tensor here.
    _summed: tuple[Tensor, Tensor, Tensor, Tensor] | None = field(
        default=None, repr=False, compare=False
    )

    def summed(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(Crr (V, P1²), SXr (V, P1², Kc), SXX (P1², Kc²), X_head (R, P, Kc))``
        summed over runs and restricted to the columns these runs support."""
        if self._summed is None:
            cols = self.support.any(dim=0)
            V, P1 = self.Crr.shape[0], self.P + 1
            Kc = int(cols.sum())
            self._summed = (
                self.Crr.sum(1).reshape(V, P1 * P1),
                self.SXr[..., cols].sum(1).reshape(V, P1 * P1, Kc),
                self.SXX[..., cols][..., cols, :].sum(0).reshape(P1 * P1, Kc * Kc),
                self.X_head[..., cols],
            )
        return self._summed

    def runs(self, idx: Sequence[int] | Tensor) -> LagStats:
        """These runs only (stats keep their run axis)."""
        i = torch.as_tensor(idx, dtype=torch.long, device=self.n.device)
        return LagStats(
            self.P,
            self.n[i],
            self.Crr[:, i],
            self.SXr[:, i],
            self.SXX[i],
            self.r_head[:, i],
            self.X_head[i],
            self.support[i],
        )

    def voxels(self, sl: slice | Tensor) -> LagStats:
        cached = None
        if self._summed is not None:
            c, sx, sxx, xh = self._summed
            cached = (c[sl], sx[sl], sxx, xh)
        return LagStats(
            self.P,
            self.n,
            self.Crr[sl],
            self.SXr[sl],
            self.SXX,
            self.r_head[sl],
            self.X_head,
            self.support,
            cached,
        )


def lag_stats(
    resid: Tensor,
    X: Tensor,
    bounds: Sequence[tuple[int, int]],
    P: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> LagStats:
    """One pass: lagged cross-products of residuals ``resid`` (V, T) and design
    ``X`` (T, K) out to lag ``P``, per run. The GEMMs run in the inputs' dtype;
    results are stored in ``dtype`` (float64 by default — every later step works
    on small matrices, where float64 is cheap and the likelihood surface is flat).
    """
    V, _ = resid.shape
    K = X.shape[1]
    R = len(bounds)
    dev = resid.device
    P1 = P + 1
    Crr = torch.empty(V, R, P1, P1, dtype=dtype, device=dev)
    SXr = torch.empty(V, R, P1, P1, K, dtype=dtype, device=dev)
    SXX = torch.empty(R, P1, P1, K, K, dtype=dtype, device=dev)
    r_head = torch.empty(V, R, P, dtype=dtype, device=dev)
    X_head = torch.empty(R, P, K, dtype=dtype, device=dev)
    support = torch.empty(R, K, dtype=torch.bool, device=dev)
    n = torch.tensor([e - s for s, e in bounds], device=dev)
    Xc = X.to(device=dev, dtype=resid.dtype)
    for ri, (s, e) in enumerate(bounds):
        m = e - s
        if m <= P:
            raise ValueError(f"run {ri} has {m} timepoints; AR order needs more than P={P}")
        r = resid[:, s:e]
        x = Xc[s:e]
        # lag l view: samples t-l for t = P..m-1
        rlag = torch.stack([r[:, P - lag : m - lag] for lag in range(P1)], dim=1)  # (V, P1, m-P)
        xlag = torch.stack([x[P - lag : m - lag] for lag in range(P1)], dim=0)  # (P1, m-P, K)
        Crr[:, ri] = (rlag @ rlag.transpose(1, 2)).to(dtype)
        # S_Xr[v,k,l,:] = xlag[k]' rlag[v,l] — one GEMM: ((P1·K), m-P) @ (m-P, V·P1)
        xs = xlag.permute(0, 2, 1).reshape(P1 * K, m - P)
        rs = rlag.permute(2, 0, 1).reshape(m - P, V * P1)
        SXr[:, ri] = (xs @ rs).reshape(P1, K, V, P1).permute(2, 0, 3, 1).to(dtype)
        SXX[ri] = (xs @ xs.T).reshape(P1, K, P1, K).permute(0, 2, 1, 3).to(dtype)
        r_head[:, ri] = r[:, :P].to(dtype)
        X_head[ri] = x[:P].to(dtype)
        support[ri] = x.abs().amax(dim=0) > 0
    return LagStats(P, n, Crr, SXr, SXX, r_head, X_head, support)


def levinson(kappa: Tensor, P: int) -> tuple[Tensor, Tensor]:
    """Innovations filters for every order 0..p from partial autocorrelations.

    ``kappa`` (..., p). Returns ``(taps, v)``: ``taps`` (..., p+1, P+1) — row j is
    the order-j whitening filter ``[1, −φ^{(j)}_1, …, −φ^{(j)}_j, 0…] / √v_j``
    applied as ``Σ_i taps[i]·x_{t−i}`` — and ``v`` (..., p+1), the one-step error
    variances relative to a unit marginal variance (v_0 = 1)."""
    p = kappa.shape[-1]
    batch = kappa.shape[:-1]
    phi = kappa.new_zeros(*batch, 0)
    v = [kappa.new_ones(batch)]
    rows = []
    for j in range(p + 1):
        if j > 0:
            k = kappa[..., j - 1 : j]
            phi = torch.cat([phi - k * phi.flip(-1), k], dim=-1)
            v.append(v[-1] * (1 - kappa[..., j - 1] ** 2))
        row = torch.cat([kappa.new_ones(*batch, 1), -phi, kappa.new_zeros(*batch, P - j)], dim=-1)
        rows.append(row / v[-1].sqrt().unsqueeze(-1))
    return torch.stack(rows, dim=-2), torch.stack(v, dim=-1)


def _head_filter(taps: Tensor, p: int, P: int) -> Tensor:
    """(..., P, P) lower-triangular whitening of a run's first P samples: row t
    uses the order-min(t, p) taps over samples t, t-1, …, 0."""
    rows = []
    for t in range(P):
        c = taps[..., min(t, p), : t + 1]  # (..., t+1): taps for lags 0..t
        rows.append(torch.cat([c.flip(-1), c.new_zeros(*c.shape[:-1], P - t - 1)], dim=-1))
    return torch.stack(rows, dim=-2)


def _quadratic_parts(theta: Tensor, st: LagStats) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """(rw2, g, M, logdetV) per voxel for κ = tanh(θ), θ (V, p), summed over runs,
    on the columns these runs support."""
    p, P = theta.shape[-1], st.P
    kappa = torch.tanh(theta)
    taps, v = levinson(kappa, P)  # (V, p+1, P+1), (V, p+1)
    c = taps[:, p]  # steady taps (V, P1)
    H = _head_filter(taps, p, P)  # (V, P, P)

    V, P1 = c.shape
    Crr, SXr, SXX, X_head = st.summed()
    K = SXr.shape[-1]
    cc = (c.unsqueeze(2) * c.unsqueeze(1)).reshape(V, P1 * P1)  # tap outer product
    rw2 = (cc * Crr).sum(-1)
    g = torch.bmm(cc.unsqueeze(1), SXr).squeeze(1)
    M = (cc @ SXX).reshape(V, K, K)
    hr = torch.einsum("vts,vrs->vrt", H, st.r_head)  # (V, R, P) whitened heads
    hx = torch.einsum("vts,rsK->vrtK", H, X_head)  # (V, R, P, K)
    rw2 = rw2 + hr.pow(2).sum((1, 2))
    g = g + torch.einsum("vrtK,vrt->vK", hx, hr)
    M = M + torch.einsum("vrtK,vrtJ->vKJ", hx, hx)

    logv = torch.log(v)  # (V, p+1)
    n = st.n.to(theta.dtype)
    head = torch.cumsum(logv[:, :p], dim=1)  # Σ_{j<min(n,p)} log v_j; runs are longer than P ≥ p
    logdetV = (head[:, -1:] if p else theta.new_zeros(theta.shape[0], 1)).squeeze(1) * len(
        st.n
    ) + logv[:, p] * (n - p).sum()
    return rw2, g, M, logdetV


def _columns(st: LagStats) -> Tensor:
    """Design columns with data in any of these runs."""
    return st.support.any(dim=0)


def reml_criterion(theta: Tensor, st: LagStats) -> Tensor:
    """``log|V| + log|X'V⁻¹X| + (N−K)·log RSS`` per voxel (smaller is better) —
    ``glm.arma``'s criterion, for AR(p) with κ = tanh(θ)."""
    rw2, g, M, logdetV = _quadratic_parts(theta, st)
    # Near |κ| → 1 the taps blow up and M stops being numerically PD; such a θ is
    # +inf, so a line search backs off it rather than the whole batch failing.
    L, info = torch.linalg.cholesky_ex(M)
    bad = info != 0
    if bool(bad.any()):
        L = torch.where(
            bad[:, None, None], torch.eye(M.shape[-1], dtype=M.dtype, device=M.device), L
        )
    a = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    rss = rw2 - (g * a).sum(-1)
    N, K = int(st.n.sum()), M.shape[-1]
    logdetM = 2 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    crit = logdetV + logdetM + (N - K) * torch.log(rss)
    return torch.where(bad | ~(rss > 0), torch.full_like(crit, float("inf")), crit)


def yule_walker_init(st: LagStats, p: int, clamp: float = 0.95) -> Tensor:
    """θ from the residual autocorrelations — Newton's starting point. Biased by
    the design projection (that is what REML then removes), close enough to start."""
    C = st.Crr.sum(1)
    gamma = C[:, 0, : p + 1] / C[:, 0, :1]  # lag-k autocorrelation, k = 0..p
    kappa = []
    phi = gamma.new_zeros(gamma.shape[0], 0)
    v = gamma.new_ones(gamma.shape[0])
    for j in range(1, p + 1):
        num = gamma[:, j] - (phi * gamma[:, 1:j].flip(-1)).sum(-1)
        k = (num / v).clamp(-clamp, clamp)
        kappa.append(k)
        phi = torch.cat([phi - k.unsqueeze(-1) * phi.flip(-1), k.unsqueeze(-1)], dim=-1)
        v = v * (1 - k**2)
    return torch.atanh(torch.stack(kappa, dim=-1)) if p else gamma.new_zeros(gamma.shape[0], 0)


def _grad(theta: Tensor, st: LagStats) -> Tensor:
    """Per-voxel gradient of the criterion. The criterion is a sum of independent
    voxel terms, so one backward pass of the sum gives every voxel's own."""
    with torch.enable_grad():
        x = theta.detach().requires_grad_(True)
        (gr,) = torch.autograd.grad(reml_criterion(x, st).sum(), x)
    return gr


def _hessian(theta: Tensor, st: LagStats, h: float = 1e-4) -> Tensor:
    """Per-voxel p×p Hessian by central differences of :func:`_grad`.

    2p backward passes, each the size of one evaluation — forward-over-reverse
    through a K×K Cholesky per voxel does not fit in memory at real K. float64
    keeps the O(h²) error far below anything Newton or the variance needs."""
    cols = []
    for j in range(theta.shape[-1]):
        e = torch.zeros_like(theta)
        e[:, j] = h
        cols.append((_grad(theta + e, st) - _grad(theta - e, st)) / (2 * h))
    H = torch.stack(cols, dim=-1)
    return 0.5 * (H + H.transpose(1, 2))


def theta_covariance(theta: Tensor, st: LagStats) -> Tensor:
    """Asymptotic covariance of θ̂ per voxel: the criterion is −2·log-likelihood
    (up to a constant), so the information is half its Hessian."""
    return torch.linalg.inv(0.5 * _hessian(theta, st))


def fit_ar_reml(
    st: LagStats,
    p: int,
    *,
    n_iter: int = 60,
    tol: float = 1e-4,
    theta0: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Per-voxel REML AR(p): quasi-Newton on θ (κ = tanh θ) from Yule–Walker.

    Returns ``(theta (V, p), criterion (V,))``; ``tol`` is the predicted remaining
    decrease, in criterion units, at which a voxel stops. The p×p Hessian is taken once by
    central differences at the start, then kept current by BFGS updates — one
    gradient per iteration instead of 2p+1, which is what the cost is made of.
    Steps are Levenberg-damped where the model is not positive definite, capped
    at 1 in θ (tanh saturates), and backtracked until the criterion drops.
    """
    if p == 0:
        z = torch.zeros(st.Crr.shape[0], 0, dtype=st.Crr.dtype, device=st.Crr.device)
        return z, reml_criterion(z, st)
    theta = (yule_walker_init(st, p) if theta0 is None else theta0).clone()
    st.summed()
    crit = reml_criterion(theta, st)
    gr = _grad(theta, st)
    B = _hessian(theta, st)
    eye = torch.eye(p, dtype=theta.dtype, device=theta.device)
    active = torch.ones(theta.shape[0], dtype=torch.bool, device=theta.device)
    fresh = torch.ones_like(active)  # B is an exact Hessian, not a BFGS model
    for _ in range(n_iter):
        idx = active.nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        th, sub, g0, Bi, c0 = theta[idx], st.voxels(idx), gr[idx], B[idx], crit[idx]
        lam_min = torch.linalg.eigvalsh(Bi)[:, 0]
        damp = (1e-6 - lam_min).clamp(min=0) + 1e-9
        step = torch.linalg.solve(Bi + damp[:, None, None] * eye, g0.unsqueeze(-1)).squeeze(-1)
        # tanh saturates: a θ step past ~1 is a jump to |κ| ≈ 1, never progress
        step = step / step.norm(dim=-1, keepdim=True).clamp(min=1.0)
        scale = torch.ones_like(c0)
        accepted = torch.zeros_like(c0, dtype=torch.bool)
        new_th, new_c = th.clone(), c0.clone()
        for _ in range(12):
            todo = ~accepted
            if not bool(todo.any()):
                break
            w = todo.nonzero().squeeze(1)
            cand = th[w] - scale[w, None] * step[w]
            cc = reml_criterion(cand, sub.voxels(w))
            ok = torch.isfinite(cc) & (cc <= c0[w])
            new_th[w[ok]] = cand[ok]
            new_c[w[ok]] = cc[ok]
            accepted[w[ok]] = True
            scale[w[~ok]] *= 0.5
        g1 = _grad(new_th, sub)
        # BFGS update of the Hessian model; skipped where curvature is not positive
        sv, yv = new_th - th, g1 - g0
        sy = (sv * yv).sum(-1)
        Bs = (Bi @ sv.unsqueeze(-1)).squeeze(-1)
        sBs = (sv * Bs).sum(-1)
        upd = (sy > 1e-12) & (sBs > 1e-12)
        Bn = Bi + torch.where(
            upd[:, None, None],
            yv.unsqueeze(-1) * yv.unsqueeze(-2) / sy.clamp(min=1e-12)[:, None, None]
            - Bs.unsqueeze(-1) * Bs.unsqueeze(-2) / sBs.clamp(min=1e-12)[:, None, None],
            torch.zeros_like(Bi),
        )
        theta[idx], crit[idx], gr[idx], B[idx] = new_th, new_c, g1, Bn
        # Converged when the model predicts less than `tol` left to gain (criterion
        # units, i.e. −2·log-likelihood). A relative-change test stops BFGS while
        # it is still creeping along a shallow valley.
        pred = 0.5 * (
            g1 * torch.linalg.solve(Bn + damp[:, None, None] * eye, g1.unsqueeze(-1)).squeeze(-1)
        ).sum(-1)
        moved = pred.abs() > tol
        # A stale BFGS model can point nowhere useful: give those voxels an exact
        # Hessian and another go. Stop only when an exact-Hessian step fails too.
        retry = ~accepted & ~fresh[idx]
        if bool(retry.any()):
            r = idx[retry]
            B[r] = _hessian(theta[r], st.voxels(r))
        fresh[idx] = retry
        active[idx] = (moved & accepted) | retry
    return theta, crit


def gls_from_stats(theta: Tensor, st: LagStats, beta_ols: Tensor) -> dict[str, Tensor]:
    """GLS at the fitted AR(p): betas, their variances and σ² from the same stats.

    ``β = β_OLS + M⁻¹g`` (because X'V⁻¹y = Mβ_OLS + g), ``Var β = σ²M⁻¹``,
    ``σ² = RSS/(N−K)``. ``beta_ols`` (V, K) must be the fit ``resid`` came from."""
    cols = _columns(st)
    rw2, g, M, _ = _quadratic_parts(theta, st)
    L = torch.linalg.cholesky(M)
    a = torch.cholesky_solve(g.unsqueeze(-1), L).squeeze(-1)
    rss = rw2 - (g * a).sum(-1)
    N, K = int(st.n.sum()), int(cols.sum())
    sigma2 = rss / (N - K)
    Minv_diag = torch.diagonal(torch.cholesky_inverse(L), dim1=-2, dim2=-1)
    beta = beta_ols[:, cols].to(a.dtype) + a
    var = sigma2[:, None] * Minv_diag
    return {"beta": beta, "var": var, "sigma2": sigma2, "t": beta / var.sqrt(), "rss": rss}
