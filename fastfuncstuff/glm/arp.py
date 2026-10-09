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
    """Lagged cross-products of OLS residuals and design, per run, on each run's
    own columns.

    A run only touches the design columns that are nonzero in it (its own
    polynomials and motion, plus the shared task columns), so the design side is
    stored on those *local* columns: ``cols`` (R, kl) maps each run's local slot
    to a design column (−1 = padding). 100 runs × 401 columns with 5 per run
    stays 5 wide instead of 401. ``r`` terms are indexed (V, R, ...)."""

    P: int
    n: Tensor  # (R,) run lengths
    Crr: Tensor  # (V, R, P+1, P+1)
    SXr: Tensor  # (V, R, P+1, P+1, kl)
    SXX: Tensor  # (R, P+1, P+1, kl, kl)
    r_head: Tensor  # (V, R, P)
    X_head: Tensor  # (R, P, kl)
    cols: Tensor  # (R, kl) long, −1 = padding
    K: int  # columns in the full design
    # The criterion's data, built once per instance (see `compact`).
    _crit: _Crit | None = field(default=None, repr=False, compare=False)

    @property
    def support(self) -> Tensor:
        """(R, K) bool: design columns with data in each run."""
        sup = torch.zeros(len(self.n), self.K, dtype=torch.bool, device=self.cols.device)
        r, j = torch.nonzero(self.cols >= 0, as_tuple=True)
        sup[r, self.cols[r, j]] = True
        return sup

    def compact(self) -> _Crit:
        """The criterion's data for these runs, run-summed and structured.

        r'V⁻¹r, X'V⁻¹r and X'V⁻¹X are each linear in one feature vector per
        voxel: the steady taps' outer product c⊗c and G = H'H of the head
        filter. Their coefficients are summed over runs here, so an evaluation
        is a few GEMMs with no per-run or per-row work. Both feature blocks are
        symmetric and are folded to one triangle.

        M = X'V⁻¹X is split by column support: columns that live in one run only
        (per-run polynomials, per-run motion — block-diagonal nuisance) give one
        block per run, and V⁻¹ is block-diagonal by run, so M is *exactly* zero
        between different runs' blocks. Only the shared columns (the task)
        couple them. Factoring through that structure (Schur complement on the
        shared block) replaces one K³ per voxel with R·k³ + k_s³."""
        if self._crit is None:
            self._crit = _build_crit(self)
        return self._crit

    def runs(self, idx: Sequence[int] | Tensor) -> LagStats:
        """These runs only (stats keep their run axis)."""
        i = torch.as_tensor(idx, dtype=torch.long, device=self.n.device)
        return LagStats(
            self.P, self.n[i], self.Crr[:, i], self.SXr[:, i], self.SXX[i],
            self.r_head[:, i], self.X_head[i], self.cols[i], self.K,
        )  # fmt: skip

    def voxels(self, sl: slice | Tensor) -> LagStats:
        cached = self._crit.voxels(sl) if self._crit is not None else None
        return LagStats(
            self.P, self.n, self.Crr[sl], self.SXr[sl], self.SXX, self.r_head[sl],
            self.X_head, self.cols, self.K, cached,
        )  # fmt: skip


@dataclass
class _Crit:
    """Run-summed, column-structured criterion data for a voxel batch.

    ``F`` folded features. Private columns are padded to ``k`` per run (``pad``
    marks the padding, given an identity diagonal so the blocks stay PD);
    ``order`` gives each structured column's position in ``ucols``: the R·k
    private slots (−1 for padding), then the shared ones."""

    P: int
    n: Tensor  # (R,)
    a_r: Tensor  # (V, F)
    g_priv: Tensor  # (V, F, R, k)
    g_shared: Tensor  # (V, F, ks)
    D: Tensor  # (F, R, k, k)
    B: Tensor  # (F, R, k, ks)
    S: Tensor  # (F, ks, ks)
    pad: Tensor  # (R, k) bool
    order: Tensor  # (R·k + ks,) long, positions in ``ucols``
    ucols: Tensor  # (n_cols,) the design columns these runs support, sorted

    @property
    def n_cols(self) -> int:
        return int(self.ucols.numel())

    def voxels(self, sl: slice | Tensor) -> _Crit:
        return _Crit(
            self.P, self.n, self.a_r[sl], self.g_priv[sl], self.g_shared[sl],
            self.D, self.B, self.S, self.pad, self.order, self.ucols,
        )  # fmt: skip

    def compact(self) -> _Crit:
        return self


def _build_crit(st: LagStats) -> _Crit:
    V, R = st.Crr.shape[:2]
    P, P1, kl = st.P, st.P + 1, st.cols.shape[1]
    dev, dt = st.Crr.device, st.Crr.dtype
    rh, xh = st.r_head, st.X_head
    Fm = _feature_fold(P, dt, dev)  # (F_full, F)
    F = Fm.shape[1]
    a_r = (
        torch.cat(
            [
                st.Crr.sum(1).reshape(V, P1 * P1),
                torch.einsum("vrs,vrt->vst", rh, rh).reshape(V, P * P),
            ],
            dim=1,
        )
        @ Fm
    )
    # per-run coefficients on local columns, features folded
    Ag = torch.einsum(
        "vrfk,fs->vrsk",
        torch.cat(
            [
                st.SXr.reshape(V, R, P1 * P1, kl),
                torch.einsum("rsK,vrt->vrstK", xh, rh).reshape(V, R, P * P, kl),
            ],
            dim=2,
        ),
        Fm,
    )  # (V, R, F, kl)
    Am = torch.einsum(
        "rfx,fs->rsx",
        torch.cat(
            [
                st.SXX.reshape(R, P1 * P1, kl * kl),
                torch.einsum("rsK,rtJ->rstKJ", xh, xh).reshape(R, P * P, kl * kl),
            ],
            dim=1,
        ),
        Fm,
    ).reshape(R, F, kl, kl)

    # global column structure: private to one run, or shared by several
    valid = st.cols >= 0
    ucols = torch.unique(st.cols[valid])
    gpos = torch.searchsorted(ucols, st.cols.clamp(min=0))  # position in ucols
    count = torch.bincount(gpos[valid], minlength=ucols.numel())
    is_shared = count > 1
    shared_pos = torch.nonzero(is_shared).squeeze(1)  # positions in ucols
    spos_of = torch.full((ucols.numel(),), -1, dtype=torch.long, device=dev)
    spos_of[shared_pos] = torch.arange(shared_pos.numel(), device=dev)
    ks = shared_pos.numel()
    priv_loc = [torch.nonzero(valid[r] & ~is_shared[gpos[r]]).squeeze(1) for r in range(R)]
    k = max((int(x.numel()) for x in priv_loc), default=0)

    D = torch.zeros(F, R, k, k, dtype=dt, device=dev)
    B = torch.zeros(F, R, k, ks, dtype=dt, device=dev)
    S = torch.zeros(F, ks, ks, dtype=dt, device=dev)
    g_priv = torch.zeros(V, F, R, k, dtype=dt, device=dev)
    g_shared = torch.zeros(V, F, ks, dtype=dt, device=dev)
    slots = torch.full((R, k), -1, dtype=torch.long, device=dev)
    for r in range(R):
        pl = priv_loc[r]
        sl = torch.nonzero(valid[r] & is_shared[gpos[r]]).squeeze(1)
        sp = spos_of[gpos[r, sl]]
        npv = pl.numel()
        ip = torch.arange(npv, device=dev)
        D[:, r, ip[:, None], ip[None, :]] = Am[r][:, pl[:, None], pl[None, :]]
        B[:, r, ip[:, None], sp[None, :]] = Am[r][:, pl[:, None], sl[None, :]]
        S[:, sp[:, None], sp[None, :]] += Am[r][:, sl[:, None], sl[None, :]]
        g_priv[:, :, r, :npv] = Ag[:, r][:, :, pl]
        g_shared[:, :, sp] += Ag[:, r][:, :, sl]
        slots[r, :npv] = gpos[r, pl]
    pad = slots < 0
    order = torch.cat([slots.reshape(-1), shared_pos])
    return _Crit(P, st.n, a_r, g_priv, g_shared, D, B, S, pad, order, ucols)


def _solve_structured(u: Tensor, c: _Crit):
    """Assemble M's blocks for features ``u`` (V, F), factor, and solve M a = g.

    Returns a dict with the factors and ``a`` (aD (V,R,k), aS (V,ks)),
    ``ga`` = g'a, ``logdetM``, ``rw2`` and ``bad`` (non-PD)."""
    V, F = u.shape
    R, k = c.pad.shape
    ks = c.S.shape[-1]
    eye_pad = torch.diag_embed(c.pad.to(u.dtype))  # (R, k, k): 1 on padded diagonal
    Dm = (u @ c.D.reshape(F, -1)).reshape(V, R, k, k) + eye_pad
    Bm = (u @ c.B.reshape(F, -1)).reshape(V, R, k, ks)
    Sm = (u @ c.S.reshape(F, -1)).reshape(V, ks, ks)
    gD = torch.einsum("vf,vfrk->vrk", u, c.g_priv)
    gS = torch.einsum("vf,vfk->vk", u, c.g_shared)
    rw2 = (u * c.a_r).sum(-1)

    LD, infoD = torch.linalg.cholesky_ex(Dm)
    bad = (infoD != 0).any(-1)
    eye_k = torch.eye(k, dtype=u.dtype, device=u.device)
    LD = torch.where((infoD != 0)[..., None, None], eye_k, LD)
    Y = torch.cholesky_solve(Bm, LD)  # D⁻¹B  (V, R, k, ks)
    zD = torch.cholesky_solve(gD.unsqueeze(-1), LD).squeeze(-1)  # D⁻¹gD
    BtY = torch.einsum("vrki,vrkj->vij", Bm, Y)
    Sig = Sm - BtY
    LS, infoS = torch.linalg.cholesky_ex(Sig)
    bad = bad | (infoS != 0)
    eye_s = torch.eye(ks, dtype=u.dtype, device=u.device)
    LS = torch.where((infoS != 0)[:, None, None], eye_s, LS)
    rhs = gS - torch.einsum("vrki,vrk->vi", Bm, zD)
    aS = torch.cholesky_solve(rhs.unsqueeze(-1), LS).squeeze(-1)
    aD = zD - torch.einsum("vrki,vi->vrk", Y, aS)
    ga = (gD * aD).sum((-1, -2)) + (gS * aS).sum(-1)
    logdetM = 2 * torch.log(torch.diagonal(LD, dim1=-2, dim2=-1)).sum((-1, -2)) + 2 * torch.log(
        torch.diagonal(LS, dim1=-2, dim2=-1)
    ).sum(-1)
    return {"LD": LD, "LS": LS, "Y": Y, "aD": aD, "aS": aS, "ga": ga, "rw2": rw2,
            "logdetM": logdetM, "bad": bad}  # fmt: skip


def _minv_blocks(f: dict, c: _Crit) -> tuple[Tensor, Tensor, Tensor]:
    """M⁻¹ on the structured blocks (the only entries any coefficient touches):
    S-block Σ⁻¹, B-block −D⁻¹BΣ⁻¹, D-blocks D⁻¹ + D⁻¹BΣ⁻¹B'D⁻¹. Padding zeroed."""
    Si = torch.cholesky_inverse(f["LS"])
    Y = f["Y"]
    YS = torch.einsum("vrki,vij->vrkj", Y, Si)
    Mb = -YS
    Md = torch.cholesky_inverse(f["LD"]) + torch.einsum("vrki,vrli->vrkl", YS, Y)
    keep = (~c.pad).to(Md.dtype)
    Md = Md * (keep[:, :, None] * keep[:, None, :])
    return Md, Mb, Si


Stats = LagStats | _Crit


def lag_stats(
    resid: Tensor,
    X: Tensor,
    bounds: Sequence[tuple[int, int]],
    P: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> LagStats:
    """One pass: lagged cross-products of residuals ``resid`` (V, T) and design
    ``X`` (T, K) out to lag ``P``, per run, on each run's own columns. The GEMMs
    run in the inputs' dtype; results are stored in ``dtype`` (float64 by
    default — every later step works on small matrices, where float64 is cheap
    and the likelihood surface is flat)."""
    V, _ = resid.shape
    K = X.shape[1]
    R = len(bounds)
    dev = resid.device
    P1 = P + 1
    Xc = X.to(device=dev, dtype=resid.dtype)
    local = [torch.nonzero(Xc[s:e].abs().amax(dim=0) > 0).squeeze(1) for s, e in bounds]
    kl = max(int(c.numel()) for c in local)
    cols = torch.full((R, kl), -1, dtype=torch.long, device=dev)
    Crr = torch.empty(V, R, P1, P1, dtype=dtype, device=dev)
    SXr = torch.zeros(V, R, P1, P1, kl, dtype=dtype, device=dev)
    SXX = torch.zeros(R, P1, P1, kl, kl, dtype=dtype, device=dev)
    r_head = torch.empty(V, R, P, dtype=dtype, device=dev)
    X_head = torch.zeros(R, P, kl, dtype=dtype, device=dev)
    n = torch.tensor([e - s for s, e in bounds], device=dev)
    for ri, (s, e) in enumerate(bounds):
        m = e - s
        if m <= P:
            raise ValueError(f"run {ri} has {m} timepoints; AR order needs more than P={P}")
        c = local[ri]
        kc = c.numel()
        cols[ri, :kc] = c
        r = resid[:, s:e]
        x = Xc[s:e][:, c]
        # lag l view: samples t-l for t = P..m-1
        rlag = torch.stack([r[:, P - lag : m - lag] for lag in range(P1)], dim=1)  # (V, P1, m-P)
        xlag = torch.stack([x[P - lag : m - lag] for lag in range(P1)], dim=0)  # (P1, m-P, kc)
        Crr[:, ri] = (rlag @ rlag.transpose(1, 2)).to(dtype)
        # S_Xr[v,k,l,:] = xlag[k]' rlag[v,l] — one GEMM: ((P1·kc), m-P) @ (m-P, V·P1)
        xs = xlag.permute(0, 2, 1).reshape(P1 * kc, m - P)
        rs = rlag.permute(2, 0, 1).reshape(m - P, V * P1)
        SXr[:, ri, :, :, :kc] = (xs @ rs).reshape(P1, kc, V, P1).permute(2, 0, 3, 1).to(dtype)
        SXX[ri, :, :, :kc, :kc] = (xs @ xs.T).reshape(P1, kc, P1, kc).permute(0, 2, 1, 3).to(dtype)
        r_head[:, ri] = r[:, :P].to(dtype)
        X_head[ri, :, :kc] = x[:P].to(dtype)
    return LagStats(P, n, Crr, SXr, SXX, r_head, X_head, cols, K)


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


def _feature_index(P: int, device) -> Tensor:
    """Upper-triangle positions of the two symmetric feature blocks in the full
    (P+1)² + P² feature vector."""
    P1 = P + 1
    a, b = torch.triu_indices(P1, P1, device=device)
    c, d = torch.triu_indices(P, P, device=device)
    return torch.cat([a * P1 + b, P1 * P1 + c * P + d])


def _feature_fold(P: int, dtype, device) -> Tensor:
    """(F_full, F) matrix adding each coefficient to its mirror: Σ_kl u_kl A_kl
    = Σ_{k≤l} u_kl (A_kl + A_lk) for symmetric u."""
    P1 = P + 1
    idx = _feature_index(P, device)
    F = torch.zeros(P1 * P1 + P * P, idx.numel(), dtype=dtype, device=device)
    cols = torch.arange(idx.numel(), device=device)
    F[idx, cols] = 1
    # mirror of each kept entry within its block
    a, b = torch.triu_indices(P1, P1, device=device)
    c, d = torch.triu_indices(P, P, device=device)
    mirror = torch.cat([b * P1 + a, P1 * P1 + d * P + c])
    F[mirror, cols] = 1
    return F


def _features(theta: Tensor, st: Stats) -> tuple[Tensor, Tensor]:
    """(u (V, F), log|V| (V,)) for κ = tanh(θ): the per-voxel feature vector the
    quadratic forms are linear in (see :meth:`LagStats.summed`), and log|V|."""
    p, P = theta.shape[-1], st.P
    taps, v = levinson(torch.tanh(theta), P)  # (V, p+1, P+1), (V, p+1)
    c = taps[:, p]  # steady taps
    H = _head_filter(taps, p, P)  # (V, P, P)
    V = theta.shape[0]
    u = torch.cat(
        [(c.unsqueeze(2) * c.unsqueeze(1)).reshape(V, -1), (H.transpose(1, 2) @ H).reshape(V, -1)],
        dim=1,
    )[:, _feature_index(P, theta.device)]
    # per run: Σ_{j<p} log v_j + (n_r − p)·log v_p   (every run is longer than P ≥ p)
    logv = torch.log(v)
    R, N = len(st.n), st.n.sum().to(theta.dtype)
    logdetV = R * logv[:, :p].sum(-1) + (N - R * p) * logv[:, p]
    return u, logdetV


class _Core(torch.autograd.Function):
    """``log|M| + m·log RSS`` from the features u, with an analytic gradient.

    RSS = rw2 − g'M⁻¹g. With a = M⁻¹g: ∂/∂M = M⁻¹ + (m/RSS)·aa',
    ∂/∂g = −2(m/RSS)·a, ∂/∂rw2 = m/RSS, contracted with the (structured)
    coefficients. Non-PD M or RSS ≤ 0 → +inf (a line search backs off it);
    those voxels get zero gradient."""

    @staticmethod
    def forward(ctx, u, c: _Crit, m: int):
        f = _solve_structured(u, c)
        rss = f["rw2"] - f["ga"]
        bad = f["bad"] | ~(rss > 0)
        out = f["logdetM"] + m * torch.log(rss)
        out = torch.where(bad, torch.full_like(out, float("inf")), out)
        ctx.f, ctx.c, ctx.m = f, c, m
        ctx.save_for_backward(rss, bad)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs: Tensor):
        (grad_out,) = grad_outputs
        rss, bad = ctx.saved_tensors
        f, c, m = ctx.f, ctx.c, ctx.m
        g0 = torch.where(bad, torch.zeros_like(grad_out), grad_out)
        w = g0 * m / torch.where(bad, torch.ones_like(rss), rss)
        Md, Mb, Ms = _minv_blocks(f, c)
        aD, aS = f["aD"], f["aS"]
        dD = g0[:, None, None, None] * Md + w[:, None, None, None] * aD.unsqueeze(
            -1
        ) * aD.unsqueeze(-2)
        dB = (
            g0[:, None, None, None] * Mb
            + w[:, None, None, None] * aD.unsqueeze(-1) * aS[:, None, None, :]
        )
        dS = g0[:, None, None] * Ms + w[:, None, None] * aS.unsqueeze(-1) * aS.unsqueeze(-2)
        V, F = c.a_r.shape
        du = dD.reshape(V, -1) @ c.D.reshape(F, -1).T
        du = du + 2 * (dB.reshape(V, -1) @ c.B.reshape(F, -1).T)  # B and B' both in M
        du = du + dS.reshape(V, -1) @ c.S.reshape(F, -1).T
        du = du - 2 * w[:, None] * (
            torch.einsum("vfrk,vrk->vf", c.g_priv, aD) + torch.einsum("vfk,vk->vf", c.g_shared, aS)
        )
        du = du + w[:, None] * c.a_r
        return du, None, None


def reml_criterion(theta: Tensor, st: Stats) -> Tensor:
    """``log|V| + log|X'V⁻¹X| + (N−K)·log RSS`` per voxel (smaller is better) —
    ``glm.arma``'s criterion, for AR(p) with κ = tanh(θ). Non-stationary-looking
    or numerically singular points (|κ| → 1) are +inf."""
    c = st.compact()
    u, logdetV = _features(theta, st)
    m = int(st.n.sum()) - c.n_cols
    return logdetV + _Core.apply(u, c, m)


def _columns(st: LagStats) -> Tensor:
    """Design columns with data in any of these runs (sorted indices)."""
    return st.compact().ucols


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


def _grad(theta: Tensor, st: Stats) -> Tensor:
    """Per-voxel gradient of the criterion. The criterion is a sum of independent
    voxel terms, so one backward pass of the sum gives every voxel's own."""
    with torch.enable_grad():
        x = theta.detach().requires_grad_(True)
        (gr,) = torch.autograd.grad(reml_criterion(x, st).sum(), x)
    return gr


def _hessian(theta: Tensor, st: Stats, h: float = 1e-4) -> Tensor:
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


def theta_covariance(theta: Tensor, st: Stats) -> Tensor:
    """Asymptotic covariance of θ̂ per voxel: the criterion is −2·log-likelihood
    (up to a constant), so the information is half its Hessian."""
    cov, info = torch.linalg.inv_ex(0.5 * _hessian(theta, st))
    # A voxel whose fit never left a non-finite point has a zero Hessian: no
    # information, so no covariance — NaN rather than failing the batch.
    return torch.where((info != 0)[:, None, None], torch.full_like(cov, float("nan")), cov)


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
    st = st.compact()
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
    ``σ² = RSS/(N−K)``. ``beta_ols`` (V, K) must be the fit ``resid`` came from;
    the outputs cover the columns these runs support, in their original order."""
    c = st.compact()
    u, _ = _features(theta, st)
    f = _solve_structured(u, c)
    rss = f["rw2"] - f["ga"]
    sigma2 = rss / (int(st.n.sum()) - c.n_cols)
    Md, _, Ms = _minv_blocks(f, c)
    V = u.shape[0]
    a_struct = torch.cat([f["aD"].reshape(V, -1), f["aS"]], dim=1)
    dg_struct = torch.cat(
        [torch.diagonal(Md, dim1=-2, dim2=-1).reshape(V, -1), torch.diagonal(Ms, dim1=-2, dim2=-1)],
        dim=1,
    )
    real = c.order >= 0
    a = u.new_zeros(V, c.n_cols)
    dg = u.new_zeros(V, c.n_cols)
    a[:, c.order[real]] = a_struct[:, real]
    dg[:, c.order[real]] = dg_struct[:, real]
    beta = beta_ols[:, _columns(st)].to(a.dtype) + a
    var = sigma2[:, None] * dg
    return {"beta": beta, "var": var, "sigma2": sigma2, "t": beta / var.sqrt(), "rss": rss}
