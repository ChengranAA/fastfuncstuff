"""AR(p) REML from lagged statistics must equal dense REML exactly.

The statistics route never forms V; if any of the head-row filters, the lag
bookkeeping or the log|V| recursion is off, these disagree at O(1), not 1e-13."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance
from fastfuncstuff.glm.arp import fit_ar_reml, gls_from_stats, lag_stats, levinson, reml_criterion

F64 = torch.float64
P = 4


@pytest.fixture(scope="module")
def problem():
    torch.manual_seed(1)
    lens = [60, 45, 70]
    bounds, s = [], 0
    for m in lens:
        bounds.append((s, s + m))
        s += m
    n = s
    X = torch.zeros(n, 3 + 2 * len(lens), dtype=F64)
    X[:, :3] = torch.randn(n, 3, dtype=F64)
    for r, (a, b) in enumerate(bounds):  # per-run constant + linear drift, block-diagonal
        X[a:b, 3 + 2 * r] = 1
        X[a:b, 4 + 2 * r] = torch.linspace(-1, 1, b - a, dtype=F64)
    Y = torch.randn(5, n, dtype=F64).cumsum(1) * 0.3 + torch.randn(5, n, dtype=F64)
    beta_ols = torch.linalg.lstsq(X, Y.T).solution.T
    st = lag_stats(Y - beta_ols @ X.T, X, bounds, P)
    return X, Y, bounds, beta_ols, st


def _dense_whitener(theta, p, bounds, n):
    taps, _ = levinson(torch.tanh(theta), P)
    A = torch.zeros(n, n, dtype=F64)
    for a, b in bounds:
        for t in range(b - a):
            for i in range(min(t, P) + 1):
                A[a + t, a + t - i] = taps[min(t, p), i]
    return A


def _dense_reml(A, X, y):
    Xw, yw = A @ X, A @ y
    beta = torch.linalg.lstsq(Xw, yw).solution
    rss = ((yw - Xw @ beta) ** 2).sum()
    logdet_v = -2 * torch.log(torch.diagonal(A)).sum()
    return logdet_v + torch.logdet(Xw.T @ Xw) + (len(y) - X.shape[1]) * torch.log(rss), beta


@pytest.mark.parametrize("p", [0, 1, 2, P])
def test_stats_criterion_equals_dense_reml(problem, p):
    X, Y, bounds, _, st = problem
    theta = torch.randn(Y.shape[0], p, dtype=F64) * 0.5
    ref = torch.stack(
        [
            _dense_reml(_dense_whitener(theta[v], p, bounds, X.shape[0]), X, Y[v])[0]
            for v in range(5)
        ]
    )
    torch.testing.assert_close(reml_criterion(theta, st), ref, rtol=0, atol=1e-9)


def test_ar1_reproduces_the_arma_grid_point(problem):
    """Levinson's AR(1) and glm.arma's (a, 0) covariance are built independently;
    they must give the same likelihood, or the two noise models cannot be compared."""
    X, Y, bounds, _, st = problem
    a = 0.55
    R = build_arma11_covariance(
        a, 0.0, X.shape[0], torch.device("cpu"), F64, [b[0] for b in bounds]
    )
    L_inv = torch.linalg.inv(torch.linalg.cholesky(R))
    ref = torch.stack([_dense_reml(L_inv, X, Y[v])[0] for v in range(5)])
    theta = torch.full((5, 1), float(np.arctanh(a)), dtype=F64)
    torch.testing.assert_close(reml_criterion(theta, st), ref, rtol=0, atol=1e-9)


def test_newton_reaches_the_dense_optimum_and_gls_matches(problem):
    from scipy.optimize import minimize

    X, Y, bounds, beta_ols, st = problem
    theta, crit = fit_ar_reml(st, 2, tol=1e-10)
    v = 0
    res = minimize(
        lambda x: _dense_reml(_dense_whitener(torch.tensor(x), 2, bounds, X.shape[0]), X, Y[v])[
            0
        ].item(),
        np.zeros(2),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
    )
    assert crit[v].item() == pytest.approx(res.fun, abs=1e-6)
    np.testing.assert_allclose(theta[v].numpy(), res.x, atol=1e-4)
    _, beta = _dense_reml(_dense_whitener(theta[v], 2, bounds, X.shape[0]), X, Y[v])
    torch.testing.assert_close(
        gls_from_stats(theta, st, beta_ols)["beta"][v], beta, rtol=0, atol=1e-10
    )


def test_run_subsets_drop_columns_they_have_no_data_for(problem):
    """A single run's criterion is that run alone with its own columns: the
    other runs' drift columns are zero there and must not make M singular."""
    X, Y, bounds, _, st = problem
    one = st.runs([1])
    theta = torch.zeros(5, 1, dtype=F64)
    a, b = bounds[1]
    Xr = X[a:b][:, X[a:b].abs().amax(0) > 0]
    ref = torch.stack(
        [_dense_reml(torch.eye(b - a, dtype=F64), Xr, Y[v, a:b])[0] for v in range(5)]
    )
    torch.testing.assert_close(reml_criterion(theta, one), ref, rtol=0, atol=1e-9)
