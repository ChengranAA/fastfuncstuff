import numpy as np
import pytest

from fastfuncstuff.design.hrf_derive import build_pc_basis_design_per_run


def _reference(onsets_per_run, pcs, lag_times, tr, n_tps, basis, microtime_offset=0.0):
    """The per-PC loop this function used to be: np.interp / np.convolve per column."""
    K, n_lags = pcs.shape
    lo, hi = float(lag_times[0]), float(lag_times[-1])
    out = []
    for onsets, n in zip(onsets_per_run, n_tps, strict=False):
        block = np.zeros((n, K))
        if onsets.size == 0:
            out.append(block)
            continue
        tr_times = np.arange(n) * tr + microtime_offset
        if basis == "FIR":
            idx = np.round((onsets - microtime_offset) / tr).astype(int)
            idx = idx[(idx >= 0) & (idx < n)]
            vec = np.zeros(n)
            vec[idx] = 1.0
            tr_lags = np.arange(n_lags) * tr
            for k in range(K):
                block[:, k] = np.convolve(vec, np.interp(tr_lags, lag_times, pcs[k]))[:n]
        else:
            rel = tr_times[:, None] - onsets[None, :]
            inside = (rel >= lo) & (rel <= hi)
            for k in range(K):
                vals = np.zeros_like(rel)
                if rel[inside].size:
                    vals[inside] = np.interp(rel[inside], lag_times, pcs[k])
                block[:, k] = vals.sum(axis=1)
        out.append(block)
    return out


def _case(seed, n_lags=20, K=7, lag0=0.0, dt=1.75):
    rng = np.random.default_rng(seed)
    lag_times = lag0 + np.arange(n_lags) * dt
    pcs = rng.standard_normal((K, n_lags))
    n_tps = [60, 45, 60]
    onsets = [np.sort(rng.uniform(-5, 120, 12)), np.array([]), np.array([0.0, 0.0, 10.5, 200.0])]
    return pcs, lag_times, onsets, n_tps


@pytest.mark.parametrize("basis", ["FIR", "TENT", "TENTzero"])
@pytest.mark.parametrize(
    ("n_lags", "lag0", "dt", "tr"),
    [
        (20, 0.0, 1.75, 1.75),  # lag grid is TR-spaced
        (25, 0.0, 1.0, 1.75),  # finer grid than TR
        (12, -2.0, 2.0, 1.75),  # does not start at 0: FIR lags extrapolate to end values
        (1, 0.0, 1.0, 1.75),  # a single knot
    ],
)
def test_matches_the_per_pc_loops(basis, n_lags, lag0, dt, tr):
    pcs, lag_times, onsets, n_tps = _case(3, n_lags=n_lags, lag0=lag0, dt=dt)
    kw = dict(tr=tr, n_timepoints_per_run=n_tps, basis=basis, microtime_offset=0.3)
    got = build_pc_basis_design_per_run(onsets, pcs, lag_times, **kw)
    want = _reference(onsets, pcs, lag_times, tr, n_tps, basis, 0.3)
    for g, w in zip(got, want, strict=True):
        assert g.shape == w.shape
        np.testing.assert_allclose(g, w, rtol=0, atol=1e-10)


def test_duplicate_onsets_count_once_in_fir_as_before():
    pcs, lag_times, _, _ = _case(1)
    onsets = [np.array([3.5, 3.5, 3.5, 10.0])]
    got = build_pc_basis_design_per_run(onsets, pcs, lag_times, 1.75, [50], basis="FIR")
    want = _reference(onsets, pcs, lag_times, 1.75, [50], "FIR")
    np.testing.assert_allclose(got[0], want[0], atol=1e-10)
