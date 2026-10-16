import numpy as np
import pytest

from fastfuncstuff.decomposition.icasso import _clusterstat


def _reference(mat, labels):
    """The pairwise-loop implementation this replaced (GIFT clusterstat.m port)."""
    ids = np.unique(labels)
    K = len(ids)
    internal = np.full(K, np.nan)
    external = np.full(K, np.nan)
    between = np.zeros((K, K))
    sizes = np.zeros(K, dtype=int)
    for i, ci in enumerate(ids):
        mi = labels == ci
        ni = mi.sum()
        sizes[i] = ni
        s_in = mat[np.ix_(mi, mi)]
        if ni > 1:
            internal[i] = s_in[~np.eye(ni, dtype=bool)].mean()
        if K > 1:
            s_out = mat[np.ix_(mi, ~mi)]
            external[i] = s_out.mean() if s_out.size > 0 else np.nan
    for i in range(K):
        mi = labels == ids[i]
        for j in range(i + 1, K):
            vals = mat[np.ix_(mi, labels == ids[j])]
            between[i, j] = vals.mean() if vals.size > 0 else 0.0
    between = between + between.T
    np.fill_diagonal(between, np.inf)
    return sizes, internal, external, between


@pytest.mark.parametrize("n,k", [(40, 5), (23, 23), (30, 1), (60, 12)])
@pytest.mark.parametrize("symmetric", [True, False])
def test_matches_the_pairwise_loop(n, k, symmetric):
    rng = np.random.default_rng(n * 7 + k)
    mat = rng.random((n, n))
    if symmetric:
        mat = (mat + mat.T) / 2
    labels = rng.integers(1, k + 1, n) * 3  # non-contiguous ids; some clusters may be singletons
    if k == n:
        labels = np.arange(1, n + 1)  # all singletons: no intra values at all
    sizes, internal, external, between = _reference(mat, labels)
    got = _clusterstat(mat, labels)
    np.testing.assert_array_equal(got["N"], sizes)
    np.testing.assert_allclose(got["internal"]["avg"], internal, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(got["external"]["avg"], external, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(got["between"]["avg"], between, rtol=1e-12, atol=1e-14)
