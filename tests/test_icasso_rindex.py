"""R-index scan early exit preserves all defined values and the NaN tail."""

import numpy as np
import pytest


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("max_clusters", [8, None])
def test_rindex_short_circuit_matches_full_scan(tied, max_clusters):
    import importlib

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist, squareform

    module = importlib.import_module("fastfuncstuff.decomposition.icasso")
    points = np.random.default_rng(17).normal(size=(24, 4))
    condensed = pdist(points)
    if tied:
        condensed = np.round(condensed)
    distance = squareform(condensed)
    tree = linkage(condensed, method="average")
    n = max_clusters or distance.shape[0]
    expected = np.full(n, np.nan)
    for k in range(2, n + 1):
        labels = module.fcluster(tree, k, criterion="maxclust")
        if (np.unique(labels, return_counts=True)[1] == 1).any():
            continue
        stat = module._clusterstat(distance, labels)
        expected[k - 1] = (stat["internal"]["avg"] / stat["between"]["avg"].min(axis=1)).mean()
    actual = module.compute_rindex(distance, tree, max_clusters)
    np.testing.assert_array_equal(actual, expected)


def test_rindex_stops_after_first_singleton(monkeypatch):
    import importlib

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist, squareform

    module = importlib.import_module("fastfuncstuff.decomposition.icasso")
    distances = pdist(np.r_[np.linspace(0.0, 1.0, 11), 100.0][:, None])
    tree = linkage(distances, method="average")
    original = module.fcluster
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "fcluster", counted)
    result = module.compute_rindex(squareform(distances), tree)
    assert len(calls) == 1
    assert np.isnan(result).all()
