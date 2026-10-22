"""Temporal similarity diagnostic: chunked top-N selection and lag structure."""

import numpy as np
import torch

from fastfuncstuff.glm.temporal_similarity import (
    TopVoxelTracker,
    plot_temporal_similarity,
    temporal_similarity,
    write_temporal_similarity_tsv,
)


def test_tracker_matches_global_topk_across_chunks():
    rng = np.random.default_rng(0)
    score = torch.from_numpy(rng.normal(size=1000)).float()
    ols = torch.from_numpy(rng.normal(size=(1000, 7))).float()
    for higher in (True, False):
        tracker = TopVoxelTracker(50, higher_is_better=higher)
        for c0 in range(0, 1000, 137):  # chunk edges that don't align with n_keep
            idx = torch.arange(c0, min(c0 + 137, 1000))
            tracker.add(score[idx], idx, ols[idx])
        expected = torch.topk(score if higher else -score, 50).indices
        assert set(tracker.index.tolist()) == set(expected.tolist())
        assert torch.equal(tracker.ols, ols[tracker.index])


def test_nan_scores_are_never_kept():
    tracker = TopVoxelTracker(3)
    score = torch.tensor([float("nan"), 0.1, float("nan"), 0.2, 0.3])
    tracker.add(score, torch.arange(5), torch.zeros(5, 2))
    assert sorted(tracker.index.tolist()) == [1, 3, 4]


def test_shared_slow_noise_shows_as_positive_short_lag_correlation():
    # Two runs of 40 trials, presented in shuffled column order. Each run has
    # its own spatial pattern with an AR(1) amplitude, so nearby trials are
    # alike, the likeness decays with distance, and runs are independent.
    rng = np.random.default_rng(1)
    n_vox, n_per_run = 500, 40
    cols, runs = [], []
    for run in range(2):
        pattern = rng.normal(size=n_vox)
        amp = np.zeros(n_per_run)
        for t in range(1, n_per_run):
            amp[t] = 0.7 * amp[t - 1] + rng.normal()
        for t in range(n_per_run):
            cols.append(amp[t] * pattern + rng.normal(size=n_vox))
            runs.append(run)
    betas = np.stack(cols, 1)
    runs = np.array(runs)
    shuffle = rng.permutation(betas.shape[1])
    order = np.argsort(shuffle)  # presentation order of the shuffled columns
    res = temporal_similarity(betas[:, shuffle], runs[shuffle], order, max_lag=5)
    assert res.lag_mean[0] > 0.2
    assert res.lag_mean[0] > res.lag_mean[-1]
    assert abs(res.between_run_mean) < res.lag_mean[0] / 2
    np.testing.assert_allclose(np.diag(res.matrix), 1.0)
    assert np.all(np.diff(res.run_ids) >= 0)


def test_outputs_written(tmp_path):
    rng = np.random.default_rng(2)
    runs = np.repeat([0, 1], 12)
    res = temporal_similarity(rng.normal(size=(30, 24)), runs, np.arange(24), max_lag=4)
    results = {"OLS": res, "ridge": res}
    write_temporal_similarity_tsv(str(tmp_path / "ts.tsv"), results)
    lines = (tmp_path / "ts.tsv").read_text().splitlines()
    assert lines[0] == "trials_apart\tOLS\tridge"
    assert len(lines) == 1 + 4 + 1
    plot_temporal_similarity(str(tmp_path / "ts.png"), results, "test")
    assert (tmp_path / "ts.png").stat().st_size > 0
