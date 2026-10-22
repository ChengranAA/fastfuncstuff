"""Temporal similarity of single-trial beta maps (GLMsingle Fig. 5 diagnostic).

Correlating every pair of trial beta maps, in presentation order, shows how much
a trial's estimate leaks into its neighbours: OLS on closely spaced trials gives
anticorrelated successive maps (collinearity), slow shared noise gives positive
correlation that decays over many trials. Comparing OLS against the final ridge
betas shows what regularisation did to that structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


class TopVoxelTracker:
    """Keep the ``n_keep`` best-scoring voxels (and their OLS betas) across chunks.

    Scores arrive chunk by chunk, so a running top-N avoids holding OLS betas for
    every voxel — at production scale that is a second full beta volume.
    """

    def __init__(self, n_keep: int, higher_is_better: bool = True):
        self.n_keep = n_keep
        self.sign = 1.0 if higher_is_better else -1.0
        self.score = torch.empty(0)
        self.index = torch.empty(0, dtype=torch.long)
        self.ols: torch.Tensor | None = None

    def add(self, score: torch.Tensor, voxel_index: torch.Tensor, ols: torch.Tensor) -> None:
        score = torch.nan_to_num(score.detach().float().cpu() * self.sign, nan=-torch.inf)
        ols = ols.detach().float().cpu()
        if self.ols is None:
            self.ols = ols.new_empty((0, ols.shape[1]))
        all_score = torch.cat([self.score, score])
        all_index = torch.cat([self.index, voxel_index.detach().cpu().long()])
        all_ols = torch.cat([self.ols, ols])
        keep = torch.topk(all_score, min(self.n_keep, len(all_score))).indices
        self.score, self.index, self.ols = all_score[keep], all_index[keep], all_ols[keep]


@dataclass
class TemporalSimilarity:
    """Trial-by-trial pattern correlation, in presentation order."""

    matrix: np.ndarray  # (n_trials, n_trials)
    lag_mean: np.ndarray  # (max_lag,) mean correlation of same-run trials k apart
    between_run_mean: float
    run_ids: np.ndarray = field(repr=False)  # (n_trials,) in presentation order


def temporal_similarity(
    betas: np.ndarray, run_ids: np.ndarray, order: np.ndarray, max_lag: int = 10
) -> TemporalSimilarity:
    """Correlate trial beta maps across voxels, ordered by presentation time.

    Each voxel is z-scored across trials first; otherwise the evoked response
    common to every trial makes all pairs correlate and hides the temporal
    structure.
    """
    b = np.asarray(betas, dtype=np.float64)[:, order]
    sd = b.std(axis=1, keepdims=True)
    z = (b - b.mean(axis=1, keepdims=True)) / np.where(sd > 0, sd, 1.0)
    c = np.corrcoef(z.T)
    runs = np.asarray(run_ids)[order]
    same_run = runs[:, None] == runs[None, :]
    pos = np.arange(len(order))
    lag = np.abs(pos[:, None] - pos[None, :])
    lag_mean = np.array(
        [
            c[same_run & (lag == k)].mean() if (same_run & (lag == k)).any() else np.nan
            for k in range(1, max_lag + 1)
        ]
    )
    between = float(c[~same_run].mean()) if (~same_run).any() else float("nan")
    return TemporalSimilarity(c, lag_mean, between, runs)


def write_temporal_similarity_tsv(path: str, results: dict[str, TemporalSimilarity]) -> None:
    names = list(results)
    max_lag = len(next(iter(results.values())).lag_mean)
    with open(path, "w") as fh:
        fh.write("trials_apart\t" + "\t".join(names) + "\n")
        for k in range(max_lag):
            fh.write(
                f"{k + 1}\t" + "\t".join(f"{results[n].lag_mean[k]:.5f}" for n in names) + "\n"
            )
        fh.write(
            "between_runs\t" + "\t".join(f"{results[n].between_run_mean:.5f}" for n in names) + "\n"
        )


def plot_temporal_similarity(
    path: str, results: dict[str, TemporalSimilarity], subtitle: str
) -> None:
    """Matrices side by side (shared diverging scale) over the decay-with-lag curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, muted, surface = "#21211f", "#55554f", "#fcfcfb"
    colors, markers = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], ["o", "s", "^", "D"]
    names = list(results)
    off = np.concatenate([r.matrix[~np.eye(len(r.matrix), dtype=bool)] for r in results.values()])
    lim = float(np.quantile(np.abs(off), 0.99)) or 1.0
    cmap = LinearSegmentedColormap.from_list("div", ["#184f95", "#f0efec", "#b5302f"])

    fig = plt.figure(figsize=(5.5 * len(names) + 1, 9), facecolor=surface)
    gs = fig.add_gridspec(2, len(names), height_ratios=[1.2, 1], hspace=0.35, wspace=0.1)
    for i, name in enumerate(names):
        r = results[name]
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(r.matrix, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        for b in np.flatnonzero(np.diff(r.run_ids)) + 0.5:
            ax.axhline(b, color=muted, lw=0.4)
            ax.axvline(b, color=muted, lw=0.4)
        ax.set_title(name, fontsize=11, color=ink)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("trial (presentation order, runs boxed)", fontsize=9, color=muted)
    fig.text(
        0.5,
        0.505,
        f"pattern correlation between trials; colour scale ±{lim:.2f} (blue negative, red positive)",
        ha="center",
        fontsize=9,
        color=muted,
    )

    ax = fig.add_subplot(gs[1, :])
    max_lag = len(results[names[0]].lag_mean)
    lags = np.arange(1, max_lag + 1)
    ends = []
    for name, c, mk in zip(names, colors, markers, strict=False):
        y = results[name].lag_mean
        ax.plot(lags, y, color=c, lw=2, marker=mk, ms=8, mec=surface, mew=2, label=name)
        ax.axhline(results[name].between_run_mean, color=c, lw=1, ls=":")
        ends.append([float(y[-1]), name])
    # Direct labels, nudged apart so close curve ends never overprint.
    ends.sort()
    ys = np.concatenate([results[n].lag_mean for n in names])
    gap = 0.07 * float(np.nanmax(ys) - np.nanmin(ys) or 1.0)
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y_lab, name in ends:
        ax.text(max_lag + 0.35, y_lab, name, va="center", fontsize=9, color=ink)
    ax.axhline(0, color="#8d8c85", lw=1)
    ax.set_xlim(0.5, max_lag + 2.5)
    ax.set_xticks(lags)
    ax.set_xlabel("trials apart within a run", fontsize=10, color=muted)
    ax.set_ylabel("mean pattern correlation", fontsize=10, color=muted)
    ax.set_title(
        "Correlation between nearby trials (dotted: mean between runs)",
        fontsize=11,
        loc="left",
        color=ink,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.grid(axis="y", color="#e4e3dd", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(surface)
    fig.suptitle(subtitle, fontsize=10, color=muted, y=0.995)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=surface)
    plt.close(fig)
