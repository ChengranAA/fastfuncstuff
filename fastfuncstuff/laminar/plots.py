"""Figures for a laminar fit: one page per ROI, the notebook's plots as functions.

Every panel is a function drawing onto an axes it is given, so a figure can be rebuilt
from any subset; :func:`plot_roi` composes the page ``ffs_laminar -plots`` writes.

Depth is drawn the way the model orders it -- superficial (pial side) first -- with the
ends named rather than numbered, so nobody has to remember which way a convention runs:
the tables say "depth 1 = deepest", the model says "k = 0 = surface", and a figure that
shows only the integer would be wrong half the time. Colour runs from superficial
(yellow) to deep (purple) in every panel that has depth.

matplotlib is imported inside the functions, on the Agg backend: a CLI must not need a
display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.laminar.pipeline import RoiFit, depth_names

BEST = "#c44e52"
OTHER = "#4c72b0"
GOOD = "#55a868"


def _req(a, what: str):
    """A figure input of the fit, or a plain error when the fit was made without it."""
    if a is None:
        raise ValueError(f"this RoiFit has no {what}; refit it with fit_roi to plot it")
    return a


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True}
    )
    return plt


def _depth_colours(k: int):
    cmap = _plt().get_cmap("viridis_r")  # superficial (index 0) bright, deep dark
    return [cmap(i / max(k - 1, 1)) for i in range(k)]


def _blocks(fit: RoiFit) -> list[slice]:
    block, gap = int(fit.meta["block"]), int(fit.meta["gap"])
    return [slice(1, block), slice(block + gap + 1, block + gap + block)]


def _ends(ax, axis: str = "y") -> None:
    """Label the two ends of a depth axis instead of numbering it."""
    ticks, labels = [0, 1], ["superficial", "deep"]
    if axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xticks(ticks, labels)


def panel_input(ax, fit: RoiFit) -> None:
    """What the model was given: both responses, the padding between, and what was masked."""
    y = _req(fit.y_obs, "data")
    t = np.arange(len(y))
    for k, c in enumerate(_depth_colours(y.shape[1])):
        ax.plot(t, y[:, k], color=c, lw=1.2)
    ax.fill_between(t, np.nanmin(y) - 0.2, np.nanmax(y) + 0.2, where=~_req(fit.rows, "row mask"), color="0.85",
                    zorder=0, label="masked out of the fit")  # fmt: skip
    ax.set(xlabel="sample (concatenated)", ylabel="signal change (%)", title="Model input")
    ax.legend(fontsize=7, loc="upper right")


def panel_free_energy(ax, fit: RoiFit) -> None:
    """F of every model at every vascular resolution, relative to the best model at that K."""
    rel = fit.per_k_F - fit.per_k_F.max(axis=0, keepdims=True)
    im = ax.imshow(rel, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(fit.k_values)), [f"K={k}" for k in fit.k_values])
    ax.set_yticks(range(len(fit.models)), fit.models, fontsize=7)
    ax.grid(False)
    ax.set_title("Free energy relative to the best model at each K")
    ax.figure.colorbar(im, ax=ax, label="dF (nats)", fraction=0.046)


def panel_model_probability(ax, fit: RoiFit) -> None:
    colours = [BEST if i == fit.best else OTHER for i in range(len(fit.models))]
    ax.bar(range(len(fit.models)), fit.p_model, color=colours)
    ax.set_xticks(range(len(fit.models)), fit.models, rotation=40, ha="right", fontsize=7)
    ax.set(ylabel="posterior probability", title="Model comparison (averaged over K)")


def panel_modulation(ax, fit: RoiFit) -> None:
    """The best model's B per neuronal depth with 90% credible intervals, and the
    probability each depth is modulated at all (marginalised over the model space)."""
    names = depth_names(len(fit.B_mean))
    x = np.arange(len(names))
    ax.bar(x, fit.B_mean, yerr=1.6448536 * fit.B_sd, capsize=4, color=GOOD, alpha=0.85)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, [f"{n}\nP={p:.2f}" for n, p in zip(names, fit.p_depth, strict=True)])
    ax.set(ylabel="B (log change in excitability)",
           title=f"Modulation, '{fit.models[fit.best]}' (P = P(depth modulated), all models)")  # fmt: skip


def panel_fit(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    """Data (dotted) against the model (solid) at every depth, both conditions."""
    y_obs, y_hat = _req(fit.y_obs, "data"), _req(fit.y_hat, "fit")
    cols = _depth_colours(y_obs.shape[1])
    offset = 0
    for name, sl in zip(conditions, _blocks(fit), strict=True):
        x = np.arange(sl.stop - sl.start) + offset
        for k, c in enumerate(cols):
            ax.plot(x, y_obs[sl, k], ":o", color=c, ms=3, lw=1, alpha=0.8)
            ax.plot(x, y_hat[sl, k], "-", color=c, lw=1.8)
        ax.text(x.mean(), ax.get_ylim()[1] * 0.92, name, ha="center", fontsize=8)
        offset = x[-1] + 3
    r2 = fit.r2[fit.best]
    ax.set(xlabel="sample", ylabel="signal change (%)",
           title=f"Data (dotted) and fit (solid); variance explained {r2:.3f}")  # fmt: skip


def _era(fit: RoiFit, which: int) -> tuple[np.ndarray, slice]:
    sl = _blocks(fit)[which]
    return np.arange(sl.stop - sl.start) * float(fit.meta["TR"]), sl


def panel_measured(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    t, sl = _era(fit, 1)
    y = _req(fit.y_obs, "data")
    for k, c in enumerate(_depth_colours(y.shape[1])):
        ax.plot(t, y[sl, k], color=c, lw=1.8)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set(xlabel="time (s)", ylabel="signal change (%)",
           title=f"Measured BOLD, {conditions[1]}")  # fmt: skip


def panel_neuronal(ax, fit: RoiFit, conditions: tuple[str, str]) -> None:
    """The deveined result: the model's neuronal states at each neuronal depth."""
    t, sl = _era(fit, 1)
    for i, name in enumerate(depth_names(fit.neuronal.shape[1])):
        ax.plot(t, fit.neuronal[sl, i], lw=2, label=name)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.legend(fontsize=7)
    ax.set(xlabel="time (s)", ylabel="excitatory activity (a.u.)",
           title=f"Deveined neuronal response, {conditions[1]}")  # fmt: skip


def panel_profiles(ax, fit: RoiFit) -> None:
    """Peak response by depth: the measured BOLD ramps toward the surface, the inferred
    neuronal profile is what is left once the draining is explained."""
    _, sl = _era(fit, 1)
    meas = _req(fit.y_obs, "data")[sl].max(axis=0)
    neur = fit.neuronal[sl].max(axis=0)
    ax.plot(meas, np.linspace(0, 1, len(meas)), "o-", color=BEST, lw=2, label="measured BOLD (%)")
    ax2 = ax.twiny()
    ax2.plot(neur, np.linspace(0, 1, len(neur)), "s-", color=GOOD, lw=2, label="neuronal (a.u.)")
    ax2.grid(False)
    ax.invert_yaxis()
    _ends(ax)
    ax.set(xlabel="peak signal change (%)", title="Depth profile: measured vs deveined")
    ax2.set_xlabel("peak neuronal activity (a.u.)", color=GOOD)
    ax.legend(fontsize=7, loc="lower right")


def panel_operator(ax, fit: RoiFit) -> None:
    """The forward mixing, rows normalised: strictly triangular -- the deepest depth is its own
    neurons, the superficial one draws much of its signal from below. A per-depth scalar
    cannot express that."""
    w = np.abs(fit.W)
    w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-300)
    im = ax.imshow(w, cmap="magma", aspect="auto", vmin=0, vmax=1)
    names = depth_names(w.shape[1])
    ax.set_xticks(range(len(names)), names, rotation=20, ha="right", fontsize=7)
    _ends_rows(ax, w.shape[0])
    ax.grid(False)
    ax.set(
        xlabel="neuronal depth", ylabel="measured depth", title="Forward mixing W (row-normalised)"
    )
    ax.figure.colorbar(im, ax=ax, fraction=0.046)


def _ends_rows(ax, n: int) -> None:
    ax.set_yticks([0, n - 1], ["superficial", "deep"])


def panel_drainage(ax, fit: RoiFit) -> None:
    """How much deeper each depth sources its signal than the depth mapping alone gives --
    only a draining vein can cause it; zero at the deepest depth."""
    shift = _req(fit.drainage_shift, "drainage shift")
    k = len(shift)
    ax.plot(shift, np.linspace(0, 1, k), "o-", color=OTHER, lw=2)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.invert_yaxis()
    _ends(ax)
    ax.set(
        xlabel="neuronal depths deeper than the mapping", title="Vein-attributable contamination"
    )


def panel_transit(ax, fit: RoiFit) -> None:
    """The transit delay and smearing a static operator discards: BOLD at three depths in
    response to the same neuronal drive, summed over neuronal depths."""
    ir = _req(fit.impulse, "impulse response")
    tr = float(fit.meta["TR"])
    t = np.arange(ir.shape[2]) * tr
    k = ir.shape[0]
    for idx, name in ((0, "superficial"), (k // 2, "middle"), (k - 1, "deep")):
        ax.plot(t, ir[idx].sum(axis=0), lw=2, label=name)
    ax.legend(fontsize=7)
    ax.set(xlabel="time after drive (s)", ylabel="BOLD (a.u.)", title="Transit delay and smearing")


def panel_psf(ax, fit: RoiFit) -> None:
    """The depth point-spread function at each vascular resolution: a 1-D kernel as the
    reference builds it, or the central row of a measured sampling matrix."""
    drawn = False
    for k, kern in (fit.kernels or {}).items():
        if kern is None:
            continue
        kern = np.asarray(kern)
        y = kern[kern.shape[0] // 2] if kern.ndim == 2 else kern
        ax.plot(np.linspace(0, 1, 2 * len(y) + 1)[1::2], y, "o-", lw=1.6, label=f"K={k}")
        drawn = True
    if drawn:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "no depth PSF", ha="center", va="center", transform=ax.transAxes)
    ax.set(
        xlabel="depth (superficial to deep)", ylabel="weight", title="Depth point-spread function"
    )


def plot_roi(
    fit: RoiFit, path: str | Path, conditions: tuple[str, str] = ("cond 1", "cond 2")
) -> str:
    """One page for one ROI: input, evidence, modulation, fit, deveining. Returns the path."""
    plt = _plt()
    fig, axes = plt.subplots(4, 3, figsize=(16, 15))
    a = axes.ravel()
    panel_input(a[0], fit)
    panel_free_energy(a[1], fit)
    panel_model_probability(a[2], fit)
    panel_modulation(a[3], fit)
    panel_fit(a[4], fit, conditions)
    panel_profiles(a[5], fit)
    panel_measured(a[6], fit, conditions)
    panel_neuronal(a[7], fit, conditions)
    panel_operator(a[8], fit)
    panel_drainage(a[9], fit)
    panel_transit(a[10], fit)
    panel_psf(a[11], fit)
    fig.suptitle(
        f"{fit.name} ({fit.hemi}):  best model '{fit.models[fit.best]}' "
        f"(p = {fit.p_model[fit.best]:.3f}),  K = {fit.k_values}",
        y=0.995, fontsize=12,
    )  # fmt: skip
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


__all__ = [
    "panel_drainage",
    "panel_fit",
    "panel_free_energy",
    "panel_input",
    "panel_measured",
    "panel_model_probability",
    "panel_modulation",
    "panel_neuronal",
    "panel_operator",
    "panel_profiles",
    "panel_psf",
    "panel_transit",
    "plot_roi",
]
