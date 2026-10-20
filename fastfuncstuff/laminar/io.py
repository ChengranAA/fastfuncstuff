"""ROI x depth response curves from the toolbox, in the shape the laminar model inverts.

``ffs_deconvolve`` (and ``ffs_reml``) on ROIs sampled by depth write tidy curve tables:
``{prefix}_iresp_{condition}.tsv``, one line per time x ROI x depth, depth 1 deepest.
The model wants a ``(n_scans, K)`` matrix in percent signal change, **superficial
depth first**, and -- in the published design ([[Faes 2026]]) -- two conditions'
responses concatenated with white-noise padding that is masked out of the fit,
driven by one event per condition and a modulatory input marking the second. This
module does that conversion and nothing else.

The layout copies the reference driver (``make_modelspace_oracle.m``): each
condition's block is one zero baseline row followed by the response, the event
sits on the response's first sample, the modulatory input is a 0.1 s pulse ending
one sample after the second event, and the baseline and padding rows are masked.

Units are the caller's responsibility to have right at the source: the curves must
be percent signal change (``ffs_deconvolve``'s default since it scales; not with
``-dont_scale``). A curve with a mean far above 10 is refused as data units.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.laminar.params import P0, ModelSpec


@dataclass
class DepthCurves:
    """One ROI's response, ``values`` ``(n_times, K)`` with **depth 1 (deepest) first**,
    as the toolbox writes it."""

    times: np.ndarray
    values: np.ndarray
    roi: str
    name: str
    hemi: str
    coverage: np.ndarray | None = None


def read_iresp_table(path: str | Path) -> dict[tuple[str, str, str], DepthCurves]:
    """Every ROI's curves in a tidy ``_iresp_<condition>.tsv`` (keyed ``(roi, name, hemi)``)."""
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows or "depth" not in rows[0]:
        raise ValueError(f"{path}: not an ROI x depth curve table (no 'depth' column)")
    value_col = list(rows[0])[-1]
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["roi"], r["name"], r.get("hemi", "-")), []).append(r)
    out = {}
    for key, rs in groups.items():
        times = np.array(sorted({float(r["time"]) for r in rs}))
        depths = sorted({int(r["depth"]) for r in rs})
        if depths != list(range(1, len(depths) + 1)):
            raise ValueError(f"{path}: ROI {key[1]} has depths {depths}, not 1..K")
        t_ix = {t: i for i, t in enumerate(times)}
        vals = np.full((len(times), len(depths)), np.nan)
        cov = np.ones(len(depths))
        for r in rs:
            vals[t_ix[float(r["time"])], int(r["depth"]) - 1] = float(r[value_col])
            if "coverage" in r:
                cov[int(r["depth"]) - 1] = float(r["coverage"])
        if np.isnan(vals).any():
            raise ValueError(f"{path}: ROI {key[1]} is missing time x depth entries")
        out[key] = DepthCurves(times, vals, key[0], key[1], key[2], cov)
    return out


def pick_roi(
    table: dict[tuple[str, str, str], DepthCurves], roi: str, hemi: str | None = None
) -> DepthCurves:
    """The ROI whose id or name is ``roi`` (and hemisphere, when two share a name)."""
    hits = [c for (rid, name, h), c in table.items() if roi in (rid, name) and hemi in (None, h)]
    if not hits:
        have = ", ".join(sorted({f"{n} ({h})" for (_, n, h) in table}))
        raise ValueError(f"no ROI {roi!r}{f' in {hemi}' if hemi else ''}; have {have}")
    if len(hits) > 1:
        raise ValueError(f"ROI {roi!r} is in several hemispheres: give hemi")
    return hits[0]


@dataclass
class LaminarData:
    """Everything one inversion needs, in the model's orientation."""

    y: torch.Tensor  # (n_scans, K), percent change, superficial first
    rows: torch.Tensor  # (n_scans,) bool, False on baselines and padding
    u: torch.Tensor  # (n_micro, 2): modulatory (second condition), then driving
    spec: ModelSpec
    #: Keyword arguments for ``integrate`` / ``variational_laplace``.
    integrate_kwargs: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def two_condition_data(
    first: DepthCurves,
    second: DepthCurves,
    *,
    duration: float,
    n_neuronal: int = 3,
    gap: int = 20,
    noise_sd: float = 0.1,
    seed: int = 20260918,
    delay_seconds: float | None = 0.0,
    TE: float = 0.028,
    B0: float = 7.0,
    min_coverage: float = 0.99,
    p0=None,
    dtype: torch.dtype = torch.float64,
) -> LaminarData:
    """Faes et al.'s design from two conditions' curves of one ROI.

    ``duration`` (s) is the event the curves respond to. The scan interval of the
    model is the curves' own sample spacing (the TENT knot spacing), which must be
    uniform and the same for both. ``delay_seconds`` is where in each sample the
    prediction is read: TENT knots sit at exact times after the event, so 0 by
    default; ``None`` restores the reference's slice-timing ``TR / 2``.
    """
    for c in (first, second):
        if c.values.shape[1] != first.values.shape[1]:
            raise ValueError("the two conditions have different numbers of depths")
        if c.coverage is not None and (c.coverage < min_coverage).any():
            raise ValueError(
                f"ROI {c.name}: some depths were read only {100 * c.coverage.min():.0f}% "
                "inside the EPI; a cut depth profile cannot be fitted"
            )
        if np.abs(c.values).mean() > 10:
            raise ValueError(
                f"ROI {c.name}: curves look like data units, not percent change "
                "(rerun ffs_deconvolve without -dont_scale)"
            )
    if not np.array_equal(first.times, second.times):
        raise ValueError("the two conditions are sampled at different times")
    step = np.diff(first.times)
    if step.size == 0 or not np.allclose(step, step[0], rtol=1e-6):
        raise ValueError("curve samples must be evenly spaced (uniform TENT knots)")
    tr = float(step[0])
    k = first.values.shape[1]
    block = len(first.times) + 1  # one zero baseline row, then the response

    rng = np.random.default_rng(seed)
    pad = noise_sd * rng.standard_normal((gap, k))
    flip = slice(None, None, -1)  # depth 1 (deepest) first -> superficial first
    y = np.concatenate(
        [np.zeros((1, k)), first.values[:, flip], pad, np.zeros((1, k)), second.values[:, flip]]
    )
    rows = np.zeros(len(y), bool)
    rows[1:block] = True
    rows[block + gap + 1 :] = True

    spec = ModelSpec(
        N=n_neuronal, K=k, n_inputs=2, n_mod=1, TE=TE, B0=B0, TR=tr, dtype=dtype,
        p0=p0 if p0 is not None else P0(),  # e.g. pipeline.reference_p0(): the driver's baselines
    )  # fmt: skip
    from fastfuncstuff.laminar.integrate import build_input

    onset_a = tr  # the event on the first response sample
    onset_b = (block + gap + 1) * tr
    n_micro = int(round(len(y) * tr / spec.dt))
    # The reference's modulatory pulse: 0.1 s, ending one sample after the second event.
    u = build_input(
        spec, [[onset_b + tr - 0.1], [onset_a, onset_b]], [[0.1], [duration, duration]], n_micro
    )
    kwargs = {} if delay_seconds is None else {"delay_seconds": float(delay_seconds)}
    return LaminarData(
        y=torch.as_tensor(y, dtype=dtype),
        rows=torch.as_tensor(rows),
        u=u,
        spec=spec,
        integrate_kwargs=kwargs,
        meta={"roi": first.name, "hemi": first.hemi, "TR": tr, "K": k, "block": block, "gap": gap},
    )


def read_depth_psf(path: str | Path) -> dict[tuple[str, str], np.ndarray]:
    """A ``*_depth_psf.tsv`` (written by reml/deconvolve/nwarp) as ``{(name, hemi): M}``,
    each ``(K, K)`` with depth 1 deepest first and the weights as written."""
    out: dict[tuple[str, str], dict] = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        out.setdefault((r["name"], r["hemi"]), {})[(int(r["depth"]), int(r["true_depth"]))] = float(
            r["weight"]
        )
    mats = {}
    for key, entries in out.items():
        k = max(d for d, _ in entries)
        m = np.zeros((k, k))
        for (d, j), wgt in entries.items():
            m[d - 1, j - 1] = wgt
        mats[key] = m
    return mats


def model_kernel(
    matrix_deep_first: np.ndarray, *, normalise: bool = True, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """A ``(K, K)`` depth sampling matrix from
    :func:`processing.roi_depth.depth_sampling_matrices` (depth 1 deepest, as the
    toolbox writes it) in the model's superficial-first order, ready for ``kernel=``."""
    m = np.asarray(matrix_deep_first, np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not np.isfinite(m).all():
        raise ValueError("a depth sampling matrix must be a finite (K, K) array")
    if normalise:  # the reference's convention: no dilution by white matter or CSF
        s = m.sum(axis=1, keepdims=True)
        if (s <= 0).any():
            raise ValueError("a depth sampling matrix row sums to zero")
        m = m / s
    return torch.as_tensor(m[::-1, ::-1].copy(), dtype=dtype)


__all__ = [
    "DepthCurves",
    "LaminarData",
    "model_kernel",
    "pick_roi",
    "read_depth_psf",
    "read_iresp_table",
    "two_condition_data",
]
