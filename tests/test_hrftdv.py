"""ffs_hrftdv primitives: lag/width recovery, column pairs, patches, the pair library."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing import hrftdv as tdv

CPU = torch.device("cpu")
DT, TR = 0.1, 1.0


def _hrf() -> np.ndarray:
    from scipy.stats import gamma

    t = np.arange(0, 30, DT)
    h = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    return h / h.max()


def _events(n_runs=2, T=200, seed=0):
    """Jittered 2 s events for 2 conditions, per run (onsets in seconds)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_runs):
        on = np.cumsum(rng.uniform(5, 9, 40))
        on = on[on < T * TR - 20]
        cond = rng.integers(0, 2, on.size)
        out.append([on[cond == c] for c in range(2)])
    return out


def _design_for(events, T, dur=2.0):
    n_fine = int(round(T * TR / DT))
    step = int(round(TR / DT))

    def build(curves: np.ndarray) -> torch.Tensor:
        curves = np.atleast_2d(curves)
        runs = []
        for ev in events:
            cols = []
            for on in ev:
                box = np.zeros(n_fine)
                for o in on:
                    box[int(o / DT) : int((o + dur) / DT)] = 1.0
                for c in curves:  # condition-major, curve-minor, as build_task_design
                    cols.append(np.convolve(box, c)[:n_fine][::step] * DT)
            runs.append(np.stack(cols, 1))
        return torch.as_tensor(np.vstack(runs))

    return build


def _projector(n_runs, T):
    q = tdv.orthonormal_nuisance([torch.ones(T, 1)] * n_runs, CPU)
    starts = [i * T for i in range(n_runs)]
    return lambda x: tdv.project_runs(x.double(), q, starts), q, starts


def test_stretch_shift_moves_the_peak_by_lag_and_keeps_it_under_stretch():
    h = _hrf()
    t_pk = np.argmax(h) * DT
    for lag, w in [(0.0, 1.0), (1.3, 1.0), (-0.8, 1.3), (0.5, 0.75)]:
        c = tdv.stretch_shift(h, DT, lag, w)
        assert abs(np.argmax(c) * DT - (t_pk + lag)) < DT + 1e-9
        assert c.max() == pytest.approx(1.0)
    np.testing.assert_allclose(tdv.stretch_shift(h, DT, 0.0, 1.0), h, atol=1e-12)
    # wider stretch about the peak -> larger full width at half maximum
    fwhm = [np.sum(tdv.stretch_shift(h, DT, 0.0, w) > 0.5) for w in (0.75, 1.0, 1.3)]
    assert fwhm[0] < fwhm[1] < fwhm[2]


def test_fit_lag_width_recovers_known_lags():
    T, n_runs = 200, 2
    h = _hrf()
    events = _events(n_runs, T)
    build = _design_for(events, T)
    project, _, _ = _projector(n_runs, T)
    grid = tdv.LagGrid.make(2.0, 0.1, (0.8, 1.0, 1.25))
    rng = np.random.default_rng(1)
    true = rng.uniform(-1.5, 1.5, 60)
    ys = []
    for lag in true:
        x = build(tdv.stretch_shift(h, DT, lag, 1.0)).numpy()
        ys.append(x @ np.array([1.0, 0.6]) + 0.05 * rng.standard_normal(T * n_runs))
    y = project(torch.as_tensor(np.stack(ys, 1))[None])[0].float()
    q = tdv.bank_q(project(tdv.as_bank(build(tdv.lag_width_bank(h, DT, grid)), grid.n)))
    fit = tdv.fit_lag_width(y, q, grid, CPU)
    err = fit.lag - true
    assert np.sqrt(np.mean(err**2)) < 0.12, err
    assert np.mean(fit.width == 1.0) > 0.7
    assert not fit.at_bound.any()


def test_fit_lag_width_flags_lags_beyond_the_grid():
    T, n_runs = 200, 2
    h = _hrf()
    build = _design_for(_events(n_runs, T), T)
    project, _, _ = _projector(n_runs, T)
    grid = tdv.LagGrid.make(1.0, 0.1, (1.0,))
    y = build(tdv.stretch_shift(h, DT, 2.5, 1.0)) @ torch.tensor([1.0, 1.0], dtype=torch.float64)
    y = project(y[None, :, None])[0].float()
    q = tdv.bank_q(project(tdv.as_bank(build(tdv.lag_width_bank(h, DT, grid)), grid.n)))
    fit = tdv.fit_lag_width(y, q, grid, CPU)
    assert fit.at_bound[0] and fit.lag[0] == pytest.approx(1.0)


def test_column_pair_takes_weighted_percentiles_not_extremes():
    rng = np.random.default_rng(0)
    lag = np.r_[rng.normal(-0.5, 0.1, 80), rng.normal(0.8, 0.1, 20), [5.0]]  # one wild voxel
    width = np.r_[np.full(80, 0.875), np.full(20, 1.3), [1.0]]
    r2 = np.r_[np.full(100, 0.5), [0.01]]
    p = tdv.column_pair(lag, width, r2, (10, 90), tail=15)
    assert p is not None
    assert -0.7 < p.early_lag < -0.4 and 0.5 < p.late_lag < 1.0
    assert p.early_width == 0.875 and p.late_width == 1.3
    assert tdv.column_pair(lag[:5], width[:5], r2[:5], min_voxels=8) is None


def test_surface_patches_are_geodesic_and_cover_the_seeds():
    """A plane folded into a U: the two arms are 2 mm apart in space but 20 mm
    along the mesh, so no patch may hold vertices of both arms."""
    n = 41
    s = np.linspace(0, 40, n)
    xs, ys = np.meshgrid(s, np.linspace(0, 10, 11), indexing="ij")
    # fold: x in [0,20] is arm A at z=0; x in [20,40] folds back over at z=2
    xa = np.where(xs <= 20, xs, 40 - xs)
    z = np.where(xs <= 20, 0.0, 2.0)
    verts = np.c_[xa.ravel(), ys.ravel(), z.ravel()]
    idx = np.arange(verts.shape[0]).reshape(xs.shape)
    faces = []
    for i in range(n - 1):
        for j in range(10):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i, j + 1], idx[i + 1, j + 1]
            faces += [(a, b, c), (b, d, c)]
    faces = np.array(faces)
    labels = tdv.surface_patches(verts, faces, np.arange(verts.shape[0]), radius_mm=3.0)
    assert (labels >= 0).all()
    arm = (xs <= 19).ravel(), (xs >= 21).ravel()
    for lab in np.unique(labels):
        m = labels == lab
        assert not ((m & arm[0]).any() and (m & arm[1]).any()), "patch crossed the fold"


def test_extend_labels_splits_a_gap_between_two_banks():
    lab = np.full((9, 1, 1), -1)
    lab[0], lab[8] = 0, 1
    out = tdv.extend_labels(lab, (1.0, 1.0, 1.0), extend_mm=3.0)[:, 0, 0]
    assert list(out) == [0, 0, 0, 0, -1, 1, 1, 1, 1]


def test_volume_tiles_group_by_cube():
    ijk = np.array([[0, 0, 0], [1, 1, 1], [4, 0, 0], [3, 3, 3]])
    lab = tdv.volume_tiles(ijk, (1.0, 1.0, 1.0), 4.0)
    assert lab[0] == lab[1] == lab[3] and lab[2] != lab[0]


def test_pair_library_snaps_and_shares_entries():
    shapes = np.stack([_hrf(), tdv.stretch_shift(_hrf(), DT, 1.0, 1.0)])
    pairs = [
        tdv.ColumnPair(-0.42, 0.61, 0.9, 1.2, 10),
        tdv.ColumnPair(-0.38, 0.58, 0.85, 1.18, 12),  # same entry after snapping
        tdv.ColumnPair(-0.42, 0.61, 0.9, 1.2, 10),  # same timing, other curve
    ]
    lib = tdv.build_pair_library(shapes, np.array([0, 0, 1]), pairs, DT, 0.2, tdv.DEFAULT_WIDTHS)
    assert lib.curves.shape == (2, 2, shapes.shape[1])
    assert lib.index[0] == lib.index[1] != lib.index[2]
    e = lib.entries[lib.index[0]]
    assert e[1] == pytest.approx(-0.4) and e[2] == pytest.approx(0.6)
    assert e[3] == 0.875 and e[4] == 1.15
    early, late = lib.curves[lib.index[0]]
    assert np.argmax(late) > np.argmax(early)


def test_fit_hrftdv_finds_a_later_pair_where_voxels_are_later():
    """Two units: one with voxels spread over lags -0.6..0.6 s, the other all late
    (0.4..1.4 s). The second unit's pair must sit later; every voxel gets an entry."""
    T, n_runs = 200, 2
    h = _hrf()
    events = _events(n_runs, T, seed=3)
    build = _design_for(events, T)
    project, _, _ = _projector(n_runs, T)
    rng = np.random.default_rng(2)
    lags = np.r_[np.linspace(-0.6, 0.6, 30), np.linspace(0.4, 1.4, 30)]
    unit = np.r_[np.zeros(30, int), np.ones(30, int)]
    ys = [
        build(tdv.stretch_shift(h, DT, lag, 1.0)).numpy() @ np.array([1.0, 0.7])
        + 0.05 * rng.standard_normal(T * n_runs)
        for lag in lags
    ]
    y = project(torch.as_tensor(np.stack(ys, 1))[None])[0].float()
    shapes = np.stack([h, tdv.stretch_shift(h, DT, 0.0, 1.3)])
    res = tdv.fit_hrftdv(
        y,
        unit,
        np.zeros(2, int),
        shapes,
        build,
        project,
        np.ones(60, bool),
        np.ones(60),
        tdv.LagGrid.make(2.5, 0.1, (0.875, 1.0, 1.15)),
        DT,
        min_voxels=8,
        lag_step=0.1,
        device=CPU,
        verbose=False,
    )
    a, b = res.pairs
    assert not a.fallback and not b.fallback
    assert b.early_lag > a.early_lag + 0.5 and b.late_lag > a.late_lag + 0.5
    assert a.early_lag < 0 < a.late_lag
    assert res.voxel_entry.shape == (60,) and set(res.voxel_entry) == {0, 1}
    assert np.nanmax(np.abs(res.rel_lag[:30])) < 0.9
