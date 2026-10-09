"""Blurring TO a smoothness: the time-point sample, the residual master, the search."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.stats.blur_to import residual_master, sample_columns
from fastfuncstuff.surface.acf import blur_to_fwhm, detrend
from fastfuncstuff.surface.mesh import vertex_areas
from fastfuncstuff.surface.smooth import HeatSmoother, surface_fwhm


def _sheet(n: int, s: float):
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    v = np.c_[ii.ravel() * s, jj.ravel() * s, np.zeros(n * n)]
    idx = ii * n + jj
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
    f = np.r_[np.c_[a.ravel(), b.ravel(), c.ravel()], np.c_[b.ravel(), d.ravel(), c.ravel()]]
    return v, f.astype(np.int64)


@pytest.fixture(scope="module")
def sheet():
    v, f = _sheet(101, 0.8)
    inner = (np.abs(v[:, 0] - 40) < 30) & (np.abs(v[:, 1] - 40) < 30)
    return v, f, inner


def _noise(v, f, t, fwhm, seed):
    white = np.random.default_rng(seed).normal(size=(len(v), t))
    return HeatSmoother(v, f, fwhm)(white / np.sqrt(vertex_areas(v, f))[:, None])


def _design(lengths, task):
    """Per-run Legendre drift (degree 2) plus one task column across all runs."""
    t = sum(lengths)
    x = np.zeros((t, 3 * len(lengths) + 1))
    s = 0
    for r, n in enumerate(lengths):
        x[s : s + n, 3 * r : 3 * r + 3] = np.polynomial.legendre.legvander(np.linspace(-1, 1, n), 2)
        s += n
    x[:, -1] = task
    return x


def test_sample_spreads_over_runs_and_never_takes_a_censored_point():
    lengths = [100, 300, 200]
    good = np.r_[np.arange(0, 50), np.arange(100, 600)]  # second half of run 1 censored
    cols = sample_columns(lengths, 40, good=good, seed=3)
    assert len(cols) == 40 and len(np.unique(cols)) == 40
    assert np.isin(cols, good).all()
    per_run = np.histogram(cols, bins=[0, 100, 400, 600])[0]
    # kept lengths 50 : 300 : 200 -> about 3.6 : 21.8 : 14.5
    assert per_run.tolist() in ([4, 22, 14], [4, 21, 15], [3, 22, 15])
    assert sample_columns(lengths, 10_000, good=good).tolist() == good.tolist()


def test_residual_master_is_the_ols_residual_with_censored_rows_out_of_the_fit():
    rng = np.random.default_rng(0)
    lengths = [60, 80]
    x = _design(lengths, rng.normal(size=140))
    y = rng.normal(size=(7, 140)) + 50
    y[:, 10] += 1e4  # a spike, censored: it must not move the fit
    good = np.delete(np.arange(140), 10)
    cols = np.array([3, 70, 139])
    beta = np.linalg.lstsq(x[good], y[:, good].T, rcond=None)[0]
    want = (y - (x @ beta).T)[:, cols]
    np.testing.assert_allclose(residual_master(y, x, cols, good=good, chunk=3), want, atol=1e-6)


def test_blurring_commutes_with_the_residual_so_no_refit_is_needed(sheet):
    v, f, inner = sheet
    lengths = [50, 50]
    rng = np.random.default_rng(1)
    x = _design(lengths, rng.normal(size=100))
    y = _noise(v, f, 100, 2.0, 1) + np.outer(rng.normal(size=len(v)), x[:, -1])
    hs = HeatSmoother(v, f, 4.0, mask=inner)
    cols = sample_columns(lengths, 12)
    np.testing.assert_allclose(
        residual_master(hs(y), x, cols), hs(residual_master(y, x, cols)), atol=1e-4
    )


def test_a_detrended_master_mistakes_the_task_for_smoothness_and_residuals_do_not(sheet):
    # A broad task activation is spatially smooth: left in the master, it reads as
    # extra smoothness, and blur-to then under-blurs. The FWHM moves only a few
    # percent, but near the target the kernel is a difference of squares, so the
    # kernel moves several times as much (here ~17%).
    v, f, inner = sheet
    lengths = [120, 120]
    t = np.arange(240)
    task = ((t % 30) < 15).astype(float)
    x = _design(lengths, task)
    noise = _noise(v, f, 240, 3.0, 2)
    noise /= noise[inner].std()
    blob = np.exp(-((v[:, 0] - 40) ** 2 + (v[:, 1] - 40) ** 2) / (2 * 15.0**2))
    y = noise + 1.5 * np.outer(blob, task)
    truth = surface_fwhm(noise, v, f, inner)[0]
    cols = sample_columns(lengths, 40)
    resid = residual_master(y, x, cols)
    detr = detrend(y, lengths, degree=2)[:, cols]
    assert surface_fwhm(resid, v, f, inner)[0] == pytest.approx(truth, rel=0.03)
    assert surface_fwhm(detr, v, f, inner)[0] > 1.03 * truth
    _, k_resid, _ = blur_to_fwhm(resid, v, f, 6.0, inner)
    _, k_detr, _ = blur_to_fwhm(detr, v, f, 6.0, inner)
    assert k_detr < 0.9 * k_resid


def test_forty_time_points_pin_the_kernel(sheet):
    # The default -blur_to_columns: across seeds, the kernel 40 points choose stays
    # within a few percent of the one all 400 choose.
    v, f, inner = sheet
    lengths = [100] * 4
    x = _design(lengths, np.zeros(400))[:, :-1]
    y = _noise(v, f, 400, 2.5, 5)
    _, k_all, _ = blur_to_fwhm(
        residual_master(y, x, np.arange(400)), v, f, 6.0, inner, n_columns=400, tol=0.005
    )
    ks = [
        blur_to_fwhm(
            residual_master(y, x, sample_columns(lengths, 40, seed=s)), v, f, 6.0, inner, tol=0.005
        )[1]
        for s in range(5)
    ]
    assert np.max(np.abs(np.array(ks) / k_all - 1)) < 0.04
