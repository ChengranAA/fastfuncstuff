"""Detrended and percent-signal-change frames of a time series, for display."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer.derive import detrend_trace
from fastfuncstuff.viewer.timeview import build_time_view


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    t = np.arange(40.0)
    base = rng.uniform(500, 1500, size=(6, 5, 4))[..., None]
    drift = rng.normal(size=(6, 5, 4))[..., None] * t / 10
    s = base + drift + rng.normal(size=(6, 5, 4, 40))
    s[0, 0, 0] = 0.0  # outside the head
    return s.astype(np.float32)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_every_frame_is_the_detrended_series_at_that_time(series, order):
    view = build_time_view(series, order, psc=False)
    for t in (0, 17, 39):
        frame = view.frame(series[..., t], t)
        want = np.stack([detrend_trace(v, order) for v in series.reshape(-1, 40)])[:, t]
        np.testing.assert_allclose(frame.reshape(-1), want, atol=2e-3)
        assert frame.shape == series.shape[:3]


def test_psc_is_relative_to_each_voxels_mean_and_zero_outside(series):
    raw = build_time_view(series, -1, psc=True)
    mean = series.mean(-1)
    t = 9
    frame = raw.frame(series[..., t], t)
    inside = mean > 1
    np.testing.assert_allclose(
        frame[inside], 100 * (series[..., t] - mean)[inside] / mean[inside], rtol=1e-4, atol=1e-4
    )
    assert frame[0, 0, 0] == 0.0
    # Detrended PSC: the residual over the mean.
    both = build_time_view(series, 1, psc=True)
    resid = build_time_view(series, 1, psc=False).frame(series[..., t], t)
    np.testing.assert_allclose(
        both.frame(series[..., t], t)[inside],
        100 * resid[inside] / mean[inside],
        rtol=1e-3,
        atol=1e-3,
    )


def test_no_detrend_and_no_psc_is_the_raw_frame(series):
    view = build_time_view(series, -1, psc=False)
    assert view.identity
    np.testing.assert_array_equal(view.frame(series[..., 3], 3), series[..., 3])
