"""Deming regression against a float64 reference of the same closed form."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.phasereg.deming import deming_regression


def _reference_slope(x: np.ndarray, y: np.ndarray, phi: np.ndarray) -> np.ndarray:
    xc, yc = x - x.mean(0), y - y.mean(0)
    sxx, syy, sxy = (xc * xc).sum(0), (yc * yc).sum(0), (xc * yc).sum(0)
    diff = syy - phi * sxx
    return (diff + np.sqrt(diff * diff + 4 * phi * sxy * sxy)) / (2 * sxy)


def test_weak_coupling_at_fmri_phi_is_not_lost_to_cancellation():
    # Raw-unit magnitude (~200 noise) on radian phase (~0.05 noise) with phi
    # ~1e7 and weak coupling: phi*SXX dwarfs SYY, the regime where the
    # textbook numerator diff + sqrt(...) cancels in float32.
    rng = np.random.default_rng(0)
    n_tp, n_vox = 400, 2000
    x = rng.normal(0, 0.05, (n_tp, n_vox))
    y = 30.0 * x * rng.uniform(-1, 1, n_vox) + rng.normal(0, 200, (n_tp, n_vox))
    phi = np.full(n_vox, 1.6e7) * rng.uniform(0.5, 2, n_vox)

    expected = _reference_slope(x, y, phi)
    slope, _ = deming_regression(
        torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(phi).float()
    )
    rel = np.abs(slope.double().numpy() - expected) / np.abs(expected).clip(min=1e-3)
    assert np.median(rel) < 1e-4
    assert rel.max() < 1e-2


def test_strong_coupling_recovers_the_true_slope():
    rng = np.random.default_rng(1)
    n_tp, n_vox = 500, 50
    truth = rng.normal(0, 1, (n_tp, n_vox))
    sx, sy, b1 = 0.1, 0.5, 3.0
    x = truth + rng.normal(0, sx, truth.shape)
    y = b1 * truth + rng.normal(0, sy, truth.shape)
    slope, _ = deming_regression(
        torch.from_numpy(x).float(), torch.from_numpy(y).float(), (sy / sx) ** 2
    )
    assert abs(float(slope.median()) - b1) < 0.05
