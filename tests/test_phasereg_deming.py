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


def test_slope_cap_never_leaves_a_voxel_noisier():
    """Uncorrelated magnitude/phase with SYY > phi*SXX is where uncapped Deming
    diverges; at 2x OLS the corrected series can be no noisier than the input.
    The cap is opt-in (default off since 2026-10-14) and kept for comparison."""
    from fastfuncstuff.phasereg.core import phase_regress

    g = torch.Generator().manual_seed(0)
    n_vox, n_tp = 400, 120
    pha = 0.01 * torch.randn(n_vox, n_tp, generator=g)
    mag = 1000 + 50 * torch.randn(n_vox, n_tp, generator=g)
    kw = dict(phi=1e4, keep_drift=True, max_poly_degree=0, device="cpu")
    capped = phase_regress(mag, pha, 2.0, max_slope_ratio=2.0, **kw)
    free = phase_regress(mag, pha, 2.0, **kw)
    assert free.slope.abs().max() > 1e5
    ratio = capped.magnitude_corrected.std(dim=1) / mag.std(dim=1)
    assert ratio.max() <= 1.0 + 1e-4


def test_thermal_phi_recovers_c_one_and_the_vein_slope():
    """-phi_method thermal: complex data with pure thermal noise (SD sigma per
    channel), so magnitude SD = sigma and phase SD = sigma/M, phi = M^2 and c = 1.
    c is calibrated on non-vein voxels only; the vein voxels' Deming slope must
    then come out unbiased, where OLS is attenuated by the phase noise."""
    import numpy as np
    import torch

    from fastfuncstuff.phasereg.core import phase_regress

    rng = np.random.default_rng(3)
    n_tp, n_vein, n_cal, m0, sigma, slope = 400, 200, 600, 1000.0, 10.0, 2000.0
    drive = np.convolve(rng.standard_normal(n_tp + 20), np.ones(8) / 8, "same")[:n_tp]
    pha_true = np.zeros((n_tp, n_vein + n_cal))
    pha_true[:, :n_vein] = 0.01 * drive[:, None]
    mag_true = m0 + slope * pha_true
    z = mag_true * np.exp(1j * pha_true)
    z = z + sigma * (rng.standard_normal(z.shape) + 1j * rng.standard_normal(z.shape))
    mag = torch.tensor(np.abs(z).T, dtype=torch.float32)  # (voxels, time)
    pha = torch.tensor(np.angle(z).T, dtype=torch.float32)
    calib = torch.zeros(n_vein + n_cal, dtype=torch.bool)
    calib[n_vein:] = True
    res = phase_regress(
        mag, pha, 2.0, phi_method="thermal", phi_calib=calib, max_slope_ratio=None,
        phase_unwrap=False, device=torch.device("cpu"),
    )  # fmt: skip
    c = float((res.phi[n_vein:] / mag[n_vein:].mean(1) ** 2).median())
    assert abs(c - 1.0) < 0.1
    vein_slope = float(res.slope[:n_vein].median())
    ols = phase_regress(
        mag, pha, 2.0, regression="ols", phase_unwrap=False, device=torch.device("cpu")
    )
    assert abs(vein_slope / slope - 1.0) < 0.1
    assert float(ols.slope[:n_vein].median()) < 0.9 * slope  # attenuated: what Deming fixes


def test_shrink_slope_damps_unreliable_slopes_and_odr_is_phaseprep():
    """-shrink slope (default): s = 1/(1 + A^2/phi) is ~1 in a well-coupled voxel, so
    the full A*phase comes out, and ~0 where Deming fits phase noise, so at most a
    bounded bit of noise is added where -shrink none explodes. -shrink odr is phaseprep's output: the voxel
    mean plus the residual scaled by s."""
    from fastfuncstuff.phasereg.core import phase_regress

    g = torch.Generator().manual_seed(3)
    n_tp = 200
    # 50 coupled voxels (magnitude = -400 * true phase + noise), 400 uncoupled. The
    # coupled signal is slow (< 0.08 Hz at TR 2 s), as vein fluctuations are, so the
    # FFT noise band (> 0.15 Hz) sees only noise.
    t = torch.arange(n_tp, dtype=torch.float32) * 2.0
    f = 0.005 + 0.075 * torch.rand(50, 6, 1, generator=g)
    ph0 = 6.283 * torch.rand(50, 6, 1, generator=g)
    truth = 0.02 * torch.sin(6.283 * f * t + ph0).sum(1)
    pha_c = truth + 0.001 * torch.randn(50, n_tp, generator=g)
    mag_c = 1000 - 400 * truth + 2 * torch.randn(50, n_tp, generator=g)
    pha_u = 0.01 * torch.randn(400, n_tp, generator=g)
    mag_u = 1000 + 50 * torch.randn(400, n_tp, generator=g)
    mag, pha = torch.cat([mag_c, mag_u]), torch.cat([pha_c, pha_u])
    kw = dict(keep_drift=True, max_poly_degree=0, device="cpu")
    slope = phase_regress(mag, pha, 2.0, shrink_mode="slope", **kw)
    full = phase_regress(mag, pha, 2.0, shrink_mode="none", **kw)
    removed = (mag - slope.magnitude_corrected).std(dim=1) / mag.std(dim=1)
    assert float(removed[:50].median()) > 0.9  # the coupled signal comes out
    # Uncoupled: s*A = A*phi/(phi + A^2) peaks at sqrt(phi)/2, so at most a quarter of
    # the noise variance is added (SD x sqrt(1.25)) -- bounded, never an explosion.
    noisier = slope.magnitude_corrected[50:].std(dim=1) / mag[50:].std(dim=1)
    assert float(noisier.max()) < 1.25**0.5 + 0.01 and float(noisier.median()) < 1.03
    assert float((full.magnitude_corrected[50:].std(dim=1) / mag[50:].std(dim=1)).max()) > 10

    odr = phase_regress(
        mag, pha, 2.0, shrink_mode="odr", keep_drift=False, max_poly_degree=0, device="cpu"
    )
    s = 1.0 / (1.0 + odr.slope**2 / odr.phi)
    m0, p0 = mag - mag.mean(1, keepdim=True), pha - pha.mean(1, keepdim=True)
    r = m0 - odr.slope[:, None] * p0
    want = mag.mean(1, keepdim=True) + s[:, None] * r
    assert torch.allclose(odr.magnitude_corrected, want, rtol=1e-4, atol=1e-2)


def test_spr_regression_ols_inside_the_vein_mask_only():
    """-spr_regression ols: inside spr_allow the slope is the plain least-squares one
    (a correction that can never add variance); outside, the global Deming + shrink."""
    from fastfuncstuff.phasereg.core import phase_regress

    g = torch.Generator().manual_seed(5)
    n_vox, n_tp = 300, 150
    pha = 0.01 * torch.randn(n_vox, n_tp, generator=g)
    mag = 1000 - 300 * pha + 10 * torch.randn(n_vox, n_tp, generator=g)
    allow = torch.zeros(n_vox, dtype=torch.bool)
    allow[:150] = True
    kw = dict(keep_drift=True, max_poly_degree=0, device="cpu")
    split = phase_regress(mag, pha, 2.0, spr_allow=allow, spr_regression="ols", **kw)
    glob = phase_regress(mag, pha, 2.0, spr_allow=allow, spr_regression=None, **kw)
    m0, p0 = mag - mag.mean(1, keepdim=True), pha - pha.mean(1, keepdim=True)
    ols = (m0 * p0).sum(1) / (p0 * p0).sum(1)
    assert torch.allclose(split.slope[:150], ols[:150], rtol=1e-4)
    assert torch.equal(split.magnitude_corrected[150:], glob.magnitude_corrected[150:])
    ratio = split.magnitude_corrected[:150].std(1) / mag[:150].std(1)
    assert float(ratio.max()) <= 1.0 + 1e-5  # OLS removes, never adds
