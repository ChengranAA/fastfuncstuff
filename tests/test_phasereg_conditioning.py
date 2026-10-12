"""Phase conditioning and slope-fit series for phase regression.

Each test reproduces a way real 7T phase made the slope too small, so veins
survived: wrapped input, a brain-wide per-volume phase wobble, and fitting the
slope after the task-locked vein covariance had been projected out.
"""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.phasereg import phase_regress
from fastfuncstuff.phasereg.conditioning import remove_global_phase, temporal_unwrap

CPU = torch.device("cpu")


def test_temporal_unwrap_recovers_wrapped_series():
    torch.manual_seed(0)
    n_tp, n_vox = 150, 40
    offset = torch.rand(n_vox) * 40 - 20  # static field, many radians
    walk = torch.cumsum(torch.randn(n_tp, n_vox) * 0.4, dim=0)
    truth = offset + walk
    wrapped = torch.remainder(truth + torch.pi, 2 * torch.pi) - torch.pi

    out = temporal_unwrap(wrapped)
    diff = out - truth
    # Equal up to a per-voxel multiple of 2*pi, constant in time.
    assert torch.allclose(diff, diff[:1].expand_as(diff), atol=1e-4)
    k = diff[0] / (2 * torch.pi)
    assert torch.allclose(k, k.round(), atol=1e-4)


def test_temporal_unwrap_leaves_non_wrap_steps_alone():
    # A step that no multiple of 2*pi makes small is not a wrap (motion, seam,
    # noise-floor phase); "correcting" it only adds a random level shift.
    p = torch.zeros(10, 1)
    p[5:] = 3.6  # 3.6 - 2*pi = -2.68: still large either way
    assert torch.equal(temporal_unwrap(p), p)


def test_temporal_unwrap_leaves_continuous_phase_alone():
    torch.manual_seed(1)
    p = torch.randn(100, 20) * 0.3 + 7.0
    out = temporal_unwrap(p, run_lengths=[60, 40])
    for a, b in ((0, 60), (60, 100)):
        d = out[a:b] - p[a:b]
        assert torch.allclose(d, d[:1].expand_as(d), atol=1e-5)


def test_remove_global_phase_keeps_local_signal():
    torch.manual_seed(2)
    n_tp = 120
    g = torch.cumsum(torch.randn(n_tp) * 0.1, 0)
    local = torch.sin(torch.linspace(0, 12, n_tp)) * 0.1
    pha = torch.randn(n_tp, 500) * 0.02 + g[:, None]
    pha[:, 7] += local
    out, term = remove_global_phase(pha, order=0, fit_mask=torch.ones(500, dtype=torch.bool))
    assert torch.corrcoef(torch.stack([term, g]))[0, 1] > 0.99
    r = torch.corrcoef(torch.stack([out[:, 7], local]))[0, 1]
    assert r > 0.9, "a single voxel's local phase must survive global removal"


def _grid_vein_data(seed, global_std=0.0, spontaneous=0.3):
    """Voxels whose magnitude = A * true phase + noise; phase seen with noise."""
    rng = np.random.default_rng(seed)
    n_tp, side = 240, 12
    n_vox = side * side
    tr = 2.0
    t = np.arange(n_tp) * tr
    onsets = np.arange(20.0, t[-1] - 30.0, 40.0)
    box = np.zeros(n_tp)
    for o in onsets:
        box[(t >= o) & (t < o + 20.0)] = 1.0
    task = np.convolve(box, np.exp(-np.arange(10) / 3.0), mode="full")[:n_tp]
    task /= task.std()
    # One voxel in ten is a vein; the rest are grey matter with a magnitude
    # response and no phase response, as the PR premise has it.
    vein = rng.random(n_vox) < 0.1
    true_ph = 0.05 * task[:, None] + spontaneous * 0.05 * rng.standard_normal((n_tp, n_vox))
    true_ph[:, ~vein] = 0.0
    slope = 200.0
    mag = 1000.0 + slope * true_ph + 2.0 * rng.standard_normal((n_tp, n_vox))
    mag[:, ~vein] += 5.0 * task[:, None]
    glob = np.cumsum(rng.standard_normal(n_tp))
    glob = global_std * (glob - glob.mean()) / glob.std()
    pha = true_ph + 0.01 * rng.standard_normal((n_tp, n_vox)) + glob[:, None]
    to_t = lambda a: torch.tensor(a.T, dtype=torch.float32)  # noqa: E731
    return to_t(mag), to_t(pha), tr, (side, side, 1), slope, [[onsets]], torch.from_numpy(vein)


def test_global_phase_wobble_attenuates_slope_and_removal_restores_it():
    mag, pha, tr, shape, slope, _, vein = _grid_vein_data(3, global_std=0.15)
    common = dict(tr=tr, regression="ols", max_poly_degree=2, volume_shape=shape, device=CPU)
    raw = phase_regress(mag, pha, **common).slope[vein].median().item()
    fixed = phase_regress(mag, pha, phase_global_order=0, **common).slope[vein].median().item()
    assert raw < 0.3 * slope, f"expected attenuation, got {raw:.1f}"
    assert abs(fixed - slope) < 0.15 * slope, f"global removal gave {fixed:.1f}"


def test_slope_fit_on_task_preserved_data_with_task_removal():
    # Coupling is almost entirely task-locked: projecting the task out first
    # leaves the slope to a little spontaneous variance under the noise floor.
    mag, pha, tr, shape, slope, onsets, vein = _grid_vein_data(4, spontaneous=0.05)
    common = dict(
        tr=tr,
        regression="ols",
        max_poly_degree=2,
        task_removal="tent",
        tent_window=30.0,
        onsets_per_condition=onsets,
        device=CPU,
    )
    on_data = phase_regress(mag, pha, slope_on="data", **common).slope[vein].median().item()
    on_resid = phase_regress(mag, pha, slope_on="residual", **common).slope[vein].median().item()
    assert abs(on_data - slope) < 0.15 * slope, f"slope_on=data gave {on_data:.1f}"
    assert on_resid < 0.5 * slope, f"residual fit should be attenuated, got {on_resid:.1f}"
