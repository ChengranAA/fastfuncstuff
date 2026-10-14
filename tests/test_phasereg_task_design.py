"""phasereg task model: the design phi and the residual fit are built from."""


def test_canonical_task_design_uses_event_durations():
    """Bug of record: -task_removal canonical modelled every event as an impulse,
    so a 15 s block's plateau stayed in the task residual that phi comes from."""
    import numpy as np
    import torch

    from fastfuncstuff.phasereg.core import _build_task_design_canonical

    onsets = [[np.arange(15.0, 240.0, 30.0)]]
    block = _build_task_design_canonical(onsets, 2.5, [102], torch.device("cpu"), [15.0])[0][:, 0]
    impulse = _build_task_design_canonical(onsets, 2.5, [102], torch.device("cpu"))[0][:, 0]
    # A 15 s block holds near its peak for several TRs; an impulse response does not.
    assert int((block[:18] > 0.8).sum()) >= 4
    assert int((impulse[:18] > 0.8).sum()) == 1


def test_empty_condition_changes_nothing():
    """Bug of record: a condition with no events in a run is an all-zero design
    column, and QR kept an arbitrary direction for it, so projecting the task out
    removed real signal (up to 0.79 scanner units here before the fix)."""
    import numpy as np
    import torch

    from fastfuncstuff.phasereg.core import phase_regress

    rng = np.random.default_rng(0)
    n_tp, n_vox, tr = 120, 50, 2.0
    task = np.sin(np.arange(n_tp) * tr * 2 * np.pi / 40.0)
    pha = 0.05 * task[:, None] + 0.02 * rng.standard_normal((n_tp, n_vox))
    mag = 1000 + 30 * pha / 0.05 + 3 * rng.standard_normal((n_tp, n_vox))
    mag_t = torch.tensor(mag, dtype=torch.float32)
    pha_t = torch.tensor(pha, dtype=torch.float32)
    onsets = [[np.arange(10.0, 230.0, 40.0)]]

    def run(ons):
        return phase_regress(
            mag_t, pha_t, tr, task_removal="canonical", onsets_per_condition=ons,
            durations_per_condition=[20.0] * len(ons), phi_method="residual",
            slope_on="residual", device=torch.device("cpu"),
        ).magnitude_corrected  # fmt: skip

    base = run(onsets)
    padded = run([*onsets, [np.array([])]])
    assert float((base - padded).abs().max()) < 1e-3
