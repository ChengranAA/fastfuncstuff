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
