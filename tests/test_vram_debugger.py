import pytest
import torch

from fastfuncstuff.memory import (
    VRAMDebugger,
    compute_moco_resample_batch_size,
    moco_resample_bytes,
)


def test_resample_prediction_and_planner_share_one_model(monkeypatch):
    """The debugger once compared reality to (3 + B) volumes while the planner
    sized the batch with 5 per volume plus kernel overhead, so a correct plan was
    reported as "under-predicted 2.6x"."""
    cpu = torch.device("cpu")
    budget = 8 * 1024**3
    monkeypatch.setattr("fastfuncstuff.memory.get_available_memory", lambda d: budget)
    shape = (90, 100, 80)
    for interp in ("wsinc5", "cubic", "linear"):
        batch = compute_moco_resample_batch_size(*shape, 10_000, cpu, interp=interp)
        assert batch >= 1
        assert moco_resample_bytes(*shape, batch, interp) <= budget
        assert moco_resample_bytes(*shape, batch + 1, interp) > budget


@pytest.mark.gpu
def test_debugger_reports_the_allocation_it_watched():
    """A sampling thread got one sample in 4.7 s of a GPU loop and reported 0.00x;
    the allocator's own peak counter cannot miss a short-lived buffer."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    dev = torch.device("cuda")
    nbytes = 256 * 1024**2
    dbg = VRAMDebugger(dev, nbytes, operation="test")
    dbg.start()
    buf = torch.empty(nbytes // 4, dtype=torch.float32, device=dev)
    del buf  # freed before stop(): only a peak counter can still see it
    stats = dbg.stop()
    assert stats["net_peak_gb"] * 1024**3 == pytest.approx(nbytes, rel=0.05)
    assert stats["ratio"] == pytest.approx(1.0, rel=0.05)
