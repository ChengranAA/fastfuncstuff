"""
Pytest configuration for fastfuncstuff tests.
"""

import os

# Cap CPU threads for the whole test session. Test problems are tiny, so torch's
# default intra-op pool (one thread per core) mostly spins up threads to do
# near-nothing: it momentarily pegs every core without speeding the suite up, and
# it oversubscribes badly when a test spawns its own worker pool. A small fixed cap
# keeps the suite well-behaved (and reduction order deterministic across machines)
# without slowing the little real work there is. The env vars also cover numpy/BLAS
# and are inherited by spawned worker processes. FFS_TEST_NUM_THREADS is the one
# explicit escape hatch; ambient BLAS defaults must not silently claim the machine.
try:
    _TEST_THREADS = max(1, int(os.environ.get("FFS_TEST_NUM_THREADS", "4")))
except ValueError:
    _TEST_THREADS = 4
_TEST_THREADS = min(_TEST_THREADS, os.cpu_count() or 1)
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_var] = str(_TEST_THREADS)

import pytest  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(_TEST_THREADS)
torch.set_num_interop_threads(min(2, _TEST_THREADS))

# Bump dynamo recompile cache size: many tests trigger compiled kernels
# (e.g. glm.xval._cod_kernel_compiled) with varying shapes, dtypes, devices,
# and grad-mode states. The default limit (8) is fine for production but
# fills up across a full test run and turns into FailOnRecompileLimitHit
# pollution in unrelated downstream tests.
try:
    import torch._dynamo

    torch._dynamo.config.cache_size_limit = 256
    torch._dynamo.config.accumulated_cache_size_limit = 1024
except Exception:
    pass


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line(
        "markers", "benchmark_validation: benchmark validation tests (need existing outputs)"
    )
    config.addinivalue_line(
        "markers", "benchmark_full: full benchmark execution tests (need AFNI + data)"
    )


@pytest.fixture(scope="session")
def device():
    """Ordinary tests are CPU-only; GPU tests must opt in with @pytest.mark.gpu."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_backend_by_default(monkeypatch, request):
    """Prevent unmarked tests and library defaults from discovering accelerators."""
    if request.node.get_closest_marker("gpu") is None:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        if hasattr(torch.backends, "mps"):
            monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture(scope="session")
def _surface_cache_root(tmp_path_factory):
    return tmp_path_factory.mktemp("surface_cache")


@pytest.fixture(autouse=True)
def surface_cache_in_tmp(monkeypatch, _surface_cache_root):
    """Keep surface/diskcache entries out of the real ~/.cache.

    Patched rather than via XDG_CACHE_HOME, which also roots the Triton and
    compile-cost caches that GPU tests want warm.
    """
    from fastfuncstuff.surface import diskcache

    monkeypatch.setattr(diskcache, "_dir", lambda kind: _surface_cache_root / kind)


@pytest.fixture(autouse=True)
def reset_random_seed(request):
    """Reset only the RNGs that the selected test is allowed to use."""
    torch.manual_seed(42)
    if request.node.get_closest_marker("gpu") is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture(autouse=True)
def isolate_float32_matmul_precision():
    """Contain the global float32 matmul precision within each test.

    CLI entry points call ``utils.configure_torch_backends``, which runs
    ``torch.set_float32_matmul_precision("high")`` — a good default that
    enables TF32 on CUDA for GLM/registration-style workloads. But it is a
    *global, persistent* switch: once a CLI-exercising test flips it, every
    later test inherits TF32's ~1e-3 matmul error. Precision-sensitive tests
    (e.g. permutation t-stats validated against scipy at 1e-4) then pass or
    fail purely on collection order. Snapshot before / restore after so no
    test leaks the setting into the next; the chain keeps every test at the
    import-time default unless it opts in explicitly.
    """
    prev = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@pytest.fixture(autouse=True)
def isolate_cudnn_tf32():
    """Run every test with cuDNN TF32 off, as the CLIs do (``configure_torch_backends``).

    PyTorch leaves ``cudnn.allow_tf32`` True, and from torch 2.14.1 / cuDNN 9.24 the
    float32 convolutions behind our smoothing and cost terms really use it, so a library
    test that never calls ``configure_torch_backends`` would measure TF32 noise (~1e-3)
    instead of the algorithm.  Restored afterwards so no test leaks the flag.
    """
    prev = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = prev
