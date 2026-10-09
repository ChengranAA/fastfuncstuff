"""Run-block whitener: per-run L⁻¹ must equal the dense block-diagonal one.

The dense path factors the whole T×T run-block-diagonal R; the whitener factors
one max-run-length Toeplitz block (or one block per run under censoring) and
reuses leading sub-blocks. Same numbers, so float64 must agree to rounding."""

from __future__ import annotations

import pytest
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance_batch
from fastfuncstuff.glm.whiten import arma11_whiteners, run_bounds, stack_cols, whitener_elements

CPU = torch.device("cpu")
F64 = torch.float64
A = torch.tensor([0.0, 0.3, 0.6, 0.8], dtype=F64)
B = torch.tensor([-0.4, 0.0, 0.2], dtype=F64)


def _dense(n, run_starts, tau):
    R, _, pairs = build_arma11_covariance_batch(A, B, n, CPU, F64, run_starts=run_starts, tau=tau)
    L = torch.linalg.cholesky(R)
    eye = torch.eye(n, dtype=F64).expand(len(pairs), -1, -1)
    L_inv = torch.linalg.solve_triangular(L, eye, upper=False)
    logdet = 2 * torch.log(torch.diagonal(L, dim1=1, dim2=2) + 1e-10).sum(1)
    return L_inv, logdet, pairs


@pytest.mark.parametrize(
    "lens, censor",
    [([50, 50, 50], False), ([40, 60, 40, 25], False), ([40, 60, 40], True)],
    ids=["equal", "unequal", "censored"],
)
def test_whitener_equals_dense_block_diagonal(lens, censor):
    torch.manual_seed(0)
    n = sum(lens)
    run_starts = [sum(lens[:i]) for i in range(len(lens))]
    tau = None
    if censor:  # drop 6 random TRs per run: survivors keep their true time index
        tau = torch.cat([torch.sort(torch.randperm(m + 6)[:m]).values for m in lens])
    L_inv, logdet, pairs = _dense(n, run_starts, tau)
    ws, logdet_w, pairs_w = arma11_whiteners(
        A, B, n, run_starts=run_starts, tau=tau, device=CPU, dtype=F64
    )
    assert pairs_w == pairs
    torch.testing.assert_close(logdet_w, logdet, rtol=0, atol=1e-10)
    X, Y = torch.randn(n, 7, dtype=F64), torch.randn(13, n, dtype=F64)
    for i, w in enumerate(ws):
        torch.testing.assert_close(w.cols(X), L_inv[i] @ X, rtol=0, atol=1e-10)
        torch.testing.assert_close(w.rows(Y), Y @ L_inv[i].T, rtol=0, atol=1e-10)
    torch.testing.assert_close(stack_cols(ws, X), L_inv @ X, rtol=0, atol=1e-10)


def test_storage_is_per_run_not_total_length():
    """The point of the whitener: 17 equal runs store one run's block, not 17²."""
    n, m = 17 * 40, 40
    starts = list(range(0, n, m))
    ws, _, _ = arma11_whiteners(A, B, n, run_starts=starts, device=CPU, dtype=F64)
    assert ws[0].nbytes == m * m * 8
    assert whitener_elements(n, starts) == m * m
    # moving keeps one shared block per pair
    assert ws[0].to(CPU, torch.float32).nbytes == m * m * 4


def test_run_bounds_tolerates_missing_zero_and_single_run():
    assert run_bounds(10, None) == [(0, 10)]
    assert run_bounds(10, [4]) == [(0, 4), (4, 10)]


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_single_pair_whitener_builds_on_the_requested_device():
    """The GLS step builds one whitener per (a,b) on the compute device when the
    search grid was too large to keep; its parameter tensors must live there too."""
    from fastfuncstuff.glm.whiten import arma11_whitener

    w = arma11_whitener(0.4, -0.1, 60, run_starts=[0, 30], device=torch.device("cuda"))
    assert w.device.type == "cuda"
