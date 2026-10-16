"""Batched VB traces retain binning, lower medians and exceptional-bin behavior."""

import numpy as np
import pytest
import torch

from fastfuncstuff.design.flobs import compute_vb_block_trace


def _legacy(design, covariance, n_blocks, n_basis, lambdas, sigma2, n_bins):
    x = design.double()
    c = torch.from_numpy(np.linalg.inv(covariance)).double()
    p = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float64)
    for b in range(n_blocks):
        p[b * n_basis : (b + 1) * n_basis, b * n_basis : (b + 1) * n_basis] = c
    gram = x.T @ x
    lam = torch.from_numpy(lambdas.astype(np.float64))
    n_bins = max(1, min(n_bins, len(lam)))
    if n_bins > 1:
        qs = torch.linspace(0.0, 1.0, n_bins + 1)[1:-1].double()
        bins = torch.bucketize(lam, torch.quantile(lam, qs))
    else:
        bins = torch.zeros(len(lam), dtype=torch.long)
    values = np.zeros(n_bins)
    matrices = []
    for b in range(n_bins):
        mask = bins == b
        if not mask.any():
            continue
        a = gram + lam[mask].median().item() * p
        matrices.append(a)
        try:
            inv = torch.cholesky_solve(
                torch.eye(len(a), dtype=torch.float64), torch.linalg.cholesky(a)
            )
        except torch.linalg.LinAlgError:
            inv = torch.linalg.pinv(a)
        total = 0.0
        for block in range(n_blocks):
            section = slice(block * n_basis, (block + 1) * n_basis)
            total += (c * inv[section, section]).sum().item()
        values[b] = total
    return (sigma2.astype(np.float64) * values[bins.numpy()]).astype(np.float32), matrices


@pytest.mark.parametrize("n_bins", [1, 7, 20])
@pytest.mark.parametrize("n_basis", [2, 3])
def test_vb_trace_matches_old_bins_and_lower_medians(monkeypatch, n_bins, n_basis):
    g = torch.Generator().manual_seed(181)
    design = torch.randn(60, 3 * n_basis + 2, generator=g, requires_grad=True)
    rng = np.random.default_rng(99)
    a = rng.normal(size=(n_basis, n_basis))
    prior = a @ a.T + np.eye(n_basis)
    lambdas = (np.arange(48) % 17 + 1).astype(np.float64)
    sigma2 = rng.uniform(0.1, 2.0, 48)
    expected, matrices = _legacy(design, prior, 3, n_basis, lambdas, sigma2, n_bins)
    factors = []
    cholesky_ex = torch.linalg.cholesky_ex

    def record(a, *args, **kwargs):
        factors.extend(a.unbind(0))
        return cholesky_ex(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", record)
    actual = compute_vb_block_trace(
        design, prior, 3, n_basis, lambdas, sigma2, n_bins=n_bins, device=torch.device("cpu")
    )
    assert len(factors) == len(matrices)
    for actual_matrix, expected_matrix in zip(factors, matrices, strict=True):
        # Exact matrices prove identical active bins and lower (not averaged) medians.
        torch.testing.assert_close(actual_matrix, expected_matrix, rtol=0, atol=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_vb_trace_falls_back_only_for_rank_deficient_bin(monkeypatch):
    design = torch.zeros(12, 4)
    lambdas = np.repeat([0.0, 1.0, 2.0], 3)
    sigma2 = np.ones(9)
    expected, _ = _legacy(design, np.eye(2), 2, 2, lambdas, sigma2, 3)
    calls = []
    pinv = torch.linalg.pinv

    def record(a, *args, **kwargs):
        calls.append(a.clone())
        return pinv(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "pinv", record)
    actual = compute_vb_block_trace(
        design, np.eye(2), 2, 2, lambdas, sigma2, n_bins=3, device=torch.device("cpu")
    )
    assert len(calls) == 1
    assert torch.count_nonzero(calls[0]) == 0
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_vb_trace_injected_failure_keeps_other_bins_batched(monkeypatch):
    design = torch.zeros(12, 4)
    lambdas = np.repeat([1.0, 2.0, 3.0], 3)
    sigma2 = np.ones(9)
    expected, _ = _legacy(design, np.eye(2), 2, 2, lambdas, sigma2, 3)
    cholesky_ex, pinv = torch.linalg.cholesky_ex, torch.linalg.pinv
    calls = []

    def fail_one(a, *args, **kwargs):
        l, info = cholesky_ex(a, *args, **kwargs)
        l[1] = float("nan")
        info[1] = 1
        return l, info

    def record(a, *args, **kwargs):
        calls.append(a.clone())
        return pinv(a, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "cholesky_ex", fail_one)
    monkeypatch.setattr(torch.linalg, "pinv", record)
    actual = compute_vb_block_trace(
        design, np.eye(2), 2, 2, lambdas, sigma2, n_bins=3, device=torch.device("cpu")
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], 2.0 * torch.eye(4, dtype=torch.float64), rtol=0, atol=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


def test_vb_trace_common_path_transfers_only_once(monkeypatch):
    transfers = []
    cpu = torch.Tensor.cpu

    def record(tensor, *args, **kwargs):
        transfers.append(tensor.shape)
        return cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record)
    compute_vb_block_trace(
        torch.eye(4),
        np.eye(2),
        2,
        2,
        np.arange(1.0, 10.0),
        np.ones(9),
        n_bins=3,
        device=torch.device("cpu"),
    )
    assert transfers == [torch.Size([3, 2])]


def test_vb_trace_planner_can_limit_factor_batch(monkeypatch):
    from fastfuncstuff import memory

    calls = []

    def one_bin(**kwargs):
        calls.append(kwargs)
        return 1

    monkeypatch.setattr(memory, "estimate_chunk_size", one_bin)
    design = torch.eye(4)
    lambdas, sigma2 = np.arange(1.0, 10.0), np.ones(9)
    expected, _ = _legacy(design, np.eye(2), 2, 2, lambdas, sigma2, 3)
    actual = compute_vb_block_trace(
        design, np.eye(2), 2, 2, lambdas, sigma2, n_bins=3, device=torch.device("cpu")
    )
    assert calls[0]["operation"] == "vb_block_trace"
    assert calls[0]["use_double"] is True
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
