"""FLOBS whole-series paths: offloaded + chunked must match resident.

``cv_basis_constrained_ridge``, ``fit_basis_lss``, ``fit_basis_fracridge`` and
``estimate_arma11_per_voxel`` keep the concatenated series on the compute device
only when it fits; otherwise it stays on the CPU and voxel chunks stream to the
device. The streamed path must give the resident path's numbers.

No GPU here: the compute device is ``cpu:0``, which torch treats as a different
device from where CPU tensors report living (``cpu``), so every residency
comparison takes the offloaded branch while the arithmetic stays on the CPU. The
chunk planner is pinned to a tiny chunk so the streaming loops really iterate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import fastfuncstuff.memory as memory
from fastfuncstuff.design.flobs import (
    cv_basis_constrained_ridge,
    estimate_arma11_per_voxel,
    fit_basis_fracridge,
    fit_basis_lss,
    generate_flobs_basis,
)

CPU = torch.device("cpu")
OFFLOAD = torch.device("cpu", 0)
N_VOX = 23  # not a multiple of the chunk, so the ragged last chunk is covered


@pytest.fixture(scope="module")
def basis():
    return generate_flobs_basis(n_basis=3, n_samples=500, duration=32.0, dt=0.1, seed=3)


@pytest.fixture
def tiny_chunks(monkeypatch):
    monkeypatch.setattr(memory, "estimate_chunk_size", lambda *a, **k: 5)


def _runs(basis, n_trials_per_run=1, n_runs=3, n_tp=90, seed=0, rho=None):
    """Per-run data and a design with one K-column block per trial."""
    rng = np.random.default_rng(seed)
    K = basis.basis_functions.shape[0]
    hrf = basis.basis_functions[:, :: int(round(1.0 / basis.dt))]
    datas, designs = [], []
    for _ in range(n_runs):
        onsets = np.sort(rng.choice(np.arange(5, n_tp - 30), n_trials_per_run * 2, replace=False))
        X = np.zeros((n_tp, n_trials_per_run * K))
        for t in range(n_trials_per_run):
            on = np.zeros(n_tp)
            on[onsets[2 * t : 2 * t + 2]] = 1.0
            for b in range(K):
                X[:, t * K + b] = np.convolve(on, hrf[b])[:n_tp]
        coefs = rng.multivariate_normal(basis.m, basis.C * 0.3, size=(N_VOX, n_trials_per_run))
        y = np.einsum("tk,vk->vt", X, coefs.reshape(N_VOX, -1))
        eps = rng.standard_normal((N_VOX, n_tp))
        if rho is not None:
            for t in range(1, n_tp):
                eps[:, t] += rho * eps[:, t - 1]
        y = y + 0.8 * y.std() * eps
        datas.append(torch.from_numpy(y).float())
        designs.append(torch.from_numpy(X).float())
    return datas, designs


def test_cv_offloaded_matches_resident(basis, tiny_chunks):
    datas, designs = _runs(basis)
    kw = dict(
        per_run_data=datas,
        per_run_task_designs=designs,
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        n_blocks=1,
        polort=2,
        weight_grid=[0.5, 2.0],
        include_ols=True,
        verbose=False,
    )
    resident = cv_basis_constrained_ridge(device=CPU, **kw)
    offloaded = cv_basis_constrained_ridge(device=OFFLOAD, **kw)
    np.testing.assert_allclose(offloaded.r2_per_weight, resident.r2_per_weight, atol=1e-5)


def test_fracridge_offloaded_matches_resident(basis, tiny_chunks):
    datas, designs = _runs(basis)
    kw = dict(
        per_run_data=datas,
        per_run_task_designs=designs,
        n_blocks=1,
        n_basis=3,
        polort=2,
        fracs=np.array([0.3, 0.7, 1.0]),
        verbose=False,
    )
    resident = fit_basis_fracridge(device=CPU, **kw)
    offloaded = fit_basis_fracridge(device=OFFLOAD, **kw)
    np.testing.assert_allclose(offloaded.r2_by_frac, resident.r2_by_frac, atol=1e-5)
    np.testing.assert_allclose(offloaded.betas, resident.betas, rtol=1e-5, atol=1e-6)
    assert offloaded.sigma2_mean == pytest.approx(resident.sigma2_mean, rel=1e-5)


def test_lss_offloaded_matches_resident(basis, tiny_chunks):
    datas, designs = _runs(basis, n_trials_per_run=4)
    labels = ["a_trial0", "a_trial1", "b_trial0", "b_trial1"]
    kw = dict(
        per_run_data=datas,
        per_run_designs=designs,
        block_labels=labels,
        condition_labels=["a", "b"],
        basis_functions=basis.basis_functions,
        prior_mean=basis.m,
        prior_cov=basis.C,
        polort=2,
        verbose=False,
    )
    resident = fit_basis_lss(device=CPU, **kw)
    offloaded = fit_basis_lss(device=OFFLOAD, **kw)
    np.testing.assert_allclose(offloaded.betas, resident.betas, rtol=1e-5, atol=1e-6)


def test_arma_search_offloaded_matches_resident(basis, tiny_chunks):
    datas, designs = _runs(basis, rho=0.5)
    resident = estimate_arma11_per_voxel(datas, designs, polort=2, device=CPU, verbose=False)
    offloaded = estimate_arma11_per_voxel(datas, designs, polort=2, device=OFFLOAD, verbose=False)
    np.testing.assert_array_equal(offloaded, resident)
