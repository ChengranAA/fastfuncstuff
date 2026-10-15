"""REML parameter selection must survive a fitted baseline on raw fMRI data."""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import _residualize_reml_search_data, fit_glm_arma11
from fastfuncstuff.glm.core import construct_polynomial_matrix


def _case():
    n_run, length = 3, 60
    t = np.arange(length) * 2.0
    task = ((t % 30.0) < 10.0).astype(np.float32)
    task = np.tile(task - task.mean(), n_run)
    poly = construct_polynomial_matrix(length, 2, torch.device("cpu"))
    design = torch.cat([torch.from_numpy(task)[:, None], torch.block_diag(*([poly] * n_run))], 1)
    noise = np.random.default_rng(3).standard_normal(length * n_run) * 0.5
    data = torch.from_numpy((1000.0 + task.astype(np.float64) + noise).astype(np.float32))[None]
    return data, design


@pytest.mark.parametrize("surface", [False, True])
def test_float32_raw_baseline_selects_float64_parameters(surface):
    data, design = _case()
    kwargs = dict(
        tr=2.0,
        run_starts=[0, 60, 120],
        task_indices=[0],
        device=torch.device("cpu"),
        verbose=False,
        save_profile_likelihoods=surface,
    )
    expected = fit_glm_arma11(data, design, use_double=True, **kwargs)
    got = fit_glm_arma11(data, design, **kwargs)
    torch.testing.assert_close(got.arma_params.double(), expected.arma_params, rtol=0, atol=1e-6)
    torch.testing.assert_close(got.sigma2.double(), expected.sigma2, rtol=1e-3, atol=1e-5)


def test_conditioning_preserves_input_and_handles_redundant_columns():
    torch.manual_seed(5)
    design = torch.randn(40, 3)
    design = torch.cat([torch.ones(40, 1), design, design[:, :1]], 1)
    data = torch.randn(8, 40) + 1000.0
    original = data.clone()
    got = _residualize_reml_search_data(design, data)
    expected = (
        data.double() - (data.double() @ torch.linalg.pinv(design.double()).T) @ design.double().T
    )
    assert torch.equal(data, original)
    torch.testing.assert_close(got.double(), expected, rtol=0, atol=5e-4)


def test_conditioning_without_fixed_effects_preserves_data():
    data = torch.randn(3, 20)
    assert _residualize_reml_search_data(torch.empty(20, 0), data) is data


@pytest.mark.gpu
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_baseline_selection_matches_float64():
    data, design = _case()
    kwargs = dict(tr=2.0, run_starts=[0, 60, 120], task_indices=[0], verbose=False)
    expected = fit_glm_arma11(data, design, device=torch.device("cpu"), use_double=True, **kwargs)
    got = fit_glm_arma11(data, design, device=torch.device("mps"), **kwargs)
    torch.testing.assert_close(got.arma_params.double(), expected.arma_params, rtol=0, atol=1e-6)
    torch.testing.assert_close(got.sigma2.double(), expected.sigma2, rtol=1e-3, atol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_conditioning_preserves_full_fp32_under_tf32():
    data, design = _case()
    expected = _residualize_reml_search_data(design, data)
    torch.set_float32_matmul_precision("high")
    got = _residualize_reml_search_data(design.cuda(), data.cuda())
    assert torch.get_float32_matmul_precision() == "high"
    torch.testing.assert_close(got.cpu(), expected, rtol=0, atol=5e-4)
