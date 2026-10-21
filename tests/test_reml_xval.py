"""ARMA-aware condition-level cross-validation for ffs_reml.

The anchor test is that with white noise (a = b = 0) the ARMA path must
reproduce the OLS path exactly -- the whitening transform is the identity there,
so any disagreement is a bug in the fold bookkeeping rather than a modelling
choice.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.arma import build_arma11_covariance
from fastfuncstuff.glm.reml_xval import (
    _whiten_per_run,
    censor_xval_inputs,
    compute_xval_r2_arma,
)
from fastfuncstuff.glm.xval import compute_xval_r2


def _design(n_runs=6, run_length=160, n_conditions=4, seed=0):
    generator = np.random.default_rng(seed)
    n_timepoints = n_runs * run_length
    n_columns = n_conditions + n_runs  # conditions + per-run intercept (nuisance)
    design = np.zeros((n_timepoints, n_columns), dtype=np.float32)
    run_starts = [run * run_length for run in range(n_runs)]
    for run in range(n_runs):
        for slot, condition in enumerate(generator.permutation(n_conditions)):
            onset = run * run_length + slot * (run_length // n_conditions)
            design[onset : onset + 8, condition] = 1.0
        design[run * run_length : (run + 1) * run_length, n_conditions + run] = 1.0
    return design, run_starts, n_timepoints, n_conditions


def _splits(n_runs):
    return [([r for r in range(n_runs) if r != held], [held]) for held in range(n_runs)]


class TestArmaXval:
    def test_matches_ols_when_noise_is_white(self):
        """a = b = 0 makes whitening the identity, so the two paths must agree."""
        design, run_starts, n_timepoints, n_conditions = _design()
        generator = torch.Generator().manual_seed(4)
        design_t = torch.from_numpy(design)
        betas = torch.randn(120, design.shape[1], generator=generator)
        data = betas @ design_t.T + torch.randn(120, n_timepoints, generator=generator) * 2.0

        common = dict(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=list(range(n_conditions, design.shape[1])),
            cv_splits=_splits(len(run_starts)),
            device=torch.device("cpu"),
            verbose=False,
        )
        arma = compute_xval_r2_arma(**common, arma_params=torch.zeros(120, 2))
        ols = compute_xval_r2(**common, metric="cod")

        assert isinstance(ols["r2"], torch.Tensor)
        torch.testing.assert_close(arma["r2"], ols["r2"].float(), atol=2e-3, rtol=2e-3)

    def test_groups_voxels_by_arma_pair(self):
        """Distinct (a, b) values must produce distinct groups, not one blob."""
        design, run_starts, n_timepoints, n_conditions = _design(n_runs=4, run_length=120)
        data = torch.randn(30, n_timepoints)
        params = torch.zeros(30, 2)
        params[10:20, 0] = 0.3
        params[20:, 0] = 0.6
        result = compute_xval_r2_arma(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=list(range(n_conditions, design.shape[1])),
            cv_splits=_splits(4),
            arma_params=params,
            device=torch.device("cpu"),
            verbose=False,
        )
        assert result["n_groups"] == 3

    def test_voxel_chunking_does_not_change_the_score(self, monkeypatch):
        """Chunks that split (a, b) groups across their boundaries must score
        exactly as one whole-brain pass: every quantity is per voxel or per group."""
        import fastfuncstuff.glm.reml_xval as reml_xval

        design, run_starts, n_timepoints, n_conditions = _design(n_runs=4, run_length=120)
        generator = torch.Generator().manual_seed(7)
        betas = torch.randn(37, design.shape[1], generator=generator)
        data = betas @ torch.from_numpy(design).T + torch.randn(
            37, n_timepoints, generator=generator
        )
        params = torch.zeros(37, 2)
        params[::3, 0] = 0.4
        params[1::3, 1] = 0.2
        common = dict(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=list(range(n_conditions, design.shape[1])),
            cv_splits=_splits(4),
            arma_params=params,
            device=torch.device("cpu"),
            verbose=False,
        )
        whole = compute_xval_r2_arma(**common)["r2"]
        monkeypatch.setattr(reml_xval, "estimate_chunk_size", lambda **_: 5)
        chunked = compute_xval_r2_arma(**common)["r2"]
        # BLAS blocking may differ with the column count: ULP-level tolerance.
        torch.testing.assert_close(chunked, whole, rtol=1e-5, atol=1e-6)

    def test_gls_beats_ols_under_strong_autocorrelation(self):
        """The point of prewhitening: correlated noise should favour the GLS fit."""
        design, run_starts, n_timepoints, n_conditions = _design(n_runs=8, run_length=200, seed=3)
        generator = torch.Generator().manual_seed(17)
        design_t = torch.from_numpy(design)
        n_voxels = 200
        betas = torch.randn(n_voxels, design.shape[1], generator=generator)
        signal = betas @ design_t.T

        # AR(1) noise at a = 0.7, generated run by run so runs stay independent.
        a = 0.7
        noise = torch.zeros(n_voxels, n_timepoints)
        run_length = 200
        for run in range(len(run_starts)):
            innovation = torch.randn(n_voxels, run_length, generator=generator) * 3.0
            series = torch.zeros(n_voxels, run_length)
            series[:, 0] = innovation[:, 0]
            for t in range(1, run_length):
                series[:, t] = a * series[:, t - 1] + innovation[:, t]
            noise[:, run * run_length : (run + 1) * run_length] = series
        data = signal + noise

        common = dict(
            data=data,
            design_matrix=design,
            run_starts=run_starts,
            stim_indices=list(range(n_conditions)),
            nuisance_indices=list(range(n_conditions, design.shape[1])),
            cv_splits=_splits(len(run_starts)),
            device=torch.device("cpu"),
            verbose=False,
        )
        params = torch.zeros(n_voxels, 2)
        params[:, 0] = a
        gls = compute_xval_r2_arma(**common, arma_params=params)["r2"]
        ols = compute_xval_r2(**common, metric="cod")["r2"]

        assert isinstance(ols, torch.Tensor)
        assert gls.median().item() > ols.median().item()

    def test_rejects_unsupported_metric(self):
        design, run_starts, _, n_conditions = _design(n_runs=3, run_length=100)
        with pytest.raises(ValueError, match="only 'cod'"):
            compute_xval_r2_arma(
                data=torch.randn(5, 300),
                design_matrix=design,
                run_starts=run_starts,
                stim_indices=list(range(n_conditions)),
                nuisance_indices=list(range(n_conditions, design.shape[1])),
                cv_splits=_splits(3),
                arma_params=torch.zeros(5, 2),
                metric="corr",
                device=torch.device("cpu"),
                verbose=False,
            )

    def test_rejects_mismatched_arma_params(self):
        design, run_starts, _, n_conditions = _design(n_runs=3, run_length=100)
        with pytest.raises(ValueError, match="voxels"):
            compute_xval_r2_arma(
                data=torch.randn(5, 300),
                design_matrix=design,
                run_starts=run_starts,
                stim_indices=list(range(n_conditions)),
                nuisance_indices=list(range(n_conditions, design.shape[1])),
                cv_splits=_splits(3),
                arma_params=torch.zeros(9, 2),
                device=torch.device("cpu"),
                verbose=False,
            )


class TestCensoredXval:
    def test_censored_trs_cannot_influence_the_score(self):
        """Two datasets differing only at censored TRs must score identically,
        across both runs, under ARMA whitening."""
        design, run_starts, n_timepoints, n_conditions = _design(n_runs=3, run_length=120)
        generator = torch.Generator().manual_seed(1)
        betas = torch.randn(40, design.shape[1], generator=generator)
        clean = betas @ torch.from_numpy(design).T + torch.randn(
            40, n_timepoints, generator=generator
        )
        censored = [5, 6, 130, 250, 251, 252]  # holes in all three runs
        good = [t for t in range(n_timepoints) if t not in set(censored)]

        def score(data):
            d, x, starts, tau = censor_xval_inputs(data, design, run_starts, good)
            assert tau is not None and d.shape[1] == len(good) == x.shape[0]
            return compute_xval_r2_arma(
                data=d,
                design_matrix=x,
                run_starts=starts,
                stim_indices=list(range(n_conditions)),
                nuisance_indices=list(range(n_conditions, design.shape[1])),
                cv_splits=_splits(len(starts)),
                arma_params=torch.tensor([[0.6, 0.2]]).expand(40, 2),
                device=torch.device("cpu"),
                verbose=False,
                tau=tau,
            )["r2"]

        spiked = clean.clone()
        spiked[:, censored] += 1e4
        torch.testing.assert_close(score(clean), score(spiked), rtol=0, atol=0)

    def test_no_censoring_is_a_passthrough(self):
        design, run_starts, n_timepoints, _ = _design(n_runs=2, run_length=50)
        data = torch.randn(3, n_timepoints)
        d, x, starts, tau = censor_xval_inputs(data, design, run_starts, None)
        assert d is data and x is design and starts == run_starts and tau is None

    def test_whitening_across_a_hole_is_gls_on_the_kept_submatrix(self):
        """With tau, the whitened Gram is X_k' S_kk^-1 X_k for the kept rows of
        the full-run covariance S -- the exact marginal, not a lag-1 squeeze."""
        a, b, n = 0.7, 0.3, 30
        keep = torch.tensor([t for t in range(n) if t not in (10, 11, 20)])
        x = torch.randn(n, 3, dtype=torch.float64)
        full = build_arma11_covariance(a, b, n, torch.device("cpu"), dtype=torch.float64)
        assert full is not None
        s_kk = full[keep][:, keep]
        xk = x[keep]
        expected = xk.T @ torch.linalg.solve(s_kk, xk)

        xw, _ = _whiten_per_run(xk, xk.clone(), [(0, 0, len(keep))], a, b, {}, tau=keep)
        torch.testing.assert_close(xw.T @ xw, expected)
