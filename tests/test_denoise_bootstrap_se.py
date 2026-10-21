"""compute_bootstrap_se streams its replicates; it must match the stored-replicate std."""

from __future__ import annotations

import numpy as np
import torch

from fastfuncstuff.cli.denoise import compute_bootstrap_se


def _stored_reference(data, design, n_task, n_boots):
    """The former implementation: keep every replicate, then np.std."""
    n_vox, n_t = data.shape
    n_reg = design.shape[1]
    boots = np.zeros((n_boots, n_vox, n_task), dtype=np.float32)
    for b in range(n_boots):
        idx = torch.randint(0, n_t, (n_t,))
        xb = design[idx, :]
        inv = torch.linalg.inv(xb.T @ xb + 1e-6 * torch.eye(n_reg))
        boots[b] = (inv @ xb.T @ data[:, idx].T).T[:, :n_task].numpy()
    return np.std(boots, axis=0)


def test_streamed_bootstrap_se_matches_stored_replicates():
    gen = torch.Generator().manual_seed(0)
    n_vox, n_t = 37, 80
    design = torch.randn(n_t, 5, generator=gen)
    data = torch.randn(n_vox, n_t, generator=gen) + (design[:, :2] @ torch.randn(2, n_vox)).T

    torch.manual_seed(123)
    ref = _stored_reference(data, design, n_task=2, n_boots=25)
    torch.manual_seed(123)
    # chunk_size=10: several ragged chunks per replicate.
    got = compute_bootstrap_se(
        data, design, n_task=2, n_boots=25, chunk_size=10, device=torch.device("cpu"), verbose=False
    )
    assert got.shape == (n_vox, 2)
    # Welford in float64 vs np.std over float32 replicates: agree to float32 rounding.
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-6)
