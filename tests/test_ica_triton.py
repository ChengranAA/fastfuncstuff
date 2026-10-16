"""The fused FastICA data pass against the portable update it replaces."""

import pytest
import torch

pytestmark = pytest.mark.gpu

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _sources(k: int, v: int, seed: int = 0, n_gauss: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.distributions.Laplace(0.0, 1.0).sample((k, v))
    n_gauss = k // 2 if n_gauss is None else n_gauss
    s[:n_gauss] = torch.randn(n_gauss, v, generator=g)
    a = torch.linalg.qr(torch.randn(k, k, generator=g))[0]
    x = a @ s
    x = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
    return x.cuda()


@cuda
@pytest.mark.parametrize(
    ("k", "n_feat", "n_restarts", "n_samp"),
    [(30, 30, 1, 70_001), (7, 12, 3, 5_000), (64, 64, 2, 20_000)],
)
def test_fused_terms_match_the_portable_update(k, n_feat, n_restarts, n_samp):
    from fastfuncstuff.decomposition import ica_triton
    from fastfuncstuff.decomposition.ica import _nonlinearity_batch

    x = _sources(n_feat, n_samp)
    w = torch.linalg.qr(torch.randn(n_restarts, n_feat, n_feat, device="cuda"))[0][:, :k]
    term, g_prime = ica_triton.logcosh_terms(x, w)

    x64, w64 = x.double(), w.double()
    g64, gp64 = _nonlinearity_batch(w64 @ x64, "logcosh")
    torch.testing.assert_close(term.double(), (g64 @ x64.T) / n_samp, atol=2e-6, rtol=0)
    torch.testing.assert_close(g_prime.double(), gp64, atol=2e-6, rtol=0)


@cuda
def test_fastica_batch_fused_and_portable_find_the_same_unmixing(monkeypatch):
    from fastfuncstuff.decomposition.ica import fastica_batch

    x = _sources(12, 40_000, seed=3, n_gauss=0)  # Gaussian sources never converge
    monkeypatch.setenv("FFS_NO_FUSED_FASTICA", "1")
    w_ref, it_ref, conv_ref = fastica_batch(x, 2, 12, max_iter=400, base_seed=5)
    monkeypatch.delenv("FFS_NO_FUSED_FASTICA")
    w, it, conv = fastica_batch(x, 2, 12, max_iter=400, base_seed=5)

    assert conv.all() and conv_ref.all()
    # Same start, same fixed point: rows agree up to float32 rounding (no sign or
    # permutation freedom, since both runs follow the same trajectory).
    match = (w.double() @ w_ref.double().transpose(-1, -2)).abs().amax(-1)
    assert match.min() > 1 - 1e-6
    assert (it - it_ref).abs().max() <= 2
