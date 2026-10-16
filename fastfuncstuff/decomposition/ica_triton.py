"""Fused CUDA kernel for one symmetric FastICA (logcosh) iteration's data pass.

The portable update ``E[g(Wx) x^T]`` and ``E[g'(Wx)]`` writes the ``(R, k, V)``
projection, rewrites it as ``tanh``, reads it again for the squares and once more
for the ``g x^T`` product -- about 1.9 GB of traffic per iteration at k = 30,
V = 1.8M, for a ``k x k`` result.  Nothing between ``x`` and those sums needs to
be written down: each program loads a tile of ``x``, projects it, takes ``tanh``
and accumulates both sums in registers, so ``x`` is read once.

As in the qwarp kernels, voxels are split into a fixed number of ranges whose
partial sums torch adds (in float64), so the result does not depend on atomic
ordering and a fit is reproducible run to run.  The dots are IEEE float32 --
TF32 noise (~3e-4) sits above FastICA's 1e-4 tolerance, and 3xTF32 measured no
faster here.

Reference: ``ica._nonlinearity_batch`` with ``fun="logcosh"``, which this must
match to float32 rounding.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

from fastfuncstuff.triton_key import install_triton_key_cache

install_triton_key_cache()

#: Widest padded component / feature count the kernel takes.  At 128 the ``k x F``
#: accumulator no longer fits a 64-voxel tile in shared memory, and at 100
#: components the portable path measured faster anyway.
MAX_WIDTH = 64
_BLOCK_V = 64
_MAX_SPLIT = 512


@triton.jit
def _logcosh_terms_kernel(
    x_ptr,  # (F, V), row stride stride_x, unit column stride
    w_ptr,  # (R, K, F) contiguous
    out_ptr,  # (R, n_split, KP, FP + 1)
    K,
    F,
    V,
    stride_x,
    n_split,
    KP: tl.constexpr,
    FP: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_s = tl.program_id(0)
    r = tl.program_id(1)
    ok = tl.arange(0, KP)
    of = tl.arange(0, FP)
    w = tl.load(
        w_ptr + r * K * F + ok[:, None] * F + of[None, :],
        mask=(ok[:, None] < K) & (of[None, :] < F),
        other=0.0,
    )
    per_split = tl.cdiv(tl.cdiv(V, BLOCK_V), n_split) * BLOCK_V
    v_start = pid_s * per_split
    v_end = tl.minimum(v_start + per_split, V)

    acc = tl.zeros((KP, FP), dtype=tl.float32)
    gsq = tl.zeros((KP,), dtype=tl.float32)
    for s in tl.range(v_start, v_end, BLOCK_V):
        ov = s + tl.arange(0, BLOCK_V)
        # Masked voxels and padded features load as zero, so their projection --
        # and its tanh -- is zero and adds nothing to either sum.
        x = tl.load(
            x_ptr + of[:, None] * stride_x + ov[None, :],
            mask=(of[:, None] < F) & (ov[None, :] < v_end),
            other=0.0,
        )
        t = libdevice.tanh(tl.dot(w, x, input_precision="ieee"))
        gsq += tl.sum(t * t, axis=1)
        acc += tl.dot(t, tl.trans(x), input_precision="ieee")

    base = out_ptr + (r * n_split + pid_s) * KP * (FP + 1)
    tl.store(base + ok[:, None] * (FP + 1) + of[None, :], acc)
    tl.store(base + ok * (FP + 1) + FP, gsq)


def fits(n_components: int, n_features: int) -> bool:
    """Whether :func:`logcosh_terms` takes this width."""
    return max(16, triton.next_power_of_2(max(n_components, n_features))) <= MAX_WIDTH


def logcosh_terms(x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """``(E[tanh(w x) x^T], 1 - E[tanh(w x)^2])`` for ``x (F, V)``, ``w (R, K, F)``.

    Both are float32: ``(R, K, F)`` and ``(R, K, 1)``, the two terms of the
    symmetric FastICA update.
    """
    n_restarts, k, n_feat = w.shape
    n_samp = x.shape[1]
    if x.stride(1) != 1:
        x = x.contiguous()
    width = max(16, triton.next_power_of_2(max(k, n_feat)))
    n_split = max(1, min(triton.cdiv(n_samp, _BLOCK_V), _MAX_SPLIT))
    out = torch.empty(n_restarts, n_split, width, width + 1, device=x.device, dtype=torch.float32)
    _logcosh_terms_kernel[(n_split, n_restarts)](
        x,
        w.contiguous(),
        out,
        k,
        n_feat,
        n_samp,
        x.stride(0),
        n_split,
        KP=width,
        FP=width,
        BLOCK_V=_BLOCK_V,
    )
    sums = out.sum(dim=1, dtype=torch.float64) / n_samp
    term = sums[:, :k, :n_feat].float()
    g_prime_mean = (1.0 - sums[:, :k, width : width + 1]).float()
    return term, g_prime_mean
