"""One-pass CUDA kernel for a symmetric 1-D filter along any axis.

``smooth3d.blur_axis`` is one strided ``add_`` per tap: a 4 mm Gaussian at
0.8 mm voxels has 19 taps, so each axis reads and rewrites the whole array ~19
times -- 176 ms for three axes of a 50 x 246 x 68 x 246 block that one read and
one write per axis does in 12.  Here each output element gathers its taps
straight from the input (the neighbouring rows are L2-resident) and is written
once.  Summation is float32 in tap order, centre first -- the shift-add order,
so a run blurs to the same bits -- and the radius is a runtime loop bound, so
a new FWHM does not compile a new kernel.

The array is viewed as ``(outer, n, inner)`` around the filtered axis.  When
``inner`` is wide the vector runs along it (``_rows_kernel``: each program one
``(outer, i)`` row, coalesced loads at every tap); when it is narrow -- the
filtered axis is the contiguous one or nearly -- the vector runs along the axis
itself instead (``_along_kernel``), or every program would hold a handful of
elements.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from fastfuncstuff.triton_key import install_triton_key_cache

install_triton_key_cache()

#: ``inner`` at or above which the row kernel is used.
_ROWS_MIN_INNER = 64


@triton.jit
def _rows_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n,
    inner,
    R,
    REFLECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)  # o * n + i
    o = row // n
    i = row % n
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = offs < inner
    base = o.to(tl.int64) * n * inner + offs
    acc = tl.load(x_ptr + base + i.to(tl.int64) * inner, mask=m, other=0.0) * tl.load(w_ptr + R)
    for j in range(1, R + 1):
        w = tl.load(w_ptr + R + j)
        ip = i + j
        im = i - j
        if REFLECT:
            # Mirror without repeating the edge sample (torch's ``reflect``).
            ip = tl.where(ip > n - 1, 2 * (n - 1) - ip, ip)
            im = tl.where(im < 0, -im, im)
            mp = m
            mm = m
        else:
            mp = m & (ip < n)
            mm = m & (im >= 0)
        acc += w * tl.load(x_ptr + base + ip.to(tl.int64) * inner, mask=mp, other=0.0)
        acc += w * tl.load(x_ptr + base + im.to(tl.int64) * inner, mask=mm, other=0.0)
    tl.store(out_ptr + base + i.to(tl.int64) * inner, acc, mask=m)


@triton.jit
def _along_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    n,
    inner,
    R,
    REFLECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    line = tl.program_id(0)  # o * inner + c
    o = line // inner
    c = line % inner
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    base = o.to(tl.int64) * n * inner + c
    acc = tl.load(x_ptr + base + i.to(tl.int64) * inner, mask=m, other=0.0) * tl.load(w_ptr + R)
    for j in range(1, R + 1):
        w = tl.load(w_ptr + R + j)
        ip = i + j
        im = i - j
        if REFLECT:
            ip = tl.where(ip > n - 1, 2 * (n - 1) - ip, ip)
            im = tl.where(im < 0, -im, im)
            mp = m
            mm = m
        else:
            mp = m & (ip < n)
            mm = m & (im >= 0)
        acc += w * tl.load(x_ptr + base + ip.to(tl.int64) * inner, mask=mp, other=0.0)
        acc += w * tl.load(x_ptr + base + im.to(tl.int64) * inner, mask=mm, other=0.0)
    tl.store(out_ptr + base + i.to(tl.int64) * inner, acc, mask=m)


def blur_axis_cuda(x: Tensor, taps: Tensor, axis: int, *, reflect: bool = False) -> Tensor:
    """``smooth3d.blur_axis`` for a float32 CUDA tensor, in one pass.  Returns a new tensor."""
    x = x.contiguous()
    axis = axis % x.dim()
    n = int(x.shape[axis])
    inner = math.prod(x.shape[axis + 1 :])
    outer = math.prod(x.shape[:axis])
    r = (int(taps.numel()) - 1) // 2
    w = taps.to(device=x.device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x)
    if x.numel() == 0:
        return out
    if inner >= _ROWS_MIN_INNER:
        block = min(512, triton.next_power_of_2(inner))
        grid = (outer * n, triton.cdiv(inner, block))
        _rows_kernel[grid](x, w, out, n, inner, R=r, REFLECT=reflect, BLOCK=block)
    else:
        block = min(256, triton.next_power_of_2(n))
        grid = (outer * inner, triton.cdiv(n, block))
        _along_kernel[grid](x, w, out, n, inner, R=r, REFLECT=reflect, BLOCK=block)
    return out
