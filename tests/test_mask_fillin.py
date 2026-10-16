"""3dAutomask's fill-in loop: the candidate-only passes must equal full sweeps."""

import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from fastfuncstuff.processing.mask import _afni_fillin_completely, _afni_fillin_once


def _fillin_once_full_grid(mask, nside):
    """THD_mask_fillin_once as a plain sweep of the whole grid (the original port)."""
    nz, ny, nx = mask.shape
    ns = [min((n - 1) // 2, nside) for n in (nz, ny, nx)]
    if not any(ns):
        return mask, 0
    fill = np.zeros_like(mask)
    core = tuple(slice(s, n - s) for s, n in zip(ns, (nz, ny, nx), strict=True))
    for axis in range(3):
        s, n = ns[axis], mask.shape[axis]
        if s == 0:
            continue
        plus = np.zeros_like(mask)
        minus = np.zeros_like(mask)
        for d in range(1, s + 1):
            src = [slice(None)] * 3
            dst = [slice(None)] * 3
            src[axis], dst[axis] = slice(d, n), slice(0, n - d)
            plus[tuple(dst)] |= mask[tuple(src)]
            minus[tuple(src)] |= mask[tuple(dst)]
        fill[core] |= (plus & minus)[core]
    fill &= ~mask
    return mask | fill, int(fill.sum())


def _blobby(shape, seed, density):
    rng = np.random.default_rng(seed)
    m = rng.random(shape) < density
    m = binary_dilation(m, iterations=2)
    return m & (rng.random(shape) > 0.25)  # punch holes for the fill-in to close


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shape", [(24, 30, 28), (5, 40, 33), (31, 2, 26), (17, 19, 1)])
@pytest.mark.parametrize("nside", [1, 2, 4, 8])
def test_fillin_completely_equals_repeated_full_sweeps(seed, shape, nside):
    m = _blobby(shape, seed, density=0.02 + 0.01 * seed)
    ref = m
    while True:
        ref, n = _fillin_once_full_grid(ref, nside)
        if n == 0:
            break
    np.testing.assert_array_equal(_afni_fillin_completely(m, nside), ref)
    one, n_one = _afni_fillin_once(m, nside)
    ref_one, n_ref = _fillin_once_full_grid(m, nside)
    assert n_one == n_ref
    np.testing.assert_array_equal(one, ref_one)
