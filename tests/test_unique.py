"""fastfuncstuff.unique must be np.unique, result for result."""

import itertools

import numpy as np
import pytest

from fastfuncstuff.unique import sorted_unique as unique

FLAGS = list(itertools.product([False, True], repeat=3))


def _same(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("flags", FLAGS)
@pytest.mark.parametrize(
    "make",
    [
        lambda r: r.integers(0, 50, 1000),  # many repeats
        lambda r: r.integers(-(10**12), 10**12, 5000),  # nearly all distinct, negative
        lambda r: r.integers(0, 9, (13, 7)).astype(np.int32),  # 2-D: inverse keeps the shape
        lambda r: r.integers(0, 3, 400).astype(np.uint8),
        lambda r: r.random(300) > 0.5,  # bool
        lambda r: np.zeros(0, np.int64),
        lambda r: np.array([7]),
        lambda r: r.integers(0, 4, (2, 3, 4)).astype(np.int16),
    ],
)
def test_matches_numpy(make, flags):
    a = make(np.random.default_rng(1))
    ri, rv, rc = flags
    _same(
        unique(a, return_index=ri, return_inverse=rv, return_counts=rc),
        np.unique(a, return_index=ri, return_inverse=rv, return_counts=rc),
    )


def test_non_integer_and_axis_calls_are_numpy_itself():
    f = np.array([3.0, np.nan, 1.0, np.nan])  # NaNs collapse in np.unique, not in a sort
    _same(unique(f), np.unique(f))
    m = np.array([[1, 2], [1, 2], [0, 1]])
    _same(unique(m, axis=0), np.unique(m, axis=0))
    s = np.array(["b", "a", "b"])
    _same(unique(s), np.unique(s))
    assert unique([3, 1, 3]).tolist() == [1, 3]  # list input, as np.unique takes
