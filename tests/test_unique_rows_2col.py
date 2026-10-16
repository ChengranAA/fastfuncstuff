import pytest
import torch

from fastfuncstuff.glm.arma import _unique_rows_2col


def _grid_pairs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    grid = torch.arange(-9, 10) / 10.0
    a = grid[torch.randint(0, len(grid), (n,), generator=g)]
    b = grid[torch.randint(0, len(grid), (n,), generator=g)]
    return torch.stack([a, b], 1).float().contiguous()


def test_same_rows_order_and_inverse_as_unique_dim0():
    """The ARMA grouping consumes unique rows AND the inverse, so both must match."""
    x = _grid_pairs(5000)
    x[::50] = 0.0
    want_rows, want_inv = torch.unique(x, dim=0, return_inverse=True)
    rows, inv = _unique_rows_2col(x, return_inverse=True)
    assert torch.equal(rows, want_rows)
    assert torch.equal(inv, want_inv)
    assert torch.equal(_unique_rows_2col(x), want_rows)


@pytest.mark.parametrize(
    "x",
    [
        torch.empty(0, 2),
        torch.tensor([[0.0, float("nan")], [0.0, 1.0], [0.0, float("nan")]]),
        torch.ones(4, 3),
    ],
    ids=["empty", "nan", "three-columns"],
)
def test_inputs_it_cannot_rank_delegate_to_unique(x):
    """Empty, NaN-bearing and non-pair tables keep torch.unique's own behaviour."""
    want_rows, want_inv = torch.unique(x, dim=0, return_inverse=True)
    rows, inv = _unique_rows_2col(x, return_inverse=True)
    assert rows.shape == want_rows.shape
    assert torch.equal(inv, want_inv)
