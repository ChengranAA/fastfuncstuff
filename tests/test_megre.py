"""ffs_megre's image-space steps: background removal, SWI mask, mIP."""

import math

import torch

from fastfuncstuff.processing.megre import (
    masked_highpass,
    minimum_intensity_projection,
    swi_phase_mask,
)


def test_highpass_removes_smooth_background_keeps_vein():
    """A smooth background field inside a mask comes out ~0 right up to the mask
    edge (the normalised blur), while a one-voxel 'vein' survives."""
    z, y, x = torch.meshgrid(
        torch.arange(20.0), torch.arange(40.0), torch.arange(40.0), indexing="ij"
    )
    mask = ((y - 20) ** 2 + (x - 20) ** 2) < 15**2
    background = 30.0 + 0.5 * x + 0.2 * y
    vein = torch.zeros_like(background)
    vein[:, 20, 20] = 10.0
    local = masked_highpass(background + vein, mask, fwhm_mm=4.0, voxel_mm=(1.0, 1.0, 1.0))
    assert local[~mask].abs().max() == 0
    assert local[mask & (vein == 0)].abs().median() < 0.05
    edge = mask & ~torch.roll(mask, 1, 2)
    assert local[edge].abs().max() < 1.0  # an unnormalised blur leaves ~30 here
    assert local[10, 20, 20] > 5.0


def test_swi_phase_mask_only_darkens_venous_sign():
    phase = torch.tensor([-math.pi, -math.pi / 2, 0.0, math.pi / 2])
    m = swi_phase_mask(phase, sign=-1, power=1)
    assert torch.allclose(m, torch.tensor([0.0, 0.5, 1.0, 1.0]))
    assert torch.allclose(swi_phase_mask(-phase, sign=1, power=1), m)


def test_mip_is_sliding_minimum_with_same_shape():
    v = torch.tensor([5.0, 1.0, 4.0, 3.0, 2.0]).reshape(5, 1, 1)
    out = minimum_intensity_projection(v, 3, axis=0).flatten()
    assert out.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
