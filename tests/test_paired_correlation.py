"""Tests for stats.paired_correlation (per-voxel r between datasets on one grid)."""

import numpy as np
import pytest
import torch

from fastfuncstuff.glm.core import construct_polynomial_matrix
from fastfuncstuff.stats.paired_correlation import pair_list, paired_voxel_correlation

CPU = torch.device("cpu")


def _reference(a, b, polort):
    """Plain NumPy: project the Legendre drift out of each voxel, then corrcoef."""
    t = a.shape[-1]
    p = construct_polynomial_matrix(t, polort, CPU, dtype=torch.float64).numpy()
    q, _ = np.linalg.qr(p)
    ra = a.reshape(-1, t).astype(np.float64)
    rb = b.reshape(-1, t).astype(np.float64)
    ra = ra - ra @ q @ q.T
    rb = rb - rb @ q @ q.T
    num = (ra * rb).sum(1)
    den = np.linalg.norm(ra, axis=1) * np.linalg.norm(rb, axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0).reshape(a.shape[:3])


def _datasets(n=3, shape=(7, 5, 4), t=60, seed=0):
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=(*shape, t))
    drift = np.linspace(0, 3, t) ** 2  # shared drift the detrend has to remove
    return [
        (shared * (k + 1) * 0.5 + rng.normal(size=(*shape, t)) + 5 * drift + 100).astype(np.float32)
        for k in range(n)
    ]


def test_matches_numpy_on_every_pair_and_tolerates_f_order():
    data = _datasets()
    # nibabel's get_fdata hands back F-order arrays; the slab streaming must not care.
    data[1] = np.asfortranarray(data[1])
    r = paired_voxel_correlation(data, polort=2, device=CPU, verbose=False)
    assert r.shape == (7, 5, 4, 3)
    for p, (a, b) in enumerate(pair_list(3)):
        np.testing.assert_allclose(r[..., p].numpy(), _reference(data[a], data[b], 2), atol=1e-5)


def test_detrend_is_what_removes_shared_drift():
    rng = np.random.default_rng(1)
    drift = np.linspace(-1, 1, 80) ** 2 * 10
    a = (rng.normal(size=(3, 3, 3, 80)) + drift).astype(np.float32)
    b = (rng.normal(size=(3, 3, 3, 80)) + drift).astype(np.float32)
    raw = paired_voxel_correlation([a, b], polort=0, device=CPU, verbose=False)
    clean = paired_voxel_correlation([a, b], polort=2, device=CPU, verbose=False)
    assert float(raw.mean()) > 0.8  # independent noise, correlated only through drift
    assert abs(float(clean.mean())) < 0.1


def test_mask_constant_voxels_and_fisher():
    data = _datasets(n=2)
    data[0][0, 0, 0] = 3.0  # constant: no direction to correlate -> 0, not NaN
    mask = np.ones(data[0].shape[:3], bool)
    mask[6] = False
    r = paired_voxel_correlation(data, mask=mask, device=CPU, verbose=False)[..., 0]
    assert float(r[0, 0, 0]) == 0.0
    assert torch.all(r[6] == 0)
    ref = _reference(data[0], data[1], 0)
    ref[0, 0, 0] = 0.0  # the reference correlates the constant's rounding residue
    np.testing.assert_allclose(r[:6].numpy(), ref[:6], atol=1e-5)
    z = paired_voxel_correlation(data, fisher=True, device=CPU, verbose=False)[..., 0]
    full = paired_voxel_correlation(data, device=CPU, verbose=False)[..., 0]
    assert torch.allclose(z, torch.atanh(full.clamp(-0.999999, 0.999999)), atol=1e-6)


def test_pair_modes():
    assert pair_list(3, "all") == [(0, 1), (0, 2), (1, 2)]
    assert pair_list(4, "adjacent") == [(0, 1), (1, 2), (2, 3)]
    assert pair_list(3, "ref:2") == [(1, 0), (1, 2)]
    for bad in ("ref:0", "ref:9", "ref:x", "every"):
        with pytest.raises(ValueError):
            pair_list(3, bad)
    with pytest.raises(ValueError):
        pair_list(1)


def test_cli_writes_labelled_maps_before_and_delta(tmp_path):
    import nibabel as nib

    from fastfuncstuff.cli.util_tcorr import main
    from fastfuncstuff.io.afni import read_brick_labels, read_brick_stataux

    def save(arrs, tag, tr):
        paths = []
        for k, a in enumerate(arrs):
            img = nib.Nifti1Image(a, np.eye(4))
            img.header.set_zooms((1.0, 1.0, 1.0, tr))
            paths.append(str(tmp_path / f"{tag}{k}.nii.gz"))
            nib.save(img, paths[-1])
        return paths

    after = save(_datasets(n=3), "a", 2.0)
    before = save(_datasets(n=3, seed=4), "b", 2.0)
    out = tmp_path / "r.nii.gz"
    argv = ["-input", *after, "-before", *before, "-prefix", str(out), "-device", "cpu"]
    assert main([*argv, "-quiet"]) == 0
    r = nib.load(out)
    assert r.shape == (7, 5, 4, 3)
    assert read_brick_labels(r) == ["1-2", "1-3", "2-3"]
    # 60 frames at TR 2 is 120 s: AFNI's rule gives polort 1, so two ort params.
    assert read_brick_stataux(r)[0] == (2, (60.0, 1.0, 2.0))
    b = nib.load(tmp_path / "r_before.nii.gz").get_fdata()
    d = nib.load(tmp_path / "r_delta.nii.gz").get_fdata()
    np.testing.assert_allclose(d, r.get_fdata() - b, atol=1e-6)

    # A header TR that is really a shot time must not silently pick a polort.
    bad = save(_datasets(n=2), "c", 0.05)
    assert (
        main(["-input", *bad, "-prefix", str(tmp_path / "x.nii.gz"), "-device", "cpu", "-quiet"])
        == 2
    )
