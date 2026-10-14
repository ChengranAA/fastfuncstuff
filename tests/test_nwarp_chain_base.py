"""-chain_base: a chain applied onto an oblique master lands where the header says.

ffs_nwarp maps matrices through the output grid's CARDINAL affine, as AFNI does. On
two oblique grids with the same rotation but different origins (an anat and the
EPI-resolution box cut from it, say), that frame differs from the base's by
(R^-1 - I)(origin offset): ds003427's 42-degree slab put every final EPI 1.2-1.7 mm
off its header. The source is linear in TRUE scanner mm, so trilinear sampling is
exact and any frame slip shows as an error.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.nwarpforge import nwarpforge

DEV = torch.device("cpu")
COEF = np.array([0.7, -0.4, 1.1])


def _oblique(vox: float, origin) -> np.ndarray:
    t = np.deg2rad(40.0)
    rot = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    a = np.eye(4)
    a[:3, :3] = rot * vox
    a[:3, 3] = origin
    return a


def _run(tmp_path, chain_base: bool) -> tuple[np.ndarray, np.ndarray]:
    n = 24
    src_aff = _oblique(1.0, (-10.0, -12.0, -8.0))
    ijk = np.indices((n, n, n)).reshape(3, -1)
    xyz = src_aff[:3, :3] @ ijk + src_aff[:3, 3:4]
    vals = (COEF @ xyz + 50.0).reshape(n, n, n).astype(np.float32)
    src = tmp_path / "src.nii"
    nib.Nifti1Image(vals, src_aff).to_filename(str(src))

    # Same rotation, coarser voxels, origin moved 4 mm along the slab's oblique axes.
    mast_aff = _oblique(1.5, src_aff[:3, 3] + src_aff[:3, :3] @ np.array([2.0, 3.0, 2.0]))
    mast = tmp_path / "mast.nii"
    nib.Nifti1Image(np.zeros((10, 10, 10), np.float32), mast_aff).to_filename(str(mast))
    ident = tmp_path / "ident.aff12.1D"
    ident.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")

    out = tmp_path / ("fixed.nii" if chain_base else "afni.nii")
    nwarpforge(
        source_path=str(src),
        nwarp_specs=[str(ident)],
        prefix=str(out),
        master_path=str(mast),
        interp="linear",
        device=DEV,
        verb=0,
        chain_base=str(src) if chain_base else None,
    )
    img = nib.load(str(out))
    got = np.asarray(img.dataobj, np.float32)
    mijk = np.indices(got.shape).reshape(3, -1)
    want = (COEF @ (img.affine[:3, :3] @ mijk + img.affine[:3, 3:4]) + 50.0).reshape(got.shape)
    return got, want


def test_chain_base_puts_data_where_the_oblique_header_says(tmp_path):
    got, want = _run(tmp_path, chain_base=True)
    np.testing.assert_allclose(got, want, atol=2e-3)


def test_without_chain_base_the_cardinal_frame_slips(tmp_path):
    # AFNI parity is kept by default; this pins that the slip is real, so the flag
    # is not a no-op on this geometry.
    got, want = _run(tmp_path, chain_base=False)
    assert np.abs(got - want).max() > 0.3


def test_surface_points_get_the_same_correction(tmp_path):
    # The surface route samples true-mm points on its own master (the raw anat)
    # through a chain fitted on the cropped anat box: the same slip, the same fix.
    _run(tmp_path, chain_base=True)  # writes src.nii and mast.nii
    src, mast = tmp_path / "src.nii", tmp_path / "mast.nii"
    ident = tmp_path / "ident.aff12.1D"
    m = nib.load(str(mast)).affine
    rng = np.random.default_rng(1)
    pts = (m @ np.c_[rng.uniform(1, 8, size=(100, 3)), np.ones(100)].T)[:3].T
    want = pts @ COEF + 50.0
    for base, ok in ((str(src), True), (None, False)):
        got = nwarpforge(
            source_path=str(src),
            nwarp_specs=[str(ident)],
            prefix="",
            master_path=str(mast),
            interp="linear",
            device=DEV,
            verb=0,
            points=pts,
            chain_base=base,
        )
        assert got is not None
        err = np.abs(got.numpy() - want).max()
        assert (err < 2e-3) if ok else (err > 0.3)


def test_chain_base_keeps_jac_on_the_named_fieldmap(tmp_path):
    """Bug of record (2026-10-14): -chain_base inserted its correction at the front of
    the transforms but not of the names, so -jac AXIS:FIELDMAP (found by name, then
    indexed into the transforms) took the Jacobian of the transform BEFORE the
    fieldmap: on autoproc chains the PE-only anat nonlinear warp, which stippled every
    stage10 image. Chain = [other PE warp, fieldmap PE warp] on cardinal grids, where
    the correction is the identity, so -chain_base must change nothing; a constant
    source makes the output the applied Jacobian itself."""
    n = 16
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    src = tmp_path / "const.nii"
    nib.Nifti1Image(np.full((n, n, n), 100.0, np.float32), aff).to_filename(str(src))

    def pe_ramp(name, slope_mm):
        jj = np.arange(n, dtype=np.float32)[None, :, None]
        d = np.zeros((n, n, n, 3), np.float32)
        d[..., 1] = np.broadcast_to(slope_mm * (jj - n / 2), (n, n, n))
        p = tmp_path / name
        nib.Nifti1Image(d, aff).to_filename(str(p))
        return str(p)

    other = pe_ramp("anat_nl_warp.nii", 0.6)  # Jacobian 1 -/+ 0.3
    fmap = pe_ramp("fmap_warp.nii", 0.2)  # Jacobian 1 -/+ 0.1
    outs = {}
    for cb in (False, True):
        out = tmp_path / f"out_{cb}.nii"
        nwarpforge(
            source_path=str(src), nwarp_specs=[other, fmap], prefix=str(out),
            master_path=str(src), interp="linear", device=DEV, verb=0,
            jac_axis=1, jac_match="fmap_warp", auto_pad=False,
            chain_base=str(src) if cb else None,
        )  # fmt: skip
        outs[cb] = np.asarray(nib.load(str(out)).dataobj, np.float32)[3:-3, 3:-3, 3:-3]
    assert abs(float(np.median(outs[False])) - 100.0) == pytest.approx(10.0, abs=0.5)
    np.testing.assert_allclose(outs[True], outs[False], atol=1e-3)
