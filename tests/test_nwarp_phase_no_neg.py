"""-no_neg must clamp only the magnitude, never the phase channel.

With -no_neg, the per-frame warp path once clamped every warped channel at zero:
unwrapped phase in radians lost every negative value, and the complex modes lost
three quadrants of real/imag/cos/sin. The space-time path already had a per-channel
mask; the per-frame path (no -tpattern, e.g. autoproc -slicetiming_method first)
did not. Phase regression downstream then fits a half-rectified regressor.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.processing.nwarpforge import nwarpforge


@pytest.mark.parametrize("phase_warp", ["direct", "split", "circular", "complex"])
def test_no_neg_leaves_phase_signed(tmp_path, phase_warp):
    nx, ny, nz, nt = 8, 9, 7, 3
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    mag = 100.0 + rng.random((nx, ny, nz, nt), dtype=np.float32)
    xx = np.arange(nx, dtype=np.float32)[:, None, None, None]
    tt = np.arange(nt, dtype=np.float32)[None, None, None, :]
    # Spans negative and positive radians, but stays inside (-pi, pi) so the
    # wrapped modes can return it unchanged.
    phase = (-2.5 + 0.6 * xx + 0.1 * tt) * np.ones((nx, ny, nz, nt), np.float32)

    src, ph = tmp_path / "mag.nii", tmp_path / "ph.nii"
    nib.Nifti1Image(mag, affine).to_filename(str(src))
    nib.Nifti1Image(phase, affine).to_filename(str(ph))
    ident = tmp_path / "ident.aff12.1D"
    ident.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")

    out, ph_out = tmp_path / "out.nii", tmp_path / "out_phase.nii"
    nwarpforge(
        source_path=str(src),
        nwarp_specs=[str(ident)],
        prefix=str(out),
        phase_path=str(ph),
        phase_prefix=str(ph_out),
        phase_units="rad",
        phase_warp=phase_warp,
        interp="linear",
        no_neg=True,
        device=torch.device("cpu"),
        verb=0,
        auto_pad=False,
    )
    got = np.asarray(nib.load(str(ph_out)).dataobj, dtype=np.float32)
    assert got.min() < -2.0, "negative phase was clamped"
    np.testing.assert_allclose(got[1:-1, 1:-1, 1:-1], phase[1:-1, 1:-1, 1:-1], atol=1e-4)
