"""ffs_phasedrift: per-slice field wobble out, local (vessel) phase kept."""

from __future__ import annotations

import nibabel as nib
import numpy as np

from fastfuncstuff.cli.phasedrift import main


def test_per_slice_wobble_removed_local_signal_kept(tmp_path):
    rng = np.random.default_rng(0)
    nx, ny, nz, nt = 16, 16, 6, 80
    static = rng.uniform(-20, 20, (nx, ny, nz, 1))  # unwrapped static field
    # Each slice sees its own field state per volume, with an in-plane gradient.
    off = 0.3 * rng.standard_normal((1, 1, nz, nt))
    grad = 0.1 * rng.standard_normal((1, 1, nz, nt)) * np.linspace(-1, 1, nx)[:, None, None, None]
    local = 0.05 * np.sin(np.linspace(0, 10, nt))
    phase = static + off + grad + 0.01 * rng.standard_normal((nx, ny, nz, nt))
    phase[8, 8, 3] += local
    src = tmp_path / "ph.nii"
    nib.Nifti1Image(phase.astype(np.float32), np.eye(4)).to_filename(str(src))

    out = tmp_path / "clean.nii"
    trace = tmp_path / "trace.1D"
    assert main(["-input", str(src), "-prefix", str(out), "-save_trace", str(trace), "-verb", "0"]) == 0
    got = np.asarray(nib.load(str(out)).dataobj, dtype=np.float64)

    dyn = got - got.mean(-1, keepdims=True)
    assert np.median(dyn.std(-1)) < 0.02, "per-slice wobble survived"
    r = np.corrcoef(dyn[8, 8, 3], local)[0, 1]
    assert r > 0.95, f"local phase lost (r={r:.3f})"
    assert np.loadtxt(trace).shape == (nt, nz)
