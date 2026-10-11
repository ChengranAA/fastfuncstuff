"""Paired volume + surface denoising: one noise pool, outputs split per domain."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli import denoise as denoise_cli
from fastfuncstuff.cli.denoise import _denoise_output_extension
from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data

TR = 1.0
N_TP = 80
SHAPE = (4, 4, 2)
N_VERTS = 24
MESH = {"mesh_fingerprint": "V24_F1_test", "TR_seconds": "1"}
ONSETS = [[5.0, 17.0, 29.0, 41.0, 53.0, 65.0], [7.0, 19.0, 31.0, 43.0, 55.0, 67.0]]


def _dataset(tmp_path: Path):
    """Shared physiological-like noise in both domains; task in a few of each."""
    rng = np.random.default_rng(3)
    t = np.arange(N_TP) * TR
    vols, surfs = [], []
    for run, onsets in enumerate(ONSETS):
        noise = np.sin(2 * np.pi * t / 7.0 + run) + 0.5 * np.sin(2 * np.pi * t / 3.3)
        task = np.zeros(N_TP)
        for onset in onsets:
            task[int(onset) + 3 : int(onset) + 8] = 1.0
        n_vox = int(np.prod(SHAPE))
        vol = 100 + 2 * rng.standard_normal((n_vox, N_TP)) + 4 * rng.random((n_vox, 1)) * noise
        vol[:4] += 15 * task
        surf = 100 + 2 * rng.standard_normal((N_VERTS, N_TP)) + 4 * rng.random((N_VERTS, 1)) * noise
        surf[:4] += 15 * task
        img = nib.Nifti1Image(vol.reshape(*SHAPE, N_TP).astype(np.float32), np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((1, 1, 1, TR))
        vols.append(tmp_path / f"vol{run}.nii.gz")
        nib.save(img, vols[-1])
        surfs.append(tmp_path / f"surf{run}.func.gii")
        save_gifti_data(surfs[-1], surf.astype(np.float32), MESH)
    onset_file = tmp_path / "faces.1D"
    onset_file.write_text("\n".join(" ".join(f"{o:g}" for o in row) for row in ONSETS) + "\n")
    surface_mask = np.ones(N_VERTS, np.float32)
    surface_mask[-2:] = 0  # e.g. medial wall
    mask_path = tmp_path / "cortex.shape.gii"
    save_gifti_data(mask_path, surface_mask, {"mesh_fingerprint": MESH["mesh_fingerprint"]})
    return [str(v) for v in vols], [str(s) for s in surfs], str(onset_file), str(mask_path)


def test_paired_validation():
    vols = ["a.nii.gz", "b.nii.gz"]
    assert _denoise_output_extension(vols, "fit", paired=True) == ".nii.gz"
    for kwargs, match in (
        ({"automask": True}, "automask"),
        ({"cv_blur": 4.0}, "cv_blur"),
        ({"noise_pool_blur": 6.0}, "noise_pool_blur"),
    ):
        with pytest.raises(ValueError, match=match):
            _denoise_output_extension(vols, "fit", paired=True, **kwargs)
    with pytest.raises(ValueError, match="NIfTI -prefix"):
        _denoise_output_extension(vols, "fit.func.gii", paired=True)
    with pytest.raises(ValueError, match="-input takes the volume runs"):
        _denoise_output_extension(["a.func.gii", "b.func.gii"], "fit", paired=True)


def test_paired_denoise_pools_both_domains(tmp_path, monkeypatch):
    vols, surfs, onsets, surface_mask = _dataset(tmp_path)
    prefix = str(tmp_path / "out")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ffs_denoise", "-input", *vols, "-surface_input", *surfs,
            "-surface_mask", surface_mask, "-onsets", onsets, "-durations", "4",
            "-tr", str(TR), "-prefix", prefix, "-device", "cpu", "-max_comps", "2",
            "-min_noise_voxels", "10", "-plots", "no",
        ],
    )  # fmt: skip
    denoise_cli.main()

    pool_vol = np.asanyarray(nib.load(f"{prefix}_noise_pool_mask_volume.nii.gz").dataobj)
    pool_surf, meta = load_gifti_data(f"{prefix}_noise_pool_mask_surface.func.gii")
    assert pool_vol.shape == SHAPE
    assert pool_surf.shape == (N_VERTS,)
    assert meta["mesh_fingerprint"] == MESH["mesh_fingerprint"]
    # The point of pairing: quiet voxels and quiet vertices feed the same components.
    assert pool_vol.sum() > 0 and pool_surf.sum() > 0
    assert np.all(pool_surf[-2:] == 0)  # masked-out vertices never enter the pool
    assert not list(tmp_path.glob("out_noise_pool_mask.*"))
