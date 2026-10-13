"""ffs_veinmask: EPI darkness against the anat's tissue prediction.

A synthetic slab: GM/WM/CSF fractions, a smooth coil bias, and a few GM voxels
darkened like veins. White matter is darker than GM too: the model must predict
that and flag only the veins, which a local darkness map cannot.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.processing.veinmask import tissue_darkness, vein_mask, vein_region

DEV = torch.device("cpu")


def _phantom():
    n = 40
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float32),) * 3, indexing="ij")
    gm = ((x // 4) % 2 == 0).astype(np.float32) * 0.9 + 0.05  # GM stripes
    csf = np.where(x > n - 5, 0.9, 0.0).astype(np.float32)
    gm = np.where(csf > 0, 0.05, gm)
    wm = np.clip(1 - gm - csf, 0, 1)
    tissue = np.stack([gm, wm, csf])
    img = 1000 * gm + 650 * wm + 1300 * csf
    bias = 1 + 0.4 * np.sin(x / 15) * np.cos(y / 18)
    img = img * bias
    rng = np.random.default_rng(0)
    gm_idx = np.argwhere((gm > 0.5) & (x > 2) & (x < n - 8))
    veins = gm_idx[rng.choice(len(gm_idx), 60, replace=False)]
    vein = np.zeros_like(img, bool)
    vein[tuple(veins.T)] = True
    img = np.where(vein, img * 0.45, img)
    img = img * (1 + 0.02 * rng.standard_normal(img.shape))
    return (
        torch.from_numpy(img.astype(np.float32)),
        torch.from_numpy(tissue),
        torch.from_numpy(vein),
    )


def test_veins_flagged_and_dark_white_matter_predicted():
    img, tissue, vein = _phantom()
    brain = torch.ones_like(vein)
    ratio, expected, levels = tissue_darkness(img, tissue, brain, (4.0, 4.0, 4.0))
    # Centred by construction (the median log ratio is divided out); the fit's own
    # quality shows in the levels and in what gets flagged below.
    assert abs(float(ratio[~vein].median()) - 1.0) < 0.02
    region = vein_region(tissue, brain)
    m = vein_mask(ratio, region, darker=0.3, dilate=0)
    assert float(m[vein].float().mean()) > 0.95  # the veins
    wm = tissue[1] > 0.5
    assert int(m[wm].sum()) == 0  # dark white matter is predicted, not flagged
    assert int(m.sum()) < 2 * int(vein.sum())
    assert levels[1] < levels[0] < levels[2]


def test_cli_writes_the_mask_from_tissue_files(tmp_path):
    from fastfuncstuff.cli.veinmask import main

    img, tissue, vein = _phantom()
    aff = np.diag([0.8, 0.8, 0.8, 1.0])
    # load_image is (z, y, x); NIfTI is (x, y, z)
    epi = tmp_path / "epi.nii.gz"
    nib.Nifti1Image(img.numpy().transpose(2, 1, 0), aff).to_filename(str(epi))
    paths = []
    for k, name in enumerate(("c1", "c2", "c3")):
        p = tmp_path / f"{name}.nii.gz"
        nib.Nifti1Image(tissue[k].numpy().transpose(2, 1, 0), aff).to_filename(str(p))
        paths.append(str(p))
    pre = tmp_path / "vm"
    main(["-epi", str(epi), "-tissue", *paths, "-prefix", str(pre), "-dilate", "0",
          "-bias_fwhm", "8", "-device", "cpu", "-verb", "0"])  # fmt: skip
    got = np.asarray(nib.load(f"{pre}_mask.nii.gz").dataobj).transpose(2, 1, 0) > 0
    v = vein.numpy()
    assert got[v].mean() > 0.9
    assert got.sum() < 2 * v.sum()
