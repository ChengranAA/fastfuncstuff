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


def test_local_levels_survive_contrast_that_drifts_across_the_slab():
    """The global model's failure on ds003427: GM/WM contrast that changes across the
    slab reads as darkness in a whole region. Fitted locally, the levels follow it and
    only the veins stay dark."""
    from fastfuncstuff.processing.veinmask import local_tissue_darkness

    img, tissue, vein = _phantom()
    n = img.shape[0]
    # GM brightness halves from the top slice to the bottom one, WM and CSF do not
    drift = torch.linspace(0.5, 1.0, n)[:, None, None]
    gm, wm, csf = tissue
    flat = 1000 * gm + 650 * wm + 1300 * csf
    img = img * (1000 * gm * drift + 650 * wm + 1300 * csf) / flat
    brain = torch.ones_like(vein)
    region = vein_region(tissue, brain)
    g_ratio, _, _ = tissue_darkness(img, tissue, brain, (6.0, 6.0, 6.0))
    l_ratio, _ = local_tissue_darkness(img, tissue, brain, (2.0, 2.0, 2.0))
    gm_ok = (gm > 0.5) & ~vein
    false_global = float((region & gm_ok & (g_ratio < 0.8)).float().mean() / gm_ok.float().mean())
    false_local = float((region & gm_ok & (l_ratio < 0.8)).float().mean() / gm_ok.float().mean())
    assert false_global > 0.05  # the bottom slices read dark
    assert false_local < 0.005
    assert float((l_ratio[vein] < 0.7).float().mean()) > 0.95


def _line_volume():
    """Darkness-ratio volume: a sheet, a tube through the slices and a dot, in flat 1."""
    n = 32
    r = torch.ones(n, n, n)
    r[:, 6, 4:28] = 0.5  # sheet: thin in y only
    r[2:30, 20, 20] = 0.5  # tube: thin in y and x, long in z
    r[16, 26, 8] = 0.4  # blob
    return r


def test_dark_shapes_names_plate_tube_and_blob():
    from fastfuncstuff.processing.veinmask import dark_shapes

    r = _line_volume()
    brain = torch.ones_like(r, dtype=torch.bool)
    thin, shape, ridge = dark_shapes(r, brain, (0.8, 0.8, 0.8))
    assert int(shape[16, 6, 16]) == 1
    assert int(shape[16, 20, 20]) == 2
    assert int(shape[16, 26, 8]) == 3
    # the centre line is a ridge, its neighbour across the thin direction is not
    assert bool(ridge[16, 20, 20]) and not bool(ridge[16, 21, 20])
    assert float(thin[16, 20, 20]) > 10 * float(thin[16, 20, 26])


def test_ridge_following_fills_a_faint_middle_but_not_the_halo():
    """A vessel through slices, dark at one end and only faintly dark along the rest
    (partial volume), with the smooth radial halo a real vessel has, in noise. The
    whole centre line joins the dark end; the halo voxels beside the faint part, which
    are dark enough for plain hysteresis, do not."""
    from fastfuncstuff.processing.veinmask import dark_shapes, ridge_follow_mask

    g = torch.Generator().manual_seed(0)
    n = 32
    z, y, x = torch.meshgrid(*(torch.arange(n, dtype=torch.float32),) * 3, indexing="ij")
    depth = torch.where(z < 8, 0.5, 0.22) * ((z >= 2) & (z < 30))
    r = 1 - depth * torch.exp(-((y - 16) ** 2 + (x - 16) ** 2) / (2 * 0.7**2))
    r = r + 0.03 * torch.randn(n, n, n, generator=g)
    brain = torch.ones_like(r, dtype=torch.bool)
    thin, _, ridge = dark_shapes(r, brain, (0.8, 0.8, 0.8))
    m = ridge_follow_mask(r, brain, thin, ridge, seed=0.7, grow=0.85)
    assert bool(m[2:30, 16, 16].all())  # the whole centre line
    halo = torch.stack(
        [m[10:28, 16 + dy, 16 + dx] for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    )
    assert int(halo.sum()) == 0


def test_cli_outputs_inherit_the_epi_format(tmp_path):
    """autoproc -final_format zst hands ffs_veinmask .nii.zst means and a bare prefix;
    the outputs must come back as .nii.zst, which is the name autoproc then reads."""
    from fastfuncstuff.cli.veinmask import main
    from fastfuncstuff.processing.io import save_image

    img, tissue, _ = _phantom()
    hdr = {"affine": np.diag([0.8, 0.8, 0.8, 1.0])}
    epi = tmp_path / "epi.nii.zst"
    save_image(img, str(epi), header_info=hdr)
    paths = []
    for k in range(3):
        p = tmp_path / f"c{k + 1}.nii.zst"
        save_image(tissue[k], str(p), header_info=hdr)
        paths.append(str(p))
    main(["-epi", str(epi), "-tissue", *paths, "-prefix", str(tmp_path / "vm"),
          "-device", "cpu", "-verb", "0"])  # fmt: skip
    assert (tmp_path / "vm_mask.nii.zst").exists() and not (tmp_path / "vm_mask.nii.gz").exists()
