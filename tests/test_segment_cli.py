"""Smoke tests for the ffs_segment CLI (end-to-end on a tiny phantom)."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import torch

from fastfuncstuff.cli.segment import main


def _write_nii(path, arr, affine=np.eye(4)):
    # arr is (nz, ny, nx[, t]); NIfTI wants (nx, ny, nz[, t])
    arr = np.asarray(arr.cpu() if hasattr(arr, "cpu") else arr, np.float32)
    data = arr.transpose(2, 1, 0) if arr.ndim == 3 else arr.transpose(2, 1, 0, 3)
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _phantom(n=20):
    zz, yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), torch.arange(n), indexing="ij")
    r = ((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2).sqrt()
    tissue = torch.stack(
        [(r < 4).float(), ((r >= 4) & (r < 8)).float(), ((r >= 8) & (r < 10)).float()]
    )
    means = torch.tensor([150.0, 80.0, 40.0])
    vol = sum(means[k] * tissue[k] for k in range(3))
    return vol, tissue


def test_segment_cli_writes_all_outputs(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)  # (nz,ny,nx,3)

    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input",
            str(inp),
            "-tpm",
            str(tpmp),
            "-prefix",
            str(tmp_path / "seg"),
            "-ngaus",
            "1",
            "1",
            "1",
            "-biasfwhm",
            "10",
            "-samp",
            "1",
            "-niter",
            "5",
            "-no_warp",
            "-device",
            "cpu",
            "-quiet",
        ]
    )
    assert rc == 0
    for t in range(3):
        assert (tmp_path / f"seg_c{t + 1}.nii.gz").exists()
    assert (tmp_path / "seg_biascorrected.nii.gz").exists()
    assert (tmp_path / "seg_biasfield.nii.gz").exists()

    # c1 (core) should light up the core region
    c1 = nib.load(str(tmp_path / "seg_c1.nii.gz")).get_fdata().transpose(2, 1, 0)
    core = tissue[0].numpy() > 0.5
    assert c1[core].mean() > 0.8


def test_segment_cli_writes_invwarp_that_inverts_warp(tmp_path):
    """With the deformation on, _invwarp.nii.gz is written and is the fixed-point
    inverse of _warp: composing them returns near-identity on the interior."""
    from fastfuncstuff.processing.interp import trilinear_interpolate

    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input", str(inp), "-tpm", str(tpmp), "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1", "-biasfwhm", "10", "-samp", "1", "-niter", "5",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    assert (tmp_path / "seg_warp.nii.gz").exists()
    assert (tmp_path / "seg_invwarp.nii.gz").exists()

    # Load both fields (NIfTI (nx,ny,nz,1,3) -> voxel disp per component). Only the
    # relative round-trip matters, so read them back the same way and compose.
    def _load_field(p):
        d = nib.load(str(p)).get_fdata()  # (nx,ny,nz,[1,]3)
        if d.ndim == 5:
            d = d[:, :, :, 0, :]
        d = d.transpose(2, 1, 0, 3)  # (nz,ny,nx,3)
        return torch.from_numpy(np.ascontiguousarray(d)).float()

    fwd = _load_field(tmp_path / "seg_warp.nii.gz")
    inv = _load_field(tmp_path / "seg_invwarp.nii.gz")
    nz, ny, nx, _ = fwd.shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz).float(), torch.arange(ny).float(), torch.arange(nx).float(), indexing="ij"
    )
    xf, yf, zf = xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)
    qx, qy, qz = (
        xf + inv[..., 0].reshape(-1),
        yf + inv[..., 1].reshape(-1),
        zf + inv[..., 2].reshape(-1),
    )
    rx = qx + trilinear_interpolate(fwd[..., 0].contiguous(), qx, qy, qz)
    ry = qy + trilinear_interpolate(fwd[..., 1].contiguous(), qx, qy, qz)
    rz = qz + trilinear_interpolate(fwd[..., 2].contiguous(), qx, qy, qz)
    resid = torch.sqrt((rx - xf) ** 2 + (ry - yf) ** 2 + (rz - zf) ** 2)
    m = (xf > 3) & (xf < nx - 4) & (yf > 3) & (yf < ny - 4) & (zf > 3) & (zf < nz - 4)
    assert resid[m].max() < 0.05, f"warp∘invwarp not identity: max resid {resid[m].max():.4f} vox"


def test_segment_cli_no_invwarp_flag(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp = tmp_path / "input.nii.gz"
    tpmp = tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)
    rc = main(
        [
            "-input", str(inp), "-tpm", str(tpmp), "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1", "-biasfwhm", "10", "-samp", "1", "-niter", "5",
            "-no_invwarp", "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    assert (tmp_path / "seg_warp.nii.gz").exists()
    assert not (tmp_path / "seg_invwarp.nii.gz").exists()


def test_segment_cli_ngaus_length_checked(tmp_path):
    vol, tissue = _phantom()
    tpm = (tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)).permute(1, 2, 3, 0)
    inp, tpmp = tmp_path / "i.nii.gz", tmp_path / "t.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)
    try:
        main(
            [
                "-input",
                str(inp),
                "-tpm",
                str(tpmp),
                "-prefix",
                str(tmp_path / "s"),
                "-ngaus",
                "1",
                "1",
                "-device",
                "cpu",
                "-quiet",
            ]
        )  # 2 != 3 classes
        raise AssertionError("expected SystemExit for mismatched -ngaus")
    except SystemExit:
        pass


def test_segment_cli_save_histogram(tmp_path):
    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    tpm = prob.permute(1, 2, 3, 0)
    inp, tpmp = tmp_path / "input.nii.gz", tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol)
    _write_nii(tpmp, tpm)

    rc = main(
        [
            "-input", str(inp),
            "-tpm", str(tpmp),
            "-prefix", str(tmp_path / "seg"),
            "-ngaus", "1", "1", "1",
            "-biasfwhm", "10", "-samp", "1", "-niter", "5", "-no_warp",
            "-save_histogram",  # bare → prefix_histogram.png
            "-tissue_names", "GM", "WM", "CSF",
            "-device", "cpu", "-quiet",
        ]
    )  # fmt: skip
    assert rc == 0
    hist = tmp_path / "seg_histogram.png"
    assert hist.exists() and hist.stat().st_size > 0


def _oblique_affine(angle_deg, origin, vox=1.0):
    a = np.radians(angle_deg)
    aff = np.eye(4)
    aff[:3, :3] = (
        np.array([[1.0, 0.0, 0.0], [0.0, np.cos(a), -np.sin(a)], [0.0, np.sin(a), np.cos(a)]]) * vox
    )
    aff[:3, 3] = origin
    return aff


def _seg_args(inp, tpmp, prefix, *extra):
    return [
        "-input", str(inp), "-tpm", str(tpmp), "-prefix", str(prefix),
        "-ngaus", "1", "1", "1", "-biasfwhm", "10", "-samp", "1", "-niter", "3",
        "-no_warp", "-device", "cpu", "-quiet", *extra,
    ]  # fmt: skip


def test_segment_cli_oblique_tpm_with_matrix_demands_matrix_base(tmp_path):
    """[[project_oblique_frame_slip]]: an .aff12.1D read against the wrong base grid is a
    silent multi-mm shift when that grid is oblique, so refuse instead of guessing.

    This is the bug that cost a whole subject: autoproc fitted the matrix with
    -base stage09.anat_autobox and ffs_segment read it against the TPM on the
    uncropped (5.04 deg oblique) anat -- 6.59 mm off, with nothing in the log.
    """
    import pytest

    from fastfuncstuff.processing.affine import save_matrix_1D

    vol, tissue = _phantom()
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    oblique = _oblique_affine(5.04, [-9.0, -11.0, -7.0])

    inp, tpmp = tmp_path / "input.nii.gz", tmp_path / "tpm.nii.gz"
    _write_nii(inp, vol, oblique)
    _write_nii(tpmp, prob.permute(1, 2, 3, 0), oblique)
    mat = tmp_path / "m.aff12.1D"
    save_matrix_1D(torch.eye(4), str(mat))

    with pytest.raises(SystemExit, match="-matrix_base"):
        main(_seg_args(inp, tpmp, tmp_path / "seg", "-1Dmatrix", str(mat)))

    # Naming the grid the matrix was fitted on (here, the TPM itself) lets it run.
    assert main(_seg_args(inp, tpmp, tmp_path / "ok", "-1Dmatrix", str(mat),
                          "-matrix_base", str(tpmp))) == 0  # fmt: skip
    assert (tmp_path / "ok_c1.nii.gz").exists()

    # A cardinal TPM never slips, so it keeps working with no -matrix_base.
    cinp, ctpm = tmp_path / "ci.nii.gz", tmp_path / "ct.nii.gz"
    _write_nii(cinp, vol)
    _write_nii(ctpm, prob.permute(1, 2, 3, 0))
    assert main(_seg_args(cinp, ctpm, tmp_path / "card", "-1Dmatrix", str(mat))) == 0


def test_segment_cli_matrix_base_crop_recovers_the_alignment(tmp_path):
    """-matrix_base on an oblique anat: the anat cast into input space must land on the
    input. Without it the same matrix puts it mm away."""
    from fastfuncstuff.processing.affine import save_matrix_1D

    vol, tissue = _phantom(n=24)
    prob = tissue / tissue.sum(0, keepdim=True).clamp_min(1e-6)
    # A steeper tilt than the real subject's 5.04 deg, on bigger voxels: the slip
    # scales with both, and this keeps the pass/fail gap well clear of interpolation
    # noise. 7T oblique slabs reach 42 deg, so 20 is not exotic.
    parent = _oblique_affine(20.0, [-11.0, -13.0, -9.0], vox=2.0)

    # The TPM sits on `parent`; the matrix was fitted against a CROP of it (origin
    # shifted along the tilted axes), exactly as autoproc's anat autobox is.
    crop = parent.copy()
    crop[:3, 3] = parent[:3, 3] + parent[:3, :3] @ np.array([6.0, 7.0, 5.0])

    inp, tpmp = tmp_path / "input.nii.gz", tmp_path / "tpm.nii.gz"
    cropd = tmp_path / "crop.nii.gz"
    _write_nii(inp, vol, parent)  # the input already sits on the template's grid...
    _write_nii(tpmp, prob.permute(1, 2, 3, 0), parent)
    _write_nii(cropd, vol[6:, 7:, 5:], crop)

    # ...so the alignment is perfect, and the aff12 an allineate with -base crop would
    # write for it is the crop->parent voxel identity expressed in the CROP's frame.
    # NOT a raw DICOM identity: that would mean the crop's cardinal mm equals the
    # parent's, which is exactly the false belief under test.
    mat = tmp_path / "m.aff12.1D"
    save_matrix_1D(
        torch.as_tensor(np.linalg.inv(parent) @ crop, dtype=torch.float32),
        str(mat),
        base_affine=crop,
        source_affine=parent,
    )

    def run(prefix, *extra):
        assert main(_seg_args(inp, tpmp, tmp_path / prefix, "-1Dmatrix", str(mat),
                              "-tpm_source", str(inp), *extra)) == 0  # fmt: skip
        got = nib.load(str(tmp_path / f"{prefix}_tpm_source_in_epi_initial.nii.gz"))
        return np.asarray(got.get_fdata().transpose(2, 1, 0), dtype=np.float32)

    ref = np.asarray(vol, dtype=np.float32)
    wrong = run("slip", "-matrix_base", str(tpmp))  # the old (buggy) reading
    right = run("fixed", "-matrix_base", str(cropd))  # the matrix's real base

    def corr(a):
        m = (ref > 0) | (a > 0)
        return float(np.corrcoef(ref[m].ravel(), a[m].ravel())[0, 1])

    assert corr(right) > 0.999, f"rebased cast should land on the input, r={corr(right)}"
    assert corr(wrong) < 0.9, f"mis-based cast should visibly slip, r={corr(wrong)}"
