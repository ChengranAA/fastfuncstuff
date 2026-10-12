"""ffs_phasereg output layout: per-run 4D files when several runs go in."""

import nibabel as nib
import numpy as np

from fastfuncstuff.cli.phasereg import main


def _write_runs(tmp_path, lengths, seed=0):
    rng = np.random.default_rng(seed)
    shape = (6, 5, 4)
    slope = rng.normal(0, 30, shape)
    mags, phas = [], []
    for i, nt in enumerate(lengths):
        pha = rng.normal(0, 0.3, (*shape, nt))
        mag = 1000 + slope[..., None] * pha + rng.normal(0, 1, (*shape, nt))
        for kind, arr, out in (("mag", mag, mags), ("pha", pha, phas)):
            img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
            img.header["pixdim"][4] = 2.0
            path = tmp_path / f"{kind}{i + 1}.nii.gz"
            nib.save(img, path)
            out.append(str(path))
    return mags, phas, slope


def _run(tmp_path, mags, phas, prefix, *extra):
    argv = ["-magnitude", *mags, "-phase", *phas, "-prefix", str(tmp_path / prefix)]
    assert main([*argv, "-polort", "1", "-device", "cpu", *extra]) == 0


def test_runs_are_fitted_together_and_written_apart(tmp_path):
    """Each run's corrected series lands in its own file with its own length, and
    the files are exactly the concatenated output cut at the run boundaries."""
    lengths = (40, 33)
    mags, phas, planted = _write_runs(tmp_path, lengths)
    _run(tmp_path, mags, phas, "split", "-save_macro")
    _run(tmp_path, mags, phas, "cat", "-concat_output")

    cat = np.asarray(nib.load(tmp_path / "cat_corrected.nii.gz").dataobj)
    assert cat.shape[3] == sum(lengths)
    assert not (tmp_path / "split_corrected.nii.gz").exists()
    assert not (tmp_path / "cat_macro.nii.gz").exists(), "macro 4D is opt-in"
    parts = []
    for i, nt in enumerate(lengths):
        for key in ("corrected", "macro"):
            img = nib.load(tmp_path / f"split_{key}_run-{i + 1:02d}.nii.gz")
            assert img.shape[3] == nt, (key, i)
        parts.append(
            np.asarray(nib.load(tmp_path / f"split_corrected_run-{i + 1:02d}.nii.gz").dataobj)
        )
    np.testing.assert_allclose(np.concatenate(parts, axis=3), cat, rtol=1e-5, atol=1e-3)
    # One slope map for the pooled fit, not one per run -- and it is the planted one,
    # which a loader that mixed up magnitude and phase, or voxel order, would miss.
    slope = np.asarray(nib.load(tmp_path / "split_slope.nii.gz").dataobj)
    fitted = slope != 0
    assert fitted.mean() > 0.5
    r = np.corrcoef(slope[fitted], planted[fitted])[0, 1]
    assert r > 0.95


def test_single_run_names_are_unchanged(tmp_path):
    mags, phas, _ = _write_runs(tmp_path, (40,))
    _run(tmp_path, mags, phas, "one")
    assert (tmp_path / "one_corrected.nii.gz").exists()
    assert not list(tmp_path.glob("one_corrected_run-*"))


def test_outputs_inherit_the_input_format(tmp_path):
    """.nii.zst in -> .nii.zst out with a bare prefix; an explicit extension still wins."""
    import shutil
    import subprocess

    import pytest

    if shutil.which("zstd") is None:
        pytest.skip("zstd not installed")
    mags, phas, _ = _write_runs(tmp_path, (30, 30))
    zst = []
    for f in [*mags, *phas]:
        plain = f.replace(".nii.gz", ".nii")
        nib.save(nib.load(f), plain)
        subprocess.run(["zstd", "-q", "-f", "--rm", plain, "-o", plain + ".zst"], check=True)
        zst.append(plain + ".zst")
    _run(tmp_path, zst[:2], zst[2:], "z")
    assert (tmp_path / "z_corrected_run-01.nii.zst").exists()
    assert (tmp_path / "z_slope.nii.zst").exists()
    assert not list(tmp_path.glob("z_*.nii.gz"))
    _run(tmp_path, zst[:2], zst[2:], "g.nii.gz")
    assert (tmp_path / "g_slope.nii.gz").exists()


def test_input_nifti_ext():
    from fastfuncstuff.cli_utils import input_nifti_ext

    assert input_nifti_ext("a/run.nii.zst") == ".nii.zst"
    assert input_nifti_ext("run.nii.zst[0..9]") == ".nii.zst"
    assert input_nifti_ext("run.nii") == ".nii"
    assert input_nifti_ext("run+orig.HEAD") == ".nii.gz"


def test_a_given_mask_is_applied_at_load_and_matches_the_unmasked_fit(tmp_path):
    """-mask loads only in-mask voxels; inside it the fit is the unmasked one, outside 0."""
    mags, phas, _ = _write_runs(tmp_path, (40, 33))
    rng = np.random.default_rng(9)
    mask = rng.random((6, 5, 4)) < 0.5
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask.astype(np.float32), np.eye(4)), mask_path)
    _run(tmp_path, mags, phas, "m", "-mask", str(mask_path), "-signal_thresh", "0")
    _run(tmp_path, mags, phas, "all", "-no_auto_mask", "-signal_thresh", "0")
    for i in (1, 2):
        got = np.asarray(nib.load(tmp_path / f"m_corrected_run-0{i}.nii.gz").dataobj)
        ref = np.asarray(nib.load(tmp_path / f"all_corrected_run-0{i}.nii.gz").dataobj)
        np.testing.assert_allclose(got[mask], ref[mask], rtol=1e-5)
        assert not got[~mask].any()
