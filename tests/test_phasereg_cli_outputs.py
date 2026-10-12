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
        pha = rng.normal(0, 0.1, (*shape, nt))
        mag = 1000 + slope[..., None] * pha + rng.normal(0, 5, (*shape, nt))
        for kind, arr, out in (("mag", mag, mags), ("pha", pha, phas)):
            img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
            img.header["pixdim"][4] = 2.0
            path = tmp_path / f"{kind}{i + 1}.nii.gz"
            nib.save(img, path)
            out.append(str(path))
    return mags, phas


def _run(tmp_path, mags, phas, prefix, *extra):
    argv = ["-magnitude", *mags, "-phase", *phas, "-prefix", str(tmp_path / prefix)]
    assert main([*argv, "-polort", "1", "-device", "cpu", *extra]) == 0


def test_runs_are_fitted_together_and_written_apart(tmp_path):
    """Each run's corrected series lands in its own file with its own length, and
    the files are exactly the concatenated output cut at the run boundaries."""
    lengths = (40, 33)
    mags, phas = _write_runs(tmp_path, lengths)
    _run(tmp_path, mags, phas, "split")
    _run(tmp_path, mags, phas, "cat", "-concat_output")

    cat = np.asarray(nib.load(tmp_path / "cat_corrected.nii.gz").dataobj)
    assert cat.shape[3] == sum(lengths)
    assert not (tmp_path / "split_corrected.nii.gz").exists()
    parts = []
    for i, nt in enumerate(lengths):
        for key in ("corrected", "macro"):
            img = nib.load(tmp_path / f"split_{key}_run-{i + 1:02d}.nii.gz")
            assert img.shape[3] == nt, (key, i)
        parts.append(
            np.asarray(nib.load(tmp_path / f"split_corrected_run-{i + 1:02d}.nii.gz").dataobj)
        )
    np.testing.assert_allclose(np.concatenate(parts, axis=3), cat, rtol=1e-5, atol=1e-3)
    # One slope map for the pooled fit, not one per run.
    assert (tmp_path / "split_slope.nii.gz").exists()


def test_single_run_names_are_unchanged(tmp_path):
    mags, phas = _write_runs(tmp_path, (40,))
    _run(tmp_path, mags, phas, "one")
    assert (tmp_path / "one_corrected.nii.gz").exists()
    assert not list(tmp_path.glob("one_corrected_run-*"))
