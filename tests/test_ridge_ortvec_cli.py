"""End-to-end ffs_ridge CLI fits: -ortvec* nuisance and -ridge_fracs edge cases."""

import sys

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli.ridge import main


def _fit(tmp_path, monkeypatch, single_trials, ortvec, fracs=("0.1", "1", "0.9")):
    rng = np.random.default_rng(5)
    n_t = 40
    vols = []
    for run in range(2):
        # Pure nuisance plus a little noise: once the per-run regressor is
        # projected out, nothing is left for the trial betas to pick up.
        nuis = rng.normal(0, 1, n_t)
        np.savetxt(tmp_path / f"motion_run-0{run + 1}.1D", nuis)
        data = 100 + 20 * nuis + rng.normal(0, 0.01, (2, 2, 2, n_t))
        img = nib.Nifti1Image(data.astype(np.float32), np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((1, 1, 1, 2))
        vol = tmp_path / f"vol{run}.nii.gz"
        nib.save(img, vol)
        vols.append(str(vol))
    onset = tmp_path / "onsets.1D"
    onset.write_text("4 20 36 52\n4 20 36 52\n")
    prefix = str(tmp_path / ("with" if ortvec else "without"))
    argv = ["ffs_ridge", "-input", *vols, "-onsets", str(onset), "-durations", "1"]
    argv += ["-device", "cpu", "-polort", "1", "-prefix", prefix, "-ridge_fracs", *fracs]
    if single_trials:
        argv.append("-single_trials")
    if ortvec:
        argv += ["-ortvec_glob", str(tmp_path / "motion_run-0*.1D"), "motion"]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    tag = "single_trial_betas" if single_trials else "betas_single_trial"
    return np.abs(nib.load(f"{prefix}_{tag}.nii.gz").get_fdata()).mean()


@pytest.mark.parametrize("single_trials", [False, True])
def test_ortvec_glob_projected_out(tmp_path, monkeypatch, single_trials):
    without = _fit(tmp_path, monkeypatch, single_trials, ortvec=False)
    with_ortvec = _fit(tmp_path, monkeypatch, single_trials, ortvec=True)
    assert with_ortvec < 0.05 * without


def test_single_fraction_single_trials(tmp_path, monkeypatch):
    # The no-repeats warning tells users to pass one fraction; the selection
    # used to drop the last (OLS) column regardless and crash on an empty slice.
    betas = _fit(tmp_path, monkeypatch, True, ortvec=False, fracs=("0.5", "0.5", "1"))
    assert np.isfinite(betas)


def test_single_trials_writes_temporal_similarity(tmp_path, monkeypatch):
    _fit(tmp_path, monkeypatch, True, ortvec=False)
    assert (tmp_path / "without_temporal_similarity.png").stat().st_size > 0
    header = (tmp_path / "without_temporal_similarity.tsv").read_text().splitlines()[0]
    assert header.startswith("trials_apart\tOLS (frac 1)\tridge (")
