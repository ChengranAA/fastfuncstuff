"""ffs_reml -atlas / -rois: average the ROI first, then fit that one series with ARMA(1,1)."""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.stats import gamma

TR, N_TP, SHAPE = 1.5, 160, (4, 4, 2)
ONSETS = {"a": [6, 60, 114, 168, 195], "b": [33, 87, 141, 186, 213]}


def _hrf(t):
    return np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)


@pytest.fixture
def study(tmp_path):
    rng = np.random.default_rng(11)
    t = np.arange(N_TP) * TR
    arr = 100 + rng.normal(0, 1, (*SHAPE, N_TP))
    arr[:2] += 6 * sum(_hrf(t - o) for o in ONSETS["a"])  # ROI 1 answers condition a only
    arr = arr.astype(np.float32)
    nib.save(nib.Nifti1Image(arr, np.diag([3, 3, 3, 1])), tmp_path / "run.nii.gz")
    lab = np.zeros(SHAPE, np.int16)
    lab[:2], lab[2:] = 1, 2
    nib.save(nib.Nifti1Image(lab, np.diag([3, 3, 3, 1])), tmp_path / "atlas.nii.gz")
    rows = [(o, k) for k, v in ONSETS.items() for o in v]
    ev = "onset\tduration\ttrial_type\n" + "\n".join(f"{o}\t1\t{k}" for o, k in sorted(rows))
    (tmp_path / "ev.tsv").write_text(ev + "\n")
    return tmp_path, arr


def _reml(tmp_path, bold, *extra, expect=0, hrf="TENT(0,12,5)"):
    runs = [str(b) for b in (bold if isinstance(bold, list) else [bold])]
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", *runs, "-events"]
        + [str(tmp_path / "ev.tsv")] * len(runs)
        + ["-TR", str(TR), "-hrf_model", hrf]
        + ["-tout", "-fout", "-device", "cpu", *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == expect, res.stdout[-3000:] + res.stderr[-3000:]
    return res


def _table(path):
    rows = [line.split("\t") for line in open(path).read().splitlines()]
    return rows[0], rows[1:]


def test_roi_mode_writes_table_and_curves_not_volumes(study):
    tmp_path, _ = study
    _reml(tmp_path, tmp_path / "run.nii.gz", "-atlas", str(tmp_path / "atlas.nii.gz"),
          "-Rbuck", str(tmp_path / "out"))  # fmt: skip
    header, rows = _table(tmp_path / "out_roi.tsv")
    assert header[:3] == ["roi", "name", "n_voxels"] and "Full_Fstat" in header
    fstat = {r[0]: float(r[header.index("Full_Fstat")]) for r in rows}
    assert fstat["1"] > 3 * fstat["2"]
    curve_hdr, curve = _table(tmp_path / "out_iresp_a.tsv")
    assert curve_hdr == ["time", "1", "2"]
    assert [float(r[0]) for r in curve] == [0.0, 3.0, 6.0, 9.0, 12.0]
    assert not list(tmp_path.glob("out*.nii*"))


def test_roi_fit_matches_the_fit_of_the_roi_mean_series(study):
    tmp_path, arr = study
    _reml(tmp_path, tmp_path / "run.nii.gz", "-atlas", str(tmp_path / "atlas.nii.gz"),
          "-Rbuck", str(tmp_path / "out"))  # fmt: skip
    _, roi_curve = _table(tmp_path / "out_iresp_a.tsv")
    got = np.array([float(r[1]) for r in roi_curve])

    mean = arr[:2].reshape(-1, N_TP).mean(axis=0).reshape(1, 1, 1, N_TP).astype(np.float32)
    nib.save(nib.Nifti1Image(mean, np.diag([3, 3, 3, 1])), tmp_path / "mean.nii.gz")
    _reml(tmp_path, tmp_path / "mean.nii.gz", "-Rbuck", str(tmp_path / "ref"))
    ref = nib.load(tmp_path / "ref.nii.gz")
    labels = [lab for lab in open(tmp_path / "ref.json").read().split('"') if "_Coef" in lab]
    data = ref.get_fdata().reshape(-1)
    want = np.array(
        [
            data[i]
            for i, lab in enumerate(_labels(tmp_path / "ref.json"))
            if lab.startswith("a_t") and lab.endswith("_Coef")
        ]
    )
    assert labels and len(want) == 5
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def _labels(path):
    import json

    return json.load(open(path))["SubBricks"]


def test_roi_maps_paint_the_bucket(study):
    tmp_path, _ = study
    _reml(tmp_path, tmp_path / "run.nii.gz", "-rois", str(tmp_path / "atlas.nii.gz"),
          "-Rbuck", str(tmp_path / "out"), "-roi_maps")  # fmt: skip
    vol = nib.load(tmp_path / "out.nii.gz").get_fdata()
    assert vol.shape[:3] == SHAPE
    flat = vol[:2].reshape(-1, vol.shape[-1])
    assert np.ptp(flat, axis=0).max() < 1e-5  # one value per ROI
    assert not np.allclose(vol[0, 0, 0], vol[3, 0, 0])


def test_voxel_only_options_are_refused(study):
    tmp_path, _ = study
    res = _reml(tmp_path, tmp_path / "run.nii.gz", "-atlas", str(tmp_path / "atlas.nii.gz"),
                "-clustsim", "-Rbuck", str(tmp_path / "out"), expect=1)  # fmt: skip
    assert "-clustsim" in res.stderr


def test_single_trials_per_roi_match_the_roi_mean_series(study):
    tmp_path, arr = study
    _reml(tmp_path, tmp_path / "run.nii.gz", "-atlas", str(tmp_path / "atlas.nii.gz"),
          "-Rbuck", str(tmp_path / "out"), "-single_trials", "st", hrf="spmg1")  # fmt: skip
    header, rows = _table(tmp_path / "out_single_trials.tsv")
    assert header == ["trial", "label", "onset_tr", "1", "2"]
    assert len(rows) == 10 and [int(r[2]) for r in rows] == sorted(int(r[2]) for r in rows)
    assert (tmp_path / "out_single_trials_tstat.tsv").exists()
    got = np.array([float(r[3]) for r in rows])

    mean = arr[:2].reshape(-1, N_TP).mean(axis=0).reshape(1, 1, 1, N_TP).astype(np.float32)
    nib.save(nib.Nifti1Image(mean, np.diag([3, 3, 3, 1])), tmp_path / "mean.nii.gz")
    _reml(tmp_path, tmp_path / "mean.nii.gz", "-Rbuck", str(tmp_path / "ref"),
          "-single_trials", "ref", hrf="spmg1")  # fmt: skip
    want = nib.load(tmp_path / "reml_ref_single.nii.gz").get_fdata().reshape(-1)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_surface_data_with_a_surface_roi(tmp_path):
    from fastfuncstuff.io.freesurfer import write_label
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data

    rng = np.random.default_rng(2)
    n_vert = 300
    t = np.arange(N_TP) * TR
    data = 100 + rng.normal(0, 1, (n_vert, N_TP))
    data[:40] += 6 * sum(_hrf(t - o) for o in ONSETS["a"])
    save_gifti_data(tmp_path / "lh.bold.func.gii", data.astype(np.float32),
                    {"TR_seconds": str(TR), "mesh_fingerprint": "M"})  # fmt: skip
    lab = np.zeros(n_vert)
    lab[:40], lab[40:100] = 1, 2
    save_gifti_data(tmp_path / "lh.parc.shape.gii", lab.astype(np.float32),
                    {"mesh_fingerprint": "M"}, time_series=False)  # fmt: skip
    write_label(tmp_path / "lh.V1.label", np.arange(40), np.zeros((40, 3)), np.ones(40))
    rows = [(o, k) for k, v in ONSETS.items() for o in v]
    ev = "onset\tduration\ttrial_type\n" + "\n".join(f"{o}\t1\t{k}" for o, k in sorted(rows))
    (tmp_path / "ev.tsv").write_text(ev + "\n")

    bold = tmp_path / "lh.bold.func.gii"
    _reml(tmp_path, bold, "-atlas", str(tmp_path / "lh.parc.shape.gii"),
          "-Rbuck", str(tmp_path / "s.func.gii"), "-roi_maps")  # fmt: skip
    header, table = _table(tmp_path / "s_roi.tsv")
    fstat = [float(r[header.index("Full_Fstat")]) for r in table]
    assert fstat[0] > 3 * fstat[1]
    stats, meta = load_gifti_data(tmp_path / "s.func.gii")
    assert meta["mesh_fingerprint"] == "M"
    assert np.ptp(stats[:40], axis=0).max() < 1e-5 and not stats[100:].any()

    _reml(tmp_path, bold, "-rois", str(tmp_path / "lh.V1.label"), "-Rbuck", str(tmp_path / "v"))
    _, table = _table(tmp_path / "v_roi.tsv")
    assert [r[1] for r in table] == ["lh.V1"]


@pytest.fixture
def two_runs(study):
    """The study twice over (independent noise), so leave-one-run-out has folds."""
    tmp_path, arr = study
    rng = np.random.default_rng(5)
    arr2 = (arr + rng.normal(0, 1, arr.shape)).astype(np.float32)
    nib.save(nib.Nifti1Image(arr2, np.diag([3, 3, 3, 1])), tmp_path / "run2.nii.gz")
    return tmp_path, [arr, arr2]


def _two(tmp_path, *extra, data=None, **kw):
    return _reml(
        tmp_path, data or [tmp_path / "run.nii.gz", tmp_path / "run2.nii.gz"], *extra, **kw
    )


def _save_means(tmp_path, arrs):
    paths = []
    for k, a in enumerate(arrs):
        mean = a[:2].reshape(-1, N_TP).mean(axis=0).reshape(1, 1, 1, N_TP).astype(np.float32)
        paths.append(tmp_path / f"mean{k}.nii.gz")
        nib.save(nib.Nifti1Image(mean, np.diag([3, 3, 3, 1])), paths[-1])
    return paths


def test_xval_r2_lands_in_the_roi_table_and_matches_the_mean_series(two_runs):
    tmp_path, arrs = two_runs
    _two(tmp_path, "-atlas", str(tmp_path / "atlas.nii.gz"), "-Rbuck", str(tmp_path / "out"),
         "-xval_r2", "-roi_maps")  # fmt: skip
    header, rows = _table(tmp_path / "out_roi.tsv")
    assert "xval_R2" in header
    r2 = [float(r[header.index("xval_R2")]) for r in rows]
    assert r2[0] > 0.2 and r2[0] > r2[1]
    painted = nib.load(tmp_path / "out_xval_r2.nii.gz").get_fdata()
    np.testing.assert_allclose(painted[0, 0, 0].reshape(-1)[0], r2[0], rtol=1e-4)

    _two(tmp_path, "-Rbuck", str(tmp_path / "ref"), "-xval_r2", data=_save_means(tmp_path, arrs))
    want = nib.load(tmp_path / "ref_xval_r2.nii.gz").get_fdata().reshape(-1)[0]
    np.testing.assert_allclose(r2[0], want, rtol=1e-3, atol=1e-4)


def test_beta_cv_per_roi_matches_the_mean_series(two_runs):
    tmp_path, arrs = two_runs
    _two(tmp_path, "-atlas", str(tmp_path / "atlas.nii.gz"), "-Rbuck", str(tmp_path / "out"),
         "-beta_cv", hrf="spmg1")  # fmt: skip
    header, rows = _table(tmp_path / "out_beta_cv.tsv")
    assert header == ["roi", "name", "n_voxels", "beta_cv_R2", "a_mean_beta", "b_mean_beta"]
    st_hdr, trials = _table(tmp_path / "out_single_trials.tsv")
    assert st_hdr == ["trial", "label", "condition", "run", "1", "2"] and len(trials) == 20
    assert not list(tmp_path.glob("out*.nii*"))
    got = np.array([float(r[4]) for r in trials])

    _two(tmp_path, "-Rbuck", str(tmp_path / "ref"), "-beta_cv", hrf="spmg1",
         data=_save_means(tmp_path, arrs))  # fmt: skip
    want = nib.load(tmp_path / "ref_single_trial_betas.nii.gz").get_fdata().reshape(-1)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
    want_r2 = nib.load(tmp_path / "ref_single_trial_xval_r2.nii.gz").get_fdata().reshape(-1)[0]
    np.testing.assert_allclose(float(rows[0][3]), want_r2, rtol=1e-3, atol=1e-4)
