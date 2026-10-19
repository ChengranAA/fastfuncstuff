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


def _reml(tmp_path, bold, *extra, expect=0):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(bold)]
        + ["-events", str(tmp_path / "ev.tsv"), "-TR", str(TR), "-hrf_model", "TENT(0,12,5)"]
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
