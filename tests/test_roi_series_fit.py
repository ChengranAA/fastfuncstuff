"""An ffs_nwarp ROI series (ROI x 1 x depth x time) goes straight into ffs_reml / ffs_deconvolve.

The referee for the reml side is the file itself read as a plain volume: its elements as
voxels, the same ARMA(1,1) fit, the same numbers -- so adopting it as ROI means changes
what is reported (names, depths, coverage), never what is fitted.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np

from fastfuncstuff.cli import nwarp
from fastfuncstuff.io.headers import read_brick_labels
from tests.test_nwarp_points import _write_aff1d
from tests.test_roi_depth_cli import N_TP, TR, _surfs, _table, brain  # noqa: F401


def _series(tmp):
    _write_aff1d(tmp / "ident150.aff12.1D", [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]] * N_TP)
    nwarp.main(
        [str(a) for a in (
            "-source", tmp / "bold.nii.gz", "-nwarp", tmp / "ident150.aff12.1D",
            "-master", tmp / "bold.nii.gz", "-prefix", tmp / "ser", "-atlas",
            tmp / "patch.lh.label", *_surfs(tmp), "-depths", "3", "-equivolume",
            "-interp", "cubic", "-device", "cpu", "-verb", "0",
        )]
    )  # fmt: skip
    return tmp / "ser.roi.nii.gz"


def _reml(tmp, bold, prefix, *extra):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(bold), "-events"]
        + [str(tmp / "ev.tsv"), "-TR", str(TR), "-hrf_model", "TENT(0,14,8)", "-tout", "-fout"]
        + ["-device", "cpu", "-Rbuck", str(prefix), *extra],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res


def test_reml_adopts_the_series_and_matches_the_plain_volume_fit(brain):  # noqa: F811
    tmp = brain
    ser = _series(tmp)
    res = _reml(tmp, ser, tmp / "adopt")
    assert "ROI series" in res.stdout
    header, rows = _table(tmp / "adopt_roi.tsv")
    assert header[:7] == ["roi", "name", "hemi", "depth", "depth_frac", "n_columns", "coverage"]
    assert header[7] == "volume_mm3" and [r[3] for r in rows] == ["1", "2", "3"]
    fstat = [float(r[header.index("Full_Fstat")]) for r in rows]
    assert fstat[0] < fstat[1] < fstat[2]

    # The same file as a plain (n_rois, 1, n_depths) volume, intent cleared.
    img = nib.load(ser)
    hdr = img.header.copy()
    hdr.set_intent("none", name="")
    nib.save(nib.Nifti1Image(np.asarray(img.dataobj), img.affine, hdr), tmp / "plain.nii.gz")
    _reml(tmp, tmp / "plain.nii.gz", tmp / "plainfit")
    vol = nib.load(tmp / "plainfit.nii.gz")
    labels = read_brick_labels(vol)
    plain_f = np.asarray(vol.dataobj)[0, 0, :, labels.index("Full_Fstat")]
    # ARMA(1,1) is searched per element but its optimiser is not bit-identical across
    # grid layouts (F differs 1e-3 at worst here); a divisor slip would show in the betas.
    np.testing.assert_allclose(plain_f, fstat, rtol=3e-3)
    for lab in ("vis_t4.0s#0_Coef", "vis_t6.0s#0_Coef"):
        got = np.array([float(r[header.index(lab)]) for r in rows])
        np.testing.assert_allclose(
            np.asarray(vol.dataobj)[0, 0, :, labels.index(lab)], got, rtol=5e-3
        )


def test_series_with_blur_or_a_second_atlas_is_refused(brain):  # noqa: F811
    tmp = brain
    ser = tmp / "ser.roi.nii.gz"
    if not ser.exists():
        ser = _series(tmp)
    for extra, msg in (
        (["-do_blur", "4"], "spatial"),
        (["-atlas", str(tmp / "patch.lh.label")], "already an ROI series"),
    ):
        res = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(ser), "-events"]
            + [str(tmp / "ev.tsv"), "-TR", str(TR), "-hrf_model", "TENT(0,14,8)"]
            + ["-device", "cpu", "-Rbuck", str(tmp / "bad"), *extra],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp,
        )  # fmt: skip
        assert res.returncode == 1 and msg in res.stderr, res.stdout[-800:] + res.stderr[-800:]


def test_deconvolve_reads_the_series_too(brain):  # noqa: F811
    tmp = brain
    ser = tmp / "ser.roi.nii.gz"
    if not ser.exists():
        ser = _series(tmp)
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.deconvolve", "-input", str(ser), "-onsets"]
        + [str(tmp / "stim.1D"), "-model", "TENT", "-window", "0", "14", "-prefix", str(tmp / "dec")]
        + ["-device", "cpu"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    header, rows = _table(tmp / "dec_roi.tsv")
    assert header[3] == "depth" and [r[3] for r in rows] == ["1", "2", "3"]
