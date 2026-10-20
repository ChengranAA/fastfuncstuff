"""ffs_reml -contrasts: [[contrasts]] TOML resolved against the events design, per-voxel HRFs included."""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import gamma

from fastfuncstuff.cli.util_concalc import _read_brick_labels

TR, N_TP, SHAPE = 1.5, 160, (4, 4, 2)
ONSETS = {"a": [6, 60, 114, 168, 195], "b": [33, 87, 141, 186, 213]}
TOML = """
[[contrasts]]
label = "a_minus_b"
sym = "SYM: +1*a -1*b"

[[contrasts]]
label = "mean_ab"
sym = "SYM: +0.5*a +0.5*b"
"""


def _hrf(t):
    return np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)


@pytest.fixture
def study(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(N_TP) * TR
    arr = 100 + rng.normal(0, 1, (*SHAPE, N_TP))
    arr[:2] += 6 * sum(_hrf(t - o) for o in ONSETS["a"])
    arr[2:] += 3 * sum(_hrf(t - o) for o in ONSETS["b"])
    nib.save(
        nib.Nifti1Image(arr.astype(np.float32), np.diag([3, 3, 3, 1])), tmp_path / "run.nii.gz"
    )
    rows = sorted((o, k) for k, v in ONSETS.items() for o in v)
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "\n".join(f"{o}\t1\t{k}" for o, k in rows) + "\n"
    )
    (tmp_path / "c.toml").write_text(TOML)
    # Two library curves (fast, slow) at the 0.1 s microtime grid; top half of the volume
    # uses the slow one, so the grouped (per-voxel HRF) path really has two designs.
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt * 0.7), _hrf(tt)])
    lib = lib / lib.max(axis=1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    idx = np.ones(SHAPE, np.float32)
    idx[:, :, 1] = 2
    nib.save(nib.Nifti1Image(idx, np.diag([3, 3, 3, 1])), tmp_path / "lib_hrf_index.nii.gz")
    return tmp_path


def _run(tmp_path, *extra, expect=0):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(tmp_path / "run.nii.gz")]
        + ["-events", str(tmp_path / "ev.tsv"), "-TR", str(TR), "-device", "cpu", *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == expect, res.stdout[-3000:] + res.stderr[-3000:]
    return res


def _bucket(path):
    img = nib.load(path)
    return _read_brick_labels(img), np.asanyarray(img.dataobj)


@pytest.mark.parametrize("per_voxel_hrf", [False, True])
def test_contrast_equals_the_weighted_betas(study, per_voxel_hrf):
    extra = ["-hrfopt_prefix", "lib"] if per_voxel_hrf else ["-hrf_model", "SPMG1"]
    _run(study, "-Obuck", "ols", "-Rbuck", "reml", "-contrasts", "c.toml", *extra)
    labels, data = _bucket(study / "ols.nii.gz")
    coef = {lab: data[..., i] for i, lab in enumerate(labels)}
    np.testing.assert_allclose(
        coef["a_minus_b#0_Coef"], coef["a#0_Coef"] - coef["b#0_Coef"], atol=1e-3
    )
    np.testing.assert_allclose(
        coef["mean_ab#0_Coef"], 0.5 * (coef["a#0_Coef"] + coef["b#0_Coef"]), atol=1e-3
    )
    assert "a_minus_b#0_Tstat" in coef
    reml_labels, reml_data = _bucket(study / "reml.nii.gz")
    reml = {lab: reml_data[..., i] for i, lab in enumerate(reml_labels)}
    np.testing.assert_allclose(
        reml["a_minus_b#0_Coef"], reml["a#0_Coef"] - reml["b#0_Coef"], atol=1e-3
    )


def test_contrasts_rejects_f_tests_and_unknown_conditions(study):
    (study / "f.toml").write_text('[[contrasts]]\nlabel = "f"\nsym = ["SYM: +1*a", "SYM: +1*b"]\n')
    res = _run(study, "-Obuck", "o1", "-contrasts", "f.toml", "-hrf_model", "SPMG1", expect=1)
    assert "F-test" in res.stdout + res.stderr
    (study / "u.toml").write_text('[[contrasts]]\nlabel = "u"\nsym = "SYM: +1*nope"\n')
    res = _run(study, "-Obuck", "o2", "-contrasts", "u.toml", "-hrf_model", "SPMG1", expect=1)
    assert "nope" in res.stdout + res.stderr
