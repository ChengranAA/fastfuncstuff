"""ffs_reml -hrf_opt with an ffs_hrftdv pair library: two columns per condition.

Each voxel's data is its own entry's early curve x 5 plus its late curve x 3.
Both betas must come back, under the voxel's own pair, on the OLS and REML paths.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import gamma

from fastfuncstuff.cli.util_concalc import _read_brick_labels
from fastfuncstuff.design.matrices import build_event_design_microtime

TR, NT, SHAPE = 1.0, 300, (6, 5, 4)
AFF = np.diag([3, 3, 3, 1])


def _curve(t, lag, k=6.0):
    tt = t - lag
    h = np.where(tt > 0, gamma.pdf(tt, k) - gamma.pdf(tt, k * 2.6) / 6, 0)
    return h / h.max()


def _write(tmp_path):
    tt = np.arange(320) * 0.1
    # Entry 1 a fast pair, entry 2 a slow one: different curves per voxel group.
    lib = np.stack(
        [
            np.stack([_curve(tt, -0.5), _curve(tt, 1.0)]),
            np.stack([_curve(tt, 0.3), _curve(tt, 1.8)]),
        ]
    )
    torch.save(
        {
            "hrf_library": torch.tensor(lib, dtype=torch.float32),
            "metadata": {"microtime_dt": 0.1},
            "duration_convolved": False,
        },
        tmp_path / "tdv_hrf_library.pt",
    )
    idx = (np.indices(SHAPE).sum(0) % 2) + 1
    nib.save(nib.Nifti1Image(idx.astype(np.float32), AFF), tmp_path / "tdv_hrf_index.nii.gz")
    rng = np.random.default_rng(0)
    onsets = np.cumsum(rng.uniform(5, 10, 60))
    onsets = np.round(onsets[onsets < NT - 30], 1)  # on the 0.1 s microtime grid
    # The model reml builds, so the betas are exact: early and late share the early
    # curve's unit-peak scale, and an independently rounded copy of the regressors
    # leaks ~1% between two columns this collinear.
    resp = np.stack(
        [
            build_event_design_microtime(
                all_onsets=[[onsets]],
                durations=[2.0],
                hrf_bases=torch.tensor(pair),
                n_timepoints_per_run=[NT],
                tr=TR,
                microtime_dt=0.1,
                device=torch.device("cpu"),
            ).numpy()
            @ np.array([5.0, 3.0])
            for pair in lib
        ]
    )
    data = 100 + resp[idx - 1] + rng.normal(0, 0.05, (*SHAPE, NT))
    nib.save(nib.Nifti1Image(data.astype(np.float32), AFF), tmp_path / "run.nii.gz")
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o:.1f}\t2\ta\n" for o in onsets)
    )
    return idx


@pytest.mark.parametrize("out", ["-Obuck", "-Rbuck"])
def test_pair_library_fits_early_and_late_per_voxel(tmp_path, out):
    idx = _write(tmp_path)
    res = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.reml",
            "-input",
            "run.nii.gz",
            "-events",
            "ev.tsv",
            "-TR",
            "1",
            "-device",
            "cpu",
            "-hrf_opt",
            "tdv",
            out,
            "o",
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    img = nib.load(tmp_path / "o.nii.gz")
    labels = _read_brick_labels(img)
    vol = np.asanyarray(img.dataobj)
    early = vol[..., labels.index("a_early#0_Coef")]
    late = vol[..., labels.index("a_late#0_Coef")]
    for g in (1, 2):
        np.testing.assert_allclose(early[idx == g].mean(), 5.0, rtol=0.05)
        np.testing.assert_allclose(late[idx == g].mean(), 3.0, rtol=0.05)


def test_other_consumers_refuse_a_pair_library(tmp_path):
    from fastfuncstuff.cli_utils import load_hrfopt_library

    _write(tmp_path)
    idx = torch.zeros(4, dtype=torch.long)
    with pytest.raises(SystemExit, match="early/late HRF pairs"):
        load_hrfopt_library(str(tmp_path / "tdv"), idx, microtime_dt=0.1, verbose=False)
    lib, _ = load_hrfopt_library(
        str(tmp_path / "tdv"), idx, microtime_dt=0.1, verbose=False, pairs=True
    )
    assert lib.shape[:2] == (2, 2)
