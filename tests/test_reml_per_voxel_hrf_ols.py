"""ffs_reml -hrfopt_prefix with only -Obuck: every voxel must get ITS OWN library curve.

The OLS-only path once fit one shared design (the first group's) to all voxels, so
voxels whose curve differed from it came out with a fraction of their true beta.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import gamma

from fastfuncstuff.cli.util_concalc import _read_brick_labels

TR, NT, SHAPE = 1.0, 240, (7, 6, 5)


def _hrf(t, k):
    return np.where(t > 0, gamma.pdf(t, k) - gamma.pdf(t, k * 2.6) / 6, 0)


@pytest.mark.parametrize("dsort", [False, True])
@pytest.mark.parametrize("censor", [False, True])
def test_ols_only_uses_each_voxels_own_hrf(tmp_path, dsort, censor):
    """``dsort``: OLS + -dsort took a shared-design branch of its own that
    dropped the per-voxel HRF all over again."""
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt, 3.5), _hrf(tt, 6.0)])
    lib /= lib.max(1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    # Alternates along every axis, so no axis-order mix-up can pass by luck.
    idx = (np.indices(SHAPE).sum(0) % 2) + 1
    nib.save(
        nib.Nifti1Image(idx.astype(np.float32), np.diag([3, 3, 3, 1])),
        tmp_path / "lib_hrf_index.nii.gz",
    )

    onsets = np.arange(10, NT - 25, 7.3)
    x = np.zeros(NT * 10)
    for o in onsets:
        x[int(round(o * 10)) : int(round(o * 10)) + 3] = 1
    resp = np.stack([np.convolve(x, h)[: len(x)][::10][:NT] for h in lib])
    rng = np.random.default_rng(1)
    data = 100 + 5 * resp[idx - 1].transpose(0, 1, 2, 3) + rng.normal(0, 0.5, (*SHAPE, NT))
    nib.save(
        nib.Nifti1Image(data.astype(np.float32), np.diag([3, 3, 3, 1])), tmp_path / "run.nii.gz"
    )
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t0.3\ta\n" for o in onsets)
    )
    extra = []
    if dsort:
        nib.save(
            nib.Nifti1Image(
                rng.normal(0, 1, (*SHAPE, NT)).astype(np.float32), np.diag([3, 3, 3, 1])
            ),
            tmp_path / "ds.nii.gz",
        )
        extra = ["-dsort", "ds.nii.gz"]
    if censor:
        keep = np.ones(NT, dtype=int)
        keep[[30, 31, 100, 170]] = 0
        np.savetxt(tmp_path / "censor.1D", keep, fmt="%d")
        extra += ["-censor", "censor.1D"]
    res = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.reml",
            "-input",
            "run.nii.gz",
            "-events",
            "ev.tsv",
        ]
        + [
            "-TR",
            "1",
            "-device",
            "cpu",
            "-hrfopt_prefix",
            "lib",
            "-Obuck",
            "o",
            "-round_durations",
            "1",
            *extra,
        ],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-2000:]
    img = nib.load(tmp_path / "o.nii.gz")
    beta = np.asanyarray(img.dataobj)[..., _read_brick_labels(img).index("a#0_Coef")]
    fast, slow = beta[idx == 1].mean(), beta[idx == 2].mean()
    assert abs(fast - slow) < 0.1 * max(fast, slow), (fast, slow)


def _study(tmp_path, n_lib=3):
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt, k) for k in (3.5, 5.0, 6.5)[:n_lib]])
    lib /= lib.max(1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    idx = (np.indices(SHAPE).sum(0) % n_lib) + 1
    idx[0, 0, 0] = 0  # outside ffs_hrfopt's mask: no curve assigned
    nib.save(
        nib.Nifti1Image(idx.astype(np.float32), np.diag([3, 3, 3, 1])),
        tmp_path / "lib_hrf_index.nii.gz",
    )
    onsets = np.arange(10, NT - 25, 7.3)
    rng = np.random.default_rng(2)
    nib.save(
        nib.Nifti1Image(
            (100 + rng.normal(0, 0.5, (*SHAPE, NT))).astype(np.float32), np.diag([3, 3, 3, 1])
        ),
        tmp_path / "run.nii.gz",
    )
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t0.3\ta\n" for o in onsets)
    )


@pytest.mark.parametrize("extra", [[], ["-single_trials", "st"], ["-Rbuck", "r"]])
def test_voxels_without_an_hrf_assignment_take_the_canonical_like_curve(tmp_path, extra):
    from fastfuncstuff.design.hrf_selection import closest_to_canonical_index

    _study(tmp_path)
    lib = torch.load(tmp_path / "lib_hrf_library.pt")["hrf_library"]
    expected = closest_to_canonical_index(lib, 0.1)
    res = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "fastfuncstuff.cli.reml",
            "-input",
            "run.nii.gz",
            "-events",
            "ev.tsv",
        ]
        + ["-TR", "1", "-device", "cpu", "-hrfopt_prefix", "lib", "-Obuck", "o", *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-2500:] + res.stderr[-1500:]
    assert f"closest to canonical SPMG1 (index {expected})" in res.stdout


def test_xval_r2_scores_each_voxel_with_its_own_hrf(tmp_path):
    """-xval_r2 scored every voxel against design_info["matrix"], one
    representative HRF's design, although the fit (and ffs_reml's own advice
    under -beta_cv) was per-voxel."""
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt, 3.0), _hrf(tt, 9.0)])
    lib /= lib.max(1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    idx = (np.indices(SHAPE).sum(0) % 2) + 1
    nib.save(
        nib.Nifti1Image(idx.astype(np.float32), np.diag([3, 3, 3, 1])),
        tmp_path / "lib_hrf_index.nii.gz",
    )
    nt = 120
    onsets = np.arange(6, nt - 25, 9.3)
    x = np.zeros(nt * 10)
    for o in onsets:
        x[int(round(o * 10)) : int(round(o * 10)) + 10] = 1
    resp = np.stack([np.convolve(x, h)[: len(x)][::10][:nt] for h in lib])
    rng = np.random.default_rng(3)
    for r in range(3):
        data = 100 + 3 * resp[idx - 1] + rng.normal(0, 0.5, (*SHAPE, nt))
        nib.save(
            nib.Nifti1Image(data.astype(np.float32), np.diag([3, 3, 3, 1])),
            tmp_path / f"run{r}.nii.gz",
        )
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t1\ta\n" for o in onsets)
    )
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input"]
        + [f"run{r}.nii.gz" for r in range(3)]
        + ["-events"]
        + ["ev.tsv"] * 3
        + ["-TR", "1", "-device", "cpu", "-hrfopt_prefix", "lib", "-Rbuck", "r"]
        + ["-xval_r2", "-noise_ceiling", "loro"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-2500:] + res.stderr[-1500:]
    r2 = np.asanyarray(nib.load(tmp_path / "r_xval_r2.nii.gz").dataobj)
    r2 = r2[..., 0] if r2.ndim == 4 else r2
    fast, slow = np.median(r2[idx == 1]), np.median(r2[idx == 2])
    assert min(fast, slow) > 0.8 * max(fast, slow), (fast, slow)
