"""ffs_hrftdv end to end on a sphere-shell ribbon whose response lag rises with depth,
then ffs_reml -hrf_opt on what it wrote."""

from __future__ import annotations

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from tests.viewer.test_surfaces import _subject

VOX, N, TR, NT = 1.5, 36, 1.0, 160


def _curve(lag: float) -> np.ndarray:
    from scipy.stats import gamma

    t = np.arange(320) * 0.1 - lag
    h = np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)
    return h / h.max()


def _write(tmp_path):
    from fastfuncstuff.design.matrices import build_event_design_microtime

    aff = np.diag([VOX, VOX, VOX, 1.0])
    aff[:3, 3] = -(N - 1) * VOX / 2
    ijk = np.stack(np.meshgrid(*[np.arange(N)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk * VOX + aff[:3, 3], axis=-1)
    ribbon = (r > 20) & (r < 24)
    frac = np.clip((r - 20) / 4, 0, 1)
    lag = np.round(-0.6 + 1.2 * frac, 1)  # later toward pial, as a vein would make it
    rng = np.random.default_rng(0)
    for run in (1, 2):
        on = np.round(np.cumsum(rng.uniform(5, 9, 30)), 1)
        on = on[on < NT * TR - 25]
        cond = np.arange(on.size) % 2
        (tmp_path / f"ev{run}.tsv").write_text(
            "onset\tduration\ttrial_type\n"
            + "".join(f"{o:.1f}\t2\t{'ab'[c]}\n" for o, c in zip(on, cond, strict=True))
        )
        data = 100 + rng.normal(0, 0.4, (N, N, N, NT))
        for lv in np.unique(lag[ribbon]):
            x = build_event_design_microtime(
                all_onsets=[[on[cond == 0]], [on[cond == 1]]],
                durations=[2.0, 2.0],
                hrf_bases=torch.tensor(_curve(lv), dtype=torch.float32),
                n_timepoints_per_run=[NT],
                tr=TR,
                microtime_dt=0.1,
                device=torch.device("cpu"),
            ).numpy()
            data[ribbon & (lag == lv)] += 2.0 * (x @ np.array([1.0, 0.7]))
        nib.save(nib.Nifti1Image(data.astype(np.float32), aff), str(tmp_path / f"run{run}.nii.gz"))
    return ribbon, frac


@pytest.mark.slow  # 14s
def test_hrftdv_finds_lag_rising_with_depth_and_reml_fits_its_pairs(tmp_path):
    ribbon, frac = _write(tmp_path)
    subj = _subject(tmp_path)
    common = ["-input", "run1.nii.gz", "run2.nii.gz", "-events", "ev1.tsv", "ev2.tsv"]
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.hrftdv", *common]
        + ["-fs_subj", str(subj), "-hemi", "lh", "-tr", "1", "-device", "cpu"]
        + ["-patch_mm", "6", "-prefix", "tdv"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    lib = torch.load(tmp_path / "tdv_hrf_library.pt", weights_only=False)
    assert lib["hrf_library"].ndim == 3 and lib["hrf_library"].shape[1] == 2
    assert lib["metadata"]["basis_labels"] == ["early", "late"]
    img = nib.load(tmp_path / "tdv_tdv.nii.gz")
    from fastfuncstuff.cli.util_concalc import _read_brick_labels

    labels = _read_brick_labels(img)
    b = np.asanyarray(img.dataobj)
    resp = (b[..., labels.index("responsive")] > 0) & ribbon
    assert resp.sum() > 100
    rel = b[..., labels.index("rel_lag")][resp]
    assert np.corrcoef(rel, frac[resp])[0, 1] > 0.4
    early = b[..., labels.index("unit_early_lag")][resp]
    late = b[..., labels.index("unit_late_lag")][resp]
    assert np.median(late - early) > 0.5
    index = np.asanyarray(nib.load(tmp_path / "tdv_hrf_index.nii.gz").dataobj)[..., 0]
    assert (index[ribbon] > 0).mean() > 0.95

    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", *common]
        + ["-TR", "1", "-device", "cpu", "-hrf_opt", "tdv", "-Obuck", "o"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-2000:]
    out = nib.load(tmp_path / "o.nii.gz")
    names = _read_brick_labels(out)
    assert {"a_early#0_Coef", "a_late#0_Coef", "b_early#0_Coef"} <= set(names)
    # Superficial voxels are later than their columns' median, so they load on the late curve.
    vol = np.asanyarray(out.dataobj)
    late_share = vol[..., names.index("a_late#0_Coef")] / (
        np.abs(vol[..., names.index("a_early#0_Coef")])
        + np.abs(vol[..., names.index("a_late#0_Coef")])
    )
    deep, sup = resp & (frac < 0.3), resp & (frac > 0.7)
    assert np.median(late_share[sup]) > np.median(late_share[deep]) + 0.2
