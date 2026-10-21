"""Automatic polort is per run: each run its own 1 + floor(D / 150), in its own block.

Two things are pinned here. Equal-length runs must give exactly the design (and the
fit) the old single-degree rules gave, which is what an explicit ``-polort N`` at
that degree still builds. Unequal runs (60 s + 400 s -> degrees [1, 3]) must build
blocks of different widths everywhere a block-diagonal drift is assembled, and fit.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.cli_utils import (
    assemble_per_run_nuisance,
    auto_polort_per_run,
    build_nuisance_block_diag,
    build_nuisance_per_run,
    n_polort_columns,
    polort_column_labels,
    polort_per_run,
    resolve_polort,
)

CPU = torch.device("cpu")


def test_auto_degrees_follow_each_runs_own_duration():
    # 60 s and 400 s at TR 1: 1 + floor(60/150) = 1, 1 + floor(400/150) = 3.
    assert auto_polort_per_run([60, 400], 1.0) == [1, 3]
    assert resolve_polort(None, [60, 400], 1.0, verbose=False) == [1, 3]
    # An explicit degree stays one int for every run.
    assert resolve_polort(2, [60, 400], 1.0, verbose=False) == 2
    assert polort_per_run(2, 3) == [2, 2, 2]
    assert n_polort_columns([1, 3], 2) == 6
    assert n_polort_columns(-1, 2) == 0
    assert polort_column_labels([1, 3], 2) == [
        "Run#1Pol#0",
        "Run#1Pol#1",
        "Run#2Pol#0",
        "Run#2Pol#1",
        "Run#2Pol#2",
        "Run#2Pol#3",
    ]
    with pytest.raises(ValueError, match="2 entries for 3 runs"):
        polort_per_run([1, 3], 3)


def test_block_diag_builders_take_per_run_degrees():
    run_starts, n_t = [0, 60], 460
    blockdiag = build_nuisance_block_diag(run_starts, n_t, [1, 3], CPU)
    assert blockdiag.shape == (n_t, 6)
    # Run 1's two columns are zero over run 2, and run 2's four over run 1.
    assert torch.all(blockdiag[60:, :2] == 0) and torch.all(blockdiag[:60, 2:] == 0)
    per_run = build_nuisance_per_run(run_starts, n_t, [1, 3], CPU)
    assert [m.shape for m in per_run] == [(60, 2), (400, 4)]
    names = assemble_per_run_nuisance([], run_starts, n_t, [1, 3], CPU).per_run_column_names
    assert names == [
        ["r01_poly0", "r01_poly1"],
        ["r02_poly0", "r02_poly1", "r02_poly2", "r02_poly3"],
    ]
    # A uniform list is exactly the int form.
    torch.testing.assert_close(
        build_nuisance_block_diag(run_starts, n_t, [2, 2], CPU),
        build_nuisance_block_diag(run_starts, n_t, 2, CPU),
        rtol=0,
        atol=0,
    )


def test_pack_and_design_matrix_take_per_run_degrees(tmp_path, monkeypatch):
    import fastfuncstuff.utils
    from fastfuncstuff.design.builder import build_design_matrix, pack_for_shared_task_glm

    # build_design_matrix convolves on the default device; keep it off the GPU.
    monkeypatch.setattr(fastfuncstuff.utils, "get_device", lambda *a, **k: CPU)

    data = [torch.randn(3, 60), torch.randn(3, 400)]
    task = [torch.randn(60, 2), torch.randn(400, 2)]
    packed = pack_for_shared_task_glm(data, task, [1, 3], device=CPU)
    assert packed.design_concat.shape == (460, 2 + 6)
    assert packed.column_labels[2:] == [
        "run1_poly0",
        "run1_poly1",
        "run2_poly0",
        "run2_poly1",
        "run2_poly2",
        "run2_poly3",
    ]

    timing = tmp_path / "a.1D"
    timing.write_text("10 30\n50 200 300\n")
    X, labels, _, meta = build_design_matrix(
        [timing], ["a"], [60, 400], tr=1.0, polort=[1, 3], hrf_models="SPMG1(0)"
    )
    assert X.shape[1] == 6 + 1
    assert [lab for lab in labels if "Pol#" in lab] == polort_column_labels([1, 3], 2)
    assert meta["polort_indices"] == list(range(6))


def test_shifted_and_flobs_per_run_polys():
    from fastfuncstuff.design.flobs import compute_per_voxel_residuals
    from fastfuncstuff.design.shifted_hrf import build_blockdiag_polys

    Z = build_blockdiag_polys([60, 400], [1, 3], CPU)
    assert Z is not None and Z.shape == (460, 6)
    # Residuals use each run's own slice of the block-diagonal nuisance betas.
    data = [torch.randn(2, 60), torch.randn(2, 400)]
    task = [torch.zeros(60, 1), torch.zeros(400, 1)]
    resid = compute_per_voxel_residuals(
        data, task, [1, 3], task_betas=np.zeros((2, 1)), nuisance_betas=np.ones((2, 6)), device=CPU
    )
    # Each run's prediction is the sum of its own drift columns, so a slip in the
    # column offsets would show up as the wrong run's polynomials.
    for r, (lo, hi) in enumerate(((0, 60), (60, 460))):
        expected = data[r] - Z[lo:hi].sum(dim=1).float()
        torch.testing.assert_close(resid[r], expected, rtol=1e-5, atol=1e-5)


# ── CLI regression: equal runs unchanged, unequal runs fit ─────────────────────

TR = 2.0
SHAPE = (3, 3, 2)


def _write_runs(tmp_path, n_tps, seed=0):
    rng = np.random.default_rng(seed)
    paths, events = [], []
    for r, n in enumerate(n_tps):
        t = np.arange(n) * TR
        drift = np.linspace(-1, 1, n) ** 2 * 3
        arr = 100 + drift + rng.normal(0, 1, (*SHAPE, n))
        onsets = np.arange(8.0, n * TR - 20, 24.0)
        for o in onsets:
            arr += 2 * np.exp(-0.5 * ((t - o - 5) / 2) ** 2)
        p = tmp_path / f"run{r}.nii.gz"
        img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        img.header.set_zooms((1, 1, 1, TR))
        nib.save(img, p)
        ev = tmp_path / f"ev{r}.tsv"
        ev.write_text("onset\tduration\ttrial_type\n" + "".join(f"{o}\t1\ta\n" for o in onsets))
        paths.append(str(p))
        events.append(str(ev))
    return paths, events


def _reml(tmp_path, paths, events, prefix, *extra):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", *paths, "-events", *events]
        + ["-hrf_model", "SPMG1", "-device", "cpu", "-Obuck", prefix, *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    return res.stdout, np.asanyarray(nib.load(tmp_path / f"{prefix}.nii.gz").dataobj)


def test_reml_equal_runs_match_the_single_degree_fit(tmp_path):
    # Two 300 s runs: the per-run rule and the old average rule both give 3.
    paths, events = _write_runs(tmp_path, [150, 150])
    out_auto, auto = _reml(tmp_path, paths, events, "auto")
    _, fixed = _reml(tmp_path, paths, events, "fixed", "-polort", "3")
    assert "Auto polort per run: [3, 3]" in out_auto
    np.testing.assert_array_equal(auto, fixed)


def test_reml_unequal_runs_get_their_own_degrees(tmp_path):
    # 60 s + 400 s: degrees [1, 3] -> 2 + 4 drift columns.
    paths, events = _write_runs(tmp_path, [30, 200], seed=1)
    out, bucket = _reml(tmp_path, paths, events, "uneq")
    assert "Auto polort per run: [1, 3]" in out
    assert "Nuisance columns: 6" in out
    assert np.isfinite(bucket).all()


def _ridge(tmp_path, monkeypatch, paths, prefix, *extra):
    from fastfuncstuff.cli.ridge import main

    onsets = tmp_path / "onsets.1D"
    rows = []
    for p in paths:
        n = nib.load(p).shape[-1]
        rows.append(" ".join(f"{o:g}" for o in np.arange(8.0, n * TR - 20, 24.0)))
    onsets.write_text("\n".join(rows) + "\n")
    argv = ["ffs_ridge", "-input", *paths, "-onsets", str(onsets), "-durations", "1"]
    argv += ["-device", "cpu", "-prefix", str(tmp_path / prefix), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    return np.asanyarray(nib.load(tmp_path / f"{prefix}_betas_single_trial.nii.gz").dataobj)


def test_ridge_equal_runs_match_the_single_degree_fit(tmp_path, monkeypatch, capsys):
    paths, _ = _write_runs(tmp_path, [150, 150])
    auto = _ridge(tmp_path, monkeypatch, paths, "auto")
    assert "Auto polort per run: [3, 3]" in capsys.readouterr().out
    fixed = _ridge(tmp_path, monkeypatch, paths, "fixed", "-polort", "3")
    np.testing.assert_array_equal(auto, fixed)


def test_ridge_unequal_runs_fit(tmp_path, monkeypatch, capsys):
    paths, _ = _write_runs(tmp_path, [30, 200], seed=1)
    betas = _ridge(tmp_path, monkeypatch, paths, "uneq")
    assert "Auto polort per run: [1, 3]" in capsys.readouterr().out
    assert np.isfinite(betas).all()
