"""ffs_hrfopt on a single run: in-sample selection, honestly labelled.

One run has no held-out run, so the CLI must fall back to -select full rather
than refuse — and then must not write in-sample R² under an ``_xval_r2`` name,
nor a zero-filled CV std map, nor run beta-series CV that has nothing to hold
out. These run the real CLI on CPU on a tiny synthetic volume.
"""

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.cli import hrfopt as hrfopt_cli

TR = 1.0
N_TP = 80
SHAPE = (5, 5, 2)


def _make_run(tmp_path: Path, n_runs: int = 1, trial_types=("faces", "houses")):
    """Boxcar responses in a corner of the volume, one BIDS events.tsv per run."""
    rng = np.random.default_rng(0)
    inputs, events = [], []
    for run in range(n_runs):
        vol = 100.0 + 2.0 * rng.standard_normal((*SHAPE, N_TP))
        rows = ["onset\tduration\ttrial_type"]
        for i, onset in enumerate(range(8, N_TP - 16, 12)):
            # Distinct trial types per run make conditions run-exclusive when
            # n_runs > 1, the case that used to be rejected outright.
            trial_type = f"{trial_types[i % len(trial_types)]}_r{run}"
            rows.append(f"{onset:.1f}\t2.0\t{trial_type}")
            vol[:2, :2, :, onset + 2 : onset + 6] += 6.0
        img = nib.Nifti1Image(vol.astype(np.float32), np.eye(4))
        img.header["pixdim"][4] = TR
        path = tmp_path / f"run{run + 1}.nii.gz"
        nib.save(img, str(path))
        inputs.append(str(path))
        ev = tmp_path / f"run{run + 1}_events.tsv"
        ev.write_text("\n".join(rows) + "\n")
        events.append(str(ev))
    mask = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(SHAPE, dtype=np.uint8), np.eye(4)), str(mask))
    return inputs, events, str(mask)


def _run(monkeypatch, tmp_path, inputs, events, extra):
    prefix = str(tmp_path / "hopt")
    argv = [
        "ffs_hrfopt",
        "-input",
        *inputs,
        "-events",
        *events,
        "-prefix",
        prefix,
        "-device",
        "cpu",
        "-hrf_mode",
        "library",
        "-n_hrfs",
        "3",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    hrfopt_cli.main()
    return prefix


def _assert_insample_outputs(prefix: str) -> None:
    assert Path(f"{prefix}_hrf_index.nii.gz").exists()
    assert Path(f"{prefix}_insample_r2.nii.gz").exists()
    assert Path(f"{prefix}_insample_r2_all_hrfs.nii.gz").exists()
    for cv_only in ("xval_r2", "xval_r2_std", "xval_r2_all_hrfs", "hrfopt_xval_r2"):
        assert not Path(f"{prefix}_{cv_only}.nii.gz").exists(), cv_only
    labels = nib.load(f"{prefix}_hrf_index.nii.gz")
    assert labels.shape[:3] == SHAPE


def test_reported_command_runs_in_sample(monkeypatch, tmp_path, capsys):
    """The failing command: one run, -events, -select full, -do_scale, -mask."""
    inputs, events, mask = _make_run(tmp_path)
    prefix = _run(
        monkeypatch,
        tmp_path,
        inputs,
        events,
        ["-select", "full", "-do_scale", "-mask", mask],
    )
    _assert_insample_outputs(prefix)
    out = capsys.readouterr().out
    assert "Mean in-sample R²" in out
    assert "Mean xval R²" not in out


def test_one_run_defaults_to_full(monkeypatch, tmp_path, capsys):
    inputs, events, _ = _make_run(tmp_path)
    prefix = _run(monkeypatch, tmp_path, inputs, events, [])
    _assert_insample_outputs(prefix)
    assert "-select full" in capsys.readouterr().out
    meta = json.loads(Path(f"{prefix}_metadata.json").read_text())
    assert meta["select_mode"] == "full"


def test_one_run_explicit_xval_errors(monkeypatch, tmp_path, capsys):
    inputs, events, _ = _make_run(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        _run(monkeypatch, tmp_path, inputs, events, ["-select", "xval"])
    assert excinfo.value.code == 1
    assert "-select xval needs at least 2 runs" in capsys.readouterr().out


def test_one_run_single_trials(monkeypatch, tmp_path, capsys):
    """Per-trial betas from one run; beta-series CV is skipped, not zero-filled."""
    inputs, events, _ = _make_run(tmp_path)
    prefix = _run(monkeypatch, tmp_path, inputs, events, ["-single_trials", "-cv_design", "single"])
    _assert_insample_outputs(prefix)
    assert Path(f"{prefix}_stats_single_trial.nii.gz").exists()
    assert "Beta-series CV skipped" in capsys.readouterr().out


def test_multi_run_full_without_repeats(monkeypatch, tmp_path):
    """In-sample selection does not need conditions shared across runs."""
    inputs, events, _ = _make_run(tmp_path, n_runs=2)
    prefix = _run(monkeypatch, tmp_path, inputs, events, ["-select", "full"])
    _assert_insample_outputs(prefix)
