"""ffs_fitbasis -hrf-index fits with the library its indices were chosen from.

It used to rebuild the stock 20-curve library and only range-check the
indices, so a map from any other library (PIGHS, a librarian set, a
-hrf_library run) put every voxel on an unrelated curve; unassigned voxels
took curve 0, the fastest.  The sidecar's curves are now used as stored when
their step matches -flobs-dt, resampled exactly once when it does not.
"""

import sys

import nibabel as nib
import numpy as np
import pytest
import torch

import fastfuncstuff.cli_utils as cli_utils
from fastfuncstuff.cli.fitbasis import _import_hrfopt_shapes
from fastfuncstuff.design.hrf import get_spmg1_hrf

DT, WINDOW = 0.1, 32.0
N_T = 320  # build_shape_library's length for (0.1 s, 32 s)


def _curves(dt, n_t):
    """Four positive, peak-normalised curves; the canonical one is row 3."""
    t = np.arange(n_t) * dt
    rows = [t**k * np.exp(-t / 1.2) for k in (2.0, 3.0, 9.0)]
    canon = get_spmg1_hrf(microtime_dt=dt, stim_duration=0.0, device=torch.device("cpu"))
    canon = np.pad(canon.numpy().astype(np.float64), (0, max(0, n_t - canon.numel())))[:n_t]
    rows.append(canon)
    return np.stack([r / np.abs(r).max() for r in rows])


def _study(tmp_path, *, dt=DT, n_t=N_T, duration_convolved=False, sidecar=True):
    lib = _curves(dt, n_t)
    if sidecar:
        torch.save(
            {
                "hrf_library": torch.from_numpy(lib),
                "metadata": {"microtime_dt": dt},
                "duration_convolved": duration_convolved,
            },
            tmp_path / "hopt_hrf_library.pt",
        )
    # 1-based on disk; 0 = outside ffs_hrfopt's mask.
    idx = np.array([1, 2, 0, 3], np.float32).reshape(2, 2, 1)
    nib.save(nib.Nifti1Image(idx, np.eye(4)), tmp_path / "hopt_hrf_index.nii.gz")
    return lib, str(tmp_path / "hopt_hrf_index.nii.gz")


def _import(path, durations=(2.0,)):
    return _import_hrfopt_shapes(
        path,
        dt=DT,
        duration=WINDOW,
        volume_shape=(2, 2, 1),
        mask=None,
        n_voxels=4,
        durations=list(durations),
    )


@pytest.fixture
def resample_calls(monkeypatch):
    calls = []
    real = cli_utils._resample_hrf_curves

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    monkeypatch.setattr(cli_utils, "_resample_hrf_curves", counting)
    return calls


def test_matching_step_uses_the_stored_curves_untouched(tmp_path, resample_calls):
    lib, path = _study(tmp_path)
    shapes, _, index, durations, _ = _import(path)
    assert resample_calls == []
    assert np.array_equal(shapes, lib)  # bit-identical, no interpolation
    assert durations == [2.0]
    # The unassigned voxel takes the canonical-like row 3 -- not curve 0.
    assert index.tolist() == [0, 1, 3, 2]


def test_other_step_is_resampled_exactly_once(tmp_path, resample_calls, capsys):
    _, path = _study(tmp_path, dt=0.05, n_t=2 * N_T)
    shapes, _, _, _, _ = _import(path)
    assert len(resample_calls) == 1
    assert shapes.shape == (4, N_T)
    assert "Resampled HRF library from 0.05 s to 0.1 s" in capsys.readouterr().out
    np.testing.assert_allclose(shapes, _curves(DT, N_T), atol=2e-3)


def test_duration_convolved_library_zeroes_the_durations(tmp_path):
    _, path = _study(tmp_path, duration_convolved=True)
    _, _, _, durations, prov = _import(path, durations=(2.0, 3.0))
    assert prov["duration_convolved"]
    assert durations == [0.0, 0.0]


def test_missing_sidecar_is_refused(tmp_path):
    _, path = _study(tmp_path, sidecar=False)
    with pytest.raises(FileNotFoundError, match="hrf_library.pt not found"):
        _import(path)


def _cli_inputs(tmp_path):
    rng = np.random.default_rng(0)
    inputs = []
    for r in range(2):
        p = tmp_path / f"run{r}.nii.gz"
        img = nib.Nifti1Image((100 + rng.normal(size=(2, 2, 1, 60))).astype(np.float32), np.eye(4))
        img.header.set_zooms((1, 1, 1, 1.0))
        img.header.set_xyzt_units(xyz="mm", t="sec")
        nib.save(img, p)
        inputs.append(str(p))
    onsets = tmp_path / "a.1D"
    onsets.write_text("5 25 40\n5 25 40\n")
    return ["-input", *inputs, "-onsets", str(onsets), "-durations", "1", "-tr", "1"]


def _fitbasis(monkeypatch, capsys, args):
    from fastfuncstuff.cli.fitbasis import main

    monkeypatch.setattr(sys, "argv", ["ffs_fitbasis", *args, "-device", "cpu"])
    rc = main()
    out = capsys.readouterr().out
    assert rc in (0, None), out[-2000:]
    return out


def test_cli_fits_from_the_sidecar(tmp_path, monkeypatch, capsys):
    """End to end: no -hrf set named, the sidecar supplies the curves."""
    _, path = _study(tmp_path)
    out = _fitbasis(
        monkeypatch,
        capsys,
        _cli_inputs(tmp_path) + ["-hrf-index", path, "-prefix", str(tmp_path / "fb")],
    )
    assert "hrfopt sidecar" in out


def test_fitbasis_index_map_round_trips_with_its_own_sidecar(tmp_path, monkeypatch, capsys):
    """fitbasis's own _hrf_index has to come back in through a bare -hrf-index,
    which now needs the library beside it -- so fitbasis writes one."""
    common = _cli_inputs(tmp_path)
    _fitbasis(monkeypatch, capsys, common + ["-hrf", "library", "-prefix", str(tmp_path / "a")])
    saved = torch.load(tmp_path / "a_fitbasis_hrf_library.pt", weights_only=False)
    shapes = np.loadtxt(tmp_path / "a_fitbasis_hrf_shapes.tsv").T
    np.testing.assert_allclose(saved["hrf_library"].numpy(), shapes, rtol=1e-8, atol=1e-10)
    assert saved["metadata"]["microtime_dt"] == pytest.approx(DT)
    assert saved["duration_convolved"] is False

    first = np.asanyarray(nib.load(tmp_path / "a_fitbasis_hrf_index.nii.gz").dataobj)[..., 0]
    out = _fitbasis(
        monkeypatch,
        capsys,
        common
        + ["-hrf-index", str(tmp_path / "a_fitbasis_hrf_index.nii.gz")]
        + ["-prefix", str(tmp_path / "b")],
    )
    assert "hrfopt sidecar" in out
    second = np.asanyarray(nib.load(tmp_path / "b_fitbasis_hrf_index.nii.gz").dataobj)[..., 0]
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    ("run_starts", "n_tp", "expected"),
    [
        ([0], 60, 1),  # a 1-minute run: AFNI gives 1, round(0.5) gave 0
        ([0, 60], 460, 3),  # 60 s + 400 s: the longest run sets it (mean gave 2)
        ([0, 300], 600, 3),  # equal 300 s runs
    ],
)
def test_auto_polort_is_afnis_longest_run_rule(run_starts, n_tp, expected):
    from fastfuncstuff.cli.fitbasis import _auto_polort_for_runs

    assert _auto_polort_for_runs(run_starts, n_tp, 1.0)[0] == expected


def test_shift_parametrization_writes_the_sidecar_too(tmp_path, monkeypatch, capsys):
    common = _cli_inputs(tmp_path)
    _fitbasis(
        monkeypatch,
        capsys,
        common + ["-parametrization", "shift", "-hrf", "library", "-prefix", str(tmp_path / "s")],
    )
    saved = torch.load(tmp_path / "s_fitbasis_hrf_library.pt", weights_only=False)
    assert saved["hrf_library"].shape[0] == 20
    assert saved["metadata"]["stim_durations"] == [1.0]
