"""tCorr viewer mode: per-voxel r between files, loaded by GO, re-correlated live."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import pytest
import torch

from fastfuncstuff.stats.paired_correlation import paired_voxel_correlation
from fastfuncstuff.viewer.session import ViewerSession
from fastfuncstuff.viewer.vocab import ModeAction, SetMode, SetModeParam, SetUnderlay

CPU = torch.device("cpu")
SHAPE, NT = (10, 12, 8), 50


def _write(d, name, data):
    aff = np.diag([3.0, 3.0, 3.0, 1.0])
    aff[:3, 3] = (-15.0, -18.0, -12.0)
    p = d / name
    nib.save(nib.Nifti1Image(np.asarray(data, dtype=np.float32), aff), str(p))
    return str(p)


@pytest.fixture
def files(tmp_path):
    """Three runs: a blob shares a signal across all of them, the rest is noise."""
    rng = np.random.default_rng(2)
    shared = np.sin(2 * np.pi * np.arange(NT) / 12.0).astype(np.float32)
    paths = []
    for k in range(3):
        d = rng.normal(0, 1, (*SHAPE, NT)).astype(np.float32) + 100
        d[2:5, 3:6, 2:5] += 4 * shared
        paths.append(_write(tmp_path, f"run{k + 1}.nii.gz", d))
    _write(tmp_path, "anat.nii.gz", rng.random(SHAPE) * 100)
    return tmp_path, paths


@pytest.fixture
def session(files):
    d, _ = files
    s = ViewerSession(device=CPU)
    s.do(SetUnderlay(str(d / "anat.nii.gz")))
    s.do(SetMode("tcorr"))
    yield s
    s.close()


def _map(session):
    layer = session.state.layers.find_by_source("mode:tcorr")
    return None if layer is None else session.store.get(layer.key).array[..., 0]


def _go(session, paths):
    session.do(SetModeParam("files", "|".join(f"+{p}" for p in paths)))
    session.do(ModeAction("go"))


def test_nothing_is_read_until_go(session, files):
    _, paths = files
    session.do(SetModeParam("files", f"+{paths[0]}|+{paths[1]}"))
    assert _map(session) is None
    assert "GO" in session.mode.status()


def test_go_draws_the_pair_map_and_it_matches_the_primitive(session, files):
    _, paths = files
    _go(session, paths[:2])
    vol = _map(session)
    assert vol is not None
    assert vol[3, 4, 3] > 0.8, "the shared blob must correlate"
    assert abs(float(np.median(vol))) < 0.2, "independent noise must not"
    ref = paired_voxel_correlation(
        [nib.load(p).get_fdata(dtype=np.float32) for p in paths[:2]], polort=2, verbose=False
    )[..., 0]
    np.testing.assert_allclose(vol, ref.numpy(), atol=1e-5)


def test_pair_picker_follows_the_files_and_switches_without_reloading(session, files):
    _, paths = files
    _go(session, paths)
    pick = next(c for c in session.mode.controls() if c.name == "pair")
    assert pick.choices == ("1-2", "1-3", "2-3")
    first = _map(session).copy()
    loaded = dict(session.mode._arrays)
    session.do(SetModeParam("pair", "2-3"))
    assert not np.array_equal(_map(session), first)
    assert all(session.mode._arrays[p] is a for p, a in loaded.items()), "switch reloaded"
    assert "run2" in session.mode.status() and "run3" in session.mode.status()


def test_detrend_and_blur_recorrelate_from_memory(session, files):
    _, paths = files
    _go(session, paths[:2])
    base = _map(session).copy()
    arrays = dict(session.mode._arrays)
    session.do(SetModeParam("blur", "6.0"))
    blurred = _map(session)
    assert not np.allclose(blurred, base)
    # Blur averages the independent noise away, so the noise floor rises toward
    # the blob's shared signal -- the effect the help warns about.
    assert float(np.abs(blurred).mean()) > float(np.abs(base).mean())
    assert all(session.mode._arrays[p] is a for p, a in arrays.items())


def test_mismatched_files_say_so_instead_of_drawing(session, files, tmp_path):
    _, paths = files
    odd = _write(tmp_path, "odd.nii.gz", np.zeros((*SHAPE, NT - 5)))
    _go(session, [paths[0], odd])
    assert _map(session) is None
    assert "differ" in session.mode.status()
