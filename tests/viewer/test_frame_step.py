"""`,` and `.`: time, or the selected layer's own sub-bricks.

A selected 4-D layer that does not follow time used to sit still while `.`
scrubbed everything else. The rule now: such a layer steps itself, by one
group when its threshold follows at an offset (Coef to Coef with the t
following; Coef to Coef with the z following for a triplet), and keeps the
phase it is on so a leading Full_Fstat never becomes the shown map.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel")

from fastfuncstuff.viewer.session import ViewerSession  # noqa: E402
from fastfuncstuff.viewer.vocab import (  # noqa: E402
    SelectLayer,
    SetIndex,
    SetThresholdFollow,
    SetTimeLinked,
    SetVolume,
)

CPU = torch.device("cpu")


def _write(path, n, labels=None):
    from fastfuncstuff.io.afni import save_nifti

    data = np.random.default_rng(0).random((6, 7, 5, n)).astype(np.float32)
    save_nifti(data, path, affine=np.diag([3.0, 3.0, 3.0, 1.0]), brick_labels=labels)
    return str(path)


@pytest.fixture
def session(tmp_path):
    s = ViewerSession(device=CPU)
    s.load(_write(tmp_path / "run.nii.gz", 10), key="run")
    s.do(SetTimeLinked("run", True))
    yield s
    s.close()


def _step(session, delta=1):
    cmd = session.frame_step(delta)
    if cmd is not None:
        session.do(cmd)
    return cmd


def _bucket(session, tmp_path, labels, follow):
    session.load(_write(tmp_path / "stats.nii.gz", len(labels), labels), key="stats")
    session.do(SetTimeLinked("stats", False))
    session.do(SelectLayer("stats"))
    session.do(SetThresholdFollow("stats", follow))
    return session.state.layers.get("stats")


def test_a_time_series_steps_time(session):
    session.do(SelectLayer("run"))
    assert isinstance(_step(session), SetIndex)
    assert session.state.time_index == 1


def test_a_bucket_steps_itself_not_time(session, tmp_path):
    _bucket(session, tmp_path, ["a", "b", "c"], "same")
    _step(session)
    assert session.state.layers.get("stats").volume_index == 1
    assert session.state.time_index == 0
    _step(session, -1)
    _step(session, -1)
    assert session.state.layers.get("stats").volume_index == 2  # wraps


def test_coef_tstat_pairs_step_two_and_keep_the_t(session, tmp_path):
    labels = ["Full_Fstat", "A#0_Coef", "A#0_Tstat", "B#0_Coef", "B#0_Tstat"]
    _bucket(session, tmp_path, labels, "fixed")
    session.do(SetVolume("stats", 1))
    session.do(SetThresholdFollow("stats", "next"))
    _step(session)
    layer = session.state.layers.get("stats")
    assert (layer.volume_index, layer.threshold_brick) == (3, 4)
    _step(session)  # wraps to the first Coef, not to Full_Fstat
    layer = session.state.layers.get("stats")
    assert (layer.volume_index, layer.threshold_brick) == (1, 2)


def test_plus_two_follows_a_triplet(session, tmp_path):
    labels = ["A_Coef", "A_Tstat", "A_Zstat", "B_Coef", "B_Tstat", "B_Zstat"]
    layer = _bucket(session, tmp_path, labels, "next2")
    assert layer.threshold_brick == 2
    _step(session)
    layer = session.state.layers.get("stats")
    assert (layer.volume_index, layer.threshold_brick) == (3, 5)
    assert "cut on B_Zstat" in session.brick_caption("stats")
    _step(session, -1)
    assert session.state.layers.get("stats").volume_index == 0
