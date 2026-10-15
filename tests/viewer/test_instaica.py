"""InstaICA: spatial ICA of the run you are looking at (viewer.instaica, modes.instaica)."""

from __future__ import annotations

import numpy as np
import pytest

from fastfuncstuff.viewer.instaica import rotate_to_ica
from fastfuncstuff.viewer.instapca import decompose
from fastfuncstuff.viewer.modes import registry
from fastfuncstuff.viewer.ortvec import read_columns
from fastfuncstuff.viewer.vocab import SetMode, SetModeParam
from tests.viewer.test_instapca import pca_session  # noqa: F401  (fixture)


def _two_equal_sources(seed: int = 0):
    """Two sparse sources of equal power whose maps overlap: PCA's eigenvalues
    tie, so its components are an arbitrary mix; ICA's rotation is not."""
    rng = np.random.default_rng(seed)
    shape, T = (24, 24, 12), 160
    t = np.arange(T)
    a = np.sin(2 * np.pi * t / 23)
    b = np.sign(np.sin(2 * np.pi * t / 41))
    tcs = np.stack([(a - a.mean()) / a.std(), (b - b.mean()) / b.std()], 1)
    maps = np.zeros((*shape, 2))
    maps[3:12, 3:12, 2:9, 0] = 1.0
    maps[8:18, 8:18, 3:10, 1] = 1.0
    data = 1000 + 15 * maps @ tcs.T + rng.normal(0, 3, (*shape, T))
    return data.astype(np.float32), tcs


def _best_r(truth, found):
    r = np.corrcoef(np.c_[truth, found].T)[: truth.shape[1], truth.shape[1] :]
    return np.abs(r).max(1)


def test_ica_unmixes_what_pca_leaves_mixed():
    data, tcs = _two_equal_sources()
    pca = decompose(
        data, affine=np.eye(4), tr=1.0, mask=np.ones(data.shape[:3], bool), polort=1, n_components=4
    )
    ica = rotate_to_ica(pca)
    assert ica.kind == "IC" and ica.zmap is not None
    ica_r, pca_r = _best_r(tcs, ica.timecourses).min(), _best_r(tcs, pca.timecourses).min()
    assert ica_r > 0.97 and ica_r > pca_r + 0.03, (ica_r, pca_r)


def test_the_rotated_maps_are_the_correlation_with_the_ic_time_course():
    """No second pass over the data: the algebra must match doing it directly."""
    from fastfuncstuff.viewer.derive import orthonormal_basis, read_nuisance

    data, _ = _two_equal_sources(1)
    pca = decompose(
        data, affine=np.eye(4), tr=1.0, mask=np.ones(data.shape[:3], bool), polort=1, n_components=4
    )
    ica = rotate_to_ica(pca)
    q = orthonormal_basis(read_nuisance(None, n_time=data.shape[3], polort=1).columns)
    rows = data.reshape(-1, data.shape[3]).astype(np.float64)
    c = rows - (rows @ q) @ q.T
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    for k in range(ica.n_components):
        a = ica.timecourses[:, k]
        np.testing.assert_allclose(c @ (a / np.linalg.norm(a)), ica.correlation[:, k], atol=1e-4)


def test_instaica_reviews_labels_and_saves_ica_noise(pca_session):  # noqa: F811
    s = pca_session
    s.do(SetMode("instaica"))
    s.do(SetModeParam("n_components", "4"))
    mode = s.mode
    assert mode._result is not None and mode._result.kind.startswith("IC")
    s.do(SetModeParam("map", "z"))
    layer = s.state.layers.find_by_source("mode:instaica")
    assert layer is not None and layer.name.startswith("A_IICA IC 0")
    mode.action("noise")
    mode.action("signal")
    mode.action("noise")
    mode.action("save")
    columns, _ = read_columns(str(s.tmp / "bold_ica_noise.1D"))
    np.testing.assert_allclose(columns, mode._result.timecourses[:, [0, 2]], atol=1e-5)
    first = mode._result.timecourses.copy()
    s.do(SetModeParam("seed", "7"))
    assert mode._result is not None and mode._result.timecourses.shape == first.shape


def test_denoise_takes_instaica_noise_labels(pca_session):  # noqa: F811
    s = pca_session
    s.do(SetMode("instaica"))
    s.do(SetModeParam("n_components", "4"))
    s.mode.labels = {1: "noise"}
    s.do(SetMode("denoise"))
    mix, noise, _style = s.mode._ica_columns(s.mode_named("instaica")._result.timecourses.shape[0])
    assert noise == [1]
    np.testing.assert_allclose(mix, s.mode_named("instaica")._result.timecourses)


def test_the_folder_reviewer_is_a_button_not_a_dropdown_entry():
    assert registry.get("instaica").in_selector
    assert not registry.get("ica").in_selector


def test_the_ica_button_opens_the_folder_reviewer(tmp_path):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import torch

    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    s = ViewerSession(device=torch.device("cpu"))
    win = ViewerWindow(s)
    try:
        listed = [win.mode_box.itemData(i) for i in range(win.mode_box.count())]
        assert "instaica" in listed and "ica" not in listed
        win.open_ica_folder(tmp_path)
        app.processEvents()
        assert s.mode.name == "ica"
        assert win.mode_box.currentData() == "ica"  # shown while it is active
        win._switch_mode("plain")
        app.processEvents()
        listed = [win.mode_box.itemData(i) for i in range(win.mode_box.count())]
        assert "ica" not in listed
    finally:
        win.close()
        s.close()
        app.processEvents()
