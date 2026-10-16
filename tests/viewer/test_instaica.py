"""InstaICA: spatial ICA of the run you are looking at (viewer.instaica, modes.instaica)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from fastfuncstuff.viewer.instaica import rotate_to_ica
from fastfuncstuff.viewer.instapca import Decomposition, decompose
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


def _rotate_to_ica_host_reference(
    pca: Decomposition,
    *,
    seed: int = 0,
    max_iter: int = 1000,
    fun: str = "logcosh",
    device: torch.device | None = None,
) -> Decomposition:
    """The float64 host-numpy implementation rotate_to_ica replaced (reference)."""
    import torch

    from fastfuncstuff.decomposition.ica import fastica_batch

    corr = np.asarray(pca.correlation, dtype=np.float64)  # (V, K): P, the projections
    n_vox, k = corr.shape
    if k < 2:
        raise ValueError("ICA needs at least two dimensions")
    # Whitened spatial components: each projection column at unit variance.
    centred = corr - corr.mean(0)
    d = centred.std(0)
    d = np.where(d > 0, d, 1.0)
    z = (centred / d).T  # (K, V)
    # K rows of V samples: at K = 30 and V ~ 2e5 the FastICA iterations are
    # real arithmetic, so they go where the decomposition went.
    x = torch.as_tensor(z, dtype=torch.float32, device=device or torch.device("cpu"))
    w, _, converged = fastica_batch(x, 1, k, fun=fun, max_iter=max_iter, tol=1e-4, base_seed=seed)
    W = w[0].double().cpu().numpy()  # (K, K), orthogonal unmixing
    s = W @ z  # (K, V) spatial sources

    # The time courses: A = u D W^T. ``pca.timecourses`` is u * sqrt(T) (unit
    # std, sign-fixed); the constant and the signs cancel in normalisation.
    tc = np.asarray(pca.timecourses, dtype=np.float64) @ np.diag(d) @ W.T  # (T, K)
    # Each voxel's correlation with each IC time course: P D W^T over |a|.
    proj = corr @ np.diag(d) @ W.T  # (V, K)
    norms = np.linalg.norm(tc, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    ic_corr = proj / (norms / np.sqrt(tc.shape[0]))[None, :]
    # The sign convention InstaPCA and MELODIC share: heavy tail positive.
    flip = np.where(np.sum(s**3, axis=1) < 0, -1.0, 1.0)
    s *= flip[:, None]
    tc *= flip[None, :]
    ic_corr *= flip[None, :]
    tc = (tc - tc.mean(0)) / np.where(tc.std(0) > 0, tc.std(0), 1.0)

    # Ordered by how much of the prepared variance each explains, as PCs are.
    energy = (ic_corr**2).sum(0)
    order = np.argsort(-energy)
    total = float(pca.explained.sum()) if pca.explained.size else 1.0
    explained = energy[order] / max(energy.sum(), 1e-30) * total

    zmap = s[order].T
    zmap = (zmap - zmap.mean(0)) / np.where(zmap.std(0) > 0, zmap.std(0), 1.0)
    corr_out = ic_corr[:, order].astype(np.float32)
    scale = pca.scale if pca.scale is not None else np.zeros(n_vox, np.float32)
    return replace(
        pca,
        timecourses=tc[:, order],
        correlation=corr_out,
        amplitude=(corr_out * scale[:, None]).astype(np.float32),
        explained=np.asarray(explained, dtype=np.float64),
        zmap=zmap.astype(np.float32),
        kind="IC",
        # Usual when some kept dimensions are noise: a Gaussian direction has
        # no preferred rotation. Said, not hidden -- fewer dims or another seed.
        note=""
        if bool(converged[0])
        else f"FastICA did not settle in {max_iter} iterations; fewer # dims or another seed",
    )


def test_device_rotation_matches_the_host_float64_reference():
    """rotate_to_ica now runs its (V, K) passes in float32 on the device.  Compared at the
    two real sources, where FastICA converges to a unique answer; with extra noise
    dimensions it never settles (no preferred rotation) and two correct implementations
    legitimately end in different places."""
    data, _ = _two_equal_sources(2)
    pca = decompose(
        data, affine=np.eye(4), tr=1.0, mask=np.ones(data.shape[:3], bool), polort=1, n_components=2
    )
    new, ref = rotate_to_ica(pca, seed=3), _rotate_to_ica_host_reference(pca, seed=3)
    assert new.note == ref.note == ""  # both settled
    np.testing.assert_allclose(new.explained, ref.explained, rtol=1e-3)
    for k in range(2):
        r = np.corrcoef(new.timecourses[:, k], ref.timecourses[:, k])[0, 1]
        assert r > 0.9999, (k, r)  # same component, same sign
    np.testing.assert_allclose(new.zmap, ref.zmap, rtol=0, atol=5e-3)
    np.testing.assert_allclose(new.correlation, ref.correlation, rtol=0, atol=5e-4)
    np.testing.assert_allclose(new.amplitude, ref.amplitude, rtol=0, atol=5e-3)
    assert new.correlation.dtype == np.float32 and new.zmap.dtype == np.float32


@pytest.mark.gpu
def test_cuda_rotation_matches_the_host_float64_reference():
    if not torch.cuda.is_available():
        pytest.skip("CUDA only")
    data, _ = _two_equal_sources(2)
    pca = decompose(
        data, affine=np.eye(4), tr=1.0, mask=np.ones(data.shape[:3], bool), polort=1, n_components=2
    )
    new = rotate_to_ica(pca, seed=3, device=torch.device("cuda"))
    ref = _rotate_to_ica_host_reference(pca, seed=3)
    assert new.note == ref.note == ""
    np.testing.assert_allclose(new.explained, ref.explained, rtol=1e-3)
    for k in range(2):
        assert np.corrcoef(new.timecourses[:, k], ref.timecourses[:, k])[0, 1] > 0.9999
    np.testing.assert_allclose(new.zmap, ref.zmap, rtol=0, atol=5e-3)
    np.testing.assert_allclose(new.correlation, ref.correlation, rtol=0, atol=5e-4)
