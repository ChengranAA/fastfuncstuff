"""The InstaICA engine: spatial ICA as a rotation of an InstaPCA decomposition.

MELODIC's recipe -- reduce the run to its leading temporal dimensions, whiten,
find the rotation that makes the spatial maps most non-Gaussian -- without
touching the data a second time. InstaPCA has already prepared every voxel
(drift out, unit length) and projected it onto the leading eigenvectors ``u``:
``P = c u`` is ``(V, K)``, and its columns, scaled to unit variance, *are* the
whitened spatial components. FastICA runs on those ``K`` rows of ``V`` samples
(milliseconds at K = 30), and everything else is ``K x K`` algebra:

* the data is ``c^T ~ u D Z`` with ``Z`` the whitened components and ``D`` their
  scales; ICA writes ``Z = W^T S``, so the IC time courses are ``A = u D W^T``;
* each voxel's correlation with an IC's time course is ``c a / |a|`` --
  ``P D W^T`` with columns normalised -- the same amplitude-free map InstaPCA
  draws, now for an IC;
* amplitude is that correlation times InstaPCA's per-voxel scale, unchanged;
* the z-map is the spatial source ``S`` itself, z-scored: MELODIC's picture.

So a decomposition costs one InstaPCA plus a few milliseconds, and the review
loop -- labels, the noise ortvec, Denoise's "ICA noise" -- is the same code.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from fastfuncstuff.viewer.instapca import Decomposition


def rotate_to_ica(
    pca: Decomposition,
    *,
    seed: int = 0,
    max_iter: int = 1000,
    fun: str = "logcosh",
    device: torch.device | None = None,
) -> Decomposition:
    """Spatial ICA of ``pca``'s components: same span, rotated to independence."""
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
