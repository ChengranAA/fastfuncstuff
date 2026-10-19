"""What an ROI-mode fit writes: a stats table, response curves, optional painted maps.

The fit ran on ROI means, so ``results`` has one row per ROI. The table carries the
same columns ``-Rbuck`` would put in sub-bricks (shared builder, shared names); the
curves are the FIR/TENT betas of each condition against their lag times.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from fastfuncstuff.glm.arma import OLS_VOXEL_SCATTER_ATTRS, VOXEL_SCATTER_ATTRS
from fastfuncstuff.glm.outputs import _bucket_subbricks, _ensure_numpy


def roi_bucket_columns(
    results, condition_names: list[str], contrast_names: list[str] | None = None
) -> dict[str, np.ndarray]:
    """``{sub-brick label: (n_rois,)}`` for everything ``-Rbuck`` would hold, plus R2 and noise."""
    n = int(results.betas.shape[0])
    bricks, labels, *_ = _bucket_subbricks(
        results, condition_names, contrast_names, None, (n, 1, 1), None, np.dtype(np.float32)
    )
    cols = {lab: b.reshape(n) for lab, b in zip(labels, bricks, strict=True)}
    if getattr(results, "r2", None) is not None:
        cols["R2"] = _ensure_numpy(results.r2).reshape(n)
    params = getattr(results, "arma_params", None)
    if params is not None:
        cols["arma_a"], cols["arma_b"] = (_ensure_numpy(params[:, k]) for k in (0, 1))
    if getattr(results, "sigma2", None) is not None:
        cols["StDev"] = np.sqrt(_ensure_numpy(results.sigma2)).reshape(n)
    return cols


def roi_response_curves(
    results, design_info: dict, lag_times: np.ndarray | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{condition: (lag times, (n_rois, n_lags) betas)}`` for every FIR/TENT condition."""
    if lag_times is None or getattr(results, "betas", None) is None:
        return {}
    lags = np.asarray(lag_times, dtype=float)
    fitted = getattr(results, "fitted_column_indices", None)
    position = None if fitted is None else {c: j for j, c in enumerate(fitted)}
    betas = _ensure_numpy(results.betas)
    skip = set(design_info.get("stim_vec_indices") or [])
    out = {}
    for label, bot, top in zip(
        design_info.get("stim_labels") or [],
        design_info.get("stim_bots") or [],
        design_info.get("stim_tops") or [],
        strict=False,
    ):
        cols = list(range(bot, top + 1))
        if len(cols) != len(lags) or bot in skip:
            continue
        if position is not None:
            if any(c not in position for c in cols):
                continue
            cols = [position[c] for c in cols]
        out[label] = (lags, betas[:, cols])
    return out


def expand_to_voxels(results, roi_set, *, volume_shape, affine, nifti_header) -> None:
    """Paint every per-ROI array on *results* into its voxels, in place.

    After this the volume writers see an ordinary masked analysis: ``voxel_mask`` is the
    ROI cover and each voxel carries its ROI's value.  Only a disjoint label map can be
    painted; the caller checks.
    """
    n = roi_set.n_rois
    # ljung_box and the likelihood surface are per-voxel but absent from the scatter lists.
    known = {*VOXEL_SCATTER_ATTRS, *OLS_VOXEL_SCATTER_ATTRS, "var_betas", "reml_lklhd_surface"}
    known |= {"ljung_box"}
    for attr in known:
        value = getattr(results, attr, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == n:
            painted = roi_set.paint(value.detach().cpu().numpy())
            setattr(results, attr, torch.from_numpy(painted).to(value.dtype))
    # A per-ROI array this list does not know would reach a writer unpainted and fail late.
    for attr, value in vars(results).items():
        if attr not in known and attr != "xtx_inv" and isinstance(value, torch.Tensor):
            if value.ndim >= 1 and value.shape[0] == n:
                warnings.warn(f"-roi_maps: results.{attr} was not painted", stacklevel=2)
    results.voxel_mask = torch.from_numpy(roi_set.rows.reshape(-1))
    results.original_shape = results.full_shape = tuple(volume_shape)
    results.affine = affine
    results.nifti_header = nifti_header
