"""Pair volume and surface runs in time while retaining independent geometries."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.cli_utils import LoadResult, load_and_preprocess_runs
from fastfuncstuff.io.afni import load_nifti, save_nifti
from fastfuncstuff.io.gifti import is_gifti, surface_meta


@dataclass
class PairedRuns:
    data: torch.Tensor  # (volume locations + surface locations, total time)
    run_starts: list[int]
    tr: float
    volume: LoadResult
    surface: LoadResult

    def split_map(
        self, values: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
        """Split location-first results without expanding either domain's mask."""
        if values.shape[0] != self.data.shape[0]:
            raise ValueError(f"Expected {self.data.shape[0]} locations, got {values.shape[0]}")
        return values[: self.volume.n_voxels], values[self.volume.n_voxels :]

    def save_map(
        self, values: torch.Tensor | np.ndarray, volume_path: str | Path, surface_path: str | Path
    ) -> None:
        """Write a location-first map to two files with each input's own geometry."""
        for part, domain, path, surface in zip(
            self.split_map(values),
            (self.volume, self.surface),
            (volume_path, surface_path),
            (False, True),
            strict=True,
        ):
            if is_gifti(str(path)) != surface:
                raise ValueError(
                    f"Wrong output format for {'surface' if surface else 'volume'}: {path}"
                )
            arr = (
                part.detach().cpu().numpy() if isinstance(part, torch.Tensor) else np.asarray(part)
            )
            shape = tuple(domain.volume_shape)
            if domain.mask_flat is not None:
                full = np.zeros((int(np.prod(shape)), *arr.shape[1:]), dtype=arr.dtype)
                full[domain.mask_flat] = arr
                arr = full
            arr = arr.reshape(*shape, *arr.shape[1:])
            save_nifti(arr, output_path=path, affine=domain.affine, header=domain.nifti_header)


def _declared_tr(path: str) -> float | None:
    """Read an actual time-unit declaration, never NIfTI's default pixdim=1."""
    img = load_nifti(path)
    if is_gifti(path):
        value = (surface_meta(img.header) or {}).get("TR_seconds")
        return float(value) if value else None
    if len(img.shape) != 4:
        raise ValueError(f"Expected 4-D run: {path}")
    _, time_unit = img.header.get_xyzt_units()
    scale = {"sec": 1.0, "msec": 0.001, "usec": 0.000001}.get(time_unit)
    return float(img.header.get_zooms()[3]) * scale if scale is not None else None


def load_paired_runs(
    volume_files: list[str],
    surface_files: list[str],
    *,
    tr: float | None = None,
    volume_mask: str | None = None,
    surface_mask: str | None = None,
    blur_fwhm: float | None = None,
    device: torch.device = torch.device("cpu"),
    drop_first: int = 0,
    drop_last: int = 0,
) -> PairedRuns:
    """Match two spatial domains by run index, then concatenate their locations.

    Inputs must each form a stable geometry across runs. Temporal pairing is
    positional: one volume and one surface file represent the same physical run.
    """
    if not volume_files or len(volume_files) != len(surface_files):
        raise ValueError("Pass one volume and one surface file per paired run")
    if any(is_gifti(f) for f in volume_files) or any(not is_gifti(f) for f in surface_files):
        raise ValueError("Volume files must be NIfTI/AFNI and surface files must be GIfTI")
    if tr is not None and (not np.isfinite(tr) or tr <= 0):
        raise ValueError("TR must be a positive finite number of seconds")
    if drop_first < 0 or drop_last < 0:
        raise ValueError("Dropped frame counts must be nonnegative")
    declared = [_declared_tr(f) for f in [*volume_files, *surface_files]]
    known = [v for v in declared if v is not None]
    effective_tr = tr if tr is not None else (known[0] if known else None)
    if effective_tr is None:
        raise ValueError("Neither input declares a TR; pass -tr explicitly")
    if any(
        not np.isfinite(v) or v <= 0 or not np.isclose(v, effective_tr, rtol=1e-5) for v in known
    ):
        raise ValueError("Paired runs declare inconsistent TR values")
    kwargs = dict(
        tr=effective_tr,
        blur_fwhm=blur_fwhm,
        device=device,
        drop_first=drop_first,
        drop_last=drop_last,
        verbose=False,
    )
    volume = load_and_preprocess_runs(volume_files, mask_file=volume_mask, **kwargs)
    surface = load_and_preprocess_runs(surface_files, mask_file=surface_mask, **kwargs)
    if volume.run_starts != surface.run_starts or volume.n_timepoints != surface.n_timepoints:
        raise ValueError("Paired runs must have equal time lengths after trimming")
    if not np.isclose(volume.tr, surface.tr):
        raise ValueError("Paired runs have inconsistent TR")
    return PairedRuns(
        torch.cat((volume.data, surface.data), dim=0),
        volume.run_starts,
        effective_tr,
        volume,
        surface,
    )
