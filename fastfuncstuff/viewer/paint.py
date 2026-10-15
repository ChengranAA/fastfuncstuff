"""Painting labels into a volume: the brush geometry, and nothing about Qt.

A painted ROI layer lives on a voxel grid of its own -- by default the display
grid at the moment it was made, so painting is one voxel per voxel on screen.
Everything here works in that layer's voxel indices through its affine, so the
same stroke means the same voxels whatever the display grid becomes later.

A ``disk`` brush paints the slice being looked at: voxels within the radius in
the two in-plane axes, on the one index along the plane's normal. A ``sphere``
reaches through neighbouring slices too. Radius is in mm, so a 3 mm brush is 3
mm on a 0.7 mm anatomy and on a 2 mm EPI alike.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fastfuncstuff.viewer.slicing import plane_layout
from fastfuncstuff.viewer.state import Plane

SHAPES = ("disk", "sphere")


def brush_voxels(
    centre_mm: Sequence[float],
    radius_mm: float,
    *,
    affine: np.ndarray,
    shape: tuple[int, int, int],
    plane: Plane,
    brush: str = "disk",
) -> np.ndarray:
    """``(M, 3)`` int voxel indices a brush at ``centre_mm`` covers, inside the grid.

    Always at least the voxel under the centre, so a brush smaller than a
    voxel still paints the one it is on.
    """
    aff = np.asarray(affine, dtype=np.float64)
    inv = np.linalg.inv(aff)
    centre = inv[:3, :3] @ np.asarray(centre_mm, dtype=np.float64) + inv[:3, 3]
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    reach = np.ceil(radius_mm / vox).astype(int)
    if brush == "disk":
        fixed = plane_layout(aff, plane).fixed
        reach[fixed] = 0
    elif brush != "sphere":
        raise ValueError(f"brush must be one of {SHAPES}, not {brush!r}")
    base = np.rint(centre).astype(int)
    if brush == "disk":
        # The slice under the cursor, whatever fraction of a voxel it sits at.
        centre = centre.copy()
        centre[fixed] = base[fixed]
    offsets = np.stack(
        np.meshgrid(*(np.arange(-r, r + 1) for r in reach), indexing="ij"), -1
    ).reshape(-1, 3)
    pts = base + offsets
    # Distance in mm from the true centre, through the affine, so an oblique
    # or anisotropic grid gets a round brush in space, not in index units.
    delta_mm = (pts - centre) @ aff[:3, :3].T
    keep = (delta_mm**2).sum(1) <= radius_mm**2
    keep |= (pts == base).all(1)
    inside = np.all((pts >= 0) & (pts < np.asarray(shape)), axis=1)
    return pts[keep & inside]


def stroke_centres(
    previous_mm: Sequence[float] | None, here_mm: Sequence[float], radius_mm: float
) -> np.ndarray:
    """Brush centres from the last drag position to this one, half a radius apart.

    A fast drag reports positions far apart; painting only at those leaves a
    dotted line. Filling in at half the radius keeps the stroke solid without
    repainting the same voxels many times over.
    """
    here = np.asarray(here_mm, dtype=np.float64)
    if previous_mm is None:
        return here[None]
    prev = np.asarray(previous_mm, dtype=np.float64)
    gap = float(np.linalg.norm(here - prev))
    n = max(1, int(np.ceil(gap / max(radius_mm * 0.5, 0.25))))
    t = np.linspace(0.0, 1.0, n + 1)[1:]
    return prev + t[:, None] * (here - prev)


def encode_points(points: np.ndarray) -> str:
    """Brush centres as script text: ``x,y,z;x,y,z`` in mm, to 0.01 mm."""
    return ";".join(",".join(f"{v:.2f}" for v in p) for p in np.asarray(points))


def decode_points(text: str) -> np.ndarray:
    rows = [r for r in str(text).split(";") if r.strip()]
    return np.array([[float(v) for v in r.split(",")] for r in rows], dtype=np.float64).reshape(
        -1, 3
    )
