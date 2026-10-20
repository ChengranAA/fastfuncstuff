"""The cortical ribbon cut into small cells, for reading thick depth bins.

A depth bin is a slab: with three depths, depth 1 is the deepest third of the ribbon's
*volume*, not the surface one sixth of the way out. Reading a slab takes points spread
through it, each counting by the volume it stands for. A fine isotropic grid gets that
right but has to be finer than a bin everywhere: 15 depths of 2.5 mm cortex are
0.17 mm bins, and a grid that fine is ~10^9 points per hemisphere. The meshes give a
tiling that is fine only through the depth:

* each white triangle and its pial partner bound a wedge of ribbon. The wedge is cut
  ``L x L`` along the surface, so a sub-triangle is no bigger than ``step^2`` (the
  footprint rule of :mod:`surface.projection`: no voxel face goes unread), and ``S``
  times through the depth, ``S >= 2 x n_depths`` and no thicker than ``step / 2``;
* every cell's volume is exact: its corners move linearly from white to pial, so the
  rate the triangle sweeps volume is a quadratic in the depth parameter and integrates
  in closed form. The cells tile the solid between the meshes wherever they do not
  cross, however far a pial vertex slides from its white partner.

The pairing only tiles; it never decides depth. A cell's depth comes from the mesh
distances at its white-side and pial-side faces (equidistant), or from the volume
quantile within white-foot columns (equivolume) -- the definitions
:mod:`surface.volume_depth` uses on voxels, so a cell and a voxel at one place get one
depth. A cell across a bin boundary gives each bin its share of its volume (depth
linear across the cell). A cell belongs to the column whose midthickness point is
nearest, the rule of :mod:`surface.ribbon`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

__all__ = ["RibbonCells", "patch_faces", "ribbon_cells"]

#: Nodes generated per batch: bounds the (faces, sub-triangles, sheets, 3) arrays.
_BATCH_NODES = 1 << 22


@dataclass
class RibbonCells:
    """Cells of the ribbon: where to read, how much volume each read stands for, and
    which depths and column it belongs to."""

    centre: np.ndarray  # (N, 3) scanner mm
    volume: np.ndarray  # (N,) mm^3
    lo: np.ndarray  # (N,) depth metric on the cell's white-side face (0 white .. 1 pial)
    hi: np.ndarray  # (N,) and on its pial-side face
    surface: np.ndarray  # (N,) index into the surfaces given
    vertex: np.ndarray  # (N,) owning column (vertex of that surface)
    #: Volume of cells whose white-to-pial sweep turned inside out (meshes crossing),
    #: left out of the tiling.
    inverted_mm3: float = 0.0

    def __len__(self) -> int:
        return int(self.volume.size)

    def subset(self, keep: np.ndarray) -> RibbonCells:
        return RibbonCells(
            self.centre[keep], self.volume[keep], self.lo[keep], self.hi[keep],
            self.surface[keep], self.vertex[keep], self.inverted_mm3,
        )  # fmt: skip

    def bin_shares(self, n_depths: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(cell, bin, share)``: each cell's volume split over the depth bins it
        spans, bin ``k`` (0-based, 0 deepest) holding metric ``(k/K, (k+1)/K]``."""
        k = int(n_depths)
        lo = np.minimum(self.lo, self.hi).astype(np.float64)
        hi = np.maximum(self.lo, self.hi).astype(np.float64)
        span = hi - lo
        flat = span < 1e-9
        # The LN2_LAYERS convention, ceil(metric * K), for cells with no depth extent.
        mid_bin = np.clip(np.ceil(0.5 * (lo + hi) * k) - 1, 0, k - 1).astype(np.int64)
        first = np.where(flat, mid_bin, np.clip(np.floor(lo * k), 0, k - 1).astype(np.int64))
        last = np.where(flat, mid_bin, np.clip(np.ceil(hi * k) - 1, 0, k - 1).astype(np.int64))
        last = np.maximum(last, first)
        cells, bins, shares = [], [], []
        for off in range(int((last - first).max(initial=0)) + 1):
            b = first + off
            ok = b <= last
            a = np.maximum(lo, b / k)
            c = np.minimum(hi, (b + 1) / k)
            share = np.where(flat, 1.0, np.clip(c - a, 0.0, None) / np.where(flat, 1.0, span))
            ok &= share > 0
            idx = np.flatnonzero(ok)
            cells.append(idx)
            bins.append(b[idx])
            shares.append(share[idx])
        return np.concatenate(cells), np.concatenate(bins), np.concatenate(shares)


def patch_faces(surface, members: np.ndarray, margin_mm: float) -> np.ndarray:
    """Faces to tile so every cell the ``members`` columns own is among them.

    A cell is owned through its nearest midthickness point, which can sit a column's
    length sideways of the wedge it lies in (pial vertices slide ~1 mm, p90 2.4 mm,
    from their white partners), so take every face with a corner whose midthickness
    point is within ``margin_mm`` of a member's.
    """
    from scipy.spatial import KDTree

    mid = 0.5 * (np.asarray(surface.white, np.float64) + np.asarray(surface.pial, np.float64))
    near = KDTree(mid[members]).query(mid, distance_upper_bound=margin_mm)[0]
    close = np.isfinite(near)
    return np.flatnonzero(close[np.asarray(surface.faces)].any(axis=1))


def _sub_triangles(level: int) -> np.ndarray:
    """Barycentric corners ``(L^2, 3, 3)`` of a triangle cut into ``L x L``, each
    sub-triangle wound like the parent."""
    tris = []
    for i in range(level):
        for j in range(level - i):
            tris.append(((i, j), (i + 1, j), (i, j + 1)))
            if i + j < level - 1:
                tris.append(((i + 1, j), (i + 1, j + 1), (i, j + 1)))
    uv = np.asarray(tris, np.float64) / level
    return np.concatenate([1.0 - uv.sum(-1, keepdims=True), uv], -1)


def _area(tri: np.ndarray) -> np.ndarray:
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def _wedges(
    white: np.ndarray, delta: np.ndarray, level: int, sheets: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For faces with corners ``white`` + s ``delta`` (``(F, 3, 3)``), s in [0, 1]:
    ``(nodes (F, T, S+1, 3), cell centres (F, T, S, 3), cell volumes (F, T, S))``,
    in batches of faces."""
    sub = _sub_triangles(level)  # (T, 3, 3)
    n_sub = sub.shape[0]
    s_nodes = np.arange(sheets + 1, dtype=np.float64) / sheets
    s_mid = (np.arange(sheets, dtype=np.float64) + 0.5) / sheets
    batch = max(1, _BATCH_NODES // (n_sub * (sheets + 1)))
    for f0 in range(0, white.shape[0], batch):
        w = white[f0 : f0 + batch]
        d = delta[f0 : f0 + batch]
        a0 = np.einsum("tck,fkx->ftcx", sub, w)  # sub-triangle corners on white
        av = np.einsum("tck,fkx->ftcx", sub, d)  # and how they move to pial
        c0, cv = a0.mean(2), av.mean(2)
        nodes = c0[:, :, None] + s_nodes[None, None, :, None] * cv[:, :, None]
        centres = c0[:, :, None] + s_mid[None, None, :, None] * cv[:, :, None]
        # The triangle's area vector at depth s is c_0 + s c_1 + s^2 c_2, and it sweeps
        # volume at (area vector . mean corner velocity): a quadratic, integrated exactly.
        e1, e2 = a0[:, :, 1] - a0[:, :, 0], a0[:, :, 2] - a0[:, :, 0]
        v1, v2 = av[:, :, 1] - av[:, :, 0], av[:, :, 2] - av[:, :, 0]
        g = [
            0.5 * np.einsum("ftx,ftx->ft", n, cv)
            for n in (np.cross(e1, e2), np.cross(e1, v2) + np.cross(v1, e2), np.cross(v1, v2))
        ]
        s0, s1 = s_nodes[:-1], s_nodes[1:]
        vol = (
            g[0][..., None] * (s1 - s0)
            + g[1][..., None] * (s1**2 - s0**2) / 2
            + g[2][..., None] * (s1**3 - s0**3) / 3
        )
        yield nodes, centres, vol


def ribbon_cells(
    surfaces,
    faces_of: dict[int, np.ndarray],
    step: float,
    n_depths: int = 1,
    method: str = "equidistant",
    *,
    column_voxels: float = 64.0,
    max_level: int = 32,
    max_sheets: int = 512,
    device=None,
    log: Callable[[str], None] | None = None,
) -> RibbonCells:
    """Tile the ribbon over ``faces_of[surface index]`` (face indices) into cells.

    ``step`` (mm) is the read spacing along the surface; through the depth it is at
    most ``step / 2`` and there are at least two cells per bin. ``surfaces`` are
    :class:`surface.volume_depth.RibbonSurfaces`.
    """
    import torch

    from fastfuncstuff.surface.volume_depth import column_depth_cdf, column_quantile
    from fastfuncstuff.surface.voxelize import MeshDistance

    if method not in ("equidistant", "equivolume"):
        raise ValueError(f"depth method must be equidistant or equivolume, not {method!r}")
    device = device or torch.device("cpu")
    step = float(step)
    parts = []
    inverted = 0.0
    for h, face_idx in faces_of.items():
        surf = surfaces[h]
        faces = np.asarray(surf.faces, np.int64)
        white = np.asarray(surf.white, np.float64)
        pial = np.asarray(surf.pial, np.float64)
        fv = faces[np.asarray(face_idx, np.int64)]
        if fv.size == 0:
            continue
        cw, cp = white[fv], pial[fv]
        level = np.clip(
            np.ceil(np.sqrt(np.maximum(_area(cw), _area(cp))) / step), 1, max_level
        ).astype(np.int64)
        tau = np.linalg.norm(cp - cw, axis=2).max(axis=1)
        sheets = np.clip(
            np.maximum(np.ceil(tau / (0.5 * step)), 2 * n_depths), 2, max_sheets
        ).astype(np.int64)
        nodes, centres, vols, lo_idx = [], [], [], []
        n_nodes = 0
        for key in np.unique(level * (max_sheets + 1) + sheets):
            lv, sh = divmod(int(key), max_sheets + 1)
            sel = np.flatnonzero((level == lv) & (sheets == sh))
            for nd, ce, vo in _wedges(cw[sel], cp[sel] - cw[sel], lv, sh):
                nf, nt = vo.shape[:2]
                base = n_nodes + (np.arange(nf * nt) * (sh + 1))[:, None] + np.arange(sh)
                nodes.append(nd.reshape(-1, 3))
                centres.append(ce.reshape(-1, 3))
                vols.append(vo.reshape(-1))
                lo_idx.append(base.reshape(-1))
                n_nodes += nd.shape[0] * nd.shape[1] * nd.shape[2]
        node_xyz = np.concatenate(nodes)
        centre = np.concatenate(centres)
        vol = np.concatenate(vols)
        lo_i = np.concatenate(lo_idx)
        del nodes, centres, vols, lo_idx
        if vol.sum() < 0:  # faces wound inward
            vol = -vol
        bad = vol < 0
        inverted += float(-vol[bad].sum())
        vol = np.where(bad, 0.0, vol)
        if log:
            log(
                f"{surf.name}: {fv.shape[0]:,} faces -> {vol.size:,} cells "
                f"({int(level.min())}-{int(level.max())} cuts along, "
                f"{int(sheets.min())}-{int(sheets.max())} through)"
            )

        brick = 4.0 * step
        near_w = MeshDistance(white, faces)(node_xyz, device, brick_mm=brick)
        dp = MeshDistance(pial, faces)(node_xyz, device, brick_mm=brick).distance
        dw = near_w.distance
        rho = np.clip(dw / np.maximum(dw + dp, 1e-9), 0.0, 1.0).astype(np.float32)
        del dp, node_xyz
        if method == "equivolume":
            # Each cell's volume spread over its depth range, half on the white foot of
            # each of its two faces.
            ends = np.r_[lo_i, lo_i + 1]
            other = np.r_[lo_i + 1, lo_i]

            def dev(a):
                return torch.as_tensor(a, device=device)

            cdf = column_depth_cdf(
                dev(rho[ends]), dev(near_w.face[ends]), dev(near_w.bary[ends]), faces,
                len(white), column_voxels, weight=dev(np.r_[vol, vol] * 0.5),
                rho_hi=dev(rho[other]),
            )  # fmt: skip
            metric = (
                column_quantile(cdf, dev(rho), dev(near_w.face), dev(near_w.bary), faces)
                .cpu()
                .numpy()
            )
            del cdf, ends, other
        else:
            metric = rho
        del near_w

        near_m = MeshDistance(0.5 * (white + pial), faces)(centre, device, brick_mm=brick)
        owner = faces[near_m.face, np.argmax(near_m.bary, axis=1)]
        del near_m
        keep = vol > 0
        if surf.cortex is not None:
            keep &= np.asarray(surf.cortex, bool)[owner]
        parts.append(
            RibbonCells(
                centre[keep],
                vol[keep],
                metric[lo_i[keep]],
                metric[lo_i[keep] + 1],
                np.full(int(keep.sum()), h, np.int16),
                owner[keep],
            )  # fmt: skip
        )
    if not parts:
        empty = np.zeros(0)
        return RibbonCells(
            np.zeros((0, 3)), empty, empty, empty, np.zeros(0, np.int16),
            np.zeros(0, np.int64), inverted,
        )  # fmt: skip
    return RibbonCells(
        *(np.concatenate([getattr(p, f) for p in parts]) for f in
          ("centre", "volume", "lo", "hi", "surface", "vertex")),
        inverted_mm3=inverted,
    )  # fmt: skip
