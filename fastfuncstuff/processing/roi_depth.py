"""ROI means along the cortical ribbon: one series per ROI per cortical depth.

The usual route to depth-resolved ROI signals is to upsample the data, build integer
layers on the fine grid (LayNii's LN2_LAYERS), intersect them with an ROI and average
the voxels. This does the same arithmetic straight from the white and pial meshes:

1. the data grid is upsampled (``-depth_dxyz``) over the ROIs' patch of cortex only;
2. every fine voxel between white and pial gets its depth -- equidistant, or
   equivolume as a volume quantile within its cortical column -- and its owning
   column (vertex), from :func:`surface.volume_depth.cortical_depth_volumes`, the
   code behind ``ffs_util_surf2layers``;
3. a fine voxel joins an ROI when its column is in the ROI. A surface ROI names the
   columns; a volume ROI claims the columns whose midthickness point lies in it, so
   every depth bin of an ROI averages the same patch of cortex;
4. each fine voxel reads the data by trilinear interpolation and counts by its
   volume, so a bin's value is the volume-weighted mean over its part of the ribbon
   -- what averaging an upsampled image within LayNii layers gives.

Interpolation is linear in the data, so the whole thing is one sparse
``(n_rois x n_depths, n_voxels)`` weight matrix: the ordinary weighted-ROI collapse,
and every output an ROI fit writes then comes out per ROI per depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from fastfuncstuff.processing.roi_collapse import RoiSet, _gifti_label_table, _stem


@dataclass(frozen=True)
class DepthOptions:
    n_depths: int = 1
    #: ``equidistant`` or ``equivolume``.
    method: str = "equidistant"
    #: Ribbon grid spacing, mm; None picks one from the data grid.
    dxyz: float | None = None
    column_voxels: float = 64.0


@dataclass
class _Roi:
    rid: int
    name: str
    #: index into the surfaces -> (V,) bool vertex membership.
    members: dict[int, np.ndarray]


def _read_label(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(vertex ids, white tkregister coords)`` of an ASCII FreeSurfer ``.label``."""
    lines = Path(path).read_text().splitlines()
    n = int(lines[1].split()[0])
    rows = np.array([ln.split()[:4] for ln in lines[2 : 2 + n]], dtype=np.float64).reshape(-1, 4)
    return rows[:, 0].astype(np.int64), rows[:, 1:4]


def _candidates(path: str, surfaces, n_vertices: int | None) -> list[int]:
    """Surfaces a surface ROI file could belong to, by its lh/rh name and vertex count."""
    from fastfuncstuff.io.freesurfer import infer_label

    hemi = infer_label(Path(path).name)[0]
    out = [
        i
        for i, s in enumerate(surfaces)
        if (hemi is None or s.name == hemi) and (n_vertices is None or len(s.white) == n_vertices)
    ]
    if not out:
        have = ", ".join(f"{s.name} ({len(s.white):,} vertices)" for s in surfaces)
        raise ValueError(f"{path}: matches none of the surfaces given ({have})")
    return out


def _label_surface(path: str, surfaces, notes: list[str]) -> int:
    """The surface a ``.label`` was drawn on, checked by its own white coordinates."""
    ids, coords = _read_label(path)
    if ids.size == 0:
        raise ValueError(f"{path}: empty label")
    fits = [i for i in _candidates(path, surfaces, None) if ids.max() < len(surfaces[i].white)]
    if not fits:
        raise ValueError(f"{path}: vertex ids reach {ids.max()}, beyond every surface given")
    scored = []
    for i in fits:
        s = surfaces[i]
        if s.tkr_to_scanner is None:
            continue
        to_tkr = np.linalg.inv(np.asarray(s.tkr_to_scanner, np.float64))
        white = np.asarray(s.white, np.float64)[ids] @ to_tkr[:3, :3].T + to_tkr[:3, 3]
        scored.append((float(np.median(np.linalg.norm(white - coords, axis=1))), i))
    if scored:
        err, best = min(scored)
        if err > 2.0:
            raise ValueError(
                f"{path}: its coordinates sit {err:.1f} mm (median) from the white surface "
                "of every hemisphere given -- drawn on another subject or surface"
            )
        if err > 0.01:
            notes.append(
                f"{Path(path).name}: coordinates {err:.2f} mm (median) off this white "
                "surface -- drawn on an edited copy?"
            )
        return best
    if len(fits) > 1:
        raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
    notes.append(f"{Path(path).name}: no tkregister frame to check against, matched by name")
    return fits[0]


def _surface_rois(path: str, surfaces, notes: list[str]) -> list[_Roi]:
    """ROIs from a surface file: ``.label``, ``.annot`` or a GIfTI label/shape map."""
    from fastfuncstuff.io.freesurfer import NOT_A_REGION, read_annotation

    low = path.lower()
    if low.endswith(".label"):
        i = _label_surface(path, surfaces, notes)
        ids, _ = _read_label(path)
        member = np.zeros(len(surfaces[i].white), bool)
        member[ids] = True
        return [_Roi(1, _stem(path), {i: member})]

    if low.endswith(".annot"):
        import nibabel.freesurfer as nfs

        n = len(nfs.read_annot(path)[0])
        cands = _candidates(path, surfaces, n)
        if len(cands) > 1:
            raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
        annot = read_annotation(path, n_vertices=n)
        return [
            _Roi(k + 1, name, {cands[0]: annot.labels == k})
            for k, name in enumerate(annot.names)
            if name not in NOT_A_REGION and (annot.labels == k).any()
        ]

    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.gifti import mesh_fingerprint, surface_meta

    img = load_nifti(path)
    values = np.asanyarray(img.dataobj).reshape(-1)
    if not np.array_equal(values, np.round(values)):
        raise ValueError(f"{path}: a surface ROI map must hold integer labels")
    mine = (surface_meta(img.header) or {}).get("mesh_fingerprint")
    if mine:
        cands = [
            i for i, s in enumerate(surfaces) if mesh_fingerprint(s.faces, len(s.white)) == mine
        ]
        if not cands:
            raise ValueError(f"{path}: drawn on a mesh that is none of the surfaces given")
    else:
        cands = _candidates(path, surfaces, values.size)
        if len(cands) > 1:
            raise ValueError(f"{path}: name its hemisphere (lh./rh.) -- both surfaces fit")
        notes.append(f"{Path(path).name}: no mesh fingerprint, matched by name and vertex count")
    table = _gifti_label_table(path)
    labels = [int(v) for v in np.unique(values) if v != 0]
    one = len(labels) == 1 and labels[0] not in table
    return [
        _Roi(
            lab,
            _stem(path) if one else (table[lab].name if lab in table else f"{_stem(path)}_{lab}"),
            {cands[0]: values == lab},
        )
        for lab in labels
    ]


def _volume_rois(path: str, surfaces, notes: list[str]) -> list[_Roi]:
    """A volume ROI, mapped to the columns whose midthickness point lies inside it.

    Read on the ROI file's own grid (any resolution, in register with the surfaces),
    nearest voxel: a 3-D integer map gives one ROI per label, a 4-D stack one per
    volume (any value above zero is in).
    """
    from fastfuncstuff.io.afni import load_nifti
    from fastfuncstuff.io.labels import label_table

    img = load_nifti(path)
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 4 and arr.shape[3] == 1:
        arr = arr[..., 0]
    if arr.ndim not in (3, 4):
        raise ValueError(f"{path}: ROI files must be 3-D or 4-D, got {arr.ndim}-D")
    inv = np.linalg.inv(np.asarray(img.affine, np.float64))
    sampled = []  # per surface: (V,) or (V, n) values at midthickness
    for s in surfaces:
        mid = 0.5 * (np.asarray(s.white, np.float64) + np.asarray(s.pial, np.float64))
        ijk = np.rint(mid @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
        inside = np.all((ijk >= 0) & (ijk < np.asarray(arr.shape[:3])), axis=1)
        vals = np.zeros((len(mid), *arr.shape[3:]), arr.dtype)
        vals[inside] = arr[ijk[inside, 0], ijk[inside, 1], ijk[inside, 2]]
        sampled.append(vals)
    notes.append(f"{Path(path).name}: volume ROI, each column joins by its midthickness point")
    if arr.ndim == 4:
        return [
            _Roi(j + 1, f"{_stem(path)}_{j + 1}" if arr.shape[3] > 1 else _stem(path),
                 {i: v[:, j] > 0 for i, v in enumerate(sampled)})
            for j in range(arr.shape[3])
        ]  # fmt: skip
    if not np.array_equal(arr, np.round(arr)):
        raise ValueError(f"{path}: a 3-D ROI file must hold integer labels; give weights as 4-D")
    table = label_table(path, img)
    labels = [int(v) for v in np.unique(arr) if v != 0]
    one = len(labels) == 1 and labels[0] not in table
    return [
        _Roi(
            lab,
            _stem(path) if one else (table[lab].name if lab in table else f"{_stem(path)}_{lab}"),
            {i: v == lab for i, v in enumerate(sampled)},
        )
        for lab in labels
    ]


def _read_rois(paths: list[str], surfaces, notes: list[str]) -> list[_Roi]:
    from fastfuncstuff.io.gifti import is_gifti

    rois: list[_Roi] = []
    for f in paths:
        plain = f.split("[")[0]
        on_surface = plain.lower().endswith((".label", ".annot")) or is_gifti(plain)
        rois += (_surface_rois if on_surface else _volume_rois)(plain, surfaces, notes)
    if len(paths) > 1:  # labels repeat across files; the table needs distinct ids
        for k, r in enumerate(rois):
            r.rid = k + 1
    return rois


def _trilinear(points: np.ndarray, affine: np.ndarray, shape, mask: np.ndarray | None):
    """``(flat corner indices, weights)``, each ``(N, 8)``, of reading a grid at ``points``.

    Corners off the grid or outside ``mask`` are dropped and the rest renormalised, so
    a point never reads a voxel the analysis excludes; a point with no corner left
    gets all-zero weights.
    """
    inv = np.linalg.inv(np.asarray(affine, np.float64))
    ijk = points @ inv[:3, :3].T + inv[:3, 3]
    base = np.floor(ijk).astype(np.int64)
    frac = ijk - base
    dims = np.asarray(shape[:3])
    idx = np.empty((len(points), 8), np.int64)
    w = np.empty((len(points), 8), np.float64)
    for c in range(8):
        off = np.array([(c >> 2) & 1, (c >> 1) & 1, c & 1])
        corner = base + off
        ok = np.all((corner >= 0) & (corner < dims), axis=1)
        wc = np.prod(np.where(off, frac, 1.0 - frac), axis=1)
        flat = np.ravel_multi_index(tuple(np.clip(corner, 0, dims - 1).T), tuple(dims))
        if mask is not None:
            ok &= mask.reshape(-1)[flat]
        idx[:, c], w[:, c] = flat, np.where(ok, wc, 0.0)
    total = w.sum(axis=1, keepdims=True)
    return idx, np.divide(w, total, out=np.zeros_like(w), where=total > 0)


def default_dxyz(data_affine: np.ndarray) -> float:
    vox = np.linalg.norm(np.asarray(data_affine, np.float64)[:3, :3], axis=0)
    return float(min(0.5, vox.min() / 3.0))


def build_depth_roi_set(
    paths: str | list[str],
    surfaces,
    data_affine: np.ndarray,
    data_shape: tuple[int, ...],
    mask: np.ndarray | None = None,
    opts: DepthOptions = DepthOptions(),
    *,
    device=None,
) -> RoiSet:
    """ROI x depth rows for data on ``(data_affine, data_shape)``, read along the ribbon.

    ``surfaces`` are :class:`surface.volume_depth.RibbonSurfaces` in scanner RAS --
    the data must be in register with the anatomy they were made from. ``mask`` (data
    grid) limits which voxels are ever read. Rows run ROI-major, depth 1 (deepest,
    by white matter) to ``n_depths`` (by pial).
    """
    from fastfuncstuff.surface.volume_depth import cortical_depth_volumes, crop_to_points, regrid

    files = [paths] if isinstance(paths, str) else list(paths)
    if opts.n_depths < 1:
        raise ValueError("-depths must be at least 1")
    if opts.method not in ("equidistant", "equivolume"):
        raise ValueError(f"depth method must be equidistant or equivolume, not {opts.method!r}")
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    mask3 = None if mask is None else np.asarray(mask).astype(bool).reshape(shape3)
    notes: list[str] = []
    rois = _read_rois(files, surfaces, notes)
    rois = [r for r in rois if any(m.any() for m in r.members.values())]
    if not rois:
        raise ValueError("no ROI covers any vertex of the surfaces given")

    used = sorted({i for r in rois for i, m in r.members.items() if m.any()})
    pts = np.concatenate(
        [np.r_[surfaces[i].white[m], surfaces[i].pial[m]] for r in rois for i, m in r.members.items() if m.any()]
    )  # fmt: skip
    dxyz = opts.dxyz or default_dxyz(aff)
    fine_aff, fine_shape = regrid(aff, shape3, (dxyz,) * 3)
    # The ribbon needs the whole of each ROI column, plus room for the equivolume
    # columns at the ROI's edge to fill.
    fine_aff, fine_shape, _ = crop_to_points(fine_aff, fine_shape, pts, pad_mm=3.0)
    depth = cortical_depth_volumes(
        [surfaces[i] for i in used], fine_aff, fine_shape, opts.n_depths,
        column_voxels=opts.column_voxels, owners=True, device=device,
    )  # fmt: skip
    if depth.n_gm == 0:
        raise ValueError("no ribbon point falls on the data grid: are the surfaces in register?")
    assert depth.owner_hemi is not None and depth.owner_vertex is not None
    owner_h = np.asarray(used)[depth.owner_hemi.astype(np.int64)]
    owner_v = depth.owner_vertex
    metric = depth.rho if opts.method == "equidistant" else depth.equivol
    k_bin = np.clip(np.ceil(metric * opts.n_depths), 1, opts.n_depths).astype(np.int64) - 1

    ijk = np.stack(np.unravel_index(depth.gm_idx, fine_shape), 1).astype(np.float64)
    centres = ijk @ fine_aff[:3, :3].T + fine_aff[:3, 3]
    del ijk
    corner, weight = _trilinear(centres, aff, shape3, mask3)
    _, unmasked = _trilinear(centres, aff, shape3, None)
    voxel_mm3 = float(abs(np.linalg.det(fine_aff[:3, :3])))
    read = weight.sum(axis=1) > 0

    n_depths = opts.n_depths
    rows_i, cols_i, vals = [], [], []
    n_cols, n_points, n_lost, hemi_of = [], [], [], []
    for r_idx, roi in enumerate(rois):
        inside = np.zeros(depth.n_gm, bool)
        for i, member in roi.members.items():
            here = owner_h == i
            inside[here] = member[owner_v[here]]
        n_points.append(int(inside.sum()))
        n_lost.append(int((inside & ~read & (unmasked.sum(axis=1) > 0)).sum()))
        sel = np.flatnonzero(inside & read)
        hemi_of.append("+".join(sorted({surfaces[i].name for i in np.unique(owner_h[sel])})) or "-")
        key = owner_h[sel] * (1 << 40) + owner_v[sel]
        n_cols.append([len(np.unique(key[k_bin[sel] == k])) for k in range(n_depths)])
        rows_i.append(np.repeat(r_idx * n_depths + k_bin[sel], 8))
        cols_i.append(corner[sel].reshape(-1))
        vals.append((weight[sel] * voxel_mm3).reshape(-1))
    n_vox = int(np.prod(shape3))
    w = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))),
        shape=(len(rois) * n_depths, n_vox),
    ).tocsr()
    w.eliminate_zeros()
    sizes = np.asarray(w.sum(axis=1)).ravel()
    alive = np.flatnonzero(sizes > 0)
    touched = np.flatnonzero(np.asarray((w != 0).sum(axis=0)).ravel() > 0)
    w = w[alive][:, touched].tocsr()
    rows = np.zeros(n_vox, bool)
    rows[touched] = True

    ids, names, index = [], [], {"hemi": [], "depth": [], "depth_frac": [], "n_columns": []}
    for row in alive:
        r_idx, k = divmod(int(row), n_depths)
        ids.append(rois[r_idx].rid)
        names.append(rois[r_idx].name)
        index["hemi"].append(hemi_of[r_idx])
        index["depth"].append(str(k + 1))
        index["depth_frac"].append(f"{(k + 0.5) / n_depths:.4g}")
        index["n_columns"].append(str(n_cols[r_idx][k]))
    empty = [f"{rois[int(r) // n_depths].name}@{int(r) % n_depths + 1}" for r in
             np.setdiff1d(np.arange(len(rois) * n_depths), alive)]  # fmt: skip

    in_fov = int((unmasked.sum(axis=1) > 0).sum())
    names_used = ", ".join(surfaces[i].name for i in used)
    notes += [
        f"depth sampling: {len(rois)} ROI(s) x {n_depths} {opts.method} depth(s) along the "
        f"{names_used} ribbon; depth 1 = deepest (white matter side), depth_frac = bin "
        f"centre as a fraction of {'thickness' if opts.method == 'equidistant' else 'column volume'}",
        f"ribbon grid {'x'.join(map(str, fine_shape))} at {dxyz:.3g} mm: {depth.n_gm:,} "
        f"ribbon points, {in_fov:,} inside the data's field of view",
        "the data must be in register with the anatomy the surfaces were made from",
    ]
    if in_fov < 0.5 * depth.n_gm:
        notes.append(
            f"WARNING: {100 * (1 - in_fov / depth.n_gm):.0f}% of the ribbon lies outside "
            "the data -- partial coverage, or the surfaces are not in register"
        )
    if mask3 is not None:
        notes.append(
            "ribbon reads limited to the analysis mask (-mask, or the missing-data guard's)"
            + (
                f": {sum(n_lost):,} ROI ribbon points had no voxel left, dropped"
                if any(n_lost)
                else ""
            )
        )
    if empty:
        notes.append(
            f"{len(empty)} ROI x depth row(s) hold no ribbon point, dropped: "
            + ", ".join(empty[:8])
            + (" ..." if len(empty) > 8 else "")
        )
    if not len(alive):
        raise ValueError("no ROI ribbon point reads any data voxel")
    return RoiSet(
        spec=w, ids=ids, names=names, sizes=sizes[alive], rows=rows.reshape(shape3),
        sources=[Path(f.split("[")[0]).name for f in files], notes=notes,
        index=index, size_name="volume_mm3", long_rows=True,
    )  # fmt: skip


def options_from_args(args) -> DepthOptions:
    return DepthOptions(
        n_depths=int(args.depths),
        method=str(args.depth_method),
        dxyz=args.depth_dxyz,
    )


def roi_set_from_args(
    args,
    volume_shape: tuple[int, ...],
    affine: np.ndarray | None,
    mask: np.ndarray | None = None,
    *,
    header=None,
    input_path: str | None = None,
    restrict_to_rois: bool = True,
    device=None,
) -> RoiSet:
    """The ``-atlas/-rois`` set a fit tool asked for: over ROI voxels, or -- when
    surfaces are given (``-fs_subj`` / ``-white_mesh``) -- per ROI per depth along the
    ribbon. Raises ``ValueError`` with a message for the user."""
    from fastfuncstuff.processing.roi_collapse import build_roi_set
    from fastfuncstuff.surface.ribbon_args import depth_requested

    on_surface = header is not None and _surface_data(header)
    if not depth_requested(args):
        if getattr(args, "depths", 1) != 1 or getattr(args, "depth_method", None) == "equivolume":
            raise ValueError(
                "-depths / -equivolume sample along surfaces: give -fs_subj or -white_mesh"
            )
        return build_roi_set(
            args.atlas, volume_shape, mask=mask, restrict_to_rois=restrict_to_rois,
            reference_header=header, reference_path=input_path,
        )  # fmt: skip
    if on_surface:
        raise ValueError(
            "depth sampling reads a volume along the ribbon; surface data already lie on one "
            "surface -- drop -fs_subj / -white_mesh"
        )
    if affine is None:
        raise ValueError("depth sampling needs the data's affine")
    from fastfuncstuff.surface.volume_depth import resolve_ribbon_surfaces

    log_lines: list[str] = []
    surfaces = resolve_ribbon_surfaces(
        args.fs_subj, args.white, args.pial, args.white_mesh, args.pial_mesh,
        args.cortex_label, log=log_lines.append,
    )  # fmt: skip
    roi_set = build_depth_roi_set(
        args.atlas, surfaces, affine, volume_shape, mask, options_from_args(args), device=device
    )
    roi_set.notes[:0] = [line.strip() for line in log_lines]
    return roi_set


def _surface_data(header) -> bool:
    from fastfuncstuff.io.gifti import surface_meta

    return surface_meta(header) is not None


__all__ = [
    "DepthOptions",
    "build_depth_roi_set",
    "default_dxyz",
    "options_from_args",
    "roi_set_from_args",
]
