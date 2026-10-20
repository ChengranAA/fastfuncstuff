"""ROI means along the cortical ribbon: one series per ROI per cortical depth.

The usual route to depth-resolved ROI signals is to upsample the data, build integer
layers on the fine grid (LayNii's LN2_LAYERS), intersect them with an ROI and average
the voxels. This does the same arithmetic straight from the white and pial meshes:

1. the ribbon under the ROIs is cut into small cells (:mod:`surface.slab`): about a
   voxel across along the surface (``-depth_dxyz``), much finer through the depth,
   each with its exact volume;
2. every cell gets its depth -- equidistant, or equivolume as a volume quantile
   within its cortical column, the definitions behind ``ffs_util_surf2layers`` --
   and its owning column (vertex). A cell across a bin boundary splits its volume;
3. a cell joins an ROI when its column is in the ROI. A surface ROI names the
   columns; a volume ROI claims the columns whose midthickness point lies in it, so
   every depth bin of an ROI averages the same patch of cortex;
4. each cell reads the data at its centre (cubic by default, ``-depth_interp``) and
   counts by its volume, so a bin's value is the volume-weighted mean over its slab
   of the ribbon -- what averaging an upsampled image within LayNii layers gives.

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
    #: Read spacing along the surface, mm (half that through the depth); None is the
    #: data's voxel size.
    dxyz: float | None = None
    column_voxels: float = 64.0
    #: How each read interpolates the data grid (:data:`READ_KERNELS`).
    interp: str = "cubic"


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


#: Interpolants a ribbon read can use; ``linear`` is trilinear.
READ_KERNELS = ("linear", "cubic", "quintic", "heptic", "wsinc5")


def _read_taps(points: np.ndarray, affine: np.ndarray, shape, kernel: str, mask: np.ndarray | None):
    """``(flat voxel indices, weights)``, each ``(N, taps)``, of reading a grid at
    ``points`` with a separable ``kernel``.

    Taps off the grid or outside ``mask`` are dropped and the rest renormalised, so a
    read never uses a voxel the analysis excludes; a read whose kept taps carry under
    half its weight is dropped (all-zero weights).

    Holds ``taps**3`` entries per read in several float64 temporaries (17 KB a read for
    wsinc5): call it in batches, as :func:`_compose` does, never on a whole ribbon.
    """
    import torch

    inv = np.linalg.inv(np.asarray(affine, np.float64))
    ijk = points @ inv[:3, :3].T + inv[:3, 3]
    base = np.floor(ijk).astype(np.int64)
    frac = ijk - base
    if kernel == "linear":
        offsets = np.array([0, 1])
        axis_w = [np.stack([1.0 - frac[:, a], frac[:, a]], 1) for a in range(3)]
    else:
        from fastfuncstuff.processing.interp import _KERNELS, _kernel_half_width

        half = _kernel_half_width(kernel)
        offsets = np.arange(-(half - 1), half + 1)
        fn = _KERNELS[kernel][0]
        axis_w = [fn(torch.as_tensor(frac[:, a])).numpy() for a in range(3)]
    n = offsets.size
    dims = np.asarray(shape[:3])
    ax = [base[:, a, None] + offsets[None] for a in range(3)]  # (N, n) per axis
    ok_ax = [(ax[a] >= 0) & (ax[a] < dims[a]) for a in range(3)]
    flat = (
        np.clip(ax[0], 0, dims[0] - 1)[:, :, None, None] * (dims[1] * dims[2])
        + np.clip(ax[1], 0, dims[1] - 1)[:, None, :, None] * dims[2]
        + np.clip(ax[2], 0, dims[2] - 1)[:, None, None, :]
    ).reshape(len(points), n**3)
    w = axis_w[0][:, :, None, None] * axis_w[1][:, None, :, None] * axis_w[2][:, None, None, :]
    ok = ok_ax[0][:, :, None, None] & ok_ax[1][:, None, :, None] & ok_ax[2][:, None, None, :]
    w = np.where(ok, w, 0.0).reshape(len(points), n**3)
    if mask is not None:
        w = np.where(mask.reshape(-1)[flat], w, 0.0)
    total = w.sum(axis=1, keepdims=True)
    return flat, np.divide(w, total, out=np.zeros_like(w), where=total >= 0.5)


def _compose(operator: sp.csr_matrix, cells_xyz, affine, shape, kernel, mask, n_vox):
    """``operator @ taps``: rows over grid voxels, built a batch of reads at a time.
    Also returns which reads kept no tap."""
    n_taps = (2 if kernel == "linear" else 2 * _half(kernel)) ** 3
    batch = max(1024, (1 << 22) // n_taps)
    acc, parts, held = None, [], 0
    dropped = np.zeros(cells_xyz.shape[0], bool)
    op_csc = operator.tocsc()
    for s in range(0, cells_xyz.shape[0], batch):
        e = min(cells_xyz.shape[0], s + batch)
        flat, w = _read_taps(cells_xyz[s:e], affine, shape, kernel, mask)
        dropped[s:e] = ~(w != 0).any(axis=1)
        taps = sp.csr_matrix(
            (w.reshape(-1), flat.reshape(-1), np.arange(0, (e - s) * n_taps + 1, n_taps)),
            shape=(e - s, n_vox),
        )
        parts.append((op_csc[:, s:e] @ taps).tocoo())
        held += parts[-1].nnz
        if held > (1 << 25):
            acc = _merge(acc, parts, operator.shape[0], n_vox)
            parts, held = [], 0
    return _merge(acc, parts, operator.shape[0], n_vox), dropped


def _merge(acc, parts, n_rows, n_vox):
    if not parts:
        return acc if acc is not None else sp.csr_matrix((n_rows, n_vox))
    new = sp.coo_matrix(
        (np.concatenate([p.data for p in parts]),
         (np.concatenate([p.row for p in parts]), np.concatenate([p.col for p in parts]))),
        shape=(n_rows, n_vox),
    ).tocsr()  # fmt: skip
    return new if acc is None else acc + new


def _half(kernel: str) -> int:
    from fastfuncstuff.processing.interp import _kernel_half_width

    return _kernel_half_width(kernel)


def default_step(data_affine: np.ndarray) -> float:
    """The data's voxel size (mm, geometric mean): reads one voxel apart along the
    surface and half a voxel through the depth."""
    return float(abs(np.linalg.det(np.asarray(data_affine, np.float64)[:3, :3])) ** (1 / 3))


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
    from fastfuncstuff.surface.slab import patch_faces, ribbon_cells

    files = [paths] if isinstance(paths, str) else list(paths)
    if opts.n_depths < 1:
        raise ValueError("-depths must be at least 1")
    if opts.method not in ("equidistant", "equivolume"):
        raise ValueError(f"depth method must be equidistant or equivolume, not {opts.method!r}")
    if opts.interp not in READ_KERNELS:
        raise ValueError(f"read interpolation must be one of {READ_KERNELS}, not {opts.interp!r}")
    shape3 = tuple(int(n) for n in data_shape[:3])
    aff = np.asarray(data_affine, np.float64)
    mask3 = None if mask is None else np.asarray(mask).astype(bool).reshape(shape3)
    notes: list[str] = []
    rois = _read_rois(files, surfaces, notes)
    rois = [r for r in rois if any(m.any() for m in r.members.values())]
    if not rois:
        raise ValueError("no ROI covers any vertex of the surfaces given")

    step = opts.dxyz or default_step(aff)
    n_depths = opts.n_depths
    members: dict[int, np.ndarray] = {}
    for r in rois:
        for i, m in r.members.items():
            members[i] = members.get(i, np.zeros(m.size, bool)) | m
    faces_of = {}
    for i, m in sorted(members.items()):
        if m.any():
            s = surfaces[i]
            column = np.linalg.norm(np.asarray(s.pial) - np.asarray(s.white), axis=1)[m].max()
            faces_of[i] = patch_faces(s, np.flatnonzero(m), float(column) + 3.0)
    log: list[str] = []
    cells = ribbon_cells(
        surfaces, faces_of, step, n_depths, opts.method,
        column_voxels=opts.column_voxels, device=device, log=log.append,
    )  # fmt: skip
    owned = np.zeros(len(cells), bool)
    for i, m in members.items():
        here = cells.surface == i
        owned[here] = m[cells.vertex[here]]
    cells = cells.subset(owned)
    if not len(cells):
        raise ValueError("no ribbon under the ROIs: are the surfaces' cortex labels right?")
    cell, k_bin, share = cells.bin_shares(n_depths)

    rows_i, cols_i, vals, hemi_of = [], [], [], []
    for r_idx, roi in enumerate(rois):
        inside = np.zeros(len(cells), bool)
        for i, member in roi.members.items():
            here = cells.surface == i
            inside[here] = member[cells.vertex[here]]
        pick = np.flatnonzero(inside[cell])
        hemi_of.append("+".join(sorted({surfaces[int(i)].name for i in np.unique(cells.surface[inside])})) or "-")  # fmt: skip
        rows_i.append(r_idx * n_depths + k_bin[pick])
        cols_i.append(cell[pick])
        vals.append(cells.volume[cell[pick]] * share[pick])
    n_rows = len(rois) * n_depths
    operator = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows_i), np.concatenate(cols_i))),
        shape=(n_rows, len(cells)),
    )
    n_vox = int(np.prod(shape3))
    w, dropped = _compose(operator, cells.centre, aff, shape3, opts.interp, mask3, n_vox)
    inv = np.linalg.inv(aff)
    near = np.rint(cells.centre @ inv[:3, :3].T + inv[:3, 3])
    off_grid = ~np.all((near >= 0) & (near < np.asarray(shape3)), axis=1)
    read_op = operator.multiply(~dropped[None]).tocsr()
    w.eliminate_zeros()
    sizes = np.asarray(read_op.sum(axis=1)).ravel()
    alive = np.flatnonzero((sizes > 0) & (np.diff(w.indptr) > 0))
    touched = np.flatnonzero(np.asarray((w[alive] != 0).sum(axis=0)).ravel() > 0)
    w = w[alive][:, touched].tocsr()
    voxel_rows = np.zeros(n_vox, bool)
    voxel_rows[touched] = True
    # Columns per row: owners of the cells the row reads.
    key = cells.surface.astype(np.int64) * (1 << 40) + cells.vertex
    coo = read_op.tocoo()
    pairs = np.unique(np.stack([coo.row[coo.data > 0], key[coo.col[coo.data > 0]]], 1), axis=0)
    per_row = np.bincount(pairs[:, 0], minlength=n_rows)

    ids, names, index = [], [], {"hemi": [], "depth": [], "depth_frac": [], "n_columns": []}
    for row in alive:
        r_idx, k = divmod(int(row), n_depths)
        ids.append(rois[r_idx].rid)
        names.append(rois[r_idx].name)
        index["hemi"].append(hemi_of[r_idx])
        index["depth"].append(str(k + 1))
        index["depth_frac"].append(f"{(k + 0.5) / n_depths:.4g}")
        index["n_columns"].append(str(int(per_row[row])))
    empty = [f"{rois[int(r) // n_depths].name}@{int(r) % n_depths + 1}" for r in
             np.setdiff1d(np.arange(n_rows), alive)]  # fmt: skip

    total_mm3 = float(cells.volume.sum())
    lost_fov = float(cells.volume[off_grid].sum())
    lost_mask = float(cells.volume[dropped & ~off_grid].sum())
    names_used = ", ".join(surfaces[i].name for i in faces_of)
    notes += [line.strip() for line in log]
    notes += [
        f"depth sampling: {len(rois)} ROI(s) x {n_depths} {opts.method} depth(s) along the "
        f"{names_used} ribbon; depth 1 = deepest (white matter side), depth_frac = bin "
        f"centre as a fraction of {'thickness' if opts.method == 'equidistant' else 'column volume'}",
        f"ROI ribbon {total_mm3:,.0f} mm3 in {len(cells):,} cells (reads {step:.3g} mm apart "
        f"along the surface, finer through the depth), {opts.interp} interpolation",
        "the data must be in register with the anatomy the surfaces were made from",
    ]
    if cells.inverted_mm3 > 0.001 * total_mm3:
        notes.append(
            f"{cells.inverted_mm3:,.1f} mm3 where white and pial cross (inside-out cells) left out"
        )
    if lost_fov > 0.5 * total_mm3:
        notes.append(
            f"WARNING: {100 * lost_fov / total_mm3:.0f}% of the ROI ribbon lies outside "
            "the data -- partial coverage, or the surfaces are not in register"
        )
    elif lost_fov > 0:
        notes.append(f"{100 * lost_fov / total_mm3:.1f}% of the ROI ribbon is outside the data")
    if mask3 is not None:
        notes.append(
            "ribbon reads limited to the analysis mask (-mask, or the missing-data guard's)"
            + (
                f": {100 * lost_mask / total_mm3:.1f}% of the ROI ribbon dropped"
                if lost_mask
                else ""
            )
        )
    if empty:
        notes.append(
            f"{len(empty)} ROI x depth row(s) hold no ribbon read, dropped: "
            + ", ".join(empty[:8])
            + (" ..." if len(empty) > 8 else "")
        )
    if not len(alive):
        raise ValueError("no ROI ribbon point reads any data voxel")
    return RoiSet(
        spec=w, ids=ids, names=names, sizes=sizes[alive], rows=voxel_rows.reshape(shape3),
        sources=[Path(f.split("[")[0]).name for f in files], notes=notes,
        index=index, size_name="volume_mm3", long_rows=True,
    )  # fmt: skip


def options_from_args(args) -> DepthOptions:
    return DepthOptions(
        n_depths=int(args.depths),
        method=str(args.depth_method),
        dxyz=args.depth_dxyz,
        interp=str(getattr(args, "depth_interp", "cubic")),
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
    "READ_KERNELS",
    "default_step",
    "options_from_args",
    "roi_set_from_args",
]
