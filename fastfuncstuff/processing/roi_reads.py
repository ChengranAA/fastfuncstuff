"""ROI means read through ``ffs_nwarp``'s transform chain, in the same single interpolation.

The fit tools average ROI voxels of data that were already resampled. Here the ROI
becomes a list of **reads** -- points in the anatomy the chain ends in -- and a sparse
averaging matrix; ``nwarpforge`` reads the native EPI at the points through motion,
distortion, affine and warp in one interpolation, and the matrix folds each frame's
reads into one value per ROI row. The same idea as ``-surf``, with slabs and ROIs where
that has thin surfaces and vertices.

Two kinds of ROI become reads:

* a volume ROI on the master grid: the centre of every ROI voxel, weighted by the ROI;
* ROIs on the cortical ribbon (:mod:`processing.roi_depth`): the cells of
  :mod:`surface.slab`, each weighted by its volume, one row per ROI x depth.

**Budget.** A read costs device memory in every frame and host memory while the plan is
built, so the reads per pass are capped (:func:`default_max_reads`, ``-roi_max_reads``).
Beyond the cap the ROIs are split into passes, each tiled, read and discarded in turn
over the one loaded source -- a whole-brain atlas at 0.5 mm with 7 depths is more passes,
never more memory. The only relaxation of accuracy is the sheet thickness through the
depth, and only for a single ROI that does not fit by itself; along the surface a read
never stands for more than one voxel face, or voxels go unread.

**Output.** ``PREFIX.roi.nii.gz`` is ``(n_rois, 1, n_depths, time)`` -- depth is z --
so any viewer or ``ffs_reml`` reads it as a tiny volume; ``PREFIX.roi.tsv`` says what
each ``(x, z)`` is, ``PREFIX.roi.json`` how it was made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

#: Device bytes one read costs per frame (coordinates, the composed coordinates, the warp
#: evaluated there, the operator's share): measured 133 through a fieldmap + motion chain
#: with wsinc5 at 32M reads (4.3 GB peak, 1.6 s for 6 frames) on an RTX 5070 Ti, 106 with
#: cubic; 180 leaves a third of headroom.
BYTES_PER_READ = 180
#: Host bytes per cell while a pass is tiled (nodes, centres, feet, owners, ...): measured
#: 399 on a whole left hemisphere, 34 ROIs x 7 equivolume depths, 4.7M reads (2.6 GB RSS).
HOST_BYTES_PER_CELL = 400


@dataclass
class ReadPass:
    points: np.ndarray  # (P, 3) scanner mm
    operator: sp.csr_matrix  # (R_pass, P); a row sums to 1 where it reads anything
    rows: np.ndarray  # (R_pass,) the global row each operator row is


@dataclass
class ReadPlan:
    """Every pass, and what each global row (ROI-major, depth-minor) is."""

    passes: list[ReadPass]
    ids: list[int]
    names: list[str]
    n_depths: int
    #: Per-row table columns after ``name`` (depth plans: hemi, depth, depth_frac, n_columns).
    index: dict[str, list]
    sizes: np.ndarray  # (n_rois * n_depths,) volume_mm3 (depth) or n_voxels
    size_name: str
    kind: str  # "volume" or "ribbon"
    notes: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    #: ``{roi id: {surface name: vertex ids}}`` for ribbon plans, to paint results back.
    vertices: dict = field(default_factory=dict)

    @property
    def n_rois(self) -> int:
        return len(self.ids)

    @property
    def n_rows(self) -> int:
        return self.n_rois * self.n_depths

    @property
    def n_reads(self) -> int:
        return int(sum(p.points.shape[0] for p in self.passes))


def default_max_reads(device=None, host: bool = True) -> int:
    """Reads one pass may hold: what the device and the host can take, safely.

    From :func:`memory.get_available_memory` (which applies the project's safety
    factors) over :data:`BYTES_PER_READ`; ``host`` also bounds it by the RAM a pass's
    tiling needs. Clamped to [2e5, 6e7].
    """
    import torch

    from fastfuncstuff.memory import get_available_memory

    dev = device or torch.device("cpu")
    n = get_available_memory(dev, empty_cache=False) // BYTES_PER_READ
    if host:
        try:
            import psutil

            n = min(n, int(0.5 * psutil.virtual_memory().available) // HOST_BYTES_PER_CELL)
        except ImportError:  # pragma: no cover - psutil is a dependency of the toolbox
            pass
    return int(np.clip(n, 200_000, 60_000_000))


def _normalised(op: sp.spmatrix) -> sp.csr_matrix:
    """Rows scaled to sum 1; rows with no weight stay empty."""
    op = sp.csr_matrix(op, dtype=np.float64)
    total = np.asarray(op.sum(axis=1)).ravel()
    scale = np.divide(1.0, total, out=np.zeros_like(total), where=total > 0)
    return (sp.diags(scale) @ op).tocsr().astype(np.float32)


def plan_volume_reads(roi_set, affine: np.ndarray) -> ReadPlan:
    """Reads at the centre of every ROI voxel of a :class:`RoiSet` on the master grid.

    ``affine`` is the master's real voxel -> scanner mm. A weighted or overlapping
    set works as it does everywhere else: the row is the weighted mean.
    """
    ijk = np.argwhere(roi_set.rows).astype(np.float64)  # boolean-index order = spec order
    aff = np.asarray(affine, np.float64)
    points = ijk @ aff[:3, :3].T + aff[:3, 3]
    spec = roi_set.spec
    if sp.issparse(spec):
        op = spec
    elif spec.ndim == 1:
        member = np.flatnonzero(spec >= 0)
        op = sp.csr_matrix(
            (np.ones(member.size), (spec[member], member)),
            shape=(roi_set.n_rois, spec.size),
        )
    else:
        op = sp.csr_matrix(spec)
    return ReadPlan(
        passes=[ReadPass(points, _normalised(op), np.arange(roi_set.n_rois))],
        ids=list(roi_set.ids),
        names=list(roi_set.names),
        n_depths=1,
        index={},
        sizes=np.asarray(roi_set.sizes, np.float64),
        size_name="n_voxels",
        kind="volume",
        notes=list(roi_set.notes),
    )


def plan_depth_reads(
    files: list[str],
    surfaces,
    voxel_size: float,
    opts,
    max_reads: int | None = None,
    *,
    device=None,
) -> ReadPlan:
    """ROI x depth reads along the ribbon, in passes of at most ``max_reads`` cells.

    ``voxel_size`` (mm) is the data's: reads sit one apart along the surface unless
    ``opts.dxyz`` says otherwise.
    """
    from fastfuncstuff.processing import roi_depth as rd

    notes: list[str] = []
    rois = rd.read_depth_rois(files, surfaces, notes)
    rd._check_options(opts)
    step = opts.dxyz or voxel_size
    k_n = opts.n_depths
    cap = int(max_reads) if max_reads else default_max_reads(device)

    through = 0.5
    costs = rd.tile_costs(rois, surfaces, step, k_n, through)
    if costs.max() > cap:
        # One ROI alone is over the cap: thicken the sheets (never the spacing along
        # the surface, which would leave voxels unread), keeping 2 per depth bin.
        through = 1.0
        relaxed = rd.tile_costs(rois, surfaces, step, k_n, through)
        if relaxed.max() > cap:
            big = rois[int(np.argmax(relaxed))].name
            raise ValueError(
                f"ROI {big!r} alone tiles to ~{int(relaxed.max()):,} reads (cap {cap:,}) at "
                f"{step:.3g} mm spacing and {k_n} depth(s): raise -roi_max_reads, or lower "
                "-depths, or raise -depth_dxyz"
            )
        costs = relaxed
        notes.append(
            f"sheets through the depth thickened to {through:g} of the read spacing so the largest "
            f"ROI fits the {cap:,}-read cap (2 per depth bin kept)"
        )
    groups = rd.pass_groups(costs, cap)
    n_rows = len(rois) * k_n

    sizes = np.zeros(n_rows)
    columns = np.zeros(n_rows, np.int64)
    hemi = ["-"] * len(rois)
    passes: list[ReadPass] = []
    tile_notes: list[str] = []
    n_cells = 0
    for gi, group in enumerate(groups):
        sub = [rois[i] for i in group]
        plan = rd.plan_depth_rois(sub, surfaces, step, opts, through=through, device=device)
        rows = np.concatenate([np.arange(g * k_n, (g + 1) * k_n) for g in group])
        op = plan.operator
        row_sizes = np.asarray(op.sum(axis=1)).ravel()
        sizes[rows] = row_sizes
        columns[rows] = plan.columns_per_row()
        for j, g in enumerate(group):
            hemi[g] = plan.hemi_of[j]
        n_cells += len(plan.cells)
        passes.append(ReadPass(plan.cells.centre.copy(), _normalised(op), rows))
        if gi == 0:
            tile_notes = plan.notes
        del plan

    index = {"hemi": [], "depth": [], "depth_frac": [], "n_columns": []}
    for r in range(n_rows):
        g, k = divmod(r, k_n)
        index["hemi"].append(hemi[g])
        index["depth"].append(str(k + 1))
        index["depth_frac"].append(f"{(k + 0.5) / k_n:.4g}")
        index["n_columns"].append(str(int(columns[r])))
    names_used = ", ".join(sorted({s.name for r in rois for i, m in r.members.items() if m.any() for s in [surfaces[i]]}))  # fmt: skip
    notes += tile_notes
    notes += [
        f"{len(rois)} ROI(s) x {k_n} {opts.method} depth(s) along the {names_used} ribbon; "
        "depth 1 = deepest (white matter side)",
        f"{n_cells:,} reads ({step:.3g} mm apart along the surface, finer through the depth) "
        f"in {len(passes)} pass(es) of at most {cap:,}",
    ]
    vertices = {
        r.rid: {surfaces[i].name: np.flatnonzero(m).tolist() for i, m in r.members.items() if m.any()}
        for r in rois
    }  # fmt: skip
    return ReadPlan(
        passes=passes,
        ids=[r.rid for r in rois],
        names=[r.name for r in rois],
        n_depths=k_n,
        index=index,
        sizes=sizes,
        size_name="volume_mm3",
        kind="ribbon",
        notes=notes,
        meta={
            "method": opts.method,
            "step_mm": step,
            "smooth_rounds": opts.smooth,
            "surfaces": [s.name for s in surfaces],
            "through": through,
            "max_reads": cap,
        },
        vertices=vertices,
    )


class RoiFold:
    """Fold each frame's reads into one value per row, as ``nwarpforge`` emits it.

    The mean of the reads that landed inside the EPI (nwarp reads exactly 0 outside),
    so a slab whose edge pokes out of the field of view stays at its true level
    instead of averaging in zeros. :attr:`cover` keeps, per row, the smallest share of
    the row's weight read inside the EPI in any frame -- a read that is 0 in any frame
    counts as outside, as in the surface fold.
    """

    def __init__(self, operator: sp.csr_matrix):
        self._cpu = operator
        self._op = None
        self.cover = None

    def _setup(self, device):
        from fastfuncstuff.surface.smooth import torch_csr
        from fastfuncstuff.utils import cpu_if_mps

        dev = cpu_if_mps(device, "sparse_csr_mm")
        if self._op is None or self._op.device != dev:
            self._op = torch_csr(self._cpu, dev)
        return dev

    def __call__(self, reads):
        import torch

        dev = self._setup(reads.device)
        r = reads.detach().to(dev, torch.float32).reshape(-1, 1)
        num = (self._op @ r).reshape(-1)
        cov = (self._op @ (r != 0).to(torch.float32)).reshape(-1)
        self.cover = cov if self.cover is None else torch.minimum(self.cover, cov)
        return torch.where(cov > 0, num / cov.clamp_min(1e-12), torch.zeros_like(num))

    def reset(self) -> None:
        self.cover = None


@dataclass
class ReadResult:
    values: np.ndarray  # (T, n_rows) float32
    cover: np.ndarray  # (n_rows,) smallest share read inside the EPI


def read_through_chain(
    plan: ReadPlan, source_path: str, nwarp_specs, master_path: str, **nwarp_kwargs
) -> ReadResult:
    """Run every pass of ``plan`` through ``nwarpforge`` over one read of the source."""
    from fastfuncstuff.processing.io import load_image
    from fastfuncstuff.processing.nwarpforge import nwarpforge

    source = nwarp_kwargs.pop("source_image", None) or load_image(source_path, device=None)
    out = np.zeros((0, plan.n_rows), np.float32)
    cover = np.zeros(plan.n_rows, np.float32)
    for p in plan.passes:
        fold = RoiFold(p.operator)
        res = nwarpforge(
            source_path=source_path,
            nwarp_specs=nwarp_specs,
            prefix="",
            master_path=master_path,
            points=p.points,
            point_reducer=fold,
            source_image=source,
            **nwarp_kwargs,
        )
        assert res is not None
        vals = res.cpu().numpy().astype(np.float32)
        vals = vals[None] if vals.ndim == 1 else vals
        if out.shape[0] == 0:
            out = np.zeros((vals.shape[0], plan.n_rows), np.float32)
        out[:, p.rows] = vals
        if fold.cover is not None:
            cover[p.rows] = fold.cover.cpu().numpy()
    return ReadResult(out, np.clip(cover, 0.0, 1.0))


def output_paths(prefix: str, ribbon: bool) -> list[str]:
    stem = f"{prefix}.roi"
    return [f"{stem}.nii.gz", f"{stem}.tsv", f"{stem}.json"] + (
        [f"{stem}.vertices.npz"] if ribbon else []
    )


def write_roi_outputs(
    prefix: str,
    plan: ReadPlan,
    result: ReadResult,
    tr: float | None = None,
    extra: dict | None = None,
) -> list[str]:
    """``PREFIX.roi.nii.gz`` (n_rois, 1, n_depths, T), ``.tsv`` (what each (x, z) is) and
    ``.json`` (how it was made); ribbon plans add ``.vertices.npz`` (each ROI's columns)."""
    import nibabel as nib

    from fastfuncstuff.io.afni import save_nifti

    n_t = result.values.shape[0]
    cube = result.values.reshape(n_t, plan.n_rois, plan.n_depths).transpose(1, 2, 0)
    arr = np.ascontiguousarray(cube[:, None, :, :], dtype=np.float32)
    hdr = nib.Nifti1Header()
    hdr.set_intent("none", name="ffs_roi")
    hdr["descrip"] = b"ffs_nwarp ROI x 1 x depth x time"
    paths = output_paths(prefix, plan.kind == "ribbon")
    Path(paths[0]).parent.mkdir(parents=True, exist_ok=True)
    save_nifti(arr, paths[0], affine=np.eye(4), tr=tr, header=hdr)

    cols = ["x", "z", "roi", "name", *plan.index, plan.size_name, "coverage"]
    with open(paths[1], "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in range(plan.n_rows):
            g, k = divmod(r, plan.n_depths)
            row = [str(g), str(k), str(plan.ids[g]), plan.names[g]]
            row += [str(plan.index[c][r]) for c in plan.index]
            row += [f"{plan.sizes[r]:.6g}", f"{result.cover[r]:.4f}"]
            fh.write("\t".join(row) + "\n")

    doc = {
        "tool": "ffs_nwarp",
        "kind": f"roi_timeseries_{plan.kind}",
        "axes": ["roi", "1", "depth", "time"],
        "n_rois": plan.n_rois,
        "n_depths": plan.n_depths,
        "depth_order": "depth 1 = deepest (white matter side)" if plan.kind == "ribbon" else None,
        "TR": tr,
        "size_column": plan.size_name,
        "n_reads": plan.n_reads,
        "n_passes": len(plan.passes),
        "rois": [{"x": i, "id": plan.ids[i], "name": plan.names[i]} for i in range(plan.n_rois)],
        "notes": plan.notes,
        **plan.meta,
        **(extra or {}),
    }
    with open(paths[2], "w") as fh:
        json.dump(doc, fh, indent=2, default=str)
    if plan.kind == "ribbon":
        np.savez_compressed(
            paths[3],
            **{
                f"roi{rid}_{surf}": np.asarray(v, np.int32)
                for rid, per in plan.vertices.items()
                for surf, v in per.items()
            },
        )
    return paths


# -- reading the files back --------------------------------------------------------------

#: What ``args.atlas`` holds once a fit tool has adopted an ROI-series input.
SERIES = "@roi_series"
_SERIES_SUFFIXES = (".nii.gz", ".nii.zst", ".nii")


def series_table_path(path: str) -> Path:
    """``PREFIX.roi.tsv`` for ``PREFIX.roi.nii.gz``."""
    name = str(path)
    for suf in _SERIES_SUFFIXES:
        if name.endswith(suf):
            return Path(name[: -len(suf)] + ".tsv")
    return Path(name + ".tsv")


def read_series_table(path: str) -> dict | None:
    """The table beside an ``ffs_nwarp`` ROI series, or None when ``path`` is not one.

    A series is recognised by its NIfTI intent name (``ffs_roi``) and the sidecar
    table; a file with the intent but no table is an error worth naming, not a plain
    volume.
    """
    plain = str(path).split("[")[0]
    if not plain.endswith(_SERIES_SUFFIXES):
        return None
    try:
        from fastfuncstuff.io.headers import read_nifti_header

        intent = read_nifti_header(plain)["intent_name"]
    except Exception:  # not a readable NIfTI: some other input
        return None
    if bytes(intent).rstrip(b"\0") != b"ffs_roi":
        return None
    table = series_table_path(plain)
    if not table.exists():
        raise ValueError(
            f"{plain} is an ffs_nwarp ROI series but its table {table.name} is missing"
        )
    lines = [ln.split("\t") for ln in table.read_text().splitlines() if ln]
    head, rows = lines[0], lines[1:]
    col = {k: [r[i] for r in rows] for i, k in enumerate(head)}
    size = next(k for k in ("volume_mm3", "n_voxels") if k in col)
    first = head.index("name") + 1
    index = {k: col[k] for k in head[first : head.index(size)]}
    return {
        "x": np.asarray(col["x"], int),
        "z": np.asarray(col["z"], int),
        "ids": [int(v) for v in col["roi"]],
        "names": col["name"],
        "index": {**index, "coverage": col["coverage"]},
        "sizes": np.asarray(col[size], np.float64),
        "size_name": size,
    }


def roi_set_from_series(table: dict, volume_shape: tuple[int, ...], source: str):
    """Every ``(roi, depth)`` element of an ROI series as one row of an identity :class:`RoiSet`.

    Rows are in the file's boolean-index order (roi-major, depth-minor), so the fit
    tools' machinery -- tables, curves, single trials, ``-roi_maps`` painting back
    into the same ``(n_rois, 1, n_depths)`` layout -- works unchanged.
    """
    from fastfuncstuff.processing.roi_collapse import RoiSet

    shape = tuple(int(n) for n in volume_shape[:3])
    n = int(np.prod(shape))
    if len(table["ids"]) != n:
        raise ValueError(
            f"the table has {len(table['ids'])} rows but the series holds {n} elements"
        )
    depth = "depth" in table["index"]
    return RoiSet(
        spec=np.arange(n),
        ids=list(table["ids"]),
        names=list(table["names"]),
        sizes=table["sizes"],
        rows=np.ones(shape, bool),
        sources=[Path(source).name],
        notes=[
            "input is an ffs_nwarp ROI series: each element is already an ROI (or ROI x depth) mean"
        ],
        index=table["index"],
        size_name=table["size_name"],
        long_rows=depth,
        divisor=np.ones(n),
    )


def adopt_roi_series_input(args, input_files) -> str | None:
    """Make a fit tool treat an ``ffs_nwarp`` ROI series input as ROI means.

    Sets ``args.atlas`` to :data:`SERIES` and returns None; returns an error message
    when the input is a series but the options contradict it (spatial blur, a mask, a
    second ROI set), or when reading its table fails.
    """
    try:
        table = read_series_table(str(input_files[0]))
    except ValueError as exc:
        return str(exc)
    if table is None:
        return None
    if getattr(args, "atlas", None):
        return "the input is already an ROI series (ffs_nwarp -atlas): drop -atlas/-rois"
    bad = [
        f"-{a}"
        for a in ("do_blur", "blur_to_fwhm", "blur_to_tsnr", "mask")
        if getattr(args, a, None) not in (None, False)
    ]
    if bad:
        return f"{', '.join(bad)} are spatial and mean nothing on an ROI series (one voxel per ROI)"
    args.atlas = [SERIES]
    return None


__all__ = [
    "BYTES_PER_READ",
    "SERIES",
    "ReadPass",
    "ReadPlan",
    "ReadResult",
    "RoiFold",
    "adopt_roi_series_input",
    "default_max_reads",
    "output_paths",
    "plan_depth_reads",
    "plan_volume_reads",
    "read_series_table",
    "read_through_chain",
    "roi_set_from_series",
    "series_table_path",
    "write_roi_outputs",
]
