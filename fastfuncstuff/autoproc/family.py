"""Acquisition families: the runs that share one distortion and one grid.

A family is keyed on the acquisition, not on the fieldmap — voxel size, matrix,
phase-encode axis + polarity, readout time and echo count. Those are what decide
the residual distortion a nonlinear anat correction has to undo and the voxel
size an output should keep, so runs that differ in any of them cannot share an
``anat_nl`` or a final grid. Two fieldmaps over identical runs (one repeated
mid-session) are still one family: cross-fmap alignment already reconciles them.

Grouping is per run, so one oddly-acquired run of a task shows up as its own
family instead of being folded in silently. ``-family TASK LABEL`` overrides the
grouping for whole tasks; runs it names are grouped by label alone, which can
merge acquisitions that differ — the report says so when it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import nibabel as nib
    import numpy as np

from fastfuncstuff.autoproc.bids import BoldRun


@dataclass(frozen=True)
class AcqSignature:
    """What has to match for two runs to be one family. ``None`` = unknown (no
    sidecar key, or a header that could not be read); it only matches ``None``."""

    voxel_mm: tuple[float, float, float] | None
    matrix: tuple[int, int, int] | None
    pe_dir: str | None
    readout_s: float | None
    n_echoes: int

    def describe(self) -> str:
        vox = "x".join(f"{v:g}" for v in self.voxel_mm) + " mm" if self.voxel_mm else "? mm"
        mat = "x".join(str(n) for n in self.matrix) if self.matrix else "?"
        ro = f"readout {self.readout_s:g} s" if self.readout_s is not None else "readout ?"
        echoes = f"{self.n_echoes} echo" + ("es" if self.n_echoes > 1 else "")
        return f"{vox}, {mat}, PE {self.pe_dir or '?'}, {ro}, {echoes}"


@dataclass
class Family:
    label: str
    signature: AcqSignature
    # (session, task, run) of every member, in scan order.
    members: list[tuple[str | None, str, str]] = field(default_factory=list)
    # Set by -family: the label came from the user, not from the signature.
    forced: bool = False
    # A forced family whose members do not all share ``signature``.
    mixed: bool = False
    # How this family reaches the anat (set by link_families): "root" for the
    # reference family, "xfam" for one aligned to the root's corrected anchor, "anat"
    # for one aligned to the anat on its own.
    link: str | None = None
    link_reason: str = ""
    # (fraction of this family's FoV the root covers, overlap extent in mm along
    # this family's slab normal), from the headers; None when unknown or the root.
    overlap: tuple[float, float] | None = None

    @property
    def tasks(self) -> list[str]:
        return list(dict.fromkeys(t for _s, t, _r in self.members))


def family_dir(fam: Family) -> str:
    """The folder a family's pipeline runs in (multi-family scripts only)."""
    return f"fam-{fam.label}"


def _grid(path: Path) -> tuple[tuple[float, float, float], tuple[int, int, int]] | None:
    """(voxel size, matrix) from the header alone. nibabel reads .nii/.nii.gz
    lazily; anything it cannot open (missing file, .nii.zst) is unknown rather
    than an error — the plan must still build without the data on disk."""
    try:
        import nibabel as nib

        hdr = cast("nib.Nifti1Header", nib.load(str(path)).header)
        zooms = hdr.get_zooms()[:3]
        shape = hdr.get_data_shape()[:3]
    except Exception:
        return None
    if len(zooms) < 3 or len(shape) < 3:
        return None
    return (
        (round(float(zooms[0]), 3), round(float(zooms[1]), 3), round(float(zooms[2]), 3)),
        (int(shape[0]), int(shape[1]), int(shape[2])),
    )


def signature(bold: BoldRun) -> AcqSignature:
    grid = _grid(bold.mag_path)
    ro = bold.json.get("TotalReadoutTime")
    return AcqSignature(
        voxel_mm=grid[0] if grid else None,
        matrix=grid[1] if grid else None,
        pe_dir=bold.pe_dir,
        # Sidecars print it to ~6 significant figures; one protocol gives one value.
        readout_s=round(float(ro), 6) if ro is not None else None,
        n_echoes=bold.n_echoes,
    )


def _key(bold: BoldRun) -> tuple[str | None, str, str]:
    return (bold.session, bold.task, bold.run)


def detect_families(runs: list[BoldRun], overrides: dict[str, str] | None = None) -> list[Family]:
    """Group ``runs`` into families, in order of first appearance.

    An automatic family is labelled, in order of preference, with its task, its
    ``acq-`` value, or ``famN`` — a task or acq only when the family owns it
    outright (see :func:`_owned`). The label names the family's output folder, so
    it must be unique; it never collides with a ``-family`` label.
    """
    overrides = overrides or {}
    sigs = {_key(b): signature(b) for b in runs}
    by_label: dict[str, Family] = {}
    by_sig: dict[AcqSignature, Family] = {}
    order: list[Family] = []
    for b in runs:
        sig = sigs[_key(b)]
        label = overrides.get(b.task)
        if label is not None:
            fam = by_label.get(label)
            if fam is None:
                fam = by_label[label] = Family(label, sig, forced=True)
                order.append(fam)
            elif sig != fam.signature:
                fam.mixed = True
        else:
            fam = by_sig.get(sig)
            if fam is None:
                fam = by_sig[sig] = Family("", sig)
                order.append(fam)
        fam.members.append(_key(b))

    acq_of = {_key(b): b.acq for b in runs}
    taken = set(by_label)
    n = 0
    for fam in order:
        if fam.forced:
            continue
        for name in (_owned(fam, order, lambda m: m[1]), _owned(fam, order, acq_of.get)):
            if name and name not in taken:
                fam.label = name
                break
        else:
            n += 1
            while f"fam{n}" in taken:
                n += 1
            fam.label = f"fam{n}"
        taken.add(fam.label)
    return order


def _owned(fam: Family, order: list[Family], value) -> str | None:
    """The one ``value`` every member of ``fam`` shares and no other family's
    member has — a name that identifies this family and nothing else — or None.
    Tasks and acquisitions need not line up with families (one task acquired two
    ways, two tasks acquired one way), so a name is only taken when it is exact."""
    mine = {value(m) for m in fam.members}
    if len(mine) != 1:
        return None
    (name,) = mine
    if not name:
        return None
    others = {value(m) for f in order if f is not fam for m in f.members}
    return None if name in others else name


# -family_ref modes; anything else names a family by label.
ROOT_MODES = ("highest_res", "lowest_res", "coverage", "least_distorted", "anat")
# A thinner overlap than this along the child's slab normal cannot pin rotation: a
# sliver pivots about itself, and the error grows across the rest of the slab.
MIN_OVERLAP_EXTENT_MM = 20.0


def _geometry(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, ...]] | None:
    """(sform/qform affine, matrix, voxel size) from the header alone, or None."""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(path))
        hdr = cast("nib.Nifti1Header", img.header)
        aff = np.asarray(cast("nib.Nifti1Image", img).affine, dtype=float)
        shape = tuple(int(n) for n in hdr.get_data_shape()[:3])
        zooms = tuple(float(z) for z in hdr.get_zooms()[:3])
    except Exception:
        return None
    if aff.shape != (4, 4) or len(shape) != 3 or len(zooms) != 3:
        return None
    return aff, (shape[0], shape[1], shape[2]), zooms


def overlap(child: Path, root: Path, samples: int = 24) -> tuple[float, float] | None:
    """How much of ``child``'s field of view ``root`` also covers, from headers.

    Returns (fraction of the child's FoV inside the root's, extent in mm of that
    overlap along the child's thinnest FoV axis — its slab normal). Sampled on a
    lattice of ~``samples`` points per axis through the child's voxel centres, mapped
    through both affines, so oblique and differently-tilted slabs are handled. Both
    headers are in scanner space; across sessions that ignores head movement between
    them, which is fine for a link/no-link decision.
    """
    import numpy as np

    gc, gr = _geometry(child), _geometry(root)
    if gc is None or gr is None:
        return None
    (ac, sc, zc), (ar, sr, _) = gc, gr
    axes = [np.linspace(0, n - 1, min(n, samples)) for n in sc]
    ijk = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    world = ijk @ ac[:3, :3].T + ac[:3, 3]
    rijk = (world - ar[:3, 3]) @ np.linalg.inv(ar[:3, :3]).T
    inside = np.all((rijk >= -0.5) & (rijk <= np.asarray(sr) - 0.5), axis=1)
    if not inside.any():
        return 0.0, 0.0
    normal = int(np.argmin([n * z for n, z in zip(sc, zc, strict=True)]))
    k = ijk[inside, normal]
    step = (sc[normal] - 1) / max(len(axes[normal]) - 1, 1)
    extent = (k.max() - k.min() + step) * zc[normal]
    return float(inside.mean()), float(extent)


def _rank(fam: Family, mode: str, geom) -> float:
    """Smaller is better for the root under ``mode``; inf when unknowable."""
    if geom is None:
        return math.inf
    _aff, shape, zooms = geom
    if mode == "highest_res":
        return math.prod(zooms)
    if mode == "lowest_res":
        return -math.prod(zooms)
    if mode == "coverage":
        return -math.prod(n * z for n, z in zip(shape, zooms, strict=True))
    # least_distorted: the residual shift in mm scales with readout time times the
    # voxel size along PE (displacement in voxels is field x readout).
    sig = fam.signature
    axis = {"i": 0, "j": 1, "k": 2}.get((sig.pe_dir or "")[:1])
    if sig.readout_s is None or axis is None:
        return math.inf
    return sig.readout_s * zooms[axis]


def link_families(
    families: list[Family],
    rep: dict[str, Path],
    mode: str = "highest_res",
    forced: dict[str, str] | None = None,
    min_overlap: float = 0.5,
) -> Family | None:
    """Pick the root family and decide, per child, how it reaches the anat.

    ``rep`` maps each label to one run's image (headers only). A child links to
    the root ("xfam") when the root covers at least ``min_overlap`` of its FoV over
    at least MIN_OVERLAP_EXTENT_MM along its slab normal; otherwise, or when the
    overlap cannot be read, it is aligned to the anat on its own ("anat"), which
    is the best available when the two slabs share too little brain. ``forced``
    ({label: "root"|"anat"}) overrides the decision for named children. Returns
    the root, or None under ``mode == "anat"`` (every family on its own).
    """
    forced = forced or {}
    if len(families) < 2:
        return None
    if mode == "anat":
        for fam in families:
            fam.link, fam.link_reason = "anat", "-family_ref anat"
        return None
    by_label = {f.label: f for f in families}
    if mode not in by_label and mode not in ROOT_MODES:
        raise ValueError(
            f"-family_ref {mode!r}: not a mode ({', '.join(ROOT_MODES)}) or a family "
            f"label ({', '.join(by_label)})."
        )
    for label in forced:
        if label not in by_label:
            raise ValueError(
                f"-family_link {label!r}: no such family (have {', '.join(by_label)})."
            )
    if mode in by_label:
        root = by_label[mode]
    else:
        geoms = {f.label: _geometry(rep[f.label]) for f in families}
        root = min(families, key=lambda f: _rank(f, mode, geoms[f.label]))
    root.link, root.link_reason = "root", f"-family_ref {mode}"
    for fam in families:
        if fam is root:
            continue
        fam.overlap = overlap(rep[fam.label], rep[root.label])
        want = forced.get(fam.label)
        if want is not None:
            fam.link = "xfam" if want == "root" else "anat"
            fam.link_reason = f"-family_link {fam.label} {want}"
        elif fam.overlap is None:
            fam.link, fam.link_reason = "anat", "overlap with the root unknown (header unread)"
        elif fam.overlap[0] < min_overlap or fam.overlap[1] < MIN_OVERLAP_EXTENT_MM:
            fam.link = "anat"
            fam.link_reason = (
                f"root covers {fam.overlap[0]:.0%} of its FoV over {fam.overlap[1]:.0f} mm "
                f"(needs {min_overlap:.0%} and {MIN_OVERLAP_EXTENT_MM:.0f} mm)"
            )
        else:
            fam.link = "xfam"
            fam.link_reason = (
                f"root covers {fam.overlap[0]:.0%} of its FoV over {fam.overlap[1]:.0f} mm"
            )
    return root
