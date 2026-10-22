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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import nibabel as nib

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

    @property
    def tasks(self) -> list[str]:
        return list(dict.fromkeys(t for _s, t, _r in self.members))


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
