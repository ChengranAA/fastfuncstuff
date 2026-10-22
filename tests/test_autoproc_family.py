"""Acquisition-family detection: which runs share a grid and a distortion."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from fastfuncstuff.autoproc.bids import BoldRun, Session, Subject
from fastfuncstuff.autoproc.emit import write_script
from fastfuncstuff.autoproc.family import detect_families
from fastfuncstuff.autoproc.plan import Options, build_plan


def _bold(tmp: Path, task, run, vox=2.0, shape=(8, 8, 6), pe="j-", readout=0.04, ses="01", acq=""):
    path = tmp / f"sub-X_ses-{ses}_task-{task}_acq-{acq}_run-{run}_bold.nii.gz"
    img = nib.Nifti1Image(np.zeros((*shape, 3), np.float32), np.diag([vox, vox, vox, 1]))
    img.header.set_zooms((vox, vox, vox, 2.0))
    nib.save(img, path)
    return BoldRun(
        subject="X",
        session=ses,
        task=task,
        run=run,
        mag_path=path,
        json={"RepetitionTime": 2.0, "PhaseEncodingDirection": pe, "TotalReadoutTime": readout},
        acq=acq,
    )


def test_two_resolutions_are_two_families_labelled_by_task(tmp_path):
    runs = [
        _bold(tmp_path, "floc", "1", vox=2.0),
        _bold(tmp_path, "prim", "1", vox=1.2, shape=(12, 12, 8), readout=0.05),
        _bold(tmp_path, "floc", "2", vox=2.0),
    ]
    fams = detect_families(runs)
    assert [f.label for f in fams] == ["floc", "prim"]
    assert [len(f.members) for f in fams] == [2, 1]
    assert fams[1].signature.voxel_mm == (1.2, 1.2, 1.2)


def test_identical_acquisition_across_tasks_is_one_family(tmp_path):
    # Separate fieldmaps per task do not make separate families; the acquisition does.
    runs = [_bold(tmp_path, "floc", "1"), _bold(tmp_path, "prim", "1")]
    fams = detect_families(runs)
    assert len(fams) == 1 and fams[0].label == "fam1"


def test_acq_names_a_family_its_tasks_cannot(tmp_path):
    # Two tasks share the high-res acquisition, so neither task names that family;
    # its acq does. The other family owns its task outright and takes that.
    runs = [
        _bold(tmp_path, "floc", "1", vox=1.2, acq="hires"),
        _bold(tmp_path, "prim", "1", vox=1.2, acq="hires"),
        _bold(tmp_path, "rest", "1", vox=2.0, acq="lores"),
    ]
    assert [f.label for f in detect_families(runs)] == ["hires", "rest"]


def test_shared_task_and_acq_fall_back_to_numbers(tmp_path):
    # One task, one acq label, two resolutions: nothing names either family.
    runs = [
        _bold(tmp_path, "a", "1", vox=2.0, acq="x"),
        _bold(tmp_path, "a", "2", vox=1.2, acq="x"),
    ]
    assert [f.label for f in detect_families(runs)] == ["fam1", "fam2"]


def test_polarity_alone_splits_a_family(tmp_path):
    # AP and PA residual distortions have opposite sign: one anat_nl cannot serve both.
    fams = detect_families([_bold(tmp_path, "a", "1", pe="j-"), _bold(tmp_path, "a", "2", pe="j")])
    assert len(fams) == 2
    # The task spans both, so neither family can take its name.
    assert [f.label for f in fams] == ["fam1", "fam2"]


def test_override_merges_and_flags_mixed_acquisitions(tmp_path):
    runs = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    fams = detect_families(runs, {"floc": "all", "prim": "all"})
    assert len(fams) == 1
    assert fams[0].label == "all" and fams[0].forced and fams[0].mixed


def test_auto_label_avoids_a_user_label(tmp_path):
    runs = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    fams = detect_families(runs, {"prim": "floc"})
    assert sorted(f.label for f in fams) == ["fam1", "floc"]
    assert next(f for f in fams if f.label == "floc").tasks == ["prim"]


def test_unreadable_header_does_not_split(tmp_path):
    # Plans are built from scans whose data may not be on disk; unknown grid is one key.
    runs = [_bold(tmp_path, "a", "1"), _bold(tmp_path, "a", "2")]
    for r in runs:
        r.mag_path = tmp_path / "missing" / r.mag_path.name
    assert len(detect_families(runs)) == 1


def test_plan_labels_runs_and_header_names_families_only_when_several(tmp_path):
    two = [_bold(tmp_path, "floc", "1", vox=2.0), _bold(tmp_path, "prim", "1", vox=1.2)]
    plan = build_plan(Subject("X", [Session("01", two)]), Options())
    assert [pr.family for pr in plan.runs] == ["floc", "prim"]
    assert "acq families      : 2" in write_script(plan, "wd", bids_root="/bids")

    one = [_bold(tmp_path, "floc", "1"), _bold(tmp_path, "floc", "2")]
    plan = build_plan(Subject("X", [Session("01", one)]), Options())
    assert "acq families" not in write_script(plan, "wd", bids_root="/bids")
