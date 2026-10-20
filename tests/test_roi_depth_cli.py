"""-atlas/-rois with surfaces in ffs_reml and ffs_deconvolve: one fit per ROI per depth.

A sphere "brain": white at RW, pial at RP, on a 2 mm grid. The response grows with
distance from white, so the depth bins must come out ordered deep < superficial,
and the files must be the tidy ROI x depth layout.
"""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
from scipy.stats import gamma

from fastfuncstuff.io.gifti import save_gifti_surface
from tests.test_surface_voxelize import icosphere

RW, RP, TR, N_TP = 15.0, 21.0, 2.0, 150
ONSETS = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280]


def _hrf(t):
    return np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)


@pytest.fixture(scope="module")
def brain(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("depth")
    v, f = icosphere(4)
    save_gifti_surface(tmp / "lh.white.surf.gii", v * RW, f)
    save_gifti_surface(tmp / "lh.pial.surf.gii", v * RP, f)
    cap = np.flatnonzero(v[:, 2] > 0.3)  # as the viewer saves a painted patch
    rows = "\n".join(f"{i}  0 0 0 1" for i in cap)
    (tmp / "patch.lh.label").write_text(f"#!ascii label\n{cap.size}\n{rows}\n")

    n = 26
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -(n - 1)
    ijk = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1)
    r = np.linalg.norm(ijk @ affine[:3, :3].T + affine[:3, 3], axis=-1)
    amp = np.clip(r - RW, 0, RP - RW)  # 0 at white, 6 at pial
    t = np.arange(N_TP) * TR
    resp = sum(_hrf(t - o) for o in ONSETS)
    rng = np.random.default_rng(3)
    data = 100 + amp[..., None] * resp + rng.normal(0, 0.5, (n, n, n, N_TP))
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    img.header.set_zooms((2, 2, 2, TR))
    nib.save(img, tmp / "bold.nii.gz")
    (tmp / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "".join(f"{o}\t1\tvis\n" for o in ONSETS)
    )
    (tmp / "stim.1D").write_text(" ".join(map(str, ONSETS)) + "\n")
    return tmp


def _surfs(tmp):
    return [
        "-white_mesh",
        str(tmp / "lh.white.surf.gii"),
        "-pial_mesh",
        str(tmp / "lh.pial.surf.gii"),
    ]


def _table(path):
    rows = [line.split("\t") for line in open(path).read().splitlines()]
    return rows[0], rows[1:]


def _check_depth_table(path, n):
    header, rows = _table(path)
    assert header[:7] == ["roi", "name", "hemi", "depth", "depth_frac", "n_columns", "volume_mm3"]
    assert [r[3] for r in rows] == [str(k + 1) for k in range(n)]
    assert {r[1] for r in rows} == {"patch.lh"} and {r[2] for r in rows} == {"lh"}
    return header, rows


def test_reml_fits_each_depth_and_orders_them(brain):
    tmp = brain
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(tmp / "bold.nii.gz"),
         "-events", str(tmp / "ev.tsv"), "-TR", str(TR), "-hrf_model", "TENT(0,14,8)",
         "-tout", "-fout", "-device", "cpu", "-Rbuck", str(tmp / "r"),
         "-atlas", str(tmp / "patch.lh.label"), *_surfs(tmp), "-depths", "3", "-equivolume",
         "-depth_dxyz", "0.6"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "depth 1 = deepest" in res.stdout
    header, rows = _check_depth_table(tmp / "r_roi.tsv", 3)
    fstat = [float(r[header.index("Full_Fstat")]) for r in rows]
    assert fstat[0] < fstat[1] < fstat[2]
    ch, curve = _table(tmp / "r_iresp_vis.tsv")
    assert ch == ["time", "roi", "name", "hemi", "depth", "depth_frac", "n_columns", "beta"]
    assert len(curve) == 8 * 3
    peak = {d: max(float(r[-1]) for r in curve if r[4] == d) for d in ("1", "2", "3")}
    # amp is ~1, 3, 5 in the three equal-volume shells (they sit a bit outward).
    assert peak["1"] < peak["2"] < peak["3"] and peak["3"] > 3 * peak["1"]
    assert not list(tmp.glob("r*.nii*"))
    # The depth PSF of this sampling, for the laminar model: K x K per ROI, diagonal-heavy.
    psf_hdr, psf = _table(tmp / "r_depth_psf.tsv")
    assert psf_hdr == ["roi", "name", "hemi", "depth", "true_depth", "weight"] and len(psf) == 9
    m = np.zeros((3, 3))
    for r in psf:
        m[int(r[3]) - 1, int(r[4]) - 1] = float(r[5])
    assert (np.argmax(m, axis=1) == np.arange(3)).all()


def test_deconvolve_one_depth_is_the_ribbon(brain, monkeypatch):
    from fastfuncstuff.cli import deconvolve

    tmp = brain
    monkeypatch.chdir(tmp)
    argv = ["ffs_deconvolve", "-input", str(tmp / "bold.nii.gz"), "-onsets", str(tmp / "stim.1D"),
            "-model", "TENT", "-window", "0", "14", "-prefix", str(tmp / "d"), "-device", "cpu",
            "-atlas", str(tmp / "patch.lh.label"),
            # The hemisphere-free stem form: finds lh.white.surf.gii beside it.
            "-white_mesh", str(tmp / "white.surf.gii"), "-pial_mesh", str(tmp / "pial.surf.gii")]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    _check_depth_table(tmp / "d_roi.tsv", 1)
    _, psf = _table(tmp / "d_depth_psf.tsv")
    assert len(psf) == 1 and 0.5 < float(psf[0][5]) <= 1.05  # one depth: the cortex share


def test_depth_flags_need_rois_and_volume_data(brain, monkeypatch):
    from fastfuncstuff.cli import deconvolve

    tmp = brain
    monkeypatch.chdir(tmp)
    base = ["ffs_deconvolve", "-input", str(tmp / "bold.nii.gz"), "-onsets", str(tmp / "stim.1D"),
            "-model", "TENT", "-window", "0", "14", "-prefix", str(tmp / "x"), "-device", "cpu"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", [*base, *_surfs(tmp)])
    assert deconvolve.main() == 1
    monkeypatch.setattr(sys, "argv", [*base, "-atlas", str(tmp / "patch.lh.label"), "-depths", "3"])
    assert deconvolve.main() == 1


def test_depth_flag_help_renders():
    """A bare % in a help string breaks -help and `make completions` (argparse formats it)."""
    from fastfuncstuff.cli_help import FfsArgumentParser
    from fastfuncstuff.surface.ribbon_args import add_depth_args

    parser = FfsArgumentParser(prog="x")
    add_depth_args(parser.add_argument_group("depth"))
    text = parser.format_help()
    assert "-depth_smooth" in text and "-depth_interp" in text


def test_several_depth_counts_are_one_fit_with_a_k_column(brain):
    """-depths 3 5: one load, one fit, every (K, depth) a row; K is a table column and
    each count's depth profile is ordered like its single-count run."""
    tmp = brain
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(tmp / "bold.nii.gz"),
         "-events", str(tmp / "ev.tsv"), "-TR", str(TR), "-hrf_model", "TENT(0,14,8)",
         "-tout", "-fout", "-device", "cpu", "-Rbuck", str(tmp / "m"),
         "-atlas", str(tmp / "patch.lh.label"), *_surfs(tmp), "-depths", "3", "5", "-equivolume",
         "-depth_dxyz", "0.6"],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=tmp,
    )  # fmt: skip
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    header, rows = _table(tmp / "m_roi.tsv")
    assert header[:8] == [
        "roi",
        "name",
        "K",
        "hemi",
        "depth",
        "depth_frac",
        "n_columns",
        "volume_mm3",
    ]
    assert [(r[2], r[4]) for r in rows] == [("3", str(d)) for d in (1, 2, 3)] + [
        ("5", str(d)) for d in (1, 2, 3, 4, 5)
    ]
    for k in ("3", "5"):  # the response grows outward at each count
        f = [float(r[header.index("Full_Fstat")]) for r in rows if r[2] == k]
        assert f == sorted(f)
    ch, curve = _table(tmp / "m_iresp_vis.tsv")
    assert ch[:4] == ["time", "roi", "name", "K"] and {r[3] for r in curve} == {"3", "5"}
    ph, psf = _table(tmp / "m_depth_psf.tsv")
    assert ph == ["roi", "name", "hemi", "K", "depth", "true_depth", "weight"]
    assert len(psf) == 3 * 3 + 5 * 5


def test_deconvolve_takes_several_depth_counts_too(brain, monkeypatch):
    from fastfuncstuff.cli import deconvolve

    tmp = brain
    monkeypatch.chdir(tmp)
    argv = ["ffs_deconvolve", "-input", str(tmp / "bold.nii.gz"), "-onsets", str(tmp / "stim.1D"),
            "-model", "TENT", "-window", "0", "14", "-prefix", str(tmp / "dm"), "-device", "cpu",
            "-atlas", str(tmp / "patch.lh.label"), *_surfs(tmp), "-depths", "3", "5",
            "-depth_dxyz", "0.6"]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    assert deconvolve.main() == 0
    header, rows = _table(tmp / "dm_roi.tsv")
    assert header[2] == "K" and len(rows) == 8
    _, psf = _table(tmp / "dm_depth_psf.tsv")
    assert len(psf) == 3 * 3 + 5 * 5
