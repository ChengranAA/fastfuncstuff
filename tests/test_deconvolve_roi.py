"""ffs_deconvolve -atlas / -rois: average the ROI first, then fit that one series."""

import sys

import nibabel as nib
import numpy as np
from scipy.stats import gamma

TR, N_TP, SHAPE = 2.0, 120, (4, 4, 2)


def _bold(rng, onsets, t):
    return sum(
        np.where(t - o > 0, gamma.pdf(t - o, 6) - gamma.pdf(t - o, 16) / 6, 0) for o in onsets
    )


def _dataset(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(N_TP) * TR
    runs, rows, arrays = [], [], []
    for r in range(3):
        onsets = np.sort(rng.choice(np.arange(3, 105), 14, replace=False)) * TR
        arr = 100 + rng.normal(0, 1, (*SHAPE, N_TP))
        arr[:2] += 10 * _bold(rng, onsets, t)  # x < 2 responds; x >= 2 is noise
        arr = arr.astype(np.float32)
        img = nib.Nifti1Image(arr, np.eye(4))
        img.header.set_zooms((2, 2, 2, TR))
        img.header.set_xyzt_units("mm", "sec")
        path = tmp_path / f"run{r}.nii.gz"
        nib.save(img, path)
        runs.append(str(path))
        arrays.append(arr)
        rows.append(" ".join(f"{o:.2f}" for o in onsets))
    timing = tmp_path / "stim.1D"
    timing.write_text("\n".join(rows) + "\n")
    return runs, str(timing), arrays


def _atlas(tmp_path, name="atlas.nii.gz"):
    lab = np.zeros(SHAPE, dtype=np.int16)
    lab[:2] = 1
    lab[2:] = 2
    path = str(tmp_path / name)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), path)
    return path


def _run(monkeypatch, runs, timing, prefix, *extra):
    from fastfuncstuff.cli import deconvolve

    argv = ["ffs_deconvolve", "-input", *runs, "-onsets", timing, "-model", "TENT"]
    argv += ["-window", "0", "16", "-prefix", prefix, "-device", "cpu", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return deconvolve.main()


def _table(path):
    rows = [line.split("\t") for line in open(path).read().splitlines()]
    return rows[0], rows[1:]


def test_roi_mode_writes_table_and_text_curves_and_no_volumes(monkeypatch, tmp_path):
    runs, timing, _ = _dataset(tmp_path)
    prefix = str(tmp_path / "out")
    assert _run(monkeypatch, runs, timing, prefix, "-atlas", _atlas(tmp_path)) == 0
    header, rows = _table(f"{prefix}_roi.tsv")
    assert header[:3] == ["roi", "name", "n_voxels"] and "r2" in header
    r2 = {r[0]: float(r[header.index("r2")]) for r in rows}
    assert r2["1"] > 5 * r2["2"]  # the responding ROI is the one that fits
    curves = _table(f"{prefix}_iresp_stim.tsv")
    assert curves[0] == ["time", "1", "2"]
    assert float(curves[1][0][0]) == 0.0
    assert not list(tmp_path.glob("out_iresp*.nii*"))


def test_roi_fit_is_the_fit_of_the_roi_mean_series(monkeypatch, tmp_path):
    runs, timing, arrays = _dataset(tmp_path)
    prefix = str(tmp_path / "out")
    assert _run(monkeypatch, runs, timing, prefix, "-atlas", _atlas(tmp_path)) == 0
    _, curve_rows = _table(f"{prefix}_iresp_stim.tsv")
    roi_curve = np.array([float(r[1]) for r in curve_rows])

    # The same ROI as one voxel whose series is the mean of the ROI's voxels.
    mean_runs = []
    for i, arr in enumerate(arrays):
        one = arr[:2].reshape(-1, N_TP).mean(axis=0).reshape(1, 1, 1, N_TP)
        img = nib.Nifti1Image(one.astype(np.float32), np.eye(4))
        img.header.set_zooms((2, 2, 2, TR))
        img.header.set_xyzt_units("mm", "sec")
        nib.save(img, tmp_path / f"mean{i}.nii.gz")
        mean_runs.append(str(tmp_path / f"mean{i}.nii.gz"))
    ref = str(tmp_path / "ref")
    assert _run(monkeypatch, mean_runs, timing, ref) == 0
    ref_curve = nib.load(f"{ref}_iresp_stim.nii.gz").get_fdata().reshape(-1)
    np.testing.assert_allclose(roi_curve, ref_curve, rtol=1e-3, atol=1e-3)


def test_roi_maps_paint_each_roi_into_its_voxels(monkeypatch, tmp_path):
    runs, timing, _ = _dataset(tmp_path)
    prefix = str(tmp_path / "out")
    assert _run(monkeypatch, runs, timing, prefix, "-rois", _atlas(tmp_path), "-roi_maps") == 0
    vol = nib.load(f"{prefix}_iresp_stim.nii.gz").get_fdata()
    assert vol.shape[:3] == SHAPE
    assert np.ptp(vol[:2].reshape(-1, vol.shape[-1]), axis=0).max() < 1e-5  # one value per ROI
    assert not np.allclose(vol[0, 0, 0], vol[3, 0, 0])


def test_mask_is_intersected_and_said_so(monkeypatch, tmp_path, capsys):
    runs, timing, _ = _dataset(tmp_path)
    mask = np.ones(SHAPE, dtype=np.uint8)
    mask[0] = 0
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    prefix = str(tmp_path / "out")
    args = ("-atlas", _atlas(tmp_path), "-mask", str(tmp_path / "mask.nii.gz"), "-verb", "1")
    assert _run(monkeypatch, runs, timing, prefix, *args) == 0
    assert "intersected with -mask" in capsys.readouterr().out
    _, rows = _table(f"{prefix}_roi.tsv")
    assert [r[2] for r in rows] == ["8", "16"]  # ROI 1 lost its x=0 slab (8 of 16 voxels)


def test_overlapping_rois_cannot_be_painted(monkeypatch, tmp_path, capsys):
    runs, timing, _ = _dataset(tmp_path)
    a = np.zeros(SHAPE)
    a[:2] = 1
    b = np.zeros(SHAPE)
    b[1:3] = 1
    for n, m in (("a", a), ("b", b)):
        nib.save(nib.Nifti1Image(m, np.eye(4)), tmp_path / f"{n}.nii.gz")
    rois = [str(tmp_path / "a.nii.gz"), str(tmp_path / "b.nii.gz")]
    prefix = str(tmp_path / "out")
    assert _run(monkeypatch, runs, timing, prefix, "-rois", *rois, "-roi_maps") == 1
    assert "label map" in capsys.readouterr().err
    assert _run(monkeypatch, runs, timing, prefix, "-rois", *rois) == 0
    _, rows = _table(f"{prefix}_roi.tsv")
    assert [r[1] for r in rows] == ["a", "b"]


def test_surface_runs_with_a_freesurfer_label(monkeypatch, tmp_path):
    from fastfuncstuff.io.freesurfer import write_label
    from fastfuncstuff.io.gifti import save_gifti_data

    rng = np.random.default_rng(8)
    t = np.arange(N_TP) * TR
    n_vert, runs, rows = 200, [], []
    for r in range(2):
        onsets = np.sort(rng.choice(np.arange(3, 105), 14, replace=False)) * TR
        data = 100 + rng.normal(0, 1, (n_vert, N_TP))
        data[:30] += 10 * _bold(rng, onsets, t)
        path = tmp_path / f"lh.run{r}.func.gii"
        meta = {"TR_seconds": str(TR), "mesh_fingerprint": "M"}
        save_gifti_data(path, data.astype(np.float32), meta)
        runs.append(str(path))
        rows.append(" ".join(f"{o:.2f}" for o in onsets))
    (tmp_path / "stim.1D").write_text("\n".join(rows) + "\n")
    write_label(tmp_path / "lh.V1.label", np.arange(30), np.zeros((30, 3)), np.ones(30))
    write_label(tmp_path / "lh.far.label", np.arange(150, 200), np.zeros((50, 3)), np.ones(50))
    labels = [str(tmp_path / "lh.V1.label"), str(tmp_path / "lh.far.label")]

    prefix = str(tmp_path / "out")
    assert _run(monkeypatch, runs, str(tmp_path / "stim.1D"), prefix, "-rois", *labels) == 0
    header, table = _table(f"{prefix}_roi.tsv")
    assert [r[1] for r in table] == ["lh.V1", "lh.far"]
    r2 = [float(r[header.index("r2")]) for r in table]
    assert r2[0] > 5 * r2[1]


def test_curves_are_percent_signal_change_by_default_and_data_units_with_dont_scale(
    monkeypatch, tmp_path
):
    """Scaling is the default (the laminar model and any cross-voxel comparison want
    percent); -dont_scale keeps the data's own units, as 3dDeconvolve does."""
    runs, timing, arrays = _dataset(tmp_path)
    # Baseline 1000, not ~100: at 100 percent and data units nearly coincide and the
    # test could not tell scaling from no scaling.
    for path, arr in zip(runs, arrays, strict=True):
        img = nib.Nifti1Image((arr * 10).astype(np.float32), np.eye(4))
        img.header.set_zooms((2, 2, 2, TR))
        img.header.set_xyzt_units("mm", "sec")
        nib.save(img, path)
    atlas = _atlas(tmp_path)
    assert _run(monkeypatch, runs, timing, str(tmp_path / "pct"), "-atlas", atlas) == 0
    assert (
        _run(monkeypatch, runs, timing, str(tmp_path / "raw"), "-atlas", atlas, "-dont_scale") == 0
    )
    pct = np.array([float(r[1]) for r in _table(tmp_path / "pct_iresp_stim.tsv")[1]])
    raw = np.array([float(r[1]) for r in _table(tmp_path / "raw_iresp_stim.tsv")[1]])
    baseline = 10 * np.mean([a[:2].mean() for a in arrays])  # ROI 1's mean level, ~1000
    np.testing.assert_allclose(pct, raw * 100.0 / baseline, rtol=0.02, atol=0.02)
    assert np.abs(raw).max() > 5 * np.abs(pct).max()
