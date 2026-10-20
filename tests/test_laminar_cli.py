"""ffs_laminar: the notebook's analysis as a tool, refereed by the MATLAB reference.

The inversion oracle holds the reference's two-condition responses, its shipped PSF
kernel and the free energy MATLAB reached for two models. Written out the way the
toolbox writes curves (ffs_deconvolve's tidy tables, depth 1 deepest) and run through
the CLI end to end, the same F must come back -- which pins the depth flip, the block
layout, the input, the reference baselines, the priors and the kernel all at once.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fastfuncstuff.cli import laminar
from tests.test_laminar_io import _write_table

ORACLE = Path(__file__).parent / "laminar_oracle" / "laminar_inversion_oracle.json"


@pytest.fixture
def reference(tmp_path):
    if not ORACLE.exists():  # pragma: no cover
        pytest.skip(f"no oracle at {ORACLE}")
    o = json.loads(ORACLE.read_text())
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    times = np.arange(7) * float(o["TR"])
    a = _write_table(tmp_path / "k9_iresp_c1.tsv", times, y[1:8][:, ::-1])
    b = _write_table(tmp_path / "k9_iresp_c3.tsv", times, y[29:36][:, ::-1])
    kern = tmp_path / "kernel.1D"
    np.savetxt(kern, np.asarray(o["kernel"], float).reshape(-1))
    return o, a, b, kern


def _table(path):
    rows = [ln.split("\t") for ln in open(path).read().splitlines()]
    return rows[0], rows[1:]


def test_the_cli_writes_the_four_tables(reference, tmp_path):
    o, a, b, kern = reference
    prefix = str(tmp_path / "quick")
    rc = laminar.main(
        ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-psf_kernel", str(kern),
         "-reference_delay", "-models", "null", "-max_iter", "2", "-prefix", prefix,
         "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    h, rows = _table(f"{prefix}_laminar_models.tsv")
    assert (
        h[:8] == ["roi", "name", "hemi", "model", "F", "p_model", "best", "r2"] and h[8] == "F_K9"
    )
    assert [r[3] for r in rows] == ["null"] and float(rows[0][5]) == pytest.approx(1.0)
    h, rows = _table(f"{prefix}_laminar_depths.tsv")
    assert [r[3] for r in rows] == ["superficial", "middle", "deep"]
    h, rows = _table(f"{prefix}_laminar_devein.tsv")
    assert len(rows) == 3 * 9 and {r[4] for r in rows} == {str(d) for d in range(1, 10)}
    h, rows = _table(f"{prefix}_laminar_neuronal.tsv")
    assert len(rows) == 2 * 7 * 3 and {r[3] for r in rows} == {"1", "2"}
    doc = json.load(open(f"{prefix}_laminar.json"))
    assert doc["psf"] == "kernel" and doc["K"] == [9] and doc["delay_seconds"] == "TR/2"


def test_bad_inputs_are_reported(reference, tmp_path, capsys):
    o, a, b, kern = reference
    base = ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-prefix", str(tmp_path / "x")]
    assert laminar.main([*base, "-models", "nonsense"]) == 1
    assert "unknown model" in capsys.readouterr().err
    assert laminar.main([*base, "-roi", "nowhere"]) == 1
    assert "no ROI" in capsys.readouterr().err


@pytest.mark.slow
def test_the_pipeline_reaches_matlabs_free_energy_with_its_padding(reference):
    """~1 minute: two full inversions, compared with the reference to 1e-6.

    The reference sets the noise hyperprior from the variance of the WHOLE data vector,
    masked padding included, so its F depends on the padding's random draw. With the
    oracle's own padding rows the pipeline must reproduce MATLAB exactly."""
    import torch

    from fastfuncstuff.laminar.io import pick_roi, read_iresp_table, two_condition_data
    from fastfuncstuff.laminar.pipeline import fit_roi, reference_p0

    o, a, b, kern = reference
    y = np.asarray(o["y"], float).reshape(o["ns"], o["K"])
    data = two_condition_data(
        pick_roi(read_iresp_table(a), "V1"), pick_roi(read_iresp_table(b), "V1"),
        duration=1.6, delay_seconds=None, p0=reference_p0(),
    )  # fmt: skip
    data.y[8:28] = torch.tensor(y[8:28])
    k = torch.tensor(np.loadtxt(kern))
    fit = fit_roi({9: data}, {9: k}, targets=[(1, 0, 0), (0, 0, 0)], names=["superficial", "null"])
    got = dict(zip(fit.models, fit.F, strict=True))
    for m in o["models"]:
        assert abs(got[m["name"]] - float(m["F"])) / abs(float(m["F"])) < 1e-6


@pytest.mark.slow
def test_the_cli_reaches_matlabs_model_comparison(reference, tmp_path):
    """Through the CLI, with its own padding draw: the absolute F moves by ~1 nat with
    the draw (measured), the comparison by ~0.2 -- so the comparison must match."""
    o, a, b, kern = reference
    prefix = str(tmp_path / "parity")
    rc = laminar.main(
        ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-psf_kernel", str(kern),
         "-reference_delay", "-models", "superficial", "null", "-prefix", prefix,
         "-workers", "1", "-verb", "0"]
    )  # fmt: skip
    assert rc == 0
    _, rows = _table(f"{prefix}_laminar_models.tsv")
    got = {r[3]: float(r[4]) for r in rows}
    want = {m["name"]: float(m["F"]) for m in o["models"]}
    assert abs((got["superficial"] - got["null"]) - (want["superficial"] - want["null"])) < 0.5
    assert all(abs(got[n] - want[n]) < 3.0 for n in want)


def test_a_measured_psf_matrix_is_matched_by_roi_and_k(reference, tmp_path, capsys):
    from fastfuncstuff.processing.roi_depth import write_depth_psf

    o, a, b, kern = reference
    m = np.eye(9) * 0.7 + np.eye(9, k=1) * 0.15 + np.eye(9, k=-1) * 0.15
    psf = write_depth_psf(str(tmp_path / "k9_depth_psf.tsv"), [("V1", "lh", m)], [1])
    prefix = str(tmp_path / "mat")
    base = ["-cond1", str(a), "-cond2", str(b), "-duration", "1.6", "-models", "null",
            "-max_iter", "2", "-prefix", prefix, "-workers", "1", "-verb", "0"]  # fmt: skip
    assert laminar.main([*base, "-psf_matrix", psf]) == 0
    assert json.load(open(f"{prefix}_laminar.json"))["psf"] == "matrix"
    other = write_depth_psf(str(tmp_path / "k7_depth_psf.tsv"), [("V1", "lh", np.eye(7))], [1])
    assert laminar.main([*base, "-psf_matrix", other]) == 1
    assert "K = [9]" in capsys.readouterr().err
