"""Minimum-statistic conjunctions: the planner/combiner, the TOML tables, and ffs_reml end to end."""

import subprocess
import sys

import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.stats import gamma

from fastfuncstuff.cli.util_concalc import _read_brick_labels
from fastfuncstuff.design.spec import parse_conjunctions_text, parse_contrasts_text
from fastfuncstuff.glm.conjunction import apply_conjunctions, plan_conjunctions
from fastfuncstuff.io.headers import read_brick_stataux

TR, N_TP, SHAPE = 1.5, 160, (4, 4, 2)
ONSETS = {"a": [6, 60, 114, 168, 195], "b": [33, 87, 141, 186, 213], "c": [20, 100, 150]}
TOML = """
[[contrasts]]
label = "a_minus_c"
sym = "SYM: +1*a -1*c"

[[conjunctions]]
label = "amc_and_b"
components = ["a_minus_c", "b"]
"""


def _design_info():
    # a: one column, b: two basis columns, then a polort column.
    return {
        "matrix": np.zeros((10, 4)),
        "column_labels": ["a#0", "b#0", "b#1", "Run#1Pol#0"],
        "stim_labels": ["a", "b"],
        "stim_bots": [0, 1],
        "stim_tops": [0, 2],
        "glt_labels": ["a_vs_b"],
        "glt_matrices": [np.array([[1.0, -0.5, -0.5, 0.0]])],
    }


def test_plan_appends_one_internal_glt_per_condition():
    info = _design_info()
    plan = plan_conjunctions([("ab", "a b"), ("na_vab", ["-a", "a_vs_b"])], info)
    # 'a' is shared by both conjunctions, so only two internal GLTs are added.
    assert info["glt_labels"][0] == "a_vs_b" and len(info["glt_labels"]) == 3
    np.testing.assert_array_equal(info["glt_matrices"][1], [[1, 0, 0, 0]])
    # A multi-column condition spreads over its basis like -gltsym's '+b'.
    np.testing.assert_array_equal(info["glt_matrices"][2], [[0, 1, 1, 0]])
    assert plan.n_user == 1
    assert plan.conjunctions == (("ab", ((1, 1.0), (2, 1.0))), ("na_vab", ((1, -1.0), (0, 1.0))))
    assert info["glt_conjunctions"] is plan


@pytest.mark.parametrize(
    ("conj", "message"),
    [
        (("x", "a"), "at least two"),
        (("x", "a nope"), "unknown component 'nope'"),
        (("a_vs_b", "a b"), "already a contrast"),
        (("x", "a +a"), "two distinct"),
    ],
)
def test_plan_rejects_bad_conjunctions(conj, message):
    with pytest.raises(ValueError, match=message):
        plan_conjunctions([conj], _design_info())


def test_plan_rejects_an_f_test_component():
    info = _design_info()
    info["glt_matrices"] = [np.eye(4)[:2]]
    with pytest.raises(ValueError, match="F-test"):
        plan_conjunctions([("x", "a a_vs_b")], info)


def test_apply_takes_min_t_and_min_coef_separately():
    info = _design_info()
    plan = plan_conjunctions([("ab", "a b"), ("neg", "-a -b")], info)

    class R:
        pass

    r = R()
    r.contrast_labels = list(info["glt_labels"])
    # Columns: a_vs_b, a, b. Voxel 0: a has the smaller t but the larger coef.
    r.contrast_betas = torch.tensor([[9.0, 5.0, 2.0], [9.0, -1.0, 3.0]])
    r.contrast_tstats = torch.tensor([[9.0, 2.0, 4.0], [9.0, -3.0, 1.0]])
    r.contrast_r2_partial = torch.tensor([[0.9, 0.2, 0.1], [0.9, 0.3, 0.4]])
    apply_conjunctions(r, plan)
    apply_conjunctions(r, plan)  # idempotent: the writers each call it
    assert r.contrast_labels == ["a_vs_b", "ab", "neg"]
    torch.testing.assert_close(
        r.contrast_tstats, torch.tensor([[9.0, 2.0, -4.0], [9.0, -3.0, -1.0]])
    )
    torch.testing.assert_close(
        r.contrast_betas, torch.tensor([[9.0, 2.0, -5.0], [9.0, -1.0, -3.0]])
    )
    torch.testing.assert_close(
        r.contrast_r2_partial, torch.tensor([[0.9, 0.1, 0.1], [0.9, 0.3, 0.3]])
    )


def test_toml_fragment_carries_conjunctions():
    assert [c.label for c in parse_contrasts_text(TOML)] == ["a_minus_c"]
    conj = parse_conjunctions_text(TOML)
    assert [(c.label, c.components) for c in conj] == [("amc_and_b", ["a_minus_c", "b"])]
    only = '[[conjunctions]]\nlabel = "x"\ncomponents = "a b"\n'
    assert parse_contrasts_text(only) == []
    assert parse_conjunctions_text(only)[0].components == ["a", "b"]
    with pytest.raises(ValueError, match="at least two"):
        parse_conjunctions_text('[[conjunctions]]\nlabel = "x"\ncomponents = ["a"]\n')
    with pytest.raises(ValueError, match="conjunction #1"):
        parse_conjunctions_text('[[conjunctions]]\nlabel = "x"\nparts = ["a", "b"]\n')


def _hrf(t):
    return np.where(t > 0, gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6, 0)


@pytest.fixture
def study(tmp_path):
    rng = np.random.default_rng(5)
    t = np.arange(N_TP) * TR
    arr = 100 + rng.normal(0, 1, (*SHAPE, N_TP))
    arr[:2] += 6 * sum(_hrf(t - o) for o in ONSETS["a"])
    arr[1:] += 3 * sum(_hrf(t - o) for o in ONSETS["b"])
    nib.save(
        nib.Nifti1Image(arr.astype(np.float32), np.diag([3, 3, 3, 1])), tmp_path / "run.nii.gz"
    )
    rows = sorted((o, k) for k, v in ONSETS.items() for o in v)
    (tmp_path / "ev.tsv").write_text(
        "onset\tduration\ttrial_type\n" + "\n".join(f"{o}\t1\t{k}" for o, k in rows) + "\n"
    )
    (tmp_path / "c.toml").write_text(TOML)
    return tmp_path


def _run(tmp_path, *extra, expect=0):
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(tmp_path / "run.nii.gz")]
        + ["-events", str(tmp_path / "ev.tsv"), "-TR", str(TR), "-device", "cpu", *extra],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=tmp_path,
    )
    assert res.returncode == expect, res.stdout[-3000:] + res.stderr[-3000:]
    return res


def _bucket(path):
    img = nib.load(path)
    labels = _read_brick_labels(img)
    data = np.asanyarray(img.dataobj)
    return {lab: data[..., i] for i, lab in enumerate(labels)}, img


def _assert_min(b, conj, components):
    def stat(name, kind):
        sign = -1.0 if name.startswith("-") else 1.0
        return sign * b[f"{name.lstrip('-')}#0_{kind}"]

    for kind in ("Coef", "Tstat"):
        expected = np.minimum.reduce([stat(c, kind) for c in components])
        np.testing.assert_allclose(b[f"{conj}#0_{kind}"], expected, atol=1e-4)


def test_reml_writes_conjunctions_on_ols_and_reml_buckets(study):
    _run(
        study,
        *("-Obuck", "ols", "-Rbuck", "reml", "-hrf_model", "SPMG1", "-contrasts", "c.toml"),
        *("-conj", "a_and_b", "a b", "-conj", "neg", "-a,-c"),
    )
    for name in ("ols", "reml"):
        b, img = _bucket(study / f"{name}.nii.gz")
        # The internal component GLTs never reach the bucket.
        assert not any(k.startswith("__conj__") for k in b)
        _assert_min(b, "a_and_b", ["a", "b"])
        _assert_min(b, "neg", ["-a", "-c"])
        _assert_min(b, "amc_and_b", ["a_minus_c", "b"])
        labels = list(b)
        assert labels[-8:] == [
            "a_minus_c#0_Coef",
            "a_minus_c#0_Tstat",
            "amc_and_b#0_Coef",
            "amc_and_b#0_Tstat",
            "a_and_b#0_Coef",
            "a_and_b#0_Tstat",
            "neg#0_Coef",
            "neg#0_Tstat",
        ]
        # Same t-stat code and DoF as a condition's t, so AFNI p-values apply as-is.
        aux = read_brick_stataux(img)
        for conj in ("amc_and_b", "a_and_b", "neg"):
            assert aux[labels.index(f"{conj}#0_Tstat")] == aux[labels.index("a#0_Tstat")]


def test_reml_conj_rejects_unknown_component(study):
    res = _run(study, "-Obuck", "o", "-hrf_model", "SPMG1", "-conj", "x", "a nope", expect=1)
    assert "unknown component 'nope'" in res.stdout + res.stderr


def _hrf_library(tmp_path):
    # Two library curves; top slice uses the slow one, so the per-voxel HRF path
    # really fits two designs and merges their GLTs.
    tt = np.arange(320) * 0.1
    lib = np.stack([_hrf(tt * 0.7), _hrf(tt)])
    lib = lib / lib.max(axis=1, keepdims=True)
    torch.save(
        {"hrf_library": torch.tensor(lib, dtype=torch.float32)}, tmp_path / "lib_hrf_library.pt"
    )
    idx = np.ones(SHAPE, np.float32)
    idx[:, :, 1] = 2
    nib.save(nib.Nifti1Image(idx, np.diag([3, 3, 3, 1])), tmp_path / "lib_hrf_index.nii.gz")


@pytest.mark.parametrize("route", ["ols_only", "per_voxel_hrf"])
def test_reml_conjunction_on_other_routes(study, route):
    if route == "ols_only":
        extra, buckets = ["-Obuck", "ols", "-hrf_model", "SPMG1"], ["ols"]
    else:
        _hrf_library(study)
        extra = ["-Obuck", "ols", "-Rbuck", "reml", "-hrfopt_prefix", "lib"]
        buckets = ["ols", "reml"]
    _run(study, *extra, "-conj", "a_and_b", "a b")
    for name in buckets:
        b, _ = _bucket(study / f"{name}.nii.gz")
        _assert_min(b, "a_and_b", ["a", "b"])


def test_reml_spec_conjunction(study):
    from fastfuncstuff.design.spec import (
        ConjunctionSpec,
        ContrastSpec,
        EventSpec,
        MetaSpec,
        RunSpec,
        Spec,
        load_spec,
        write_spec,
    )

    spec = Spec(
        meta=MetaSpec(
            runs=[RunSpec(bold=str(study / "run.nii.gz"), events=str(study / "ev.tsv"))],
            tr=TR,
            n_timepoints_per_run=[N_TP],
            polort=2,
        ),
        events=[EventSpec(trial_type=k, duration=1.0, hrf="SPMG1") for k in ONSETS],
        contrasts=[ContrastSpec(label="a_minus_c", sym="SYM: +1*a -1*c")],
        conjunctions=[ConjunctionSpec(label="amc_and_b", components=["a_minus_c", "b"])],
    )
    write_spec(spec, study / "design.toml")
    assert load_spec(study / "design.toml").conjunctions == spec.conjunctions
    res = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fastfuncstuff.cli.reml", "-input", str(study / "run.nii.gz")]
        + ["-spec", str(study / "design.toml"), "-Rbuck", "reml", "-device", "cpu"],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        cwd=study,
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    b, _ = _bucket(study / "reml.nii.gz")
    _assert_min(b, "amc_and_b", ["a_minus_c", "b"])
