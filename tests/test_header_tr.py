"""header_tr: the one place a NIfTI pixdim[4] becomes a TR in seconds."""

import warnings

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.io.headers import header_tr


def _img(pixdim4: float, unit: str | None, ndim: int = 4) -> nib.Nifti1Image:
    shape = (2, 2, 2, 5) if ndim == 4 else (2, 2, 2)
    img = nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4))
    if ndim == 4:
        img.header.set_zooms((1.0, 1.0, 1.0, pixdim4))
    if unit is not None:
        img.header.set_xyzt_units(xyz="mm", t=unit)
    return img


@pytest.mark.parametrize(
    ("pixdim4", "unit"), [(2.0, "sec"), (2000.0, "msec"), (2_000_000.0, "usec")]
)
def test_units_scale_to_seconds(pixdim4, unit):
    # The bug: a msec header was read as TR=2000 s and silently built the design.
    assert header_tr(_img(pixdim4, unit)) == pytest.approx(2.0)
    assert header_tr(_img(pixdim4, unit).header) == pytest.approx(2.0)


def test_unknown_unit_assumed_seconds_unless_required():
    img = _img(1.5, None)
    assert img.header.get_xyzt_units()[1] == "unknown"
    assert header_tr(img) == pytest.approx(1.5)
    assert header_tr(img, require_units=True) is None


def test_no_time_axis_is_none():
    assert header_tr(_img(0.0, "sec", ndim=3)) is None
    assert header_tr(_img(0.0, "sec")) is None


def test_implausible_tr_warns_but_returns():
    with pytest.warns(UserWarning, match="plausible"):
        assert header_tr(_img(2000.0, "sec")) == pytest.approx(2000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert header_tr(_img(2000.0, "sec"), warn=False) == pytest.approx(2000.0)
        assert header_tr(_img(0.8, "sec")) == pytest.approx(0.8)


def test_main_loader_reads_msec_header(tmp_path):
    import torch

    from fastfuncstuff.cli_utils import load_and_preprocess_runs

    path = tmp_path / "run.nii.gz"
    img = _img(2500.0, "msec")
    img = nib.Nifti1Image(
        np.random.default_rng(0).normal(100, 1, (4, 4, 4, 20)).astype(np.float32),
        np.eye(4),
        img.header,
    )
    img.header.set_zooms((1.0, 1.0, 1.0, 2500.0))
    nib.save(img, path)
    out = load_and_preprocess_runs([str(path)], device=torch.device("cpu"), verbose=False)
    assert out.tr == pytest.approx(2.5)
