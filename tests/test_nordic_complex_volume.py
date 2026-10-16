import numpy as np
import torch

from fastfuncstuff.denoise.nordic import _complex_volume


def _case(seed=0, shape=(5, 6, 4, 37)):
    rng = np.random.default_rng(seed)
    mag = np.abs(rng.standard_normal(shape)).astype(np.float32) * 50 + 1
    phase = rng.uniform(-np.pi, np.pi, shape).astype(np.float32)
    return mag, phase


def test_matches_the_numpy_form_to_float32_rounding():
    """The numpy form was 15.7 s on a real echo; the device form is this, chunked."""
    mag, phase = _case()
    want = (mag * np.exp(1j * phase)).astype(np.complex64) / 7.5
    got = _complex_volume(mag, phase, 7.5, torch.device("cpu"), chunk=16).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_chunking_does_not_change_the_answer():
    mag, phase = _case(1)
    whole = _complex_volume(mag, phase, 3.0, torch.device("cpu"), chunk=1000)
    chunked = _complex_volume(mag, phase, 3.0, torch.device("cpu"), chunk=5)
    assert torch.equal(whole, chunked)


def test_magnitude_only_has_zero_imaginary_part():
    mag, _ = _case(2)
    got = _complex_volume(mag, None, 2.0, torch.device("cpu"))
    assert got.dtype == torch.complex64
    assert float(got.imag.abs().max()) == 0.0
    np.testing.assert_allclose(got.real.numpy(), mag / 2.0, rtol=1e-6)


def test_fortran_ordered_input_is_handled():
    """nibabel's get_fdata returns F-ordered arrays."""
    mag, phase = _case(3)
    got = _complex_volume(
        np.asfortranarray(mag), np.asfortranarray(phase), 1.0, torch.device("cpu")
    )
    want = _complex_volume(mag, phase, 1.0, torch.device("cpu"))
    assert torch.equal(got, want)
