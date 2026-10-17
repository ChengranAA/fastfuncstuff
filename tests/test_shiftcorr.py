"""Tests for partition-axis inter-echo shift estimation/correction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fastfuncstuff.processing.shiftcorr import (
    PhaseConvention,
    ShiftEstimate,
    apply_shift,
    check_phase_convention,
    detect_phase_convention,
    estimate_pair_shift,
    estimate_shifts,
    fold_partition_into_matrices,
    partition_shift_from_phase,
    search_bounds,
    shift_table_from_partition_shift,
    te_regression,
)


def _phantom(
    n_t: int = 4,
    shape: tuple[int, int, int] = (24, 20, 20),
    seed: int = 0,
    noise: float = 0.01,
) -> torch.Tensor:
    """Smooth structured volumes with a bit of noise — (T, nz, ny, nx)."""
    g = torch.Generator().manual_seed(seed)
    zz, yy, xx = torch.meshgrid(
        *[torch.linspace(-1, 1, s) for s in shape],
        indexing="ij",
    )
    blob = torch.exp(-(zz**2 + yy**2 + xx**2) * 3.0) * (1.0 + 0.4 * torch.sin(4.0 * zz))
    vols = blob.unsqueeze(0).repeat(n_t, 1, 1, 1)
    vols = vols + noise * torch.randn(vols.shape, generator=g)
    return vols


def test_pair_shift_recovers_known_subvoxel_shift():
    """The estimate is the CORRECTION: the shift that undoes the displacement."""
    ref = _phantom(n_t=3)
    displacement = np.array([-0.35, 0.80, -1.25])
    mov = apply_shift(ref, displacement, axis=2)  # axis 2 == tensor dim 1 (nz)

    est, corr = estimate_pair_shift(ref, mov, axis=2, ordering="unknown", max_shift=3.0)

    assert np.allclose(est, -displacement, atol=0.01), f"{est} vs {-displacement}"
    assert (corr > 0.99).all()


def test_apply_shift_is_invertible():
    """A push by d then by -d must return the original (sinc-exact, no smoothing)."""
    # Noiseless: white noise sits at Nyquist, where the replicate pad's own edge
    # discontinuity dominates the round trip. Real image content is band-limited.
    vols = _phantom(n_t=2, noise=0.0)
    d = np.array([0.4, -0.6])
    back = apply_shift(apply_shift(vols, d, axis=2), -d, axis=2)
    # Edge partitions genuinely lose information (content shifted in from the
    # replicated pad cannot come back), so the identity only holds in the interior.
    inner = slice(6, -6)
    assert torch.allclose(back[:, inner], vols[:, inner], atol=1e-3)


def test_ordering_constrains_sign():
    lo, hi = search_bounds("ascending", 5.0)
    assert (lo, hi) == (-5.0, 0.0)
    lo, hi = search_bounds("descending", 5.0)
    assert (lo, hi) == (0.0, 5.0)
    with pytest.raises(ValueError):
        search_bounds("sideways", 5.0)


def test_te_regression_recovers_slope_and_drops_intercept():
    tes = np.array([7.61, 21.71, 35.81])
    slope = np.array([-0.02, -0.015])
    intercept = np.array([0.5, -0.3])
    cum = slope[:, None] * tes[None, :] + intercept[:, None]

    m, b = te_regression(cum, tes)

    assert np.allclose(m, slope)
    assert np.allclose(b, intercept)


def test_estimate_shifts_end_to_end_te_line():
    """Three echoes shifted by m·TE per timepoint: the fit must recover m."""
    tes = np.array([7.61, 21.71, 35.81])
    base = _phantom(n_t=3, seed=1)
    m_true = np.array([-0.03, -0.05, -0.02])  # voxels per ms of TE, per timepoint
    echoes = [apply_shift(base, m_true * te, axis=2) for te in tes]

    est = estimate_shifts(echoes, axis=2, tes=tes, ordering="unknown", max_shift=3.0, verb=0)

    assert isinstance(est, ShiftEstimate)
    # Estimates are corrections, so the recovered slope is minus the applied one.
    assert np.allclose(est.slope, -m_true, atol=2e-3)
    # The correction goes through the origin, so echo 1 is corrected too.
    assert np.allclose(est.applied, -m_true[:, None] * tes[None, :], atol=0.03)
    assert np.abs(est.applied[:, 0]).min() > 0.1
    assert np.allclose(est.frequency_drift_hz, m_true * 1e3, atol=2.0)


def test_estimate_shifts_without_tes_uses_cumulative():
    base = _phantom(n_t=2, seed=2)
    echoes = [base, apply_shift(base, np.array([-0.4, -0.4]), axis=2)]

    est = estimate_shifts(echoes, axis=2, tes=None, ordering="unknown", max_shift=3.0, verb=0)

    assert est.slope is None
    assert np.allclose(est.applied[:, 0], 0.0)  # echo 1 is the reference
    assert np.allclose(est.applied[:, 1], 0.4, atol=0.01)


def test_complex_shift_preserves_phase_structure():
    """Complex input shifts magnitude and phase together, with no global phase roll."""
    # Strictly positive magnitude: a negative real value carries its own pi of
    # phase and would make the constant-phase assertion below meaningless.
    base = _phantom(n_t=1, seed=3).abs() + 1.0
    comp = (base * torch.exp(1j * 0.3 * torch.ones_like(base))).to(torch.complex64)
    out = apply_shift(comp, np.array([1.0]), axis=2)
    inner = slice(4, -4)
    # A whole-voxel shift of a constant-phase volume must leave the phase constant.
    assert torch.allclose(
        torch.angle(out[:, inner]), torch.full_like(base[:, inner], 0.3), atol=1e-3
    )


def test_fold_into_matrices_matches_the_fourier_shift():
    """The folded matrix must move voxels the same way apply_shift does.

    This is the sign-convention gate for -me_3depi: a flipped sign here would
    DOUBLE the inter-echo shift instead of removing it, and nothing downstream
    would catch it.
    """
    from fastfuncstuff.processing.ffs_moco import MocoConfig, resample_timeseries
    from fastfuncstuff.processing.shiftcorr import fold_shift_into_matrices

    vols = _phantom(n_t=2, noise=0.0)
    d = np.array([0.45, -0.30])
    axis = 2  # NIfTI z == the (nt, nz, ny, nx) tensor's dim 1

    identity = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    folded = fold_shift_into_matrices(identity, d, axis)
    config = MocoConfig(device=torch.device("cpu"), final_interp="wsinc5", verb=0)
    via_matrix, _ = resample_timeseries(vols, folded, config, torch.device("cpu"))
    via_fourier = apply_shift(vols, d, axis)

    inner = slice(6, -6)
    assert torch.allclose(via_matrix[:, inner], via_fourier[:, inner], atol=5e-3)


def test_full_extent_sees_structure_the_inner_half_crop_throws_away():
    """Anatomy in the outer quarters must still drive the estimate.

    The reference crops to the central half because it does not pad before its
    trial shift. We replicate-pad, so the crop is unnecessary — and harmful the
    moment the FoV is not centred on the structure. Here ALL the structure sits in
    the outer quarters of the partition axis; "full" recovers the shift, the
    reference's crop is left correlating noise.
    """
    n_z = 40
    interior = slice(n_z // 4, (3 * n_z) // 4)
    # Structure deliberately in the outer quarters only — an off-centre slab, or
    # any FoV where the anatomy is not politely centred on the partition axis.
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, n_z),
        torch.linspace(-1, 1, 20),
        torch.linspace(-1, 1, 20),
        indexing="ij",
    )
    edges = (torch.exp(-((zz - 0.7) ** 2) * 60) + torch.exp(-((zz + 0.7) ** 2) * 60)) * (
        1.0 + 0.5 * torch.sin(5.0 * yy) * torch.cos(5.0 * xx)
    )
    ref = edges.unsqueeze(0).repeat(2, 1, 1, 1)
    truth = np.array([-0.6, 0.45])
    mov = apply_shift(ref, truth, axis=2)

    # Independent noise in the interior of BOTH members of the pair: the central
    # half now carries no shared signal at all, so only the outer quarters can
    # say anything about the shift.
    g = torch.Generator().manual_seed(7)
    ref[:, interior] = 0.05 * torch.randn(2, n_z // 2, 20, 20, generator=g)
    mov[:, interior] = 0.05 * torch.randn(2, n_z // 2, 20, 20, generator=g)

    full, _ = estimate_pair_shift(ref, mov, axis=2, max_shift=2.0, extent="full")
    inner, r_inner = estimate_pair_shift(ref, mov, axis=2, max_shift=2.0, extent="inner_half")

    assert np.allclose(full, -truth, atol=0.05), f"full extent got {full}"
    assert np.abs(inner - -truth).max() > 0.1, f"inner_half should be blind here but got {inner}"
    assert (r_inner < 0.2).all()  # and it knows: the peak correlation is near zero


def test_edge_taper_only_touches_the_padding_width():
    """The taper must downweight the fabricated edge and nothing else."""
    from fastfuncstuff.processing.shiftcorr import _edge_taper

    w = _edge_taper(40, 3, dim=0, ndim=1, device="cpu", dtype=torch.float32)
    assert w.shape == (40,)
    assert torch.all(w[3:-3] == 1.0)  # 34 of 40 partitions at full weight
    assert w[0] < w[1] < w[2] < 1.0
    assert torch.allclose(w[:3], w[-3:].flip(0))


def _wrapped_phase(shape=(24, 20, 20)) -> torch.Tensor:
    """Radian phase with several genuine wraps along z, like a real B0 ramp."""
    zz, yy, _ = torch.meshgrid(*[torch.arange(s).float() for s in shape], indexing="ij")
    return torch.remainder(0.4 * zz + 0.05 * yy + np.pi, 2 * np.pi) - np.pi


@pytest.mark.parametrize(
    "stored, label, scale, offset",
    [
        (_wrapped_phase(), "radians", 1.0, 0.0),
        (_wrapped_phase() * 4096 / np.pi, "signed", np.pi / 4096, 0.0),
        ((_wrapped_phase() + np.pi) * 4096 / (2 * np.pi), "unsigned", 2 * np.pi / 4096, 2048.0),
    ],
)
def test_detect_phase_convention_recognises_the_three_storage_forms(stored, label, scale, offset):
    conv = detect_phase_convention(stored)
    assert label in conv.label
    assert conv.scale == pytest.approx(scale)
    assert conv.offset == pytest.approx(offset)
    assert torch.allclose(conv.to_radians(stored), _wrapped_phase(), atol=1e-3)


def test_detect_phase_convention_refuses_unwrapped_radians():
    """Unwrapped phase is the case range-detection CANNOT name — it must not guess."""
    with pytest.raises(ValueError, match="cannot infer phase units"):
        detect_phase_convention(_wrapped_phase() * 6.0)


def test_phase_round_trip_returns_the_input_convention():
    """What goes in as 0..4095 comes back as 0..4095, not recentred on zero."""
    stored = (_wrapped_phase() + np.pi) * 4096 / (2 * np.pi)
    conv = detect_phase_convention(stored)
    back = conv.from_radians(conv.to_radians(stored))
    assert torch.allclose(back, stored, atol=1e-2)
    assert back.min() >= -1.0 and back.max() <= 4097.0


def test_wrong_phase_gain_corrupts_phase_while_magnitude_looks_fine():
    """Bug of record: the reason the gain is detected instead of assumed.

    A radians-in-file phase scaled by the old +/-4096 default makes the complex
    volume nearly real. The shifted MAGNITUDE is then still within a percent —
    which is exactly why nobody notices — but the shifted PHASE is garbage.
    """
    mag = _phantom(n_t=1, seed=5).abs()[0] + 1.0
    stored = _wrapped_phase(tuple(mag.shape))
    shift = np.array([1.37])

    good = detect_phase_convention(stored)
    assert good.scale == 1.0
    ref = apply_shift(torch.polar(mag, good.to_radians(stored)), shift, axis=2)

    bad = PhaseConvention(np.pi / 4096, 0.0, "wrong")
    got = apply_shift(torch.polar(mag, bad.to_radians(stored)), shift, axis=2)

    d = (
        torch.remainder(
            bad.from_radians(torch.angle(got)) - good.from_radians(torch.angle(ref)) + np.pi,
            2 * np.pi,
        )
        - np.pi
    )
    inner = (slice(4, -4),) * 3
    assert d[inner].abs().mean() > 0.1  # radians of phase error
    rel = ((got.abs() - ref.abs()).abs() / ref.abs())[inner]
    assert rel.mean() < 0.02  # ... and a magnitude that still looks right


def test_explicit_phase_scale_is_sanity_checked_not_overridden():
    stored = _wrapped_phase()
    narrow = check_phase_convention(stored, PhaseConvention(np.pi / 4096, 0.0, "user"))
    assert narrow and "already in radians" in narrow[0]
    unwrapped = check_phase_convention(stored * 6.0, PhaseConvention(1.0, 0.0, "user"))
    assert unwrapped and "WRAPPED" in unwrapped[0]
    assert check_phase_convention(stored, PhaseConvention(1.0, 0.0, "user")) == ()


def test_partition_shift_from_phase_returns_the_correction_apply_shift_needs():
    """Echo e reads the field at a TE-scaled displaced position. The phase-only
    estimate must come back as the correction (ShiftEstimate.applied convention),
    blind to a field change shared by all echoes."""
    nx, ny, nz, n_t = 20, 18, 24, 30
    tes = np.array([8.0, 20.0, 32.0])
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in (nx, ny, nz)], indexing="ij")
    field = 30 * np.sin(2.0 * z) + 10 * x * z + 5 * y  # Hz, gentle (< 3 Hz/vox along z)
    rng = np.random.default_rng(0)
    w_true = 0.3 * np.sin(np.linspace(0, 12, n_t))  # correction at TE1, voxels
    w_true -= w_true.mean()
    common = 2.0 * np.cos(np.linspace(0, 5, n_t))  # breathing: same Hz at every echo
    offset = 0.15 * np.sin(np.linspace(0, 3, n_t))  # receiver phase at TE 0, rad
    fld = torch.from_numpy(np.repeat(field.transpose(2, 1, 0)[None], n_t, 0).astype(np.float32))
    unwrapped = np.zeros((nx, ny, nz, len(tes), n_t), np.float32)
    for e, te in enumerate(tes):
        # observed = original pushed by MINUS the correction (so the correction restores it)
        moved = apply_shift(fld, -w_true * te / tes[0], axis=2).numpy().transpose(3, 2, 1, 0)
        f_e = moved + common[None, None, None, :]
        unwrapped[..., e, :] = 2 * np.pi * f_e * te * 1e-3 + offset + rng.normal(0, 0.02, f_e.shape)
    mag = np.ones_like(unwrapped)
    mask = np.zeros((nx, ny, nz), bool)
    mask[3:-3, 3:-3, 4:-4] = True  # clear of the shift's replicate-padded edge
    w = partition_shift_from_phase(unwrapped, mag, tes, mask, 2, device=torch.device("cpu"))
    assert np.corrcoef(w, w_true)[0, 1] > 0.99
    assert np.polyfit(w_true, w, 1)[0] == pytest.approx(1.0, abs=0.1)
    # a drifting receiver offset reads as a shift unless it is projected out too
    naive = partition_shift_from_phase(
        unwrapped, mag, tes, mask, 2, device=torch.device("cpu"), offset="none"
    )
    assert np.abs(naive - w_true).std() > 3 * np.abs(w - w_true).std()
    per_voxel = partition_shift_from_phase(
        unwrapped, mag, tes, mask, 2, device=torch.device("cpu"), offset="voxel"
    )
    assert np.corrcoef(per_voxel, w_true)[0, 1] > 0.98
    table = shift_table_from_partition_shift(w, tes)
    assert table.shape == (n_t, 3)
    assert np.allclose(table[:, 2], w * tes[2] / tes[0])


def test_partition_stretch_from_phase_and_its_fold_undo_the_displacement():
    """A shift that grows linearly along the partition axis (TE-scaled): the phase
    estimate recovers shift and stretch, and the folded matrix undoes it."""
    nx, ny, nz, n_t = 16, 14, 40, 24
    tes = np.array([8.0, 20.0, 32.0])
    zc = (nz - 1) / 2
    t = np.linspace(0, 10, n_t)
    s_true = 0.2 * np.sin(t)
    s_true -= s_true.mean()
    k_true = 0.006 * np.cos(1.3 * t)
    k_true -= k_true.mean()

    def field(x, y, z):  # Hz, analytic so it can be sampled anywhere along z
        return 25 * np.sin(0.12 * z) + 8 * x + 4 * y

    x, y, z = np.meshgrid(
        np.linspace(-1, 1, nx), np.linspace(-1, 1, ny), np.arange(nz), indexing="ij"
    )
    rng = np.random.default_rng(1)
    unwrapped = np.zeros((nx, ny, nz, len(tes), n_t), np.float32)
    for e, te in enumerate(tes):
        a = te / tes[0]
        for i in range(n_t):
            corr = a * (s_true[i] + k_true[i] * (z - zc))  # the correction at echo e
            f_e = field(x, y, z + corr) + 1.5 * np.cos(t[i])  # raw = original pushed by -corr
            unwrapped[..., e, i] = 2 * np.pi * f_e * te * 1e-3 + rng.normal(0, 0.01, f_e.shape)
    mask = np.zeros((nx, ny, nz), bool)
    mask[2:-2, 2:-2, 3:-3] = True
    sk = partition_shift_from_phase(
        unwrapped, np.ones_like(unwrapped), tes, mask, 2, device=torch.device("cpu"), stretch=True
    )
    assert sk.shape == (n_t, 2)
    assert np.corrcoef(sk[:, 0], s_true)[0, 1] > 0.99
    assert np.corrcoef(sk[:, 1], k_true)[0, 1] > 0.98
    assert np.polyfit(k_true, sk[:, 1], 1)[0] == pytest.approx(1.0, abs=0.1)

    # The fold, applied with identity motion, maps each output voxel to where the
    # original content sits in the raw frame.
    m = fold_partition_into_matrices(
        torch.eye(4, dtype=torch.float64).repeat(n_t, 1, 1), s_true, k_true, 2, zc
    ).numpy()
    zz = np.arange(nz, dtype=np.float64)
    for i in (0, n_t // 2):
        raw = field(0.0, 0.0, zz + (s_true[i] + k_true[i] * (zz - zc)))
        z_raw = m[i, 2, 2] * zz + m[i, 2, 3]
        corrected = np.interp(z_raw, zz, raw)
        inner = slice(4, -4)
        assert np.abs(corrected[inner] - field(0.0, 0.0, zz)[inner]).max() < 0.05
