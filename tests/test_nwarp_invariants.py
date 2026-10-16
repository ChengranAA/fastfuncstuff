"""ffs_nwarp's pathways must agree with each other where they should coincide.

nwarpforge has several routes to an output -- the affine-only fast path, the
static shared-coordinate batch, the per-frame loop, joint slice timing (frozen or
tissue-following), four phase modes, -jac, time-varying warps from one 5-D file
or per-frame files, -dxyz, surface points.  Each is tested on its own elsewhere;
these check the invariants that tie them together, on one synthetic data set
with an oblique-free 2 mm grid, motion, a smooth nonlinear warp and an
interleaved slice order.  A pathway that drifts from the others fails here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

nib = pytest.importorskip("nibabel")

from fastfuncstuff.processing.nwarpforge import nwarpforge  # noqa: E402

NX, NY, NZ, T = 24, 22, 18, 8
VOX = 2.0
_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param("cuda", id="cuda", marks=[pytest.mark.gpu, _cuda]),
]


def _aff():
    a = np.diag([VOX, VOX, VOX, 1.0])
    a[:3, 3] = [-(NX - 1), -(NY - 1), -(NZ - 1)]
    return a


def _line(rot=(0, 0, 0), shift=(0, 0, 0)):
    r = Rotation.from_euler("xyz", rot, degrees=True).as_matrix()
    return [*r[0], shift[0], *r[1], shift[1], *r[2], shift[2]]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("nwinv")
    rng = np.random.default_rng(0)
    aff = _aff()

    def sm(shape, sigma, scale):
        return (gaussian_filter(rng.standard_normal(shape), sigma) * scale).astype(np.float32)

    def save(name, arr):
        nib.Nifti1Image(np.asarray(arr, np.float32), aff).to_filename(str(d / name))
        return str(d / name)

    def aff1d(name, rows):
        (d / name).write_text("\n".join(" ".join(f"{v:.8f}" for v in r) for r in rows) + "\n")
        return str(d / name)

    base = np.clip(100 + sm((NX, NY, NZ), 2.0, 300), 5, None)
    series = np.stack(
        [base * (1 + 0.02 * np.sin(t / 2)) + sm((NX, NY, NZ), 1.5, 3) for t in range(T)], -1
    )
    tv = np.stack([np.stack([sm((NX, NY, NZ), 3, 3) for _ in range(3)], -1) for _ in range(T)], 3)
    for t in range(T):
        save(f"tvf_{t:03d}.nii", tv[:, :, :, t, :])
    wpe = np.zeros((NX, NY, NZ, 3), np.float32)
    wpe[..., 1] = sm((NX, NY, NZ), 3, 12)
    order = list(range(0, NZ, 2)) + list(range(1, NZ, 2))
    st = [0.0] * NZ
    for k, z in enumerate(order):
        st[z] = k * 2.0 / NZ
    return {
        "dir": d,
        "series": series,
        "src": save("src.nii", series),
        "src_const": save("src_const.nii", np.repeat(base[..., None], T, -1)),
        "ph_const": save("ph_const.nii", np.full((NX, NY, NZ, T), 0.7, np.float32)),
        "a1": aff1d("a1.aff12.1D", [_line((2, -1.5, 1), (1.2, -0.8, 0.5))]),
        "id": aff1d("id.aff12.1D", [_line()]),
        "mot": aff1d(
            "mot.aff12.1D",
            [
                _line((0.3 * np.sin(t), 0.2 * t / T, 0), (0.6 * np.cos(t), 0.1 * t, -0.3))
                for t in range(T)
            ],
        ),
        "mot_id": aff1d("mot_id.aff12.1D", [_line() for _ in range(T)]),
        "w1": save("w1.nii", np.stack([sm((NX, NY, NZ), 3, 10) for _ in range(3)], -1)),
        "w0": save("w0.nii", np.zeros((NX, NY, NZ, 3), np.float32)),
        "wpe": save("wpe.nii", wpe),
        "tv5": save("tv5.nii", tv),
        "tvglob": str(d / "tvf_*.nii"),
        "st": dict(slice_times=st, tr=2.0, tzero=0.0),
    }


def _run(data, device, specs, src=None, **kw):
    d = data["dir"]
    out = d / f"o_{len(list(d.glob('o_*')))}.nii"
    ph_out = None
    if kw.get("phase_path"):
        ph_out = str(out).replace(".nii", "_ph.nii")
        kw.setdefault("phase_prefix", ph_out)
        kw.setdefault("phase_units", "rad")
    kw.setdefault("interp", "linear")
    kw.setdefault("auto_pad", False)
    nwarpforge(
        source_path=src or data["src"],
        nwarp_specs=specs,
        prefix=str(out),
        device=torch.device(device),
        verb=0,
        **kw,
    )
    mag = np.asarray(nib.load(str(out)).dataobj, np.float64)
    ph = np.asarray(nib.load(ph_out).dataobj, np.float64) if ph_out else None
    return mag, ph


def _rel(a, b, mask=None):
    d = np.abs(a - b) if mask is None else np.abs(a - b)[mask]
    return float(d.max() / max(np.abs(b).max(), 1e-12))


def _inner():
    m = np.zeros((NX, NY, NZ), bool)
    m[5:-5, 5:-5, 4:-4] = True
    return m


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic", "wsinc5"])
def test_identity_returns_the_source(data, device, mode):
    m, _ = _run(data, device, [data["id"]], interp=mode)
    assert _rel(m, data["series"]) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_affine_fast_path_equals_the_field_path(data, device):
    fast, _ = _run(data, device, [data["a1"], data["mot"]], interp="wsinc5")
    field, _ = _run(data, device, [data["a1"], data["mot"], data["w0"]], interp="wsinc5")
    assert _rel(fast, field, _inner()) < 2e-5


@pytest.mark.parametrize("device", DEVICES)
def test_static_batch_equals_the_per_frame_loop(data, device):
    batched, _ = _run(data, device, [data["a1"], data["w1"]], interp="wsinc5")
    perframe, _ = _run(data, device, [data["a1"], data["w1"], data["mot_id"]], interp="wsinc5")
    assert _rel(batched, perframe) < 2e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("slice_timing", [False, True], ids=["plain", "slice_timing"])
@pytest.mark.parametrize("phase_warp", ["complex", "split", "direct", "circular"])
def test_constant_phase_stays_constant_and_magnitude_matches(
    data, device, slice_timing, phase_warp
):
    kw = data["st"] if slice_timing else {}
    chain = [data["a1"], data["w1"], data["mot"]]
    ref, _ = _run(data, device, chain, interp="wsinc5", no_neg=True, **kw)
    m, p = _run(
        data,
        device,
        chain,
        interp="wsinc5",
        no_neg=True,
        phase_path=data["ph_const"],
        phase_warp=phase_warp,
        **kw,
    )
    covered = ref > 1.0
    assert float(np.abs(p - 0.7)[covered].max()) < 1e-4
    # "complex" derives the magnitude from warped real/imag, so it is |m| only to
    # within the kernel's overshoot; the others carry the magnitude as is.
    assert _rel(m, ref, covered) < (5e-3 if phase_warp == "complex" else 2e-5)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("follow", [True, False], ids=["follow", "frozen"])
def test_slice_timing_of_a_constant_series_is_the_identity_in_time(data, device, follow):
    chain = [data["a1"], data["w1"], data["mot_id"]]
    plain, _ = _run(data, device, chain, src=data["src_const"], interp="wsinc5")
    st, _ = _run(
        data,
        device,
        chain,
        src=data["src_const"],
        interp="wsinc5",
        follow_tissue=follow,
        **data["st"],
    )
    assert _rel(st, plain, _inner()) < 2e-4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("follow", [True, False], ids=["follow", "frozen"])
def test_slice_timing_covers_exactly_the_voxels_the_plain_path_does(data, device, follow):
    chain = [data["a1"], data["w1"], data["mot"]]
    plain, _ = _run(data, device, chain, interp="wsinc5")
    st, _ = _run(data, device, chain, interp="wsinc5", follow_tissue=follow, **data["st"])
    assert np.array_equal(st != 0, plain != 0)


@pytest.mark.parametrize("device", DEVICES)
def test_jac_factor_is_the_same_with_and_without_slice_timing(data, device):
    chain = [data["wpe"], data["mot"]]
    jac = dict(jac_axis=1, jac_match="wpe.nii")
    j_st, _ = _run(data, device, chain, **jac, **data["st"])
    n_st, _ = _run(data, device, chain, **data["st"])
    j_pf, _ = _run(data, device, chain, **jac)
    n_pf, _ = _run(data, device, chain)
    inner = _inner()[..., None] & (n_st > 1) & (n_pf > 1)
    r_st, r_pf = j_st[inner] / n_st[inner], j_pf[inner] / n_pf[inner]
    assert float(np.abs(r_st - r_pf).max()) < 1e-4
    assert float(np.abs(r_pf - 1).max()) > 1e-2  # and it is not trivially 1


@pytest.mark.parametrize("device", DEVICES)
def test_a_time_varying_warp_reads_the_same_from_5d_and_per_frame_files(data, device):
    a, _ = _run(data, device, [data["a1"], data["tv5"], data["mot"]], interp="cubic")
    b, _ = _run(data, device, [data["a1"], data["tvglob"], data["mot"]], interp="cubic")
    assert _rel(a, b) < 1e-6


@pytest.mark.parametrize("device", DEVICES)
def test_surface_points_equal_the_volume_at_the_same_voxels(data, device):
    chain = [data["a1"], data["w1"], data["mot"]]
    vol, _ = _run(data, device, chain, interp="cubic")
    ijk = np.stack(
        np.meshgrid(np.arange(4, 20, 4), np.arange(4, 18, 4), np.arange(3, 15, 4), indexing="ij"),
        -1,
    ).reshape(-1, 3)
    mm = (_aff() @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T
    pts = nwarpforge(
        source_path=data["src"],
        nwarp_specs=chain,
        prefix="",
        device=torch.device(device),
        verb=0,
        interp="cubic",
        points=mm,
        master_path=data["src"],
        auto_pad=False,
    )
    want = vol[ijk[:, 0], ijk[:, 1], ijk[:, 2], :].T
    assert _rel(np.asarray(pts.cpu()).reshape(want.shape), want) < 1e-5
