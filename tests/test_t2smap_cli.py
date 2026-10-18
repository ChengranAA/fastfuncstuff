"""ffs_t2smap CLI: the -optcom copy is the optimal combination itself."""

import numpy as np
import torch

from fastfuncstuff.cli.t2smap import main
from fastfuncstuff.processing.io import load_image, save_image


def test_optcom_path_holds_the_optimal_combination(tmp_path):
    rng = np.random.default_rng(0)
    nt, shape = 6, (8, 10, 12)
    z, y, x = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    head = (x**2 + y**2 + z**2) < 0.8
    s0 = np.where(head, 1000.0, 5.0)
    t2s = np.where(head, 30.0, 10.0)  # ms
    tes = [10.0, 25.0, 40.0]
    paths = []
    for i, te in enumerate(tes):
        vol = s0 * np.exp(-te / t2s)
        series = vol[None] * (1 + 0.01 * rng.standard_normal((nt, *shape)))
        p = tmp_path / f"e{i + 1}.nii.gz"
        save_image(torch.from_numpy(series.astype(np.float32)), str(p), header_info=None)
        paths.append(str(p))
    main(
        ["-input", *paths, "-tes", *map(str, tes), "-prefix", str(tmp_path / "t2s_"),
         "-automask", "-no_qc", "-optcom", str(tmp_path / "combined.nii.gz"), "-device", "cpu", "-verb", "0"]
    )  # fmt: skip
    a, _ = load_image(str(tmp_path / "combined.nii.gz"))
    b, _ = load_image(str(tmp_path / "t2s_desc-optcom_bold.nii.gz"))
    assert a.shape == b.shape == (nt, *shape)
    assert torch.allclose(a, b)


def test_surface_series_combine_exactly_like_the_same_samples_in_a_volume(tmp_path):
    """GIfTI inputs (autoproc's per-echo surface reads) take the per-sample volume path:
    the combination of a vertex equals that of a voxel holding the same samples."""
    from fastfuncstuff.io.gifti import load_gifti_data, save_gifti_data

    rng = np.random.default_rng(1)
    nt, shape, tes = 5, (3, 4, 5), [10.0, 25.0, 40.0]
    t2s = rng.uniform(20, 40, shape)
    vols, gii = [], []
    for i, te in enumerate(tes):
        series = (1000 * np.exp(-te / t2s))[None] * (1 + 0.02 * rng.standard_normal((nt, *shape)))
        series = series.astype(np.float32)
        v, g = tmp_path / f"e{i}.nii.gz", tmp_path / f"e{i}.func.gii"
        save_image(torch.from_numpy(series), str(v), header_info=None)
        save_gifti_data(g, series.reshape(nt, -1).T)
        vols.append(str(v))
        gii.append(str(g))
    common = ["-tes", *map(str, tes), "-no_qc", "-device", "cpu", "-verb", "0"]
    main(["-input", *vols, "-prefix", str(tmp_path / "v_"), *common])
    main(["-input", *gii, "-prefix", str(tmp_path / "s_"), *common,
          "-optcom", str(tmp_path / "s.func.gii")])  # fmt: skip
    vol, _ = load_image(str(tmp_path / "v_desc-optcom_bold.nii.gz"))
    surf, _ = load_gifti_data(tmp_path / "s.func.gii")
    assert surf.shape == (int(np.prod(shape)), nt)
    assert np.allclose(surf, vol.numpy().reshape(nt, -1).T, rtol=1e-5)
    t2v, _ = load_image(str(tmp_path / "v_T2starmap.nii.gz"))
    t2g, _ = load_gifti_data(tmp_path / "s_T2starmap.func.gii")
    assert np.allclose(t2g, t2v.numpy().reshape(-1), rtol=1e-5)


def _field_scene(seed=3, v=400, nt=60, tes=(8.0, 22.0, 36.0, 50.0)):
    """Voxels sharing a known frequency-shift series; echo e's phase is
    2 pi df TE + a static per-voxel offset + noise ~ 1/|S_e| (|S_e| decays with TE)."""
    rng = np.random.default_rng(seed)
    te = torch.tensor(tes)
    df = torch.from_numpy(rng.normal(0, 3.0, (v, nt)).astype(np.float32))  # Hz
    t2s = torch.from_numpy(rng.uniform(20, 40, (v, 1)).astype(np.float32))
    mag = 800 * torch.exp(-te[None, :, None] / t2s[:, :, None]).expand(v, len(tes), nt)
    offset = torch.from_numpy(rng.uniform(-3, 3, (v, 1, 1)).astype(np.float32))
    noise = torch.from_numpy(rng.standard_normal((v, len(tes), nt)).astype(np.float32))
    phase = 2 * torch.pi * df[:, None, :] * te[None, :, None] / 1000 + offset + 40.0 * noise / mag
    return df, phase, mag, te


def test_phase_field_beats_every_single_echo_and_ignores_a_static_offset():
    from fastfuncstuff.processing.multiecho import phase_field_hz

    df, phase, mag, te = _field_scene()
    est = phase_field_hz(phase, mag, te)
    assert est.shape == df.shape

    def err(x):  # time-demeaned, as any regression sees it
        return (
            ((x - x.mean(1, keepdim=True)) - (df - df.mean(1, keepdim=True))).pow(2).mean().sqrt()
        )

    singles = [err(phase[:, e] * 1000 / (2 * torch.pi * te[e])) for e in range(len(te))]
    assert float(err(est)) < 0.7 * min(float(s) for s in singles)

    # no noise, no offset: exact
    clean = 2 * torch.pi * df[:, None, :] * te[None, :, None] / 1000
    assert torch.allclose(phase_field_hz(clean, mag, te), df, atol=1e-4)


def test_phase_field_uses_the_echoes_it_has():
    from fastfuncstuff.processing.multiecho import phase_field_hz

    df, _p, mag, te = _field_scene()
    clean = 2 * torch.pi * df[:, None, :] * te[None, :, None] / 1000
    holed = clean.clone()
    holed[:10, 1:, :] = 0.0  # only echo 1 left in the first ten voxels
    holed[10:20, :, 5] = float("nan")  # no echo at frame 5 in the next ten
    est = phase_field_hz(holed, mag, te)
    assert torch.allclose(est[:10], df[:10], atol=1e-4)
    assert torch.allclose(est[10:20, 5], torch.zeros(10))
    assert torch.allclose(est[10:20, :5], df[10:20, :5], atol=1e-4)


def test_cli_field_hz_matches_the_library_and_needs_every_echo(tmp_path):
    from fastfuncstuff.processing.multiecho import phase_field_hz

    rng = np.random.default_rng(1)
    nt, shape = 6, (6, 8, 10)
    tes = [10.0, 25.0, 40.0]
    mags, phs = [], []
    df = rng.normal(0, 2.0, (nt, *shape)).astype(np.float32)
    for i, te in enumerate(tes):
        m = (500 * np.exp(-te / 30.0) * (1 + 0.01 * rng.standard_normal((nt, *shape)))).astype(
            np.float32
        )
        p = (2 * np.pi * df * te / 1000 + 0.02 * rng.standard_normal(df.shape)).astype(np.float32)
        for arr, tag, lst in ((m, "m", mags), (p, "p", phs)):
            f = tmp_path / f"{tag}{i + 1}.nii.gz"
            save_image(torch.from_numpy(arr), str(f), header_info=None)
            lst.append(str(f))
    out = str(tmp_path / "field.nii.gz")
    main(["-input", *mags, "-tes", *map(str, tes), "-prefix", str(tmp_path / "t_"), "-no_qc",
          "-phase", *phs, "-field_hz", out, "-device", "cpu", "-verb", "0"])  # fmt: skip
    got, _ = load_image(out)
    assert tuple(got.shape) == (nt, *shape)
    m = torch.stack([load_image(f)[0] for f in mags])  # (E, T, z, y, x)
    p = torch.stack([load_image(f)[0] for f in phs])
    # (E,T,z,y,x) -> (V,E,T) in the same voxel order, over the whole (unmasked) grid
    want = phase_field_hz(
        p.permute(2, 3, 4, 0, 1).reshape(-1, len(tes), nt), m.permute(2, 3, 4, 0, 1).reshape(-1, len(tes), nt),
        torch.tensor(tes),
    )  # fmt: skip
    g = got.permute(1, 2, 3, 0).reshape(-1, nt)
    assert torch.allclose(g, want, atol=1e-3)
    try:
        main(["-input", *mags, "-tes", *map(str, tes), "-prefix", str(tmp_path / "u_"), "-no_qc",
              "-phase", *phs[:2], "-field_hz", out, "-device", "cpu", "-verb", "0"])  # fmt: skip
    except SystemExit as exc:
        assert "2 -phase" in str(exc)
    else:
        raise AssertionError("a missing echo's phase must be refused")


def test_field_fit_in_chunks_equals_one_pass(monkeypatch):
    """The streamed fit (phase on the host, voxel chunks on the device) must not depend
    on the chunk size: a budget small enough for a handful of voxels gives the same field."""
    import fastfuncstuff.memory as memory
    from fastfuncstuff.cli.t2smap import _field_in_chunks
    from fastfuncstuff.processing.multiecho import phase_field_hz

    df, phase, mag, te = _field_scene(v=150, nt=20)
    # lay the (V, E, T) scene on a (E, T, 5, 6, 5) grid with every voxel inside the mask
    grid = (5, 6, 5)
    ph = phase.permute(1, 2, 0).reshape(len(te), 20, *grid)
    mask = torch.ones(grid, dtype=torch.bool)
    full = phase_field_hz(phase, mag, te)
    monkeypatch.setattr(
        memory, "get_available_memory", lambda *a, **k: 6 * len(te) * 20 * 4 * 7 * 4
    )
    chunked = _field_in_chunks(ph, mask, mag, te, torch.device("cpu"))
    assert torch.allclose(chunked, full, atol=1e-5)
