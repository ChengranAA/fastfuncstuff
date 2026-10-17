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
