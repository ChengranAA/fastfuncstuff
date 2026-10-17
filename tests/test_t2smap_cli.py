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
