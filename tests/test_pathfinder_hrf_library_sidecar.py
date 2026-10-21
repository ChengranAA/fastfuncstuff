"""ffs_pathfinder's index map must come with the library it numbers into.

It writes ``{prefix}_hrf_index`` under the name every -hrf_opt consumer reads,
but never wrote ``{prefix}_hrf_library.pt``, so those consumers fitted its
indices against the stock double-gamma library -- wrong curves outright under
``-hrf_mode pighs``, which has the same 20-curve size.
"""

import numpy as np
import torch

from fastfuncstuff.cli.pathfinder import PathfinderResults, save_pathfinder_results
from fastfuncstuff.cli_utils import load_hrfopt_library


def test_saved_index_map_loads_against_its_own_library(tmp_path):
    lib = torch.rand(9, 50)
    results = PathfinderResults(
        hrf_index=torch.tensor([0, 3, 8, 2, 1, 0, 5, 6]),
        xval_r2_best=torch.rand(8),
        hrf_library=lib,
        metadata={"microtime_dt": 0.1},
    )
    save_pathfinder_results(results, str(tmp_path / "pf"), (2, 2, 2), np.eye(4))
    loaded, durations = load_hrfopt_library(
        str(tmp_path / "pf"), torch.tensor([8]), microtime_dt=0.1, model_durations=[2.0]
    )
    torch.testing.assert_close(loaded, lib)
    assert durations == [2.0]  # impulse library: the design keeps its boxcars
