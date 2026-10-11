"""Statistical utilities for neuroimaging analysis.

The names below resolve on first use rather than at import: ``fdr`` pulls in
scipy.stats (0.9 s), and every ``fastfuncstuff.stats.<module>`` import runs this
file first -- so an eager import charged the viewer's blur that much for FDR.
"""

from importlib import import_module
from typing import Any

_HOMES = {
    "add_fdrcurves_to_nifti": ".fdr",
    "compute_fdr_curve": ".fdr",
    "fdr_qvalues": ".fdr",
    "stat_to_pvalue": ".fdr",
    "consistency_report": ".spatial",
    "optimal_matching": ".spatial",
    "spatial_correlation": ".spatial",
    "spatial_correlation_matrix": ".spatial",
}

__all__ = sorted(_HOMES)


def __getattr__(name: str) -> Any:
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(home, __name__), name)
