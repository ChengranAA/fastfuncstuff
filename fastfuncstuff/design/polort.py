"""Polynomial drift degree (``-polort``): the automatic rule and per-run specs.

A degree spec is either one int for every run (an explicit ``-polort N``) or a
list with one degree per run (the automatic rule, ``1 + floor(D / 150)`` from
each run's own duration ``D``). Every block-diagonal polynomial builder takes
either form through ``polort_per_run``. numpy-only so the CLI header path can
import it cheaply.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def auto_polort(
    run_duration_sec: float,
    formula: str = "afni",
) -> int:
    """
    Auto-determine polynomial order from run duration.

    Parameters
    ----------
    run_duration_sec : float
        Duration of a single run in seconds
    formula : str, default="afni"
        Formula to use:
        - "afni": polort = 1 + floor(duration / 150) (AFNI/GLMdenoise convention)
        - "conservative": polort = max(1, round(duration / 120))

    Returns
    -------
    int
        Recommended polynomial order

    Notes
    -----
    The AFNI formula (1 + floor(duration/150)) is the standard convention:
    - 150s run → polort 2
    - 300s run → polort 3
    - 450s run → polort 4

    This provides one polynomial coefficient per ~2.5 minutes of data.
    """
    if formula == "afni":
        return int(1 + np.floor(run_duration_sec / 150.0))
    elif formula == "conservative":
        return max(1, int(round(run_duration_sec / 120.0)))
    else:
        raise ValueError(f"Unknown polort formula: {formula}")


def auto_polort_per_run(
    run_lengths: Sequence[int],
    tr: float,
    formula: str = "afni",
) -> list[int]:
    """Automatic drift degree for each run from that run's own duration.

    The one rule every tool uses when ``-polort`` is not given: each run gets
    ``auto_polort(n_tp * tr)`` in its own block-diagonal polynomial block. A
    single degree from the average, longest or shortest run under-fits the
    long runs or over-fits the short ones whenever run lengths differ.
    """
    return [auto_polort(int(n) * float(tr), formula=formula) for n in run_lengths]


def polort_per_run(polort: int | Sequence[int], n_runs: int) -> list[int]:
    """Expand a polort spec to one degree per run.

    ``polort`` is either one degree for every run (an explicit ``-polort N``)
    or an already per-run list (the automatic rule, or a stored sidecar).
    """
    if isinstance(polort, (int, np.integer)):
        return [int(polort)] * n_runs
    degrees = [int(p) for p in polort]
    if len(degrees) != n_runs:
        raise ValueError(f"per-run polort has {len(degrees)} entries for {n_runs} runs")
    return degrees


def n_polort_columns(polort: int | Sequence[int], n_runs: int) -> int:
    """Total drift columns of the block-diagonal polynomial nuisance."""
    return sum(p + 1 for p in polort_per_run(polort, n_runs) if p >= 0)


def polort_column_labels(
    polort: int | Sequence[int],
    n_runs: int,
    style: str = "afni",
) -> list[str]:
    """Labels of the drift columns, run-major (``Run#1Pol#0`` ... or ``r01_poly0`` ...)."""
    labels: list[str] = []
    for run_idx, p in enumerate(polort_per_run(polort, n_runs)):
        for k in range(p + 1):
            if style == "afni":
                labels.append(f"Run#{run_idx + 1}Pol#{k}")
            else:
                labels.append(f"r{run_idx + 1:02d}_poly{k}")
    return labels


def polort_summary(polort: int | Sequence[int]) -> str:
    """A degree spec as text: ``3`` or ``[1, 3, 3]``."""
    if isinstance(polort, (int, np.integer)):
        return str(int(polort))
    return "[" + ", ".join(str(int(p)) for p in polort) + "]"


def resolve_polort(
    polort_arg: int | None,
    run_lengths: Sequence[int],
    tr: float,
    verbose: bool = True,
    indent: str = "  ",
) -> int | list[int]:
    """``-polort`` as given, or the automatic per-run degrees when it is not.

    An explicit degree stays a single int for every run; the automatic rule
    returns a list, one degree per run. Logs the choice once when verbose.
    """
    if polort_arg is not None:
        if verbose:
            print(f"{indent}Polort: {int(polort_arg)}")
        return int(polort_arg)
    degrees = auto_polort_per_run(run_lengths, tr)
    if verbose:
        print(f"{indent}Auto polort per run: {polort_summary(degrees)}")
    return degrees


def max_polort(polort: int | Sequence[int]) -> int:
    """Largest per-run degree (for code that needs one bound, e.g. rank checks)."""
    if isinstance(polort, (int, np.integer)):
        return int(polort)
    return max(int(p) for p in polort) if len(polort) else -1
