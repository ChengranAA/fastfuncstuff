"""Checks and conversions for UNWRAPPED phase handed to a resampling tool.

Interpolating wrapped phase averages across 2π jumps and invents values that were never
measured, so every tool that resamples phase (ffs_moco -phase, ffs_locomoco -phase)
refuses anything that does not look like an unwrapped field. The tests are phrased in
terms of the wrap PERIOD in the data's own units: 2π for radians (TE-free), ``1000/TE``
for a field in Hz.
"""

from __future__ import annotations

import numpy as np


def phase_to_hz(phase: np.ndarray, te_ms: float, units: str) -> np.ndarray:
    """Unwrapped phase in ``units`` ('hz' or 'rad') as a field in Hz."""
    if units == "hz":
        return phase.astype(np.float32, copy=False)
    if units == "rad":
        return (phase / (2.0 * np.pi * te_ms * 1e-3)).astype(np.float32, copy=False)
    raise ValueError(f"unknown phase units {units!r} (hz or rad)")


def wrap_period(units: str, te_ms: float | None) -> float:
    """One wrap in the data's units: 2π rad, or 1000/TE Hz (needs the echo time)."""
    if units == "rad":
        return 2.0 * np.pi
    if te_ms is None:
        raise ValueError("phase in Hz needs the echo time to know its wrap period (-echo_times).")
    return 1000.0 / te_ms


def phase_wrap_fraction(vol: np.ndarray, period: float, mask: np.ndarray) -> float:
    """Fraction of in-mask neighbour steps larger than half a wrap period.

    Only meaningful at good SNR: a late echo's unwrapped field is noisy enough that 10-20%
    of its steps exceed half of its small period, wrapped or not (measured, 7T, TE 59 ms).
    """
    half = 0.5 * period
    n_big = n_all = 0
    for ax in range(3):
        d = np.abs(np.diff(vol, axis=ax))
        both = np.logical_and(
            np.take(mask, range(mask.shape[ax] - 1), axis=ax),
            np.take(mask, range(1, mask.shape[ax]), axis=ax),
        )
        n_big += int((d[both] > half).sum())
        n_all += int(both.sum())
    return n_big / max(n_all, 1)


def phase_input_problem(
    vols: list[np.ndarray], periods: list[float], brain: np.ndarray
) -> str | None:
    """Why one frame per echo (shortest TE first) is not an unwrapped field, or None.

    Two checks, each on the echo where it is decisive (the same echo for single-echo):

    * the LONGEST echo has the smallest wrap period. A wrapped series lives inside one
      period, while a real field spans several there (tens of Hz across a brain at any
      field strength). Radians passed as Hz land here too.
    * the SHORTEST echo has the best SNR, so an unwrapped field is smooth there; raw
      scanner units (e.g. Siemens +-4096) step by more than half a period almost
      everywhere (64% measured vs 0.5% for the unwrapped field).
    """
    last = vols[-1][brain]
    lo, hi = np.percentile(last, [0.5, 99.5])
    if hi - lo <= 1.05 * periods[-1]:
        return (
            f"echo {len(vols)} spans only {hi - lo:.3g} inside the brain, and one wrap period "
            f"is {periods[-1]:.3g}: it looks WRAPPED (or in the wrong units)."
        )
    frac = phase_wrap_fraction(vols[0], periods[0], brain)
    if frac > 0.1:
        return (
            f"echo 1 is not a smooth field: {100 * frac:.0f}% of in-brain neighbour steps "
            "exceed half a wrap period. Raw scanner phase, or the wrong -phase_units?"
        )
    return None
