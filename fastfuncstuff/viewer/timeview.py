"""A time series viewed detrended, or as percent signal change, frame by frame.

For watching activity move across a slice or a surface as time is stepped: raw
BOLD is dominated by each voxel's mean and the scanner's drift, so a frame of
it shows the anatomy, not the signal. Display only -- nothing is written and
DERIVE remains the way to keep a cleaned copy.

Stepping has to stay fast, so the expensive part is done once per (layer,
order): the drift coefficients of every voxel against the viewer's orthonormal
Legendre basis (:func:`viewer.derive.drift_basis`), ``(V, r)``. A frame is then
its raw values minus one small matrix-vector product, and percent signal change
one division more -- milliseconds even for a whole-brain grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TimeView", "build_time_view"]

#: Voxels whose mean is below this fraction of the 99th-percentile |mean| are
#: outside the head for PSC's purposes: dividing by them paints noise.
PSC_FLOOR = 1e-3


@dataclass
class TimeView:
    #: Drift order removed, -1 for none.
    order: int
    psc: bool
    #: ``(T, r)`` orthonormal basis and ``(V, r)`` coefficients; None without detrend.
    basis: np.ndarray | None
    coef: np.ndarray | None
    #: ``(V,)`` mean of each series, what PSC is relative to.
    mean: np.ndarray
    #: ``(V,)`` where PSC is defined.
    valid: np.ndarray

    @property
    def identity(self) -> bool:
        return self.order < 0 and not self.psc

    def frame(self, values: np.ndarray, t: int) -> np.ndarray:
        """Frame ``t`` (raw ``values``, any shape over V) as viewed."""
        if self.identity:
            return values
        shape = np.shape(values)
        x = np.asarray(values, np.float32).reshape(-1)
        if self.coef is not None and self.basis is not None:
            # The basis holds the constant, so this is centred on zero.
            out = x - self.coef @ self.basis[int(t)]
        else:
            out = x - self.mean if self.psc else x.copy()
        if self.psc:
            out = np.where(self.valid, 100.0 * out / np.where(self.valid, self.mean, 1.0), 0.0)
        return out.astype(np.float32).reshape(shape)


def build_time_view(series: np.ndarray, order: int, psc: bool) -> TimeView:
    """The view of ``series`` (``(..., T)``, any leading shape) for ``order`` and ``psc``."""
    from fastfuncstuff.viewer.derive import drift_basis

    s = np.asarray(series)
    n_time = s.shape[-1]
    flat = s.reshape(-1, n_time)
    mean = flat.mean(axis=1, dtype=np.float64).astype(np.float32)
    big = float(np.percentile(np.abs(mean), 99)) if mean.size else 0.0
    valid = np.abs(mean) > PSC_FLOOR * big if big > 0 else np.zeros(mean.shape, bool)
    basis = coef = None
    if order >= 0 and n_time > 1:
        basis = np.asarray(drift_basis(int(n_time), int(order)), np.float32)
        # Chunked so a whole-brain run never needs a second float copy of itself.
        coef = np.empty((flat.shape[0], basis.shape[1]), np.float32)
        step = max(1, (1 << 26) // max(n_time, 1))
        for i in range(0, flat.shape[0], step):
            coef[i : i + step] = np.asarray(flat[i : i + step], np.float32) @ basis
    return TimeView(int(order), bool(psc), basis, coef, mean, valid)
