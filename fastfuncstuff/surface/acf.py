"""Spatial autocorrelation on a surface, and blurring data TO a smoothness.

**The ACF** is measured as a function of geodesic distance and fitted to AFNI's mixed
model ``a exp(-r^2 / 2b^2) + (1 - a) exp(-r / c)``, as SUMA's ``SurfFWHM`` does
(``SUMA_SurfACF.h``), with the fit shared with the volume path (``stats.fwhmx``).
A FWHM describes the ACF near the origin; cluster inference depends on the far tail,
and real data have a heavier tail than a FWHM implies (Eklund et al. 2016). On onavg
in a real subject, a 4 mm blur took the 1-difference FWHM from 4.0 to 8.0 mm where
the kernel alone predicts ~5.7. That broad component is what the tail term is for.

Two choices of ours: correlation is taken **over time between vertices** (standardised
residual time series) rather than over space within each frame -- the same quantity
for a stationary field, and here one matrix product per block of centre vertices,
which a GPU does at once -- and distance is Dijkstra along mesh edges, as SUMA's
``SUMA_getoffsets2``.

**Blurring to a smoothness** (SurfSmooth ``-target_fwhm``, 3dBlurToFWHM): find the heat
kernel that brings a master (detrended data, or better the residuals of a fit) to the
target FWHM, then smooth the data with exactly that kernel. The search itself, and the
residual master, are geometry-free and live in :mod:`fastfuncstuff.stats.blur_to`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from .smooth import HeatSmoother, mesh_edges, surface_fwhm

__all__ = [
    "SurfaceACF",
    "SurfaceAdaptiveDiffusion",
    "blur_to_fwhm",
    "blur_to_fwhm_local",
    "detrend",
    "fit_surface_acf",
    "local_surface_fwhm",
    "surface_acf",
]


@dataclass
class SurfaceACF:
    """A measured ACF curve and its mixed-model fit (AFNI's a, b, c and FWHM)."""

    r: np.ndarray  # (B,) bin centres, mm
    curve: np.ndarray  # (B,) mean correlation, NaN where too few pairs
    a: float
    b: float
    c: float
    fwhm: float


def _edge_graph(
    vertices: np.ndarray, faces: np.ndarray, mask: np.ndarray, rings: int = 2
) -> sparse.csr_matrix:
    """Distance graph for Dijkstra: mesh edges plus chords to ``rings``-ring neighbours.

    Paths along edges zigzag: on a triangulated sheet they run 8% long at 10 mm
    (SUMA's graph distances share this), which stretches a measured ACF outward. Chords
    to second neighbours cut that to 1-2.5% (third: under 1%), and stay on the local
    sheet because they only join vertices a couple of edges apart. More rings on a
    coarse mesh would start to cut across tight folds.
    """
    e = mesh_edges(faces)
    e = e[mask[e[:, 0]] & mask[e[:, 1]]]
    n = len(vertices)
    adj = sparse.csr_matrix(
        (np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), (n, n)
    )
    reach = adj.copy()
    for _ in range(max(rings, 1) - 1):
        reach = reach + reach @ adj
    reach = sparse.triu(reach, 1).tocoo()
    w = np.linalg.norm(vertices[reach.row] - vertices[reach.col], axis=1)
    return sparse.csr_matrix(
        (np.r_[w, w], (np.r_[reach.row, reach.col], np.r_[reach.col, reach.row])), (n, n)
    )


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = x - x.mean(axis=1, keepdims=True)
    sd = z.std(axis=1)
    ok = sd > 0
    z[ok] /= sd[ok, None]
    z[~ok] = 0.0
    return z.astype(np.float32), ok


def surface_acf(
    residuals: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray | None = None,
    radius: float | None = None,
    dr: float | None = None,
    n_centres: int = 1500,
    device=None,
    seed: int = 0,
) -> SurfaceACF:
    """Mean correlation of ``(V, T)`` residuals against geodesic distance, and its fit.

    Defaults follow SUMA: bins one mean edge length wide, out to 20 of them but at
    least 30 mm (a tail several mm long is still at 0.4 after 10 mm, and the fit needs
    to see it fall). ~``n_centres`` centre vertices are enough: every one is correlated
    against every vertex within ``radius``.
    """
    import torch

    from fastfuncstuff.stats.fwhmx import fit_acf_curve

    v = np.asarray(vertices, np.float64)
    m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
    z, ok = _standardise(np.asarray(residuals, np.float64))
    m = m & ok
    graph = _edge_graph(v, faces, m)
    e = mesh_edges(faces)
    lengths = np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)
    dr = float(dr or (lengths.mean() if lengths.size else 1.0))
    radius = float(radius or max(20 * dr, 30.0))
    nb = int(radius / dr) + 1
    rng = np.random.default_rng(seed)
    inside = np.flatnonzero(m)
    centres = np.sort(rng.choice(inside, size=min(n_centres, inside.size), replace=False))

    from scipy.sparse.csgraph import dijkstra

    dev = torch.device("cpu") if device is None else device
    zt = torch.as_tensor(z, device=dev)
    sums = np.zeros(nb)
    counts = np.zeros(nb)
    t = z.shape[1]
    for start in range(0, centres.size, 128):
        block = centres[start : start + 128]
        dist = dijkstra(graph, indices=block, limit=radius)  # (C, V), inf beyond
        corr = (zt[torch.as_tensor(block, device=dev)] @ zt.T / t).cpu().numpy()
        near = np.isfinite(dist) & (dist > 0)
        near[:, ~m] = False
        ib = np.rint(dist[near] / dr).astype(np.int64)
        keep = ib < nb
        np.add.at(sums, ib[keep], corr[near][keep])
        np.add.at(counts, ib[keep], 1.0)
    curve = np.full(nb, np.nan)
    use = counts > 5  # AFNI's guard against thinly populated bins
    curve[use] = sums[use] / counts[use]
    curve[0] = 1.0
    r = np.arange(nb) * dr
    good = np.isfinite(curve)
    a, b, c, fwhm = fit_acf_curve(
        torch.as_tensor(r[good]), torch.as_tensor(curve[good]), torch.device("cpu")
    )
    return SurfaceACF(r, curve, a, b, c, fwhm)


def fit_surface_acf(r: np.ndarray, curve: np.ndarray) -> tuple[float, float, float, float]:
    """``(a, b, c, FWHM)`` of a curve, NaN bins dropped (the shared volume fitter)."""
    import torch

    from fastfuncstuff.stats.fwhmx import fit_acf_curve

    good = np.isfinite(curve)
    return fit_acf_curve(
        torch.as_tensor(np.asarray(r)[good]),
        torch.as_tensor(np.asarray(curve)[good]),
        torch.device("cpu"),
    )


def detrend(data: np.ndarray, run_lengths: list[int] | None = None, degree: int = 3) -> np.ndarray:
    """``(V, T)`` minus a per-run Legendre polynomial fit: a master for :func:`blur_to_fwhm`
    when no residuals are at hand. Task signal stays in; residuals are the better master."""
    x = np.asarray(data, np.float64)
    lengths = run_lengths or [x.shape[1]]
    out = np.empty_like(x)
    start = 0
    for n in lengths:
        tt = np.linspace(-1.0, 1.0, n)
        basis = np.polynomial.legendre.legvander(tt, min(degree, max(n - 1, 0)))
        seg = x[:, start : start + n]
        coef, *_ = np.linalg.lstsq(basis, seg.T, rcond=None)
        out[:, start : start + n] = seg - (basis @ coef).T
        start += n
    return out


def blur_to_fwhm(
    master: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    target: float,
    mask: np.ndarray | None = None,
    n_columns: int = 40,
    tol: float = 0.02,
    device=None,
    seed: int = 0,
) -> tuple[HeatSmoother | None, float, float]:
    """The heat smoother that brings ``master`` to ``target`` FWHM (1-difference estimate).

    Returns ``(smoother, kernel_fwhm, achieved)``; ``smoother`` is None when the master
    is already at least that smooth. A master wider than ``n_columns`` is cut to a
    random ``n_columns`` of its time points (``stats.blur_to.sample_columns`` makes a
    better, run-stratified sample when the run structure is known).
    """
    from fastfuncstuff.stats.blur_to import bisect_kernel

    x = np.asarray(master)
    if x.shape[1] > n_columns:
        cols = np.random.default_rng(seed).choice(x.shape[1], size=n_columns, replace=False)
        x = x[:, np.sort(cols)]

    def measure(kernel: float) -> tuple[float, HeatSmoother | None]:
        if kernel <= 0:
            return surface_fwhm(x, vertices, faces, mask)[0], None
        hs = HeatSmoother(vertices, faces, kernel, mask=mask, device=device)
        return surface_fwhm(hs(x), vertices, faces, mask)[0], hs

    return bisect_kernel(measure, target, tol)


def local_surface_fwhm(
    x: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    mask: np.ndarray | None,
    radius_mm: float,
    device=None,
) -> np.ndarray:
    """Per-vertex FWHM (mm, NaN outside ``mask``): :func:`surface_fwhm`'s edge fit,
    pooled over a heat-kernel neighbourhood of ``radius_mm`` (FWHM) along the cortex."""
    from .smooth import _LN2, surface_edge_sums

    _, num, den = surface_edge_sums(x, vertices, faces, mask)
    pooled = HeatSmoother(vertices, faces, radius_mm, mask=mask, device=device)(
        np.c_[num, den].astype(np.float64)
    )
    pooled = np.asarray(pooled, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        per = 1.0 / np.sqrt(pooled[:, 0] / (2 * _LN2 * pooled[:, 1]))
    per[~(pooled[:, 1] > 0) | ~(pooled[:, 0] > 0)] = np.nan
    if mask is not None:
        per[~np.asarray(mask, bool)] = np.nan
    return per


class SurfaceAdaptiveDiffusion:
    """Recorded implicit heat steps with a per-vertex diffusivity on a mesh: each step
    is ``(M + dt L_d) u' = M u`` with every cotan edge weight scaled by whether both
    its ends still diffuse. Replaying applies exactly what the search measured; each
    step's factorisation (CPU) or operator (GPU, Jacobi CG) is kept for every run."""

    def __init__(self, vertices, faces, mask=None, device=None, tol: float = 1e-6):
        from .mesh import vertex_areas
        from .smooth import cotan_laplacian

        n = np.asarray(vertices).shape[0]
        self.mask = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
        self._idx = np.flatnonzero(self.mask)
        lap = cotan_laplacian(vertices, faces, self.mask)[self._idx][:, self._idx].tocoo()
        off = lap.row != lap.col
        self._r, self._c, self._w = lap.row[off], lap.col[off], -lap.data[off]
        mass = vertex_areas(np.asarray(vertices, np.float64), np.asarray(faces, np.int64))
        self._mass = np.maximum(mass[self._idx], 1e-12)
        self.device = device
        self.tol = tol
        self.steps: list[tuple[float, np.ndarray]] = []
        self._ops: list = []

    def operator(self, dt: float, d: np.ndarray):
        """The solve for one step; ``d`` is a bool per masked vertex (who diffuses)."""
        n = len(self._idx)
        w = self._w * (d[self._r] & d[self._c])
        weights = sparse.csr_matrix((w, (self._r, self._c)), shape=(n, n))
        lap = sparse.diags(np.asarray(weights.sum(axis=1)).ravel()) - weights
        a = (sparse.diags(self._mass) + dt * lap).tocsr()
        if self.device is None:
            from scipy.sparse import linalg as spla

            return spla.factorized(a.tocsc())
        import torch

        from .smooth import jacobi_cg, torch_csr

        a_t = torch_csr(a, self.device)
        dinv = 1.0 / torch.as_tensor(a.diagonal(), dtype=torch.float32, device=self.device)
        return lambda b: jacobi_cg(a_t, dinv, b, b.clone(), self.tol)

    def apply(self, op, u):
        """One step on masked rows ``u`` (numpy on CPU, torch on a device)."""
        if self.device is None:
            return op(self._mass[:, None] * u)
        import torch

        m = torch.as_tensor(self._mass, dtype=torch.float32, device=self.device)
        return op(m[:, None] * u)

    def record(self, dt: float, d: np.ndarray, op=None) -> None:
        self.steps.append((float(dt), d))
        self._ops.append(op if op is not None else self.operator(dt, d))

    def _rows(self, x):
        u = np.asarray(x)[self._idx]
        if self.device is None:
            return u.astype(np.float64)
        import torch

        return torch.as_tensor(u, dtype=torch.float32, device=self.device)

    def __call__(self, data: np.ndarray) -> np.ndarray:
        x = np.asarray(data)
        vec = x.ndim == 1
        x2 = x[:, None] if vec else x
        out = np.array(x2, dtype=np.float32, copy=True)
        u = self._rows(x2)
        for op in self._ops:
            u = self.apply(op, u)
        out[self._idx] = u if self.device is None else u.cpu().numpy()
        return out[:, 0] if vec else out


def blur_to_fwhm_local(
    master: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    target: float,
    mask: np.ndarray | None = None,
    radius_mm: float = 8.0,
    device=None,
    max_steps: int = 200,
    tol: float = 0.02,
) -> tuple[SurfaceAdaptiveDiffusion | None, float, float, np.ndarray]:
    """Spatially varying blur-to on a mesh (the surface twin of
    ``stats.blur_to.volume_blur_to_fwhm_local``): heat steps that stop wherever the
    local FWHM (:func:`local_surface_fwhm`) has reached ``target``, until the global
    FWHM does; the last step trimmed to land on it. Returns ``(diffusion, start,
    achieved, local_fwhm_map)``; ``diffusion`` is None when already that smooth."""
    sad = SurfaceAdaptiveDiffusion(vertices, faces, mask, device=device)
    idx = sad._idx

    def full(u):
        out = np.zeros((len(sad.mask), u.shape[1]))
        out[idx] = u if device is None else u.cpu().numpy()
        return out

    def glob(u):
        return surface_fwhm(full(u), vertices, faces, sad.mask)[0]

    u = sad._rows(master)
    start = glob(u)
    if not np.isfinite(start) or start >= target:
        return (
            None,
            float(start),
            float(start),
            local_surface_fwhm(master, vertices, faces, sad.mask, radius_mm, device),
        )
    dt = max((target**2 - start**2) / (16 * np.log(2)) / 10.0, 1e-4)
    achieved = start
    for _ in range(max_steps):
        local = local_surface_fwhm(full(u), vertices, faces, sad.mask, radius_mm, device)[idx]
        d = np.isfinite(local) & (local < target)
        if not d.any():
            break
        op = sad.operator(dt, d)
        nxt = sad.apply(op, u)
        f = glob(nxt)
        if f >= target * (1 + tol):
            lo, hi = 0.0, dt
            for _ in range(12):
                mid = 0.5 * (lo + hi)
                if glob(sad.apply(sad.operator(mid, d), u)) >= target:
                    hi = mid
                else:
                    lo = mid
            sad.record(hi, d)
            u = sad.apply(sad._ops[-1], u)
            achieved = glob(u)
            break
        sad.record(dt, d, op)
        u, achieved = nxt, f
        if achieved >= target:
            break
    local = local_surface_fwhm(full(u), vertices, faces, sad.mask, radius_mm, device)
    return sad, float(start), float(achieved), local
