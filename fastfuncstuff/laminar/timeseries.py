"""Continuous BOLD deveining with the linearized Havlicek–Uludag vasculature.

The unknown inputs are local CBF deviations at K vascular depths, not fast
neuronal activity. Exact Jacobians of the existing vascular ODE give the
frequency response. A regularized inverse estimates the transported signal;
subtracting that prediction preserves the unexplained residual. See the wiki's
Model-derived static deveining and Laminar timeseries deveining notes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from fastfuncstuff.laminar.forward import build_cache, g_obs, rhs
from fastfuncstuff.laminar.params import ModelSpec, zero_params
from fastfuncstuff.memory import estimate_chunk_size
from fastfuncstuff.utils import accum_dtype, cpu_if_mps, linalg_device


@dataclass
class VascularSystem:
    A: torch.Tensor  # (4K, 4K), vascular log-state Jacobian, per second
    B: torch.Tensor  # (4K, K), local relative-CBF inputs
    C: torch.Tensor  # (K, 4K), GE PSC observation
    micro_c: torch.Tensor  # venule-only observation, SAME H0 as full GE


def linearized_vasculature(
    spec: ModelSpec, params: dict[str, torch.Tensor] | None = None
) -> VascularSystem:
    """Linearize at rest; omit the neuronal/NVC front end from the inverse.

    K independent vascular inputs avoid interpreting the Gaussian neuronal-depth
    basis as drainage. This is a local approximation, not a nonlinear inversion.
    Float64 is confined to generating the small physiological transfer system.
    """
    work = linalg_device(spec.device)
    s = replace(spec, dtype=torch.float64, device=work)
    p = (
        zero_params(s)
        if params is None
        else {k: v.to(device=work, dtype=s.dtype) for k, v in params.items()}
    )
    cache = build_cache(s, p)
    x = torch.zeros(s.n_states, device=work, dtype=s.dtype)
    u = torch.zeros(s.n_inputs, device=work, dtype=s.dtype)
    base, K = 4 * s.N, s.K
    # Input directly at vascular CBF, leaving local metabolism coupled by nr.
    cache = replace(cache, n2k=torch.eye(K, device=work, dtype=s.dtype))
    # rhs's neuronal blocks still need N states; use a K-input specification.
    if s.N != K:
        raise ValueError(
            "the continuous vascular system requires N=K (independent local CBF inputs)"
        )
    jac = torch.func.jacfwd(lambda z: rhs(z, u, cache, s.N, K))(x)
    obs = torch.func.jacfwd(lambda z: g_obs(z, p, s, cache=cache))(x)
    micro_cache = replace(
        cache,
        k1d=torch.zeros_like(cache.k1d),
        k2d=torch.zeros_like(cache.k2d),
        k3d=torch.zeros_like(cache.k3d),
    )
    micro_obs = torch.func.jacfwd(lambda z: g_obs(z, p, s, cache=micro_cache))(x)
    return VascularSystem(
        jac[base:, base:], jac[base:, 3 * s.N : 4 * s.N], obs[:, base:], micro_obs[:, base:]
    )


def vascular_transfer(
    system: VascularSystem, frequencies: torch.Tensor, target: str = "local"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full and target responses (F,K,K), at frequencies in Hz.

    'local' retains each depth's response to its OWN CBF (including its local AV
    contribution). 'micro' retains venules only. Neither changes vascular flow
    equations or sets an ascending-vein compartment's volume to zero.
    """
    if target not in ("local", "micro"):
        raise ValueError("target must be local or micro")
    A, B = system.A, system.B
    freq = frequencies.to(device=A.device, dtype=A.dtype)
    eye = torch.eye(A.shape[0], device=A.device, dtype=torch.complex128)
    states = torch.linalg.solve(
        2j * torch.pi * freq[:, None, None] * eye - A,
        B.to(torch.complex128).expand(freq.numel(), -1, -1),
    )
    full = system.C.to(torch.complex128) @ states
    if target == "micro":
        desired = system.micro_c.to(torch.complex128) @ states
    else:
        desired = torch.diag_embed(full.diagonal(dim1=-2, dim2=-1))
    return full, desired


@dataclass
class DynamicResult:
    correction: torch.Tensor  # (R,U,T,K), transported contribution, sample depths
    predicted: torch.Tensor  # full model reconstruction: a smoothing-only control
    grid_correction: torch.Tensor  # physiological depth grid, before sampling
    grid_predicted: torch.Tensor
    regularization: torch.Tensor  # (U,), selected on GE runs alone
    cv_r2: torch.Tensor  # (U,), leave-run-out full-model prediction
    noise_gain: torch.Tensor  # (U,), RMS gain of residual-preserving correction


def dynamic_inverse(
    profiles: torch.Tensor,
    full: torch.Tensor,
    target: torch.Tensor,
    *,
    sampling: torch.Tensor | None = None,
    noise_cov: torch.Tensor | None = None,
    strengths: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0),
    tr: float,
    pad: int,
    cutoff_hz: float = 0.1,
    device: torch.device | None = None,
    verbose: bool = True,
) -> DynamicResult:
    """Regularized spectral inverse, selecting shrinkage per unit by GE LORO.

    profiles are detrended PSC, (runs, units, time, depths). Runs used for LORO
    MUST have the same task schedule and length. With one run a single supplied
    strength is required. A fixed single strength also supports unrelated runs.
    full/target use rfftfreq(T+2*pad, TR). Reflect padding reduces wraparound;
    first/last pad samples remain boundary-sensitive and should be excluded in QC.
    sampling maps physiological depths to measured bins. noise_cov accounts for
    shared voxels in those bins; a small diagonal term stabilizes the whitening.
    """
    if profiles.ndim != 4:
        raise ValueError("profiles must be (runs, units, time, depths)")
    R, U, T, K = profiles.shape
    if U < 1 or T < 3 or not 0 <= pad < T or tr <= 0 or cutoff_hz <= 0:
        raise ValueError("need units, >=3 samples, 0<=pad<T, and positive TR/cutoff")
    if not strengths or any(x <= 0 for x in strengths) or (R < 2 and len(strengths) > 1):
        raise ValueError("positive strengths required; selection needs >=2 repeated runs")
    nf = (T + 2 * pad) // 2 + 1
    if full.shape != (nf, K, K) or target.shape != full.shape:
        raise ValueError("transfer grid must match rfftfreq(T+2*pad, tr) and K depths")
    if not torch.isfinite(profiles).all():
        raise ValueError("profiles must be finite")
    dev = device or profiles.device
    dtype = accum_dtype(dev)
    cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    H, G = full.to(device=dev, dtype=cdtype), target.to(device=dev, dtype=cdtype)
    correction, predicted = torch.empty_like(profiles), torch.empty_like(profiles)
    grid_correction, grid_predicted = torch.empty_like(profiles), torch.empty_like(profiles)
    chosen, cv, gain = [
        torch.empty(U, dtype=profiles.dtype, device=profiles.device) for _ in range(3)
    ]
    chunk = estimate_chunk_size(
        U, T + 2 * pad, K, dev, operation="laminar_dynamic", n_runs=R, n_designs=len(strengths)
    )
    freq = torch.fft.rfftfreq(T + 2 * pad, tr, device=dev)
    penalty = 1 + (freq / cutoff_hz) ** 4
    eye = torch.eye(K, device=dev, dtype=cdtype)
    weights_f = torch.full((nf,), 2.0, device=dev, dtype=dtype)
    weights_f[0] = 1
    if (T + 2 * pad) % 2 == 0:
        weights_f[-1] = 1
    # CV is scored after cropping reflection padding, in observed BOLD units.
    for start in tqdm(
        range(0, U, chunk), desc="Dynamic inverse", leave=True, disable=not verbose or U <= chunk
    ):
        stop = min(U, start + chunk)
        b = stop - start
        y = profiles[:, start:stop].to(device=dev, dtype=dtype)
        yp = F.pad(y.permute(0, 1, 3, 2).reshape(-1, 1, T), (pad, pad), mode="reflect")
        yf = torch.fft.rfft(yp.reshape(R, b, K, T + 2 * pad), dim=-1).permute(0, 1, 3, 2)
        S = (
            eye.real.expand(b, -1, -1)
            if sampling is None
            else sampling[start:stop].to(device=dev, dtype=dtype)
        )
        cov = (
            eye.real.expand(b, -1, -1)
            if noise_cov is None
            else noise_cov[start:stop].to(device=dev, dtype=dtype)
        )
        # Positive covariance required; the ridge is explicit and relative to its scale.
        cov = 0.95 * cov + 0.05 * cov.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * eye.real
        chol = torch.linalg.cholesky(cov.to(cpu_if_mps(dev, "cholesky"))).to(dev)
        whiten = torch.linalg.solve_triangular(chol, eye.real.expand(b, -1, -1), upper=False).to(
            cdtype
        )
        hs, gs = S.to(cdtype)[:, None] @ H, S.to(cdtype)[:, None] @ G
        hw = whiten[:, None] @ hs
        gram = hw.mH @ hw
        scale = (
            gram[:, 0].diagonal(dim1=-2, dim2=-1).real.mean(-1).clamp_min(torch.finfo(dtype).eps)
        )
        solve_device = cpu_if_mps(dev, "solve")
        rhs_solve = (hw.mH @ whiten[:, None]).to(solve_device)
        scores, inverses = [], []
        total = y.square().sum((0, 2, 3)).clamp_min(torch.finfo(dtype).eps)
        for strength in strengths:
            inverse = torch.linalg.solve(
                (
                    gram
                    + strength * scale[:, None, None, None] * penalty[None, :, None, None] * eye
                ).to(solve_device),
                rhs_solve,
            ).to(dev)
            inverses.append(inverse)
            if R > 1:
                train = (yf.sum(0, keepdim=True) - yf) / (R - 1)
                held = torch.einsum("bfij,rbfj->rbfi", hs @ inverse, train)
                held_t = torch.fft.irfft(held.permute(0, 1, 3, 2), n=T + 2 * pad, dim=-1)[
                    ..., pad : pad + T
                ].permute(0, 1, 3, 2)
                scores.append(1 - (held_t - y).square().sum((0, 2, 3)) / total)
            else:
                scores.append(torch.full((b,), float("nan"), device=dev, dtype=dtype))
        scores_t = torch.stack(scores)
        best = (
            scores_t.argmax(0)
            if len(strengths) > 1
            else torch.zeros(b, device=dev, dtype=torch.long)
        )
        inv = torch.stack(inverses)[best, torch.arange(b, device=dev)]
        remove = (hs - gs) @ inv
        transported = torch.einsum("bfij,rbfj->rbfi", remove, yf)
        fit = torch.einsum("bfij,rbfj->rbfi", hs @ inv, yf)

        def crop(z):
            return (
                torch.fft.irfft(z.permute(0, 1, 3, 2), n=T + 2 * pad, dim=-1)[..., pad : pad + T]
                .permute(0, 1, 3, 2)
                .to(profiles)
            )

        correction[:, start:stop], predicted[:, start:stop] = crop(transported), crop(fit)
        grid_correction[:, start:stop] = crop(torch.einsum("bfij,rbfj->rbfi", (H - G) @ inv, yf))
        grid_predicted[:, start:stop] = crop(torch.einsum("bfij,rbfj->rbfi", H @ inv, yf))
        chosen[start:stop] = torch.tensor(strengths, device=dev, dtype=dtype)[best].to(chosen)
        cv[start:stop] = scores_t[best, torch.arange(b, device=dev)].to(cv)
        filt = eye - remove
        # Integrated white-noise gain; this is a conditional diagnostic, not a CI.
        ng = (filt.abs().square().sum((-1, -2)) * weights_f).sum(-1) / (K * weights_f.sum())
        gain[start:stop] = ng.sqrt().to(gain)
    return DynamicResult(correction, predicted, grid_correction, grid_predicted, chosen, cv, gain)
