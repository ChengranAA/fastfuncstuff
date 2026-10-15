"""Settings that make region A look as different as possible from region B.

The colour a voxel gets is a differentiable function of a handful of numbers --
the 3 x N weights on the standardised channels, and each coordinate's shift and
stretch -- and Oklab is built so straight-line distance is perceived
difference. So "make A and B look different" is an objective torch can climb.

Maximising the distance between the two regions' mean colours alone has a
degenerate answer: stretch everything until A is one saturated colour, B its
opposite, and the rest of the brain is clipped flat. Three terms instead:

* **separation** -- Oklab distance between A's and B's mean colours. Bounded by
  the gamut, so it cannot run away;
* **uniformity** -- minus each region's own colour spread, so a region reads as
  one colour rather than as noise;
* **context** -- minus how much of the rest of the brain is pushed into the
  roll-off, so the anatomy stays legible. Its weight is the user's trade.

Two starts, keep the better: the current settings, and the current basis with
its weakest chroma axis replaced by Fisher's discriminant -- the channel
direction that best separates A from B in a linear world. Gradient ascent then
works through the nonlinear colour mapping LDA does not see.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.viewer.chroma import Look, paint, srgb_to_oklab, unit_coords

#: Bounds the panel's sliders have, so a result always lands on them.
SHIFT_MAX, STRETCH_MAX = 2.0, 4.0
#: Weight of within-region spread against separation.
UNIFORMITY = 0.5
#: Weight of the coordinates' correlation over the context. Without it every
#: row turns toward the one A-vs-B direction: the image becomes two colours
#: and the anatomy is gone (seen on real echoes: magenta on black).
DIVERSITY = 0.5
#: Weight of the context's mean chroma. Without it the shifts slide every
#: voxel's hue at once -- the whole brain red -- because a global tint costs
#: separation nothing; hue should mark where A and B differ, not everything.
BALANCE = 2.0
#: Temperature of the soft minimum over A-vs-B pairs, in Oklab units: well
#: below a typical separation (0.1-0.5), so it is close to the true minimum.
SOFTMIN = 0.03
#: Context samples kept per step: enough for the clipped fraction to be stable.
MAX_CONTEXT = 4_000


@dataclass(frozen=True)
class ContrastResult:
    """The settings found, and what they buy."""

    weights: np.ndarray  # (3, N), rows unit length, on the standardised channels
    shift: tuple[float, float, float]
    stretch: tuple[float, float, float]
    separation: float  # Oklab distance between A's mean and the nearest B's
    before: float  # the same at the starting settings
    clipped: float  # fraction of context voxels with a coordinate past the roll-off knee
    start: str  # "current" or "lda"
    #: Oklab distance from A to each B, in the order given.
    separations: tuple[float, ...] = ()

    def weights_text(self) -> str:
        return "; ".join(",".join(f"{v:.3f}" for v in row) for row in self.weights)


def _window(W: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Each coordinate's centre and width, and the lightness window, on the context.

    Measured exactly as :func:`viewer.chroma.fit_map` measures them on the
    stats region, so the result, written back as a custom basis, draws the
    same. Once per step: it is the expensive part.
    """
    y = W @ ctx.T
    center = y.median(dim=1).values
    width = y.std(dim=1).clamp(min=1e-6)
    q = torch.quantile(y[0], torch.tensor([0.02, 0.995], device=ctx.device))
    return center, width, q[0], q[1]


def _lab(z, W, window, look, shift, stretch, light) -> tuple[Tensor, Tensor]:
    """Oklab colours of standardised samples ``z`` (M, N), and their window units."""
    center, width, lo, hi = window
    y = W @ z.T
    rgb = paint(y, center, width, lo, hi, look, shift, stretch, light)
    return srgb_to_oklab(rgb), unit_coords(y, center, width, look.spread, shift, stretch)


def _objective(
    za, zbs, zc, W, shift, stretch, look, la, lbs, lc, context: float
) -> tuple[Tensor, Tensor, Tensor]:
    """(score, separation from each B, context units) -- higher score is better.

    With several non-target regions the separation is a soft minimum over
    the pairs, so the worst-separated B is what the gradient works on first:
    a B already far from A is not where colour should be spent, and the mean
    of the pairs would happily leave one B sitting on top of A.
    """
    window = _window(W, zc)
    # One pass over A, every B and the context together: a step is a few
    # hundred small ops, so their count, not the data, is the cost.
    z = torch.cat([za, *zbs, zc])
    light = None if la is None else torch.cat([la, *lbs, lc])
    lab, u = _lab(z, W, window, look, shift, stretch, light)
    sizes = [za.shape[0], *(zb.shape[0] for zb in zbs)]
    edges = np.cumsum([0, *sizes])
    lab_a = lab[: sizes[0]]
    lab_bs = [lab[edges[j] : edges[j + 1]] for j in range(1, len(sizes))]
    u_c = u[:, edges[-1] :]
    tint = lab[edges[-1] :, 1:].mean(0).norm()
    seps = torch.stack([torch.linalg.vector_norm(lab_a.mean(0) - b.mean(0)) for b in lab_bs])
    sep = -SOFTMIN * torch.logsumexp(-seps / SOFTMIN, 0) if len(lab_bs) > 1 else seps[0]
    spread = lab_a.std(0).norm() + torch.stack([b.std(0).norm() for b in lab_bs]).mean()
    corr = torch.corrcoef(u_c)
    redundant = (corr - torch.diag(torch.diagonal(corr))).pow(2).sum() / 6.0
    # Past ~1.5 window units tanh has flattened: a voxel there has lost its
    # detail. Smooth, so the gradient sees the knee coming.
    clipped = torch.nn.functional.softplus(8.0 * (u_c.abs() - 1.5)).mean() / 8.0
    score = sep - UNIFORMITY * spread - context * clipped - DIVERSITY * redundant - BALANCE * tint
    return score, seps, u_c


def _lda_start(za: Tensor, zb: Tensor, W0: Tensor) -> Tensor:
    """W0 with its weakest-separating chroma row replaced by Fisher's direction.

    With several B regions, ``zb`` is all of them together: the start only
    needs to point roughly away from the non-targets, the ascent does the rest.
    """
    diff = za.mean(0) - zb.mean(0)
    pooled = torch.cov(za.T) + torch.cov(zb.T)
    pooled = pooled + 1e-3 * torch.eye(pooled.shape[0], device=za.device) * pooled.diagonal().mean()
    w = torch.linalg.solve(pooled.double(), diff.double()).float()
    w = w / w.norm().clamp(min=1e-12)
    gaps = ((W0 @ za.T).mean(1) - (W0 @ zb.T).mean(1)).abs() / (W0 @ torch.cat([za, zb]).T).std(1)
    row = 1 + int(gaps[1:].argmin())  # a chroma row; lightness stays anatomy-like
    W = W0.clone()
    W[row] = w * W0[row].norm()
    return W


def optimise_contrast(
    za: Tensor,
    zb: Tensor | list[Tensor],
    zc: Tensor,
    W0: Tensor,
    look: Look,
    *,
    light: tuple[Tensor, Tensor | list[Tensor], Tensor] | None = None,
    context: float = 1.0,
    steps: int = 60,
    lr: float = 0.05,
    seed: int = 0,
) -> ContrastResult:
    """Climb the objective from two starts; return the better settings.

    ``za``/``zb``/``zc`` are standardised channel samples (M, N) for region A,
    the non-target region(s) B -- one tensor, or a list of them, each its own
    region to separate A from -- and the context (the stats region).
    ``light`` gives each set's anatomy lightness when the look takes lightness
    from anatomy, B's as a list matching ``zb``.
    """
    zbs = list(zb) if isinstance(zb, (list, tuple)) else [zb]
    if light is not None:
        lb = light[1]
        light = (light[0], list(lb) if isinstance(lb, (list, tuple)) else [lb], light[2])
    zb = zbs  # type: ignore[assignment]
    if za.device.type == "cpu":
        # A few thousand points per op: more threads only add the cost of
        # handing the work out, which on 6 cores tripled the step time.
        with _threads(2):
            return _optimise(za, zb, zc, W0, look, light, context, steps, lr, seed)
    return _optimise(za, zb, zc, W0, look, light, context, steps, lr, seed)


@contextlib.contextmanager
def _threads(n: int):
    before = torch.get_num_threads()
    torch.set_num_threads(min(n, before))
    try:
        yield
    finally:
        torch.set_num_threads(before)


def _optimise(za, zbs, zc, W0, look, light, context, steps, lr, seed) -> ContrastResult:
    if za.shape[0] < 2 or not zbs or any(zb.shape[0] < 2 for zb in zbs):
        raise ValueError("each region needs at least two voxels inside the mask")
    gen = torch.Generator().manual_seed(seed)
    if zc.shape[0] > MAX_CONTEXT:
        keep = torch.randperm(zc.shape[0], generator=gen)[:MAX_CONTEXT].to(zc.device)
        zc = zc[keep]
        if light is not None:
            light = (light[0], light[1], light[2][keep])
    la, lb, lc = light if light is not None else (None, None, None)
    dev = za.device

    def to_params(shift, stretch):
        # Bounded by construction, so the answer fits the sliders.
        s = torch.atanh((torch.tensor(shift, device=dev) / SHIFT_MAX).clamp(-0.99, 0.99))
        k = torch.atanh((torch.tensor(stretch, device=dev) / STRETCH_MAX).clamp(-0.99, 0.99))
        return s, k

    def from_params(s, k):
        return SHIFT_MAX * torch.tanh(s), STRETCH_MAX * torch.tanh(k)

    s0, k0 = to_params(look.shift, look.stretch)
    with torch.no_grad():
        shift, stretch = from_params(s0, k0)
        _, before, _ = _objective(za, zbs, zc, W0, shift, stretch, look, la, lb, lc, context)

    best: ContrastResult | None = None
    best_score = -float("inf")
    for name, W_init in (("current", W0), ("lda", _lda_start(za, torch.cat(zbs), W0))):
        W = W_init.clone().requires_grad_(True)
        s = s0.clone().requires_grad_(True)
        k = k0.clone().requires_grad_(True)
        opt = torch.optim.Adam([W, s, k], lr=lr)
        # In Oklab the first coordinate is lightness: the anatomy. It stays as
        # it is and the contrast goes into hue -- brightness for structure,
        # colour for the discriminant.
        frozen = look.space == "oklab"
        last = -float("inf")
        for step in range(steps):
            opt.zero_grad()
            shift, stretch = from_params(s, k)
            score, _, _ = _objective(za, zbs, zc, W, shift, stretch, look, la, lb, lc, context)
            (-score).backward()
            if frozen:
                assert W.grad is not None and s.grad is not None and k.grad is not None
                W.grad[0] = 0.0
                s.grad[0] = 0.0
                k.grad[0] = 0.0
            opt.step()
            # Adam reached its plateau in ~40 steps on every case tried; stop
            # once ten steps have bought less than a thousandth.
            if step % 10 == 9:
                if float(score.detach()) - last < 1e-3:
                    break
                last = float(score.detach())
        with torch.no_grad():
            shift, stretch = from_params(s, k)
            score, seps, u_c = _objective(za, zbs, zc, W, shift, stretch, look, la, lb, lc, context)
            clipped = (u_c.abs() > 1.5).any(0).float().mean()
            if float(score) > best_score:
                best_score = float(score)
                rows = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
                best = ContrastResult(
                    weights=rows.cpu().numpy(),
                    shift=tuple(float(v) for v in shift),  # type: ignore[arg-type]
                    stretch=tuple(float(v) for v in stretch),  # type: ignore[arg-type]
                    separation=float(seps.min()),
                    before=float(before.min()),
                    separations=tuple(float(v) for v in seps),
                    clipped=float(clipped),
                    start=name,
                )
    assert best is not None
    return best


def with_result(look: Look, result: ContrastResult) -> Look:
    return replace(look, shift=result.shift, stretch=result.stretch)
