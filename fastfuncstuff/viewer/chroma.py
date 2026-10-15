"""Several values per voxel, drawn as one colour.

Echoes, inversions, betas, component maps: anything with N >= 2 values at a
voxel. A colour layer is a linear map from those N values to three colour
coordinates, fitted once and applied per plane, so it costs a matmul at draw
time and plays as a movie when its channels are time series. Colour and time
never compete for one display axis, which is the problem a 4-D RGB file runs
into in every viewer that stores the colour.

Why not just put three channels into R, G and B: channels that measure the same
tissue are highly correlated, so voxels crowd the grey diagonal and colour
carries one dimension -- "how much did it decay" -- as a single brown hue. Each
basis is a different answer to that:

``pick``      three channels, each standardised. The baseline.
``dstretch``  decorrelation stretch (the Karhunen-Loeve whitening JPL used on
              planetary imagery and DStretch on rock art): whiten in the
              eigenbasis, rotate back, so what *differs* between channels gets
              the colour and hues still lean towards their channels.
``pca``       the whitened principal components themselves: most contrast,
              no fixed meaning of a hue.
``te``        a fixed basis rather than a fitted one: Legendre polynomials in
              echo time. On log signal, P0 is the level, P1 the decay rate
              (R2*) and P2 the departure from a single exponential; on percent
              change, P0 is the TE-independent (S0) part and P1 the
              TE-proportional (BOLD) part. Hues mean the same in every subject.

Two colour spaces. ``rgb`` maps the three coordinates to the three primaries.
``oklab`` puts the first coordinate in lightness and the other two on the
opponent axes (b: blue-yellow, a: green-red), so a change of hue never reads
as a change of brightness and anatomy stays legible under the tint.

The statistics a fit reads -- channel means and covariance -- come from a
region (``core``: the brain mask eroded, so the susceptibility fringe at the
edge cannot claim the whole stretch; ``brain``; or any loaded layer as a mask),
which is DStretch's "enhance this selection" in viewer terms.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch import Tensor

BASES = ("pick", "dstretch", "pca", "te")
SPACES = ("auto", "rgb", "oklab")
SIGNALS = ("raw", "log", "psc")
LIGHTNESS = ("basis", "anatomy")

#: Lightness range in Oklab. The floor keeps the darkest tissue distinguishable
#: from the masked background; the ceiling leaves headroom for chroma, since a
#: saturated colour at L=1 is out of the sRGB gamut and clips to white.
L_FLOOR, L_CEIL = 0.2, 0.8

#: How hard the anatomy can be nudged by the basis's first coordinate when the
#: lightness comes from anatomy -- enough to see S0 flicker, not enough to
#: repaint the anatomy.
ANATOMY_NUDGE = 0.15


@dataclass(frozen=True)
class Channel:
    """One channel: a layer, and which of its sub-bricks to read at time ``t``.

    The sub-brick is ``offset + t * stride``. Separate echo files are
    ``offset=0, stride=1``; echoes stacked echo-major in one file of ``nt``
    frames are ``offset=e*nt, stride=1``; interleaved, ``offset=e, stride=N``;
    a static volume or one bucket sub-brick is ``stride=0``.
    """

    key: str
    offset: int = 0
    stride: int = 1

    def brick(self, t: int, n_bricks: int) -> int:
        return max(0, min(self.offset + t * self.stride, n_bricks - 1))

    def n_frames(self, n_bricks: int) -> int:
        if self.stride <= 0:
            return 1
        return max(1, (n_bricks - 1 - self.offset) // self.stride + 1)

    def text(self) -> str:
        if self.offset == 0 and self.stride == 1:
            return self.key
        return f"{self.key}:{self.offset}" + ("" if self.stride == 1 else f"+{self.stride}")


_CHANNEL = re.compile(r"^([^:,+\s]+)(?::(\d+)(?:\+(\d+))?)?$")


def parse_channels(text: str) -> tuple[Channel, ...]:
    """``"E1,E2,E3"`` or ``"L1:0+3,L1:1+3,L1:2+3"`` -- ``KEY[:OFFSET[+STRIDE]]``.

    An offset with no stride means one fixed sub-brick (stride 0): ``L4:2`` is
    a bucket's third sub-brick, not a time series starting there.
    """
    out: list[Channel] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        m = _CHANNEL.match(part)
        if m is None:
            raise ValueError(f"bad channel {part!r}: expected KEY[:OFFSET[+STRIDE]]")
        key, offset, stride = m.groups()
        if offset is None:
            out.append(Channel(key))
        else:
            out.append(Channel(key, int(offset), 0 if stride is None else int(stride)))
    if len(out) < 2:
        raise ValueError("a colour layer needs at least two channels")
    return tuple(out)


def format_channels(channels: Sequence[Channel]) -> str:
    return ",".join(c.text() for c in channels)


@dataclass(frozen=True)
class ChromaSpec:
    """How a colour layer is made. Everything a fit depends on, and nothing else."""

    channels: tuple[Channel, ...]
    basis: str = "dstretch"
    space: str = "auto"
    signal: str = "log"
    #: Window half-width in standard deviations. Larger is calmer.
    spread: float = 2.5
    #: Oklab chroma at full swing. ~0.25 is near the edge of the sRGB gamut.
    chroma: float = 0.25
    #: Eigenvalues below ``floor * largest`` are not whitened further. Without
    #: it the weakest component -- usually noise -- is amplified the most,
    #: which on three echoes shows up as speckle.
    floor: float = 0.05
    stats: str = "core"
    lightness: str = "basis"
    #: Which three channels ``pick`` (and ``dstretch`` with N > 3) shows, as
    #: comma-separated indices. Empty: first, middle, last.
    pick: str = ""
    #: Echo times in ms, comma-separated, for the ``te`` basis. Empty: read
    #: from the channels' sidecars, or evenly spaced when they have none.
    te: str = ""
    #: Hide voxels outside the brain mask.
    masked: bool = True
    #: Frames averaged around the shown one (centred, odd). One frame of
    #: percent change at fMRI resolution is mostly thermal noise; a few
    #: seconds' average is where a breath-hold or a block shows.
    window: int = 1

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"basis must be one of {BASES}, not {self.basis!r}")
        if self.space not in SPACES:
            raise ValueError(f"space must be one of {SPACES}, not {self.space!r}")
        if self.signal not in SIGNALS:
            raise ValueError(f"signal must be one of {SIGNALS}, not {self.signal!r}")
        if self.lightness not in LIGHTNESS:
            raise ValueError(f"lightness must be one of {LIGHTNESS}, not {self.lightness!r}")
        if len(self.channels) < 2:
            raise ValueError("a colour layer needs at least two channels")
        if self.window < 1:
            raise ValueError("window must be at least one frame")
        # Checked here rather than at fit time: a value that cannot be drawn
        # must be refused by the command that sets it, not leave the layer
        # silently blank on the next repaint.
        self.picked()
        self.te_ms()

    def te_ms(self) -> list[float] | None:
        """Typed echo times, or ``None`` when they are to be read from sidecars."""
        if not self.te.strip():
            return None
        try:
            tes = [float(v) for v in self.te.split(",")]
        except ValueError:
            raise ValueError(f"te must be comma-separated numbers, got {self.te!r}") from None
        if len(tes) != len(self.channels):
            raise ValueError(f"te lists {len(tes)} echo times for {len(self.channels)} channels")
        if len(set(tes)) < 2:
            raise ValueError("echo times must differ")
        return tes

    @property
    def resolved_space(self) -> str:
        """``auto``: Oklab where the first coordinate is a level (te, pca), RGB otherwise."""
        if self.space != "auto":
            return self.space
        return "oklab" if self.basis in ("te", "pca") else "rgb"

    def picked(self) -> tuple[int, int, int]:
        n = len(self.channels)
        if self.pick.strip():
            try:
                idx = tuple(int(v) for v in self.pick.split(","))
            except ValueError:
                raise ValueError(f"pick must be three indices, got {self.pick!r}") from None
            if len(idx) != 3 or min(idx) < 0 or max(idx) >= n:
                raise ValueError(f"pick needs three indices in 0..{n - 1}, got {self.pick!r}")
            return idx  # type: ignore[return-value]
        if n == 2:
            return (0, 1, 1)
        return (0, n // 2, n - 1) if n > 3 else (0, 1, 2)

    def with_param(self, name: str, value: str) -> ChromaSpec:
        """A copy with one parameter set from its script text."""
        if name == "channels":
            return replace(self, channels=parse_channels(value))
        known = {f.name: f for f in fields(self)}
        if name not in known:
            raise KeyError(f"colour layers have no parameter {name!r}")
        current = getattr(self, name)
        if isinstance(current, bool):
            coerced: object = value not in ("0", "false", "False", "")
        elif isinstance(current, float):
            coerced = float(value)
        elif isinstance(current, int):
            coerced = int(float(value))
        else:
            coerced = "" if value == "-" else str(value)
        return replace(self, **{name: coerced})


# ---------------------------------------------------------------------------
# signal transforms
# ---------------------------------------------------------------------------


def transform(x: Tensor, signal: str, *, eps: float, mean: Tensor | None = None) -> Tensor:
    """Channels (N, ...) into the space the basis is fitted in.

    ``log`` because MR signal decays multiplicatively -- log S is linear in R2*,
    so the te basis is a straight-line fit and a covariance is not dominated by
    the brightest tissue. ``psc`` reads change from each voxel's own mean,
    which is what a breath-hold or a task is.
    """
    if signal == "raw":
        return x
    if signal == "log":
        return torch.log(x.clamp(min=eps))
    if mean is None:
        raise ValueError("percent change needs each channel's mean")
    ok = mean > eps
    return torch.where(ok, 100.0 * (x / torch.where(ok, mean, 1.0) - 1.0), 0.0)


def echo_basis(te_ms: Sequence[float]) -> np.ndarray:
    """(3, N): least-squares Legendre coefficients in TE, decay rate signed positive.

    Rows are the level, the decay rate and the curvature. P1 is negated so that
    faster decay -- shorter T2*, iron, deoxygenated blood -- is the positive
    direction and lands warm, slower decay (CSF) cool. With two echoes there
    is no curvature and its row is zero.
    """
    te = np.asarray(te_ms, dtype=np.float64)
    span = te.max() - te.min()
    if span <= 0:
        raise ValueError("echo times must differ")
    t = 2.0 * (te - te.min()) / span - 1.0
    order = min(3, len(te))
    P = np.stack([np.ones_like(t), t, 0.5 * (3.0 * t**2 - 1.0)][:order], 1)
    rows = np.linalg.pinv(P)
    out = np.zeros((3, len(te)))
    out[:order] = rows
    out[1] *= -1.0
    return out


# ---------------------------------------------------------------------------
# the fitted map
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChromaMap:
    """Transformed channels in, colour out. Fitted by :func:`fit_map`."""

    #: (3, N), applied to ``(x - mu) / sd``.
    W: Tensor
    mu: Tensor
    sd: Tensor
    #: Per coordinate: the window ``center +- width`` spans the colour range.
    center: Tensor
    width: Tensor
    space: str
    chroma: float
    #: Oklab lightness window on the first coordinate (robust percentiles,
    #: because it is a level and its distribution is skewed, not a deviation).
    l_lo: float
    l_hi: float

    def to(self, device: torch.device) -> ChromaMap:
        return replace(
            self,
            W=self.W.to(device),
            mu=self.mu.to(device),
            sd=self.sd.to(device),
            center=self.center.to(device),
            width=self.width.to(device),
        )

    def coords(self, x: Tensor) -> Tensor:
        """(N, ...) -> (3, ...): the basis coordinates, unwindowed."""
        n = x.shape[0]
        flat = x.reshape(n, -1)
        z = (flat - self.mu[:, None]) / self.sd[:, None]
        return (self.W @ z).reshape(3, *x.shape[1:])

    def unit(self, y: Tensor) -> Tensor:
        """Coordinates scaled so the window is [-1, 1]."""
        shape = (3,) + (1,) * (y.ndim - 1)
        return (y - self.center.reshape(shape)) / self.width.reshape(shape)

    def colors(self, x: Tensor, lightness: Tensor | None = None) -> Tensor:
        """(N, ...) transformed channels -> (..., 3) sRGB in [0, 1].

        ``lightness`` (in [0, 1], same spatial shape) replaces the first
        coordinate's lightness with anatomy, which the first coordinate then
        only nudges.
        """
        y = self.coords(x)
        u = self.unit(y)
        if self.space == "rgb":
            rgb = (0.5 + 0.5 * u).clamp(0.0, 1.0)
            rgb = rgb.movedim(0, -1)
            if lightness is not None:
                rgb = rgb * lightness.unsqueeze(-1)
            return rgb
        if lightness is None:
            level = ((y[0] - self.l_lo) / max(self.l_hi - self.l_lo, 1e-12)).clamp(0.0, 1.0)
            L = L_FLOOR + (L_CEIL - L_FLOOR) * level
        else:
            L = L_FLOOR + (L_CEIL - L_FLOOR) * lightness + ANATOMY_NUDGE * torch.tanh(u[0])
        # tanh rather than a clip: saturation rolls off instead of flattening,
        # so the strongest voxels are still distinguishable from strong ones.
        b = self.chroma * torch.tanh(1.25 * u[1])
        a = self.chroma * torch.tanh(1.25 * u[2])
        return oklab_to_srgb(L, a, b)


def oklab_to_srgb(L: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Oklab (Ottosson 2020) to gamma-encoded sRGB, clipped to the gamut. (..., 3)."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_**3, m_**3, s_**3
    rgb = torch.stack(
        [
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960771 * l - 0.7034186147 * m + 1.7076147010 * s,
        ],
        -1,
    ).clamp(0.0, 1.0)
    return torch.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp(min=1e-12) ** (1 / 2.4) - 0.055
    )


def _eig(cov: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of an N x N covariance, largest first, floored."""
    lam, E = np.linalg.eigh(np.atleast_2d(cov))
    lam, E = lam[::-1], E[:, ::-1]
    # Sign is arbitrary out of eigh; fixing it makes the same data always
    # draw in the same colours, and puts a component that loads every channel
    # the same way on its bright side.
    E = E * np.where(E.sum(0) < 0, -1.0, 1.0)
    lam = np.maximum(lam, max(floor, 1e-6) * max(lam[0], 1e-12))
    return lam, E


def fit_map(
    samples: Tensor | np.ndarray, spec: ChromaSpec, te_ms: Sequence[float] | None = None
) -> ChromaMap:
    """Fit a colour map to transformed channel samples ``(M, N)`` from the stats region.

    The samples stay where they are -- on the GPU for a real dataset -- and
    only the N x N covariance comes to the host, in float64, for the eigen
    decomposition; that is a few channels squared, not a million voxels.
    """
    x = torch.as_tensor(samples, dtype=torch.float32)
    x = x[torch.isfinite(x).all(dim=1)]
    if x.shape[0] < 8:
        raise ValueError("too few voxels in the statistics region to fit colours")
    n = x.shape[1]
    pick = np.zeros((3, n))
    for row, c in enumerate(spec.picked()):
        pick[row, c] = 1.0

    if spec.basis == "te":
        # A fixed basis on the signal itself: standardising each channel
        # would reweight the echoes and bend the straight-line fit.
        mu, sd = np.zeros(n), np.ones(n)
        te = np.arange(n, dtype=float) if te_ms is None or len(te_ms) != n else np.asarray(te_ms)
        W = echo_basis(te)
    else:
        mu_t = x.mean(0)
        sd_t = x.std(0)
        sd_t = torch.where(sd_t > 0, sd_t, torch.ones_like(sd_t))
        mu, sd = mu_t.double().cpu().numpy(), sd_t.double().cpu().numpy()
        if spec.basis == "pick":
            W = pick
        else:
            z = (x - mu_t) / sd_t
            cov = (z.T @ z / max(z.shape[0] - 1, 1)).double().cpu().numpy()
            lam, E = _eig(cov, spec.floor)
            if spec.basis == "dstretch":
                W = pick @ (E @ np.diag(lam**-0.5) @ E.T)
            else:  # pca
                k = min(3, n)
                W = np.zeros((3, n))
                W[:k] = (E[:, :k] / np.sqrt(lam[:k])).T

    def t(v: np.ndarray) -> Tensor:
        return torch.as_tensor(np.asarray(v), dtype=torch.float32)

    W_t, mu_t, sd_t = t(W).to(x.device), t(mu).to(x.device), t(sd).to(x.device)
    y = ((x - mu_t) / sd_t) @ W_t.T
    center = y.median(dim=0).values
    width = spec.spread * y.std(0)
    width = torch.where(width > 0, width, torch.ones_like(width))
    l_lo, l_hi = torch.quantile(y[:, 0], torch.tensor([0.02, 0.995], device=x.device)).tolist()

    return ChromaMap(
        W=W_t,
        mu=mu_t,
        sd=sd_t,
        center=center,
        width=width,
        space=spec.resolved_space,
        chroma=float(spec.chroma),
        l_lo=float(l_lo),
        l_hi=float(l_hi),
    )


# ---------------------------------------------------------------------------
# finding channels
# ---------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def sibling_order(names: Sequence[str], seed: int) -> list[int]:
    """Indices of names that differ from ``names[seed]`` only in one number, by that number.

    ``..._SBRef_e1.nii.gz``, ``_e2``, ``_e3`` are one acquisition's echoes; the
    run number, TR and voxel size elsewhere in the name are the same, and a
    different run of the same sequence differs in a different number. Returns
    just ``[seed]`` when nothing matches.
    """
    skeleton = _DIGITS.split(names[seed])
    numbers = _DIGITS.findall(names[seed])
    best: list[int] = [seed]
    for position in range(len(numbers)):
        found: list[tuple[int, int]] = []
        for i, name in enumerate(names):
            if _DIGITS.split(name) != skeleton:
                continue
            other = _DIGITS.findall(name)
            if all(
                o == s for j, (o, s) in enumerate(zip(other, numbers, strict=True)) if j != position
            ):
                found.append((int(other[position]), i))
        if len({v for v, _ in found}) > len(best):
            best = [i for _, i in sorted(found)]
    return best
