"""Colour layers in a session: making them, fitting them, drawing a plane of one.

The arithmetic is :mod:`viewer.chroma`; this is the part that knows about the
store and the stack. A colour layer holds no voxels of its own -- its channels
are other layers' sub-bricks, read at draw time -- so a run of echoes plays as
a colour movie with no 4-D RGB copy anywhere. Its store entry is lazy and
yields the first basis coordinate, which is what keeps the generic paths (the
readout's existence check, AUTO, a montage) working on it unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.viewer.chroma import (
    BASES,
    LIGHTNESS,
    SIGNALS,
    SPACES,
    Channel,
    ChromaMap,
    ChromaSpec,
    fit_map,
    format_channels,
    sibling_order,
    transform,
)
from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.layers import Layer

if TYPE_CHECKING:
    from fastfuncstuff.viewer.modes.base import Control
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.slicing import PlaneView
    from fastfuncstuff.viewer.state import DisplayGrid, Plane

#: Frames read to estimate the covariance of percent change. Enough to see
#: the dynamics; reading every frame of a long run would only add seconds.
PSC_FRAMES = 16
#: Rows of the sample matrix a fit keeps. A covariance of three to ten
#: channels is settled long before a million voxels.
MAX_SAMPLES = 400_000
#: The brain mask is eroded this many voxels for ``stats=core``.
CORE_EROSION = 2


@dataclass
class ChromaFit:
    """Everything a draw needs that does not change with time."""

    token: tuple
    map: ChromaMap
    #: (X, Y, Z) bool, on the channels' grid.
    brain: Tensor
    #: (N, X, Y, Z) per-channel mean, for percent change. ``None`` otherwise.
    mean: Tensor | None
    #: (X, Y, Z) anatomy lightness in [0, 1], for ``lightness=anatomy``.
    anatomy: Tensor | None
    eps: float
    te_ms: tuple[float, ...] | None
    #: Where the TEs came from, for the readout: "sidecar", "typed", "index".
    te_source: str


def _sidecar_echo_time(path: str) -> float | None:
    """EchoTime from the BIDS JSON beside a dataset, in ms."""
    from fastfuncstuff.viewer.tools.slicetime import sidecar_for

    try:
        sidecar = sidecar_for(Path(path))
        if sidecar is None:
            return None
        value = json.loads(sidecar.read_text()).get("EchoTime")
    except (OSError, ValueError):
        return None
    return None if value is None else float(value) * 1000.0


@dataclass
class _Base:
    """What every fit of one set of channels shares."""

    token: tuple
    #: (N, X, Y, Z) per-channel temporal mean (the volume itself, if static).
    means: Tensor
    #: (X, Y, Z) bool, automask of the first channel.
    brain: Tensor
    #: Floor for log signal: a thousandth of bright tissue.
    eps: float


def _erode(mask: Tensor, n: int) -> Tensor:
    """Peel ``n`` voxels off a mask (26-neighbour), on its device."""
    out = mask.float()[None, None]
    for _ in range(n):
        out = -torch.nn.functional.max_pool3d(-out, 3, stride=1, padding=1)
    return out[0, 0] > 0.5


class ChromaLayers:
    """The session's colour layers. One instance per session."""

    def __init__(self, session: ViewerSession) -> None:
        self.session = session
        self._fits: dict[str, ChromaFit] = {}
        self._bases: dict[tuple[Channel, ...], _Base] = {}
        self._frames: dict[str, tuple[tuple, Tensor]] = {}

    # -- making ----------------------------------------------------------
    def guess_channels(self, seed_key: str) -> tuple[Channel, ...]:
        """The channels a colour layer seeded on ``seed_key`` most plausibly wants.

        Sibling files first -- other loaded layers on the same grid whose file
        names differ from the seed's only in one number (``_e1``, ``_e2``...).
        Failing that, a non-time-series layer's own sub-bricks: a bucket, a
        component stack, MP2RAGE inversions in one file. A time series with no
        siblings cannot be guessed: echo-major and interleaved stacking look
        identical from the header.
        """
        layers = self.session.state.layers
        seed = layers.get(seed_key)
        same = [
            ly
            for ly in layers
            if ly.chroma is None and ly.shape == seed.shape and ly.n_volumes == seed.n_volumes
        ]
        names = [Path(ly.path).name for ly in same]
        order = sibling_order(names, [ly.key for ly in same].index(seed_key))
        if len(order) >= 2:
            return tuple(Channel(same[i].key) for i in order)
        if seed.n_volumes >= 2 and not seed.time_linked:
            return tuple(Channel(seed.key, i, 0) for i in range(seed.n_volumes))
        raise ValueError(
            f"cannot tell which volumes of {seed.name} are channels: name them, "
            "KEY:OFFSET+STRIDE per channel (e.g. L1:0+3,L1:1+3,L1:2+3 for three "
            "interleaved echoes)"
        )

    def default_spec(self, channels: tuple[Channel, ...]) -> ChromaSpec:
        """What a new colour layer starts as.

        Echoes with known echo times get the te basis, because its hues mean
        something fixed; anything else gets a decorrelation stretch. A time
        series starts on percent change -- what moves is what a run is for --
        and a static set on log signal, which is what decays.
        """
        tes = [self._echo_time(c) for c in channels]
        known = all(t is not None for t in tes) and len(set(tes)) == len(tes)
        timed = self._n_frames(channels) > 1
        values_positive = self._positive(channels[0])
        signal = "psc" if timed else ("log" if values_positive else "raw")
        if known and values_positive:
            return ChromaSpec(
                channels, basis="te", signal=signal, lightness="anatomy" if timed else "basis"
            )
        return ChromaSpec(channels, basis="dstretch", signal=signal)

    def make(self, seed_key: str, channels: str = "") -> Aspect:
        """Add a colour layer above ``seed_key``; replace one with the same channels."""
        from fastfuncstuff.viewer.chroma import parse_channels

        chans = parse_channels(channels) if channels.strip() else self.guess_channels(seed_key)
        self._check(chans)
        spec = self.default_spec(chans)
        layers = self.session.state.layers
        first = layers.get(chans[0].key)
        text = format_channels(chans)
        existing = next(
            (ly for ly in layers if ly.chroma is not None and ly.chroma.channels == chans), None
        )
        if existing is not None:
            self.session.state.selected = existing.key
            return Aspect.LAYERS
        # Named from the channels rather than minted from the stack's counter:
        # the channels' keys replay identically, so a recorded SET_RGB still
        # finds this layer -- a minted key depends on everything made before.
        key = f"C{chans[0].key}"
        suffix = 1
        while layers.find(key) is not None:
            suffix += 1
            key = f"C{chans[0].key}_{suffix}"
        n = self._n_frames(chans)
        self.session.store.adopt_lazy(
            key, first.shape, n, lambda i, k=key: self._lightness_volume(k, i), name=text
        )
        names = [layers.get(c.key).name for c in chans]
        layers.add(
            Layer(
                key=key,
                name=f"RGB {_common_name(names)}",
                path=f"<rgb: {text}>",
                shape=first.shape,
                n_volumes=n,
                affine=first.affine,
                range_lo=0.0,
                range_hi=1.0,
                time_linked=n > 1,
                source=f"derived:rgb:{chans[0].key}",
                chroma=spec,
            ),
            at=max(layers.index_of(c.key) for c in chans) + 1,
        )
        # The channels are under an opaque colour layer now, and each one a
        # time series repaints every frame for nothing -- 4.9 ms a frame with
        # them hidden against 42 ms drawn, on three echoes. The bottom layer
        # stays: it is the anatomy outside the mask.
        base = layers.base
        for c in {c.key for c in chans}:
            if base is None or c != base.key:
                layers.update(c, visible=False)
        self.session.state.selected = key
        return Aspect.LAYERS | Aspect.SLICES | Aspect.GRAPH

    def set_param(self, key: str, name: str, value: str) -> Aspect:
        layer = self.session.state.layers.get(key)
        if layer.chroma is None:
            raise ValueError(f"{layer.name} is not a colour layer")
        spec = layer.chroma.with_param(name, value)
        if spec == layer.chroma:
            return Aspect.NOTHING
        if spec.channels != layer.chroma.channels:
            self._check(spec.channels)
            n = self._n_frames(spec.channels)
            first = self.session.state.layers.get(spec.channels[0].key)
            self.session.store.adopt_lazy(
                key,
                first.shape,
                n,
                lambda i, k=key: self._lightness_volume(k, i),
                name=format_channels(spec.channels),
            )
            self.session.state.layers.update(
                key,
                chroma=spec,
                n_volumes=n,
                time_linked=n > 1,
                path=f"<rgb: {format_channels(spec.channels)}>",
            )
        else:
            self.session.state.layers.update(key, chroma=spec)
        self.forget(key)
        return Aspect.SLICES | Aspect.LAYERS

    def forget(self, key: str | None = None) -> None:
        if key is None:
            self._fits.clear()
            self._frames.clear()
            self._bases.clear()
            return
        self._fits.pop(key, None)
        self._frames.pop(key, None)
        self.session.invalidate(key)

    def _check(self, chans: tuple[Channel, ...]) -> None:
        layers = self.session.state.layers
        missing = [c.key for c in chans if layers.find(c.key) is None]
        if missing:
            raise KeyError(f"no layer {', '.join(missing)}")
        shapes = {layers.get(c.key).shape for c in chans}
        if len(shapes) > 1:
            raise ValueError(f"channels must share a grid; got shapes {sorted(shapes)}")

    # -- reading channels -------------------------------------------------
    def _n_bricks(self, key: str) -> int:
        return int(self.session.store.get(key).info.n_volumes)

    def _n_frames(self, chans: tuple[Channel, ...]) -> int:
        timed = [c.n_frames(self._n_bricks(c.key)) for c in chans if c.stride > 0]
        return min(timed) if timed else 1

    def _positive(self, chan: Channel) -> bool:
        vol = self.session.volume(chan.key, chan.brick(0, self._n_bricks(chan.key)))
        return bool(np.nanpercentile(vol, 5) >= 0)

    def _echo_time(self, chan: Channel) -> float | None:
        layer = self.session.state.layers.find(chan.key)
        # Sub-bricks of one file share one sidecar, so their times come back
        # equal and the caller treats them as unknown.
        return None if layer is None else _sidecar_echo_time(layer.path)

    def _raw(self, chan: Channel, t: int) -> np.ndarray:
        return np.nan_to_num(
            np.asarray(
                self.session.volume(chan.key, chan.brick(t, self._n_bricks(chan.key))),
                dtype=np.float32,
            )
        )

    def _series(self, chan: Channel) -> Tensor | None:
        """A time-series channel's whole run on the display device, when it can live there.

        Playing a colour movie reads N frames per step; out of a host array
        stored (X, Y, Z, T) each is a strided gather plus an upload, ~25 ms a
        channel. On the device a frame is a slice. The store's promote keeps
        its own budget and evicts by recency, so this is a request, not a
        claim -- ``None`` (and the host path) when the run is not in RAM yet,
        the repaint is on another device (Metal), or the card has no room.
        """
        if chan.stride <= 0:
            return None
        store = self.session.store
        if store.device.type != self.session.display_device.type:
            return None
        if store.get(chan.key).array is None:
            return None
        try:
            return store.promote(chan.key)
        except (RuntimeError, MemoryError):
            return None

    def _channel_mean(self, chan: Channel, n_frames: int) -> Tensor:
        """A channel's mean over the layer's frames (its only frame, if static)."""
        device = self.session.display_device
        if chan.stride <= 0 or n_frames <= 1:
            return torch.as_tensor(self._raw(chan, 0), device=device)
        stop = chan.offset + chan.stride * (n_frames - 1) + 1
        series = self._series(chan)
        if series is not None:
            return torch.nan_to_num(series[..., chan.offset : stop : chan.stride].mean(-1))
        arr = self.session.store.ensure_ram(chan.key)
        picked = np.asarray(arr[..., chan.offset : stop : chan.stride], dtype=np.float32)
        return torch.as_tensor(np.nan_to_num(picked).mean(-1), device=device)

    # -- fitting ----------------------------------------------------------
    def _base(self, spec: ChromaSpec, n_frames: int) -> _Base:
        """Channel means, brain mask and log floor, on the display device.

        Cached on the channels' data alone, so changing the basis, the space or
        the stats region -- the things you flip between -- reuses the automask
        and refits only a covariance of N channels.
        """
        token = (
            spec.channels,
            tuple(self.session.store.get(c.key).generation for c in spec.channels),
            n_frames,
            str(self.session.display_device),
        )
        hit = self._bases.get(spec.channels)
        if hit is not None and hit.token == token:
            return hit
        from fastfuncstuff.processing.mask import automask

        device = self.session.display_device
        means = torch.stack([self._channel_mean(c, n_frames) for c in spec.channels])
        # One mask from the first channel for every channel: per-channel masks
        # disagree at the edge, and the disagreement is a coloured fringe.
        brain = automask(means[0], dilate_extra=1).to(device=device, dtype=torch.bool)
        if not bool(brain.any()):
            brain = means[0] > 0
        eps = 1e-6
        if bool(brain.any()):
            eps = max(float(torch.quantile(means[0][brain].float(), 0.98)) * 1e-3, 1e-6)
        base = _Base(token=token, means=means, brain=brain, eps=eps)
        self._bases[spec.channels] = base
        return base

    def _token(self, layer: Layer) -> tuple:
        spec = layer.chroma
        assert spec is not None
        gens = tuple(self.session.store.get(c.key).generation for c in spec.channels)
        stats_gen = ()
        if spec.stats not in ("core", "brain"):
            found = self.session.state.layers.find(spec.stats)
            stats_gen = (
                (self.session.store.get(spec.stats).generation,) if found is not None else ()
            )
        return (spec, gens, stats_gen, layer.n_volumes, str(self.session.display_device))

    def fit(self, key: str) -> ChromaFit:
        """The layer's fit, refitted only when its spec or its channels' data changed."""
        layer = self.session.state.layers.get(key)
        token = self._token(layer)
        hit = self._fits.get(key)
        if hit is not None and hit.token == token:
            return hit
        spec = layer.chroma
        assert spec is not None
        n = layer.n_volumes
        base = self._base(spec, n)
        device = base.means.device

        if spec.stats == "core":
            region = _erode(base.brain, CORE_EROSION)
            region = region if int(region.sum()) >= 64 else base.brain
        elif spec.stats == "brain":
            region = base.brain
        else:
            region = torch.as_tensor(self.session.mask_from_layer(spec.stats, layer), device=device)
            if int(region.sum()) < 8:
                raise ValueError(f"the stats mask {spec.stats} covers fewer than 8 voxels")

        if spec.signal == "psc":
            frames = np.unique(np.linspace(0, n - 1, min(PSC_FRAMES, n)).round().astype(int))
            # Calibrated on single frames whatever the window: z-scoring the
            # averaged frames would stretch the averaged noise straight back
            # up to full colour. Against one frame's variability, averaging
            # fades noise toward grey and leaves sustained change standing.
            single = replace(spec, window=1)
            rows = [
                transform(self._upload(single, int(t)), "psc", eps=base.eps, mean=base.means)[
                    :, region
                ].T
                for t in frames
            ]
            samples = torch.cat(rows)
        else:
            samples = transform(base.means, spec.signal, eps=base.eps)[:, region].T
        if samples.shape[0] > MAX_SAMPLES:
            gen = torch.Generator().manual_seed(0)
            samples = samples[
                torch.randperm(samples.shape[0], generator=gen)[:MAX_SAMPLES].to(device)
            ]

        te_ms, te_source = self._echo_times(spec)
        cmap = fit_map(samples, spec, te_ms)

        anatomy = None
        if spec.lightness == "anatomy":
            ref = base.means[0]
            lo, hi = torch.quantile(
                ref[region].float(), torch.tensor([0.02, 0.995], device=device)
            ).tolist()
            anatomy = ((ref - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0).float()

        fit = ChromaFit(
            token=token,
            map=cmap.to(device),
            brain=base.brain if spec.masked else torch.ones_like(base.brain),
            mean=base.means if spec.signal == "psc" else None,
            anatomy=anatomy,
            eps=base.eps,
            te_ms=None if te_ms is None else tuple(te_ms),
            te_source=te_source,
        )
        self._fits[key] = fit
        self._frames.pop(key, None)
        return fit

    def _echo_times(self, spec: ChromaSpec) -> tuple[list[float] | None, str]:
        typed = spec.te_ms()
        if typed is not None:
            return typed, "typed"
        found = [self._echo_time(c) for c in spec.channels]
        if all(t is not None for t in found) and len(set(found)) == len(found):
            return [float(t) for t in found if t is not None], "sidecar"
        return None, "index"

    # -- drawing ----------------------------------------------------------
    def _upload(self, spec: ChromaSpec, t: int) -> Tensor:
        """(N, X, Y, Z) raw channels at ``t`` on the display device, window-averaged."""
        device = self.session.display_device
        half = (max(int(spec.window), 1) - 1) // 2
        vols = []
        for c in spec.channels:
            n = c.n_frames(self._n_bricks(c.key))
            times = range(max(t - half, 0), min(t + half, n - 1) + 1) if c.stride > 0 else [t]
            series = self._series(c)
            if series is None:
                frames = [torch.as_tensor(self._raw(c, u), device=device) for u in times]
                vols.append(torch.stack(frames).mean(0))
            else:
                nb = series.shape[-1]
                bricks = torch.tensor([c.brick(u, nb) for u in times], device=series.device)
                vols.append(torch.nan_to_num(series.index_select(-1, bricks).mean(-1)))
        return torch.stack(vols)

    def frame(self, key: str, t: int) -> Tensor:
        """(N + 2, X, Y, Z) at time ``t``: transformed channels, then mask and lightness.

        Mask and anatomy ride in the same tensor so a plane is one sample of
        every channel the colour needs, not one per channel.
        """
        layer = self.session.state.layers.get(key)
        fit = self.fit(key)
        t = max(0, min(int(t), layer.n_volumes - 1))
        cache = (t, fit.token)
        hit = self._frames.get(key)
        if hit is not None and hit[0] == cache:
            return hit[1]
        spec = layer.chroma
        assert spec is not None
        chans = transform(self._upload(spec, t), spec.signal, eps=fit.eps, mean=fit.mean)
        light = fit.anatomy if fit.anatomy is not None else torch.zeros_like(chans[0])
        out = torch.cat([chans, fit.brain.to(chans.dtype)[None], light[None]])
        self._frames[key] = (cache, out)
        return out

    def time_of(self, layer: Layer) -> int:
        return self.session.state.time_index if layer.time_linked else 0

    def render(
        self,
        layer: Layer,
        grid: DisplayGrid,
        plane: Plane,
        position: int,
        *,
        view: PlaneView | None,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        """One plane of a colour layer: ``(H, W, 3)`` colour and ``(H, W)`` alpha."""
        from fastfuncstuff.viewer.slicing import extract_plane

        fit = self.fit(layer.key)
        vol = self.frame(layer.key, self.time_of(layer))
        planes = extract_plane(vol, grid, layer.affine, plane, position, view=view, mode=mode)
        n = vol.shape[0] - 2
        light = planes[n + 1] if fit.anatomy is not None else None
        rgb = fit.map.colors(planes[:n], light)
        # Half-way on the interpolated mask is the nearest-neighbour edge
        # without a second sample.
        alpha = (planes[n] > 0.5).to(rgb.dtype) * float(layer.opacity)
        return rgb, alpha

    def _lightness_volume(self, key: str, index: int) -> np.ndarray:
        """The store's view of a colour layer: its first coordinate, windowed to [0, 1]."""
        layer = self.session.state.layers.find(key)
        if layer is None or layer.chroma is None:
            raise KeyError(key)
        fit = self.fit(key)
        vol = self.frame(key, index)
        u = fit.map.unit(fit.map.coords(vol[:-2]))[0]
        return (0.5 + 0.5 * u).clamp(0.0, 1.0).mul(vol[-2]).cpu().numpy()

    def rgb_volume(self, key: str, t: int) -> np.ndarray:
        """(X, Y, Z, 3) uint8 of one frame, on the layer's own grid, black outside the mask."""
        layer = self.session.state.layers.get(key)
        fit = self.fit(key)
        vol = self.frame(key, t)
        rgb = fit.map.colors(vol[:-2], fit.anatomy) * vol[-2].unsqueeze(-1)
        return (
            (rgb * 255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .reshape(*layer.shape, 3)
        )

    def save(self, key: str, path: str | Path) -> Path:
        """Write the colours as an RGB24 NIfTI (datatype 128), every frame.

        The colours, not the channels: this is for viewers and figures, and a
        file that needs the fit to be redone to look right would be neither.
        """
        import nibabel as nib

        layer = self.session.state.layers.get(key)
        frames = np.stack([self.rgb_volume(key, t) for t in range(layer.n_volumes)], axis=3)
        if frames.shape[3] == 1:
            frames = frames[:, :, :, 0]
        packed = np.empty(frames.shape[:-1], dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
        packed["R"], packed["G"], packed["B"] = frames[..., 0], frames[..., 1], frames[..., 2]
        img = nib.Nifti1Image(packed, np.asarray(layer.affine, dtype=float))
        img.header.set_xyzt_units("mm", "sec")
        out = Path(path)
        nib.save(img, str(out))
        return out

    # -- reading out ------------------------------------------------------
    def readout(self, layer: Layer, ijk: tuple[int, int, int]) -> str:
        """``e1 812  e2 540  e3 377 | lvl -0.4  R2* +1.2  curv +0.1`` at a display voxel.

        Raw channel values, then the basis coordinates in window units (0 at
        the stats region's median, +-1 at the window edge) -- the numbers the
        colour is made of, so a hue can be checked rather than guessed.
        """
        spec = layer.chroma
        assert spec is not None
        i, j, k = self.session.layer_voxel(layer.affine, ijk)
        nx, ny, nz = layer.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return "--"
        t = self.time_of(layer)
        raw = [float(self._raw(c, t)[i, j, k]) for c in spec.channels]
        fit = self.fit(layer.key)
        u = fit.map.unit(fit.map.coords(self.frame(layer.key, t)[:-2, i, j, k].reshape(-1, 1)))[
            :, 0
        ]
        one_file = len({c.key for c in spec.channels}) == 1
        tags = [f"#{c.offset}" if one_file else c.key for c in spec.channels]
        values = "  ".join(f"{tag} {v:.4g}" for tag, v in zip(tags, raw, strict=True))
        coords = "  ".join(
            f"{n} {float(v):+.2f}" for n, v in zip(coordinate_names(spec), u, strict=True)
        )
        return f"{values}  |  {coords}"

    def describe(self, layer: Layer) -> str:
        """One line on what the colours mean, for the panel."""
        spec = layer.chroma
        assert spec is not None
        try:
            fit = self.fit(layer.key)
        except (ValueError, KeyError) as exc:
            return f"cannot fit: {exc}"
        names = coordinate_names(spec)
        if spec.resolved_space == "oklab":
            where = f"lightness {('anatomy' if spec.lightness == 'anatomy' else names[0])}, blue–yellow {names[1]}, green–red {names[2]}"
        else:
            where = f"R {names[0]}, G {names[1]}, B {names[2]}"
        te = ""
        if spec.basis == "te":
            te = (
                f"; TE {', '.join(f'{v:g}' for v in fit.te_ms)} ms ({fit.te_source})"
                if fit.te_ms
                else "; TE unknown, echoes taken as evenly spaced"
            )
        return where + te


def controls(session: ViewerSession, layer: Layer) -> tuple[Control, ...]:
    """The panel a colour layer shows, as declarations (rendered by ui.controls)."""
    from fastfuncstuff.viewer.modes.base import (
        HALF,
        THIRD,
        BoolControl,
        ChoiceControl,
        FloatControl,
        IntControl,
        TextControl,
    )

    masks = tuple(
        ly.key for ly in session.state.layers if ly.chroma is None and ly.shape == layer.shape
    )
    return (
        ChoiceControl(
            name="basis",
            label="basis",
            choices=BASES,
            default="dstretch",
            style="radio",
            help="pick: three channels as R, G, B, each standardised -- the baseline, "
            "mostly one hue when channels correlate.\n"
            "dstretch: decorrelation stretch -- whiten, rotate back; what differs "
            "between channels gets the colour.\n"
            "pca: whitened principal components; most contrast, hue has no fixed meaning.\n"
            "te: Legendre fit in echo time -- level, decay rate (R2*), curvature; "
            "the same hue means the same physics in every subject.",
        ),
        ChoiceControl(
            name="signal",
            label="signal",
            choices=SIGNALS,
            default="log",
            span=THIRD,
            newline=True,
            help="What is fitted. log: decay is linear in log signal (echoes). "
            "raw: the values as stored (betas, signed maps). psc: percent change "
            "from each voxel's mean -- what moves over time; needs a time series.",
        ),
        ChoiceControl(
            name="space",
            label="space",
            choices=SPACES,
            default="auto",
            span=THIRD,
            help="rgb: coordinates to R, G, B. oklab: first coordinate to "
            "lightness, the other two to blue-yellow and green-red, so a hue "
            "never reads as brightness. auto: oklab for te and pca.",
        ),
        ChoiceControl(
            name="lightness",
            label="lightness",
            choices=LIGHTNESS,
            default="basis",
            span=THIRD,
            help="basis: brightness from the first coordinate. anatomy: from the "
            "first channel's mean, nudged by the first coordinate -- keeps a "
            "percent-change movie anatomical instead of grey.",
        ),
        FloatControl(
            name="spread",
            label="spread",
            lo=0.5,
            hi=6.0,
            step=0.1,
            default=2.5,
            span=THIRD,
            newline=True,
            unit="sd",
            help="Window half-width in standard deviations of the stats region. "
            "Smaller is more saturated and clips sooner.",
        ),
        FloatControl(
            name="chroma",
            label="chroma",
            lo=0.02,
            hi=0.4,
            step=0.01,
            default=0.25,
            span=THIRD,
            help="Oklab saturation at full swing. Past ~0.25 strong colours leave "
            "the sRGB gamut and clip.",
        ),
        FloatControl(
            name="floor",
            label="floor",
            lo=0.0,
            hi=0.5,
            step=0.01,
            default=0.05,
            span=THIRD,
            help="dstretch/pca: components weaker than this fraction of the "
            "strongest are not amplified further. 0 is a full whitening, which "
            "turns the noisiest component into speckle.",
        ),
        ChoiceControl(
            name="stats",
            label="stats from",
            choices=("core", "brain", *masks),
            default="core",
            span=HALF,
            newline=True,
            help="Where means and covariance are measured. core: the brain mask "
            "eroded, so the dropout fringe cannot take over the stretch. brain: "
            "the whole mask. A layer: its mask (an ROI, or automasked) -- "
            "stretches the colour to what differs inside it.",
        ),
        BoolControl(
            name="masked",
            label="mask",
            default=True,
            span=HALF,
            help="Hide voxels outside the first channel's automask.",
        ),
        IntControl(
            name="window",
            label="window",
            lo=1,
            hi=61,
            default=1,
            span=HALF,
            newline=True,
            help="Frames averaged around the shown one. One frame of percent "
            "change is mostly thermal noise; 5-15 frames shows a breath-hold or "
            "a block. Costs nothing on the GPU once the run is resident.",
        ),
        TextControl(
            name="channels",
            label="channels",
            default="",
            newline=True,
            help="KEY[:OFFSET[+STRIDE]] per channel, comma-separated. L1,L2,L3 for "
            "three files; L1:0+3,L1:1+3,L1:2+3 for three echoes interleaved in one "
            "run; L4:2 for one fixed sub-brick.",
        ),
        TextControl(
            name="pick",
            label="pick",
            default="",
            placeholder="first, mid, last",
            span=HALF,
            newline=True,
            help="Which three channels pick and dstretch show, as 0-based indices.",
        ),
        TextControl(
            name="te",
            label="TE ms",
            default="",
            placeholder="from sidecars",
            span=HALF,
            help="Echo times for the te basis, comma-separated. Empty reads each "
            "channel's BIDS sidecar; without them echoes are taken as evenly spaced.",
        ),
    )


def control_values(spec: ChromaSpec) -> dict[str, object]:
    return {
        "basis": spec.basis,
        "signal": spec.signal,
        "space": spec.space,
        "lightness": spec.lightness,
        "spread": spec.spread,
        "chroma": spec.chroma,
        "floor": spec.floor,
        "stats": spec.stats,
        "masked": spec.masked,
        "channels": format_channels(spec.channels),
        "pick": spec.pick,
        "te": spec.te,
        "window": spec.window,
    }


def coordinate_names(spec: ChromaSpec) -> tuple[str, str, str]:
    """What each of the three coordinates is, in the words the readout uses."""
    if spec.basis == "te":
        if spec.signal == "psc":
            return ("S0", "R2*", "curv")
        return ("level", "R2*", "curv")
    if spec.basis == "pca":
        return ("PC1", "PC2", "PC3")
    p = spec.picked()
    tag = "ds" if spec.basis == "dstretch" else "ch"
    return (f"{tag}{p[0]}", f"{tag}{p[1]}", f"{tag}{p[2]}")


def _common_name(names: list[str]) -> str:
    """The shared prefix of the channel names, or the first name when there is none worth keeping."""
    import os

    prefix = os.path.commonprefix(names).rstrip("_-. e")
    return prefix if len(prefix) >= 3 else names[0]
