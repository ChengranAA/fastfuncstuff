"""What the RGB mode draws: channels read once, a fit, and a look applied per draw.

The arithmetic is :mod:`viewer.chroma`; this is the part that knows about files,
the store and the display. Three tiers, by cost:

* :class:`ChannelSet` -- the files read (through the store, so they share its
  RAM and device budgets with everything else), each channel's mean, one brain
  mask. Seconds on a real run; made on the mode's worker, once per file list.
* the fit -- the 3 x N map from a basis and a stats region. Tens of ms.
* the :class:`~viewer.chroma.Look` -- shift, stretch, hue, gamma, spread.
  Applied per draw, so dragging a slider is a repaint.

A colour layer holds no voxels of its own. Time-series channels are promoted to
the device by the store's own promote, so a frame of a colour movie is a slice
and a matmul rather than a gather out of host memory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from fastfuncstuff.viewer.chroma import (
    ChromaMap,
    ChromaSpec,
    Look,
    fit_map,
    transform,
)

if TYPE_CHECKING:
    from fastfuncstuff.viewer.layers import Layer
    from fastfuncstuff.viewer.residency import VolumeStore
    from fastfuncstuff.viewer.slicing import PlaneView
    from fastfuncstuff.viewer.state import DisplayGrid, Plane

#: Frames read to estimate the covariance of percent change. Enough to see
#: the dynamics; reading every frame of a long run would only add time.
PSC_FRAMES = 16
#: Rows of the sample matrix a fit keeps. A covariance of a few channels is
#: settled long before a million voxels.
MAX_SAMPLES = 400_000
#: The brain mask is eroded this many voxels for ``stats=core``.
CORE_EROSION = 2


def _strip_selector(entry: str) -> str:
    from fastfuncstuff.io.headers import parse_subbrick_selector

    return str(parse_subbrick_selector(entry)[0])


def echo_time_ms(entry: str) -> float | None:
    """EchoTime from the BIDS JSON beside a dataset, in ms."""
    from fastfuncstuff.viewer.tools.slicetime import sidecar_for

    try:
        sidecar = sidecar_for(Path(_strip_selector(entry)))
        if sidecar is None:
            return None
        value = json.loads(sidecar.read_text()).get("EchoTime")
    except (OSError, ValueError):
        return None
    return None if value is None else float(value) * 1000.0


def short_name(entry: str) -> str:
    """A channel's name for readouts: the file stem, plus its selector if it has one."""
    from fastfuncstuff.viewer.tools.slicetime import dataset_stem

    base = _strip_selector(entry)
    selector = entry[len(base) :] if entry.startswith(base) else ""
    return f"{dataset_stem(Path(base))}{selector}"


def distinct_names(names: list[str]) -> tuple[str, ...]:
    """Channel names with what they all share cut off: ``e1 e2 e3``, not three run names.

    Names with nothing in common -- ``INV1``, ``UNI`` -- are left whole.
    """
    import os

    if len(names) < 2:
        return tuple(names)
    head = len(os.path.commonprefix(names))
    tail = len(os.path.commonprefix([n[::-1] for n in names]))
    # Back up to a separator, so ``_e1`` keeps its letter and ``run12`` is not ``2``.
    while head and names[0][head - 1] not in "_-. ":
        head -= 1
    cut = [n[head : len(n) - tail if tail else None] for n in names]
    if any(c.count("[") != c.count("]") for c in cut):
        # The shared tail ran into a sub-brick selector; keep selectors whole.
        cut = [n[head:] for n in names]
    if any(not c for c in cut) or len(set(cut)) < len(cut):
        return tuple(names)
    return tuple(cut)


def _erode(mask: Tensor, n: int) -> Tensor:
    """Peel ``n`` voxels off a mask (26-neighbour), on its device."""
    out = mask.float()[None, None]
    for _ in range(n):
        out = -torch.nn.functional.max_pool3d(-out, 3, stride=1, padding=1)
    return out[0, 0] > 0.5


@dataclass
class ChannelSet:
    """The channels a colour view reads, and what every fit of them shares."""

    entries: tuple[str, ...]
    names: tuple[str, ...]
    #: Store keys: a loaded layer's own key when the file is already a layer,
    #: so its RAM and device copies are shared rather than duplicated.
    keys: tuple[str, ...]
    shape: tuple[int, int, int]
    affine: np.ndarray
    #: Frames per channel. A one-volume channel is static and is read at every
    #: time point -- an anatomy beside the echoes of a run.
    frames: tuple[int, ...]
    #: (N, X, Y, Z) per-channel temporal mean, on the display device.
    means: Tensor
    #: (X, Y, Z) bool, automask of the first channel.
    brain: Tensor
    #: Floor for log signal: a thousandth of bright tissue.
    eps: float
    store: VolumeStore = field(repr=False)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    @property
    def n(self) -> int:
        return len(self.entries)

    @property
    def n_frames(self) -> int:
        timed = [f for f in self.frames if f > 1]
        return min(timed) if timed else 1

    def echo_times(self, spec: ChromaSpec) -> tuple[list[float] | None, str]:
        typed = spec.te_ms(self.n)
        if typed is not None:
            return typed, "typed"
        found = [echo_time_ms(e) for e in self.entries]
        if all(t is not None for t in found) and len(set(found)) == len(found):
            return [float(t) for t in found if t is not None], "sidecar"
        return None, "index"

    def _series(self, c: int) -> Tensor | None:
        """Channel ``c``'s whole run on the display device, when it can live there.

        Out of a host array stored (X, Y, Z, T) a frame is a strided gather
        plus an upload, ~25 ms a channel; on the device it is a slice. The
        store's promote keeps its own budget and evicts by recency, so this is
        a request -- ``None`` (and the host path) when the run is not in RAM,
        the display is on another device (Metal repaints on the CPU), or the
        card has no room.
        """
        if self.frames[c] <= 1 or self.store.device.type != self.device.type:
            return None
        if self.store.get(self.keys[c]).array is None:
            return None
        try:
            return self.store.promote(self.keys[c])
        except (RuntimeError, MemoryError):
            return None

    def _volume(self, c: int, t: int) -> Tensor:
        res = self.store.get(self.keys[c])
        t = max(0, min(t, self.frames[c] - 1))
        arr = res.array[..., t] if res.array is not None else self.store.preview(self.keys[c], t)
        return torch.as_tensor(np.nan_to_num(np.asarray(arr, dtype=np.float32)), device=self.device)

    def raw(self, t: int, window: int = 1) -> Tensor:
        """(N, X, Y, Z) channels at ``t``, averaged over ``window`` frames centred on it."""
        half = (max(int(window), 1) - 1) // 2
        vols = []
        for c in range(self.n):
            if self.frames[c] <= 1:
                vols.append(self._volume(c, 0))
                continue
            times = range(max(t - half, 0), min(t + half, self.frames[c] - 1) + 1)
            series = self._series(c)
            if series is None:
                vols.append(torch.stack([self._volume(c, u) for u in times]).mean(0))
            else:
                idx = torch.tensor(list(times), device=series.device)
                vols.append(torch.nan_to_num(series.index_select(-1, idx).mean(-1)))
        return torch.stack(vols)

    def mean_of(self, c: int) -> Tensor:
        if self.frames[c] <= 1:
            return self._volume(c, 0)
        series = self._series(c)
        if series is not None:
            return torch.nan_to_num(series.mean(-1))
        arr = self.store.ensure_ram(self.keys[c])
        return torch.as_tensor(np.nan_to_num(arr.mean(-1, dtype=np.float32)), device=self.device)


def load_channels(
    store: VolumeStore,
    entries: list[str],
    *,
    loaded: dict[str, str],
    device: torch.device,
) -> ChannelSet:
    """Read the channels named by ``entries`` (paths, AFNI selectors allowed).

    ``loaded`` maps a path to the key of a layer already holding it. Blocks
    while files inflate, so it belongs on a worker.
    """
    from fastfuncstuff.io.dsetinfo import read_info
    from fastfuncstuff.processing.mask import automask

    if len(entries) < 2:
        raise ValueError("tick at least two channels")
    keys, shapes, affines, frames = [], [], [], []
    for entry in entries:
        key = loaded.get(entry) or f"rgb:{entry}"
        if key not in store.keys():
            store.open(entry, key)
        info = store.get(key).info
        keys.append(key)
        shapes.append(tuple(int(v) for v in info.shape[:3]))
        frames.append(int(info.n_volumes))
        affines.append(np.asarray(read_info(Path(_strip_selector(entry))).affine, dtype=float))
    if len(set(shapes)) > 1:
        detail = ", ".join(f"{short_name(e)} {s}" for e, s in zip(entries, shapes, strict=True))
        raise ValueError(f"channels must share a grid: {detail}")
    timed = {f for f in frames if f > 1}
    if len(timed) > 1:
        raise ValueError(f"time-series channels differ in length: {sorted(timed)} frames")
    for key in keys:
        store.ensure_ram(key)

    chans = ChannelSet(
        entries=tuple(entries),
        names=distinct_names([short_name(e) for e in entries]),
        keys=tuple(keys),
        shape=shapes[0],  # type: ignore[arg-type]
        affine=affines[0],
        frames=tuple(frames),
        means=torch.empty(0),
        brain=torch.empty(0),
        eps=1e-6,
        store=store,
        device=device,
    )
    chans.means = torch.stack([chans.mean_of(c) for c in range(chans.n)])
    # One mask from the first channel for every channel: per-channel masks
    # disagree at the edge, and the disagreement is a coloured fringe.
    brain = automask(chans.means[0], dilate_extra=1).to(device=device, dtype=torch.bool)
    if not bool(brain.any()):
        brain = chans.means[0] > 0
    chans.brain = brain
    if bool(brain.any()):
        chans.eps = max(float(torch.quantile(chans.means[0][brain].float(), 0.98)) * 1e-3, 1e-6)
    return chans


@dataclass
class ColourSource:
    """A fitted colour view of a channel set: what a colour layer draws from."""

    chans: ChannelSet
    spec: ChromaSpec
    look: Look
    map: ChromaMap
    #: (X, Y, Z) first-channel mean as a [0, 1] lightness, for ``lightness=anatomy``.
    anatomy: Tensor
    te_ms: tuple[float, ...] | None
    te_source: str
    masked: bool = True
    window: int = 1
    _frames: dict[int, Tensor] = field(default_factory=dict, repr=False)

    def with_look(self, look: Look, *, masked: bool, window: int) -> ColourSource:
        """The same fit drawn differently. Frames survive unless the window changed."""
        frames = self._frames if window == self.window and masked == self.masked else {}
        return replace(self, look=look, masked=masked, window=window, _frames=frames)

    @property
    def n_frames(self) -> int:
        return self.chans.n_frames

    # -- frames ----------------------------------------------------------
    def frame(self, t: int) -> Tensor:
        """(N + 2, X, Y, Z) at ``t``: transformed channels, then mask and lightness.

        Mask and anatomy ride in the same tensor so a plane is one sample of
        everything the colour needs. One frame is cached -- panes share it.
        """
        t = max(0, min(int(t), self.n_frames - 1))
        hit = self._frames.get(t)
        if hit is not None:
            return hit
        mean = self.chans.means if self.spec.signal == "psc" else None
        chans = transform(
            self.chans.raw(t, self.window), self.spec.signal, eps=self.chans.eps, mean=mean
        )
        mask = self.chans.brain if self.masked else torch.ones_like(self.chans.brain)
        out = torch.cat([chans, mask.to(chans.dtype)[None], self.anatomy[None]])
        self._frames.clear()
        self._frames[t] = out
        return out

    def colors(self, vol: Tensor) -> Tensor:
        """(N + 2, ...) frame or plane -> (..., 3) colour."""
        n = self.chans.n
        light = vol[n + 1] if self.look.lightness == "anatomy" else None
        return self.map.colors(vol[:n], self.look, light)

    def render(
        self,
        layer: Layer,
        grid: DisplayGrid,
        plane: Plane,
        position: int,
        t: int,
        *,
        view: PlaneView | None,
        mode: str,
    ) -> tuple[Tensor, Tensor]:
        """One plane: ``(H, W, 3)`` colour and ``(H, W)`` alpha."""
        from fastfuncstuff.viewer.slicing import extract_plane

        vol = self.frame(t)
        planes = extract_plane(vol, grid, layer.affine, plane, position, view=view, mode=mode)
        rgb = self.colors(planes)
        # Half-way on the interpolated mask is the nearest-neighbour edge
        # without a second sample.
        alpha = (planes[self.chans.n] > 0.5).to(rgb.dtype) * float(layer.opacity)
        return rgb, alpha

    def units(self, t: int) -> Tensor:
        """(3, X, Y, Z) coordinates at ``t`` in window units, the look applied."""
        vol = self.frame(t)
        return self.map.unit(self.map.coords(vol[: self.chans.n]), self.look)

    def lightness_volume(self, t: int) -> np.ndarray:
        """A scalar stand-in for the store: the first coordinate in [0, 1]."""
        u = self.units(t)[0]
        return (0.5 + 0.5 * u).clamp(0.0, 1.0).mul(self.frame(t)[-2]).cpu().numpy()

    def rgb_volume(self, t: int) -> np.ndarray:
        """(X, Y, Z, 3) uint8 of one frame, black outside the mask."""
        vol = self.frame(t)
        rgb = self.colors(vol) * vol[-2].unsqueeze(-1)
        return (rgb * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()

    def save(self, path: str | Path) -> Path:
        """Write the colours as an RGB24 NIfTI (datatype 128), every frame.

        The colours, not the channels: this is for viewers and figures, and a
        file that needs the fit redone to look right would be neither.
        """
        import nibabel as nib

        frames = np.stack([self.rgb_volume(t) for t in range(self.n_frames)], axis=3)
        if frames.shape[3] == 1:
            frames = frames[:, :, :, 0]
        packed = np.empty(frames.shape[:-1], dtype=[("R", "u1"), ("G", "u1"), ("B", "u1")])
        packed["R"], packed["G"], packed["B"] = frames[..., 0], frames[..., 1], frames[..., 2]
        img = nib.Nifti1Image(packed, np.asarray(self.chans.affine, dtype=float))
        img.header.set_xyzt_units("mm", "sec")
        out = Path(path)
        nib.save(img, str(out))
        return out

    # -- the crosshair -------------------------------------------------------
    def at(self, ijk: tuple[int, int, int], t: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Raw channel values and the three coordinates (window units) at a voxel."""
        i, j, k = ijk
        nx, ny, nz = self.chans.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return None
        raw = self.chans.raw(t, self.window)[:, i, j, k]
        mean = self.chans.means[:, i, j, k] if self.spec.signal == "psc" else None
        x = transform(raw, self.spec.signal, eps=self.chans.eps, mean=mean)
        u = self.map.unit(self.map.coords(x.reshape(-1, 1)), self.look)[:, 0]
        return raw.cpu().numpy(), u.cpu().numpy()

    def readout(self, ijk: tuple[int, int, int], t: int) -> str:
        """``e1 812  e2 540 | level -0.4  R2* +1.2  curv +0.1`` -- what the colour is made of."""
        got = self.at(ijk, t)
        if got is None:
            return "--"
        raw, u = got
        values = "  ".join(f"{n} {v:.4g}" for n, v in zip(self.chans.names, raw, strict=True))
        coords = "  ".join(f"{n} {v:+.2f}" for n, v in zip(self.coordinate_names(), u, strict=True))
        return f"{values}  |  {coords}"

    def similar(self, ijk: tuple[int, int, int], t: int, tolerance: float) -> np.ndarray:
        """Voxels within ``tolerance`` (window units) of the crosshair's coordinates.

        The coordinates are what the colour is, so this is "select by colour"
        with the look's stretch counted: stretching a dimension makes it count
        for more in what is similar, which is the point of stretching it.
        """
        u = self.units(t)
        i, j, k = ijk
        here = u[:, i, j, k].reshape(3, 1, 1, 1)
        dist = torch.linalg.vector_norm(u - here, dim=0)
        keep = (dist <= tolerance) & self.chans.brain
        return keep.cpu().numpy()

    def coordinate_names(self) -> tuple[str, str, str]:
        return coordinate_names(self.spec, self.chans.names)

    def describe(self) -> str:
        """One line on what the colours mean."""
        names = self.coordinate_names()
        if self.look.space == "oklab":
            first = "anatomy" if self.look.lightness == "anatomy" else names[0]
            where = f"lightness {first}, blue–yellow {names[1]}, green–red {names[2]}"
        else:
            where = f"R {names[0]}, G {names[1]}, B {names[2]}"
        if self.look.hue:
            where += f" (hue turned {self.look.hue:+.0f}°)"
        if self.spec.basis == "te":
            if self.te_ms:
                tes = ", ".join(f"{v:g}" for v in self.te_ms)
                where += f"; TE {tes} ms ({self.te_source})"
            else:
                where += "; TE unknown, echoes taken as evenly spaced"
        return where


def lightness_from(ref: Tensor, region: Tensor) -> Tensor:
    """A volume as a [0, 1] lightness: floor from the whole volume, ceiling from the region.

    Split so a mask or a label image used as lightness -- every value in the
    region the same -- still has a window, rather than a zero-width one.
    """
    flat = ref.float().reshape(-1)[:16_000_000]
    vals = ref[region].float()[:16_000_000]
    lo = float(torch.quantile(flat, 0.02))
    hi = float(torch.quantile(vals, 0.995)) if vals.numel() else float(flat.max())
    if hi <= lo:
        hi = float(flat.max())
    return ((ref - lo) / max(hi - lo, 1e-12)).clamp(0.0, 1.0).float()


def fit_source(
    chans: ChannelSet,
    spec: ChromaSpec,
    look: Look,
    *,
    stats: str = "core",
    stats_mask: np.ndarray | None = None,
    fit_on: str = "auto",
    light_ref: Tensor | None = None,
    masked: bool = True,
    window: int = 1,
) -> ColourSource:
    """Fit the 3 x N map to a channel set, from the stats region.

    ``fit_on`` is static against dynamic: ``mean`` reads the covariance of
    the mean image across voxels (anatomy colour), ``frames`` of single frames
    across voxels and time (change colour). ``auto`` is frames for percent
    change, whose mean is zero by construction, and the mean otherwise.
    ``light_ref`` is the volume lightness comes from when the look asks for
    anatomy; the first channel's mean when ``None``.
    """
    device = chans.means.device
    if stats_mask is not None:
        region = torch.as_tensor(stats_mask, device=device, dtype=torch.bool)
        if int(region.sum()) < 8:
            raise ValueError("the stats region covers fewer than 8 voxels")
    elif stats == "core":
        region = _erode(chans.brain, CORE_EROSION)
        region = region if int(region.sum()) >= 64 else chans.brain
    else:
        region = chans.brain

    dynamic = fit_on == "frames" or (fit_on == "auto" and spec.signal == "psc")
    if spec.signal == "psc" and not dynamic:
        raise ValueError("percent change has a zero mean; fit it on frames")
    if dynamic and chans.n_frames < 2:
        raise ValueError("fitting on frames needs time-series channels")
    mean = chans.means if spec.signal == "psc" else None
    if dynamic:
        n = chans.n_frames
        frames = np.unique(np.linspace(0, n - 1, min(PSC_FRAMES, n)).round().astype(int))
        # Single frames whatever the window: z-scoring averaged frames would
        # stretch the averaged noise straight back up to full colour. Against
        # one frame's variability, averaging fades noise toward grey and
        # leaves sustained change standing.
        rows = [
            transform(chans.raw(int(t)), spec.signal, eps=chans.eps, mean=mean)[:, region].T
            for t in frames
        ]
        samples = torch.cat(rows)
    else:
        samples = transform(chans.means, spec.signal, eps=chans.eps)[:, region].T
    if samples.shape[0] > MAX_SAMPLES:
        gen = torch.Generator().manual_seed(0)
        samples = samples[torch.randperm(samples.shape[0], generator=gen)[:MAX_SAMPLES].to(device)]

    te_ms, te_source = chans.echo_times(spec)
    cmap = fit_map(samples, spec, te_ms).to(device)
    ref = chans.means[0] if light_ref is None else light_ref.to(device)
    return ColourSource(
        chans=chans,
        spec=spec,
        look=look,
        map=cmap,
        anatomy=lightness_from(ref, region if light_ref is None else ref != 0),
        te_ms=None if te_ms is None else tuple(te_ms),
        te_source=te_source,
        masked=masked,
        window=window,
    )


def coordinate_names(spec: ChromaSpec, names: tuple[str, ...]) -> tuple[str, str, str]:
    """What each of the three coordinates is, in the words the readout uses."""
    if spec.basis == "te":
        return ("S0", "R2*", "curv") if spec.signal == "psc" else ("level", "R2*", "curv")
    if spec.basis == "pca":
        return ("PC1", "PC2", "PC3")
    if spec.basis == "custom":
        return ("c1", "c2", "c3")
    r, g, b = spec.picked(len(names))
    tag = "ds:" if spec.basis == "dstretch" else ""
    return (f"{tag}{names[r]}", f"{tag}{names[g]}", f"{tag}{names[b]}")
