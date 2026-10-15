"""RGB: several images of one brain, as one colour image you can push around.

Echoes, MP2RAGE's INV1 / INV2 / UNI, a T1 beside a T2, three betas: any files on
one grid, chosen by hand -- in any order, any names, an echo ticked off -- and
drawn as one colour per voxel. The arithmetic is :mod:`viewer.chroma` (why
three echoes in R, G, B is grey, and the bases that fix it); the plumbing is
:mod:`viewer.chromalayer`.

The panel is three groups by cost, which is what makes it a contrast
manipulator rather than a render setting:

* **channels** -- which files. Read once, on the worker.
* **fit** -- basis, signal, stats region, R/G/B, TE, custom weights. Tens of ms.
* **look** -- each coordinate's shift and stretch, hue, gamma, spread, chroma,
  lightness. Applied per draw: dragging a slider is a repaint.

Two things aim it at finding areas. Stats can come from the current slice or a
sphere around the crosshair and refit as you move, so the stretch is a lens
tuned to wherever you are looking. And SELECT takes every voxel whose colour
coordinates are near the crosshair's as a mask layer -- select by colour, with
the stretches you set counting in what "near" means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.viewer.chroma import (
    BASES,
    SIGNALS,
    ChromaSpec,
    Look,
    resolve_space,
    sibling_order,
    transform,
)
from fastfuncstuff.viewer.chromalayer import (
    ChannelSet,
    ColourSource,
    fit_source,
    load_channels,
    short_name,
)
from fastfuncstuff.viewer.commands import Aspect, Command
from fastfuncstuff.viewer.modes.base import (
    HALF,
    THIRD,
    ActionControl,
    BoolControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    FloatControl,
    IntControl,
    Mode,
    OverlayKind,
    PathControl,
    PathListControl,
    ProgressFn,
    TextControl,
    Trace,
    mode,
)

IMAGE_FILTER = "Images (*.nii *.nii.gz *.nii.zst *.HEAD);;All (*)"
STATS_FIXED = ("core", "brain", "slice", "crosshair")
LIGHT_FIXED = ("basis", "channel 1")


def _layer_choice(layer) -> str:
    """A loaded layer as a choice: its key leads, so the choice can be parsed back."""
    return f"{layer.key} {layer.name}"


@mode
class RGBMode(Mode):
    name = "rgb"
    label = "RGB"
    tag = "RGB"
    overlay_kind = OverlayKind.VALUE
    input_kind = "any"

    def __init__(self) -> None:
        self._chans: ChannelSet | None = None
        self._loaded_for: tuple[str, ...] = ()
        self._fit: ColourSource | None = None
        self._fit_key: tuple | None = None
        self._source: ColourSource | None = None
        self._message = ""
        super().__init__()

    # -- declaration -----------------------------------------------------
    def controls(self) -> tuple[Control, ...]:
        names = self._chans.names if self._chans is not None else ()
        n = len(names)
        coords = self._coordinate_names()
        stats_layers, light_layers = self._layer_choices()
        rgb_visible = ("basis", ("pick", "dstretch"))
        return (
            PathListControl(
                name="files",
                label="channels",
                filter=IMAGE_FILTER,
                help="The images to combine, in channel order. Any files on one grid -- "
                "echoes, INV1/INV2/UNI, a T1 and a T2. Untick to leave one out. AFNI "
                "selectors pick sub-bricks: run.nii[1], or run.nii[0..$(3)] for every "
                "third volume (interleaved echoes).",
            ),
            # -- fit -----------------------------------------------------
            ChoiceControl(
                name="basis",
                label="basis",
                choices=BASES,
                default="dstretch",
                span=THIRD,
                newline=True,
                help="pick: three channels as the three coordinates, each standardised -- "
                "the baseline, near-grey when the channels correlate.\n"
                "dstretch: decorrelation stretch -- whiten, rotate back; what differs "
                "between channels gets the colour (JPL / DStretch).\n"
                "pca: whitened principal components; most contrast, no fixed meaning.\n"
                "te: Legendre fit in echo time -- level, decay rate (R2*), curvature; the "
                "same hue is the same physics in every subject.\n"
                "custom: your own weights, one row per coordinate.",
            ),
            ChoiceControl(
                name="signal",
                label="signal",
                choices=SIGNALS,
                default="log",
                span=THIRD,
                help="What is fitted and drawn. log: decay is linear in log signal "
                "(echoes, inversions). raw: values as stored (betas, signed maps). psc: "
                "percent change from each voxel's mean -- what moves over time.",
            ),
            ChoiceControl(
                name="fit_on",
                label="fit on",
                choices=("auto", "mean", "frames"),
                default="auto",
                span=THIRD,
                help="Static or dynamic statistics. mean: covariance of the mean image "
                "across voxels -- anatomy colour. frames: of single frames across voxels "
                "and time -- change colour. auto: frames for psc, mean otherwise.",
            ),
            FloatControl(
                name="floor",
                label="floor",
                lo=0.0,
                hi=0.5,
                step=0.01,
                default=0.05,
                span=THIRD,
                visible_when=("basis", ("dstretch", "pca")),
                help="Components weaker than this fraction of the strongest are not "
                "amplified further. 0 is a full whitening, which turns the noisiest "
                "component into speckle.",
            ),
            IntControl(
                name="red",
                label=f"R {names[self._picked(0)] if n else ''}".strip(),
                lo=-1,
                hi=max(n - 1, 0),
                default=-1,
                span=THIRD,
                newline=True,
                visible_when=rgb_visible,
                help="Which channel (0-based) is the first coordinate; -1 is the first.",
            ),
            IntControl(
                name="green",
                label=f"G {names[self._picked(1)] if n else ''}".strip(),
                lo=-1,
                hi=max(n - 1, 0),
                default=-1,
                span=THIRD,
                visible_when=rgb_visible,
                help="Which channel is the second coordinate; -1 is the middle one.",
            ),
            IntControl(
                name="blue",
                label=f"B {names[self._picked(2)] if n else ''}".strip(),
                lo=-1,
                hi=max(n - 1, 0),
                default=-1,
                span=THIRD,
                visible_when=rgb_visible,
                help="Which channel is the third coordinate; -1 is the last.",
            ),
            TextControl(
                name="te",
                label="TE ms",
                placeholder="from sidecars",
                newline=True,
                visible_when=("basis", ("te",)),
                help="Echo times, comma-separated, one per ticked channel. Empty reads "
                "each file's BIDS sidecar; without them echoes are taken as evenly spaced.",
            ),
            TextControl(
                name="weights",
                label="weights",
                placeholder="1,-1,0; 0,1,-1; 1,1,1",
                newline=True,
                visible_when=("basis", ("custom",)),
                help="Three rows separated by ';', one number per channel, on the "
                "standardised channels. 1,-1,0 is channel 1 minus channel 2.",
            ),
            ChoiceControl(
                name="stats",
                label="stats from",
                choices=(*STATS_FIXED, *stats_layers),
                default="core",
                span=HALF,
                newline=True,
                help="Where means and covariance are measured -- the stretch is tuned to "
                "what differs there. core: the brain mask eroded, so the dropout fringe "
                "cannot take over. brain: the whole mask. slice: the axial slice at the "
                "crosshair. crosshair: a sphere around it (refits as you move -- a "
                "lens). A layer: its mask -- an ROI, a cluster, a vein mask.",
            ),
            FloatControl(
                name="radius",
                label="radius",
                lo=2.0,
                hi=60.0,
                step=1.0,
                default=12.0,
                span=HALF,
                unit="mm",
                visible_when=("stats", ("crosshair",)),
                help="Radius of the crosshair sphere the stats are read from.",
            ),
            # -- look ----------------------------------------------------
            ChoiceControl(
                name="space",
                label="space",
                choices=("auto", "rgb", "oklab"),
                default="auto",
                span=HALF,
                newline=True,
                help="rgb: the three coordinates to R, G, B. oklab: the first to "
                "lightness, the others to blue-yellow and green-red, so hue never reads "
                "as brightness. auto: oklab for te and pca.",
            ),
            ChoiceControl(
                name="lightness",
                label="lightness",
                choices=(*LIGHT_FIXED, *light_layers),
                default="basis",
                span=HALF,
                help="Where brightness comes from. basis: the first coordinate. "
                "channel 1: the first channel's mean. A layer: any image, resampled "
                "onto the channels' grid -- a T1 under the colour. The first coordinate "
                "then only nudges it, which keeps background noise dark.",
            ),
            *self._axis_controls(coords),
            FloatControl(
                name="hue",
                label="hue",
                lo=-180.0,
                hi=180.0,
                step=5.0,
                default=0.0,
                span=THIRD,
                newline=True,
                unit="deg",
                help="Rotate the two colour coordinates: which tissue lands on which hue.",
            ),
            FloatControl(
                name="spread",
                label="spread",
                lo=0.5,
                hi=6.0,
                step=0.1,
                default=2.5,
                span=THIRD,
                unit="sd",
                help="Window half-width in standard deviations of the stats region, for "
                "every coordinate. Per-coordinate stretch multiplies it.",
            ),
            BoolControl(
                name="masked",
                label="mask",
                default=True,
                span=THIRD,
                help="Hide voxels outside the first channel's automask.",
            ),
            FloatControl(
                name="chroma",
                label="chroma",
                lo=0.02,
                hi=0.4,
                step=0.01,
                default=0.25,
                span=THIRD,
                newline=True,
                help="Oklab saturation at full swing. Past ~0.25 strong colours leave the "
                "sRGB gamut and clip.",
            ),
            FloatControl(
                name="gamma",
                label="gamma",
                lo=0.3,
                hi=3.0,
                step=0.05,
                default=1.0,
                span=THIRD,
                help="Lightness gamma: above 1 darkens the dim end, below 1 lifts it.",
            ),
            IntControl(
                name="window",
                label="window",
                lo=1,
                hi=61,
                default=1,
                span=THIRD,
                help="Frames averaged around the shown one. One frame of percent change is "
                "mostly thermal noise; 5-15 frames shows a breath-hold or a block.",
            ),
            # -- finding areas --------------------------------------------
            FloatControl(
                name="tolerance",
                label="select within",
                lo=0.05,
                hi=2.0,
                step=0.05,
                default=0.3,
                span=HALF,
                newline=True,
                help="SELECT takes voxels whose colour coordinates are within this "
                "distance (window units) of the crosshair's. Stretch weights an axis in "
                "the distance; set a noise-only axis's stretch to 0 to ignore it.",
            ),
            PathControl(
                name="save_to",
                label="save to",
                filter="NIfTI (*.nii *.nii.gz)",
                span=HALF,
                help="Where SAVE writes the colours, as an RGB24 NIfTI. Empty: beside the "
                "first channel.",
            ),
        )

    def _axis_controls(self, coords: tuple[str, str, str]) -> list[Control]:
        out: list[Control] = []
        for k, name in enumerate(coords):
            out.append(
                FloatControl(
                    name=f"shift{k}",
                    label=f"{name} shift",
                    lo=-2.0,
                    hi=2.0,
                    step=0.05,
                    default=0.0,
                    span=HALF,
                    newline=True,
                    help=f"Move the {name} window, in window units: what counts as the "
                    "middle of this colour axis.",
                )
            )
            out.append(
                FloatControl(
                    name=f"stretch{k}",
                    label="stretch",
                    lo=-4.0,
                    hi=4.0,
                    step=0.05,
                    default=1.0,
                    span=HALF,
                    help=f"Contrast along {name}: 2 doubles it, 0 flattens it, negative "
                    "flips its direction.",
                )
            )
        return out

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="siblings",
                label="+ siblings",
                help="Add the input layer's file and the files beside it whose names differ "
                "only in one number (_e1, _e2, ...).",
            ),
            ActionControl(name="add_input", label="+ input", help="Add the input layer's file."),
            ActionControl(
                name="reset_look",
                label="reset look",
                help="Shifts, stretches, hue and gamma back to neutral.",
            ),
            ActionControl(
                name="select",
                label="select",
                help="Every voxel whose colour is near the crosshair's, as a mask layer.",
            ),
            ActionControl(name="save", label="save rgb", help="Write the colours as RGB24 NIfTI."),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset({"files"})

    def panel_names(self) -> tuple[str, ...]:
        return ("channels",)

    # -- helpers ---------------------------------------------------------
    def _layer_choices(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.session is None:
            return (), ()
        layers = [ly for ly in self.session.state.layers if not ly.colour]
        return tuple(_layer_choice(ly) for ly in layers), tuple(_layer_choice(ly) for ly in layers)

    def _layer_of(self, choice: str):
        if self.session is None:
            return None
        return self.session.state.layers.find(str(choice).split(" ", 1)[0])

    def _entries(self) -> list[str]:
        return PathListControl.enabled(self.params.get("files"))

    def _spec(self) -> ChromaSpec:
        p = self.params
        return ChromaSpec(
            basis=str(p["basis"]),
            signal=str(p["signal"]),
            floor=float(p["floor"]),
            pick=(int(p["red"]), int(p["green"]), int(p["blue"])),
            te=str(p.get("te") or ""),
            weights=str(p.get("weights") or ""),
        )

    def _picked(self, k: int) -> int:
        if self._chans is None:
            return 0
        try:
            return self._spec().picked(self._chans.n)[k]
        except ValueError:
            return 0

    def _coordinate_names(self) -> tuple[str, str, str]:
        if self._source is not None:
            return self._source.coordinate_names()
        return ("axis 1", "axis 2", "axis 3")

    def _look(self) -> Look:
        p = self.params
        space = resolve_space(str(p["space"]), str(p["basis"]))
        return Look(
            space=space,
            lightness="basis" if p["lightness"] == "basis" else "anatomy",
            spread=float(p["spread"]),
            chroma=float(p["chroma"]),
            shift=(float(p["shift0"]), float(p["shift1"]), float(p["shift2"])),
            stretch=(float(p["stretch0"]), float(p["stretch1"]), float(p["stretch2"])),
            hue=float(p["hue"]),
            gamma=float(p["gamma"]),
        )

    def _crosshair_voxel(self) -> tuple[int, int, int] | None:
        if self.session is None or self._chans is None:
            return None
        return self.session.layer_voxel(self._chans.affine, self.session.state.crosshair)

    def _stats_mask(self) -> np.ndarray | None:
        """The stats region as a mask on the channels' grid, or None for core/brain."""
        stats = str(self.params["stats"])
        chans = self._chans
        if stats in ("core", "brain") or chans is None or self.session is None:
            return None
        if stats in ("slice", "crosshair"):
            ijk = self._crosshair_voxel()
            if ijk is None:
                return None
            brain = chans.brain.cpu().numpy()
            out = np.zeros(chans.shape, dtype=bool)
            if stats == "slice":
                k = min(max(ijk[2], 0), chans.shape[2] - 1)
                out[:, :, k] = brain[:, :, k]
                return out
            vox = np.sqrt((np.asarray(chans.affine)[:3, :3] ** 2).sum(0))
            grids = np.meshgrid(*(np.arange(s) for s in chans.shape), indexing="ij")
            dist2 = sum(((g - c) * v) ** 2 for g, c, v in zip(grids, ijk, vox, strict=True))
            return (dist2 <= float(self.params["radius"]) ** 2) & brain
        layer = self._layer_of(stats)
        if layer is None:
            raise ValueError(f"no layer {stats.split(' ', 1)[0]} to take stats from")
        from fastfuncstuff.viewer.session import resample_mask_nearest

        like = self.session.mask_from_layer(layer.key, layer)
        return resample_mask_nearest(like, layer.affine, chans.shape, chans.affine)

    def _light_ref(self) -> torch.Tensor | None:
        """A loaded layer resampled onto the channels' grid, for lightness."""
        choice = str(self.params["lightness"])
        if choice in LIGHT_FIXED or self._chans is None or self.session is None:
            return None
        layer = self._layer_of(choice)
        if layer is None:
            raise ValueError(f"no layer {choice.split(' ', 1)[0]} to take lightness from")
        from fastfuncstuff.viewer.slicing import sample_volume

        chans = self._chans
        vol = torch.as_tensor(
            np.nan_to_num(
                np.asarray(self.session.volume(layer.key, layer.volume_index), np.float32)
            ),
            device=chans.device,
        )
        to_layer = np.linalg.inv(np.asarray(layer.affine, float)) @ np.asarray(chans.affine, float)
        grids = torch.meshgrid(
            *(torch.arange(s, device=chans.device, dtype=torch.float32) for s in chans.shape),
            indexing="ij",
        )
        ijk = torch.stack(grids, -1)
        m = torch.as_tensor(to_layer, dtype=torch.float32, device=chans.device)
        return sample_volume(vol, ijk @ m[:3, :3].T + m[:3, 3], mode="bilinear")

    # -- lifecycle -------------------------------------------------------
    def attach(self, session) -> None:
        super().attach(session)
        # A first visit starts from the input's echoes, as an editable list --
        # a starting point, not a guess the panel hides.
        if not self.params.get("files"):
            self.params["files"] = PathListControl.encode([(p, True) for p in self._siblings()])

    def _input_path(self) -> str | None:
        layer = self.source_layer()
        if layer is None or not layer.path or str(layer.path).startswith("<"):
            return None
        return str(layer.path)

    def _siblings(self) -> list[str]:
        path = self._input_path()
        if path is None:
            return []
        p = Path(path)
        from fastfuncstuff.viewer.tools.slicetime import dataset_stem

        suffix = p.name[len(dataset_stem(p)) :]
        candidates = sorted(q for q in p.parent.glob(f"*{suffix}") if q.is_file())
        names = [q.name for q in candidates]
        if p.name not in names:
            return [path]
        return [str(candidates[i]) for i in sibling_order(names, names.index(p.name))]

    def input_layer_key(self) -> str | None:
        layer = self.source_layer()
        return None if layer is None else layer.key

    # -- the slow half ---------------------------------------------------
    def prepare(self, progress: ProgressFn | None = None) -> bool:
        if self.session is None:
            return False
        entries = tuple(self._entries())
        if self._chans is not None and entries == self._loaded_for:
            self._dirty = False
            return True
        if progress is not None:
            progress(0.1, f"reading {len(entries)} channels")
        loaded = {str(ly.path): ly.key for ly in self.session.state.layers if not ly.colour}
        try:
            self._chans = load_channels(
                self.session.store,
                list(entries),
                loaded=loaded,
                device=self.session.display_device,
            )
        except (OSError, ValueError, KeyError) as exc:
            self._chans = None
            self._message = str(exc)
            self._dirty = False
            return False
        self._loaded_for = entries
        self._fit = self._source = None
        self._fit_key = None
        self._message = ""
        self._dirty = False
        return True

    # -- the fast half ---------------------------------------------------
    def compute(self) -> ComputedOverlay | None:
        if self._chans is None:
            return None
        p = self.params
        moving_stats = str(p["stats"]) in ("slice", "crosshair")
        fit_key = (
            self._spec(),
            str(p["fit_on"]),
            str(p["stats"]),
            float(p["radius"]),
            str(p["lightness"]),
            self._crosshair_voxel() if moving_stats else None,
        )
        try:
            look = self._look()
            if self._fit is None or fit_key != self._fit_key:
                self._fit = fit_source(
                    self._chans,
                    self._spec(),
                    look,
                    stats=str(p["stats"]),
                    stats_mask=self._stats_mask(),
                    fit_on=str(p["fit_on"]),
                    light_ref=self._light_ref(),
                    masked=bool(p["masked"]),
                    window=int(p["window"]),
                )
                self._fit_key = fit_key
            self._source = self._fit.with_look(
                look, masked=bool(p["masked"]), window=int(p["window"])
            )
        except ValueError as exc:
            self._message = str(exc)
            return None
        self._message = self._source.describe()
        return ComputedOverlay(
            values=np.zeros(0, dtype=np.float32),
            affine=self._chans.affine,
            name=self.output_name(),
            colour=self._source,
        )

    def set_param(self, name: str, value) -> Aspect:
        # GRAPH: the crosshair traces are drawn in the new coordinates.
        return super().set_param(name, value) | Aspect.GRAPH

    def on_command(self, cmd: Command, dirty: Aspect) -> Aspect:
        if not dirty & (Aspect.CROSSHAIR | Aspect.TIME):
            return Aspect.NOTHING
        out = Aspect.GRAPH
        # A lens: stats read around the crosshair follow it.
        if dirty & Aspect.CROSSHAIR and str(self.params["stats"]) in ("slice", "crosshair"):
            out |= self.refresh()
        return out

    # -- actions -----------------------------------------------------------
    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name in ("siblings", "add_input"):
            current = PathListControl.parse(self.params.get("files"))
            have = {p for p, _ in current}
            new = self._siblings() if name == "siblings" else [self._input_path() or ""]
            current += [(p, True) for p in new if p and p not in have]
            if not new or not new[0]:
                self._message = "pick an input layer that is a file"
            return self.set_param("files", PathListControl.encode(current)) | Aspect.LAYERS
        if name == "reset_look":
            for k in range(3):
                self.params[f"shift{k}"] = 0.0
                self.params[f"stretch{k}"] = 1.0
            self.params["hue"], self.params["gamma"] = 0.0, 1.0
            return self.refresh() | Aspect.LAYERS
        if name == "select":
            return self._select()
        if name == "save":
            self._save()
            return Aspect.NOTHING
        return super().action(name, progress)

    def _select(self) -> Aspect:
        if self.session is None or self._source is None:
            return Aspect.NOTHING
        ijk = self._crosshair_voxel()
        if ijk is None:
            return Aspect.NOTHING
        live = self.session.state.layers.find_by_source(self.layer_source)
        if live is None:
            return Aspect.NOTHING
        t = self.session.layer_time(live)
        keep = self._source.similar(ijk, t, float(self.params["tolerance"]))
        key, dirty = self.session.install_selection(
            f"select:{self.layer_source}",
            keep,
            like=live,
            name=f"{self.output_name()} like ({ijk[0]},{ijk[1]},{ijk[2]})",
        )
        # The selection is read against the colours it was picked from: keep
        # them drawn, and outline the selection rather than paint over them.
        self.session.state.layers.update(live.key, visible=True)
        self.session.state.layers.update(key, boxed=True, opacity=0.35)
        self._message = f"selected {int(keep.sum())} voxels"
        return dirty | Aspect.LAYERS | Aspect.SLICES

    def _save(self) -> Path:
        if self._source is None:
            raise ValueError("nothing to save yet")
        chosen = str(self.params.get("save_to") or "").strip()
        if chosen:
            out = Path(chosen).expanduser()
        else:
            first = Path(self._chans.entries[0].split("[", 1)[0]) if self._chans else Path("rgb")
            out = first.with_name(f"{short_name(str(first))}_rgb.nii.gz")
        path = self._source.save(out)
        self._message = f"saved {path}"
        return path

    # -- graph --------------------------------------------------------------
    def series(self, ijk: tuple[int, int, int]) -> list[Trace]:
        """Each channel's time course at the voxel, and the three colour coordinates."""
        chans, source = self._chans, self._source
        if chans is None or source is None or chans.n_frames < 2 or self.session is None:
            return []
        i, j, k = self.session.layer_voxel(chans.affine, ijk)
        nx, ny, nz = chans.shape
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            return []
        rows = []
        for c in range(chans.n):
            res = chans.store.get(chans.keys[c])
            if res.array is None:
                return []
            rows.append(np.asarray(res.array[i, j, k, : chans.n_frames], dtype=np.float32))
        raw = torch.as_tensor(np.stack(rows), device=chans.device)
        mean = chans.means[:, i, j, k, None] if source.spec.signal == "psc" else None
        x = source.map.coords(transform(raw, source.spec.signal, eps=chans.eps, mean=mean))
        u = source.map.unit(x, source.look).cpu().numpy()
        out = [
            Trace(label=f"{name} at voxel", values=row, x_label="TR", key=f"rgb:ch{c}", short=name)
            for c, (name, row) in enumerate(zip(chans.names, rows, strict=True))
        ]
        out += [
            Trace(
                label=f"{name} (window units)",
                values=u[c],
                x_label="TR",
                key=f"rgb:axis{c}",
                short=name,
            )
            for c, name in enumerate(source.coordinate_names())
        ]
        return out

    def panels(self) -> dict[str, list[Trace]]:
        """The channels at the crosshair, against TE when there is one -- the decay curve."""
        chans, source = self._chans, self._source
        if chans is None or source is None or self.session is None:
            return {}
        ijk = self._crosshair_voxel()
        live = self.session.state.layers.find_by_source(self.layer_source)
        if ijk is None or live is None:
            return {}
        got = source.at(ijk, self.session.layer_time(live))
        if got is None:
            return {}
        raw, _ = got
        if source.te_ms is not None:
            x, x_label = np.asarray(source.te_ms, dtype=np.float32), "TE ms"
        else:
            x, x_label = np.arange(chans.n, dtype=np.float32), "channel"
        lines = [
            Trace(label="channels at crosshair", values=raw, x=x, x_label=x_label, key="values")
        ]
        if source.spec.basis == "te" and source.te_ms is not None and np.all(raw > 0):
            slope, icpt = np.polyfit(x, np.log(raw), 1)
            fine = np.linspace(float(x.min()), float(x.max()), 40, dtype=np.float32)
            lines.append(
                Trace(
                    label=f"mono-exponential fit, T2* {(-1 / slope if slope < 0 else np.inf):.1f} ms",
                    values=np.exp(icpt + slope * fine).astype(np.float32),
                    x=fine,
                    x_label=x_label,
                    key="fit",
                    short="fit",
                )
            )
        return {"channels": lines}

    def status(self) -> str:
        if self._preparing:
            return "rgb: reading channels…"
        n = len(self._entries())
        if self._chans is None:
            return f"rgb: {self._message or f'{n} channels ticked; add files to the list'}"
        frames = f", {self._chans.n_frames} frames" if self._chans.n_frames > 1 else ""
        return f"rgb: {self._chans.n} channels{frames}  ·  {self._message}"


__all__ = ["RGBMode"]
