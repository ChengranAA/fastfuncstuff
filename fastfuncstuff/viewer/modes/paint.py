"""Paint: draw ROIs on the slices, as integer labels.

While this mode is active, a left drag in an image window paints the current
label and a right drag erases (pan stays on shift/middle drag, slices on the
wheel, and ctrl+wheel sizes the brush). What is painted is an ordinary ROI
layer -- coloured by label, named in the readout, usable anywhere an ROI is,
including the RGB mode's stats -- made on the display grid so painting is one
voxel per voxel on screen, and saved as integer labels, on its own grid or
resampled onto another layer's.

The painting itself is :mod:`viewer.paintlayer`; this file is the panel and the
stroke protocol the image windows drive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastfuncstuff.viewer.commands import Aspect
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
    PathControl,
    ProgressFn,
    TextControl,
    mode,
)
from fastfuncstuff.viewer.paint import SHAPES, encode_points, stroke_centres
from fastfuncstuff.viewer.state import Plane

#: Brush radius limits, mm. The lower one is below any voxel: a 1-voxel brush.
RADIUS_MIN, RADIUS_MAX = 0.1, 40.0
OWN_GRID = "own grid"


@mode
class PaintMode(Mode):
    name = "paint"
    label = "Paint"
    tag = "PAINT"
    produces_overlay = False
    input_kind = "none"

    def __init__(self) -> None:
        self._stroke: dict | None = None
        self._message = ""
        super().__init__()

    # -- declaration -----------------------------------------------------
    def _paint_layers(self) -> list:
        if self.session is None:
            return []
        return [ly for ly in self.session.state.layers if self.session.painter.owns(ly.key)]

    def controls(self) -> tuple[Control, ...]:
        layers = self._paint_layers()
        others = [] if self.session is None else [ly for ly in self.session.state.layers]
        return (
            ChoiceControl(
                name="layer",
                label="paint into",
                choices=("new", *(f"{ly.key} {ly.name}" for ly in layers)),
                default="new",
                help="The painted layer strokes go into. 'new' makes one on the first "
                "stroke (or press NEW).",
            ),
            IntControl(
                name="value",
                label="label",
                lo=1,
                hi=999,
                default=1,
                span=HALF,
                newline=True,
                help="The integer painted. Each label has its own colour and is its own "
                "region wherever ROIs are used.",
            ),
            TextControl(
                name="label_name",
                label="name",
                placeholder="label 1",
                span=HALF,
                help="A name for the current label, shown in the readout and saved with it.",
            ),
            FloatControl(
                name="radius",
                label="radius",
                lo=RADIUS_MIN,
                hi=RADIUS_MAX,
                step=0.1,
                default=2.0,
                span=THIRD,
                newline=True,
                unit="mm",
                help="Brush radius in mm (ctrl+wheel over an image). Below half a voxel it "
                "paints the single voxel under the cursor.",
            ),
            ChoiceControl(
                name="brush",
                label="brush",
                choices=SHAPES,
                default="disk",
                span=THIRD,
                help="disk: the slice being painted on. sphere: neighbouring slices too.",
            ),
            BoolControl(
                name="overwrite",
                label="overwrite",
                default=True,
                span=THIRD,
                help="Paint over other labels. Off: only empty voxels take the label, so "
                "a region drawn against another stops at its edge.",
            ),
            ChoiceControl(
                name="save_like",
                label="save on grid of",
                choices=(OWN_GRID, *(f"{ly.key} {ly.name}" for ly in others)),
                default=OWN_GRID,
                span=HALF,
                newline=True,
                help="Save on the painted layer's own grid, or nearest-resampled onto "
                "another layer's (a label resampled any other way is a label that does "
                "not exist).",
            ),
            PathControl(
                name="save_to",
                label="save to",
                filter="NIfTI (*.nii *.nii.gz)",
                span=HALF,
                help="Where SAVE writes. Empty: the painted layer's name, beside the "
                "underlay's file.",
            ),
        )

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="new", label="new layer", help="An empty label layer to paint into."
            ),
            ActionControl(name="undo", label="undo", help="Undo the last stroke (ctrl+Z)."),
            ActionControl(name="save", label="save", help="Write the labels as NIfTI."),
        )

    def preparation_params(self) -> frozenset[str]:
        return frozenset()

    def compute(self) -> ComputedOverlay | None:
        return None

    # -- the target layer -------------------------------------------------
    def target(self) -> str | None:
        """The painted layer strokes go into, or None when one must be made."""
        choice = str(self.params.get("layer") or "new")
        if choice == "new" or self.session is None:
            return None
        key = choice.split(" ", 1)[0]
        return key if self.session.painter.owns(key) else None

    def _new_layer(self, dispatch=None) -> str:
        """Make a painted layer and point the brush at it.

        From a window (``dispatch`` given) it goes through the bus as its own
        command, so it is in the recording before the first stroke. From an
        action it is made directly: the action is the recorded command, and
        dispatching a second one inside it would make two layers on replay.
        """
        from fastfuncstuff.viewer.vocab import NewPaintLayer

        assert self.session is not None
        key = self.session.state.layers.mint_key("P")
        name = f"paint {len(self._paint_layers()) + 1}"
        if dispatch is not None:
            dispatch(NewPaintLayer(key, name))
        else:
            self.session.painter.new_layer(key, name)
        self.params["layer"] = f"{key} {self.session.state.layers.get(key).name}"
        return key

    # -- the stroke protocol (driven by image windows) ----------------------
    def begin(self, mm, plane: Plane, *, erase: bool, dispatch=None) -> Aspect:
        """Start a stroke at ``mm`` on ``plane``; make the layer if there is none."""
        if self.session is None:
            return Aspect.NOTHING
        dirty = Aspect.NOTHING
        key = self.target()
        if key is None:
            if erase:
                return Aspect.NOTHING  # nothing to erase from
            # A stroke is never inside another command, so the layer goes
            # through the bus and is recorded ahead of the stroke.
            key = self._new_layer(dispatch or self.session.do)
            dirty |= Aspect.LAYERS
        self._name_label(key)
        self._stroke = {
            "key": key,
            "value": 0 if erase else int(self.params["value"]),
            "radius": float(self.params["radius"]),
            "brush": str(self.params["brush"]),
            "plane": plane,
            "overwrite": bool(self.params["overwrite"]),
            "points": [],
            "last": None,
        }
        return dirty | self.extend(mm)

    def extend(self, mm) -> Aspect:
        stroke = self._stroke
        if stroke is None or self.session is None:
            return Aspect.NOTHING
        centres = stroke_centres(stroke["last"], mm, stroke["radius"])
        stroke["last"] = np.asarray(mm, dtype=np.float64)
        stroke["points"].extend(centres)
        changed = self.session.painter.apply(
            stroke["key"],
            stroke["value"],
            centres,
            radius_mm=stroke["radius"],
            brush=stroke["brush"],
            plane=stroke["plane"],
            overwrite=stroke["overwrite"],
        )
        return Aspect.SLICES if changed else Aspect.NOTHING

    def end(self) -> Aspect:
        """Finish the stroke: one undo step, and one line in the recording."""
        from fastfuncstuff.viewer.vocab import PaintStroke

        stroke, self._stroke = self._stroke, None
        if stroke is None or self.session is None:
            return Aspect.NOTHING
        changed = self.session.painter.close_stroke(stroke["key"])
        if stroke["points"]:
            # Recorded, not dispatched: the voxels are painted already, and
            # replay dispatches it for real (see CommandBus.record).
            self.session.bus.record(
                PaintStroke(
                    stroke["key"],
                    stroke["value"],
                    stroke["radius"],
                    stroke["brush"],
                    stroke["plane"].value,
                    stroke["overwrite"],
                    encode_points(np.asarray(stroke["points"])),
                )
            )
        verb = "erased" if stroke["value"] == 0 else f"painted label {stroke['value']} on"
        self._message = f"{verb} {changed} voxels"
        return Aspect.SLICES | Aspect.LAYERS

    @property
    def painting(self) -> bool:
        return self._stroke is not None

    def _name_label(self, key: str) -> None:
        """Give the current label the typed name, if one was typed."""
        from fastfuncstuff.viewer.paintlayer import label_entry

        assert self.session is not None
        text = str(self.params.get("label_name") or "").strip()
        value = int(self.params["value"])
        table = self.session.painter.tables[key]
        entry = table.get(value) or label_entry(value)
        if text and entry.name != text:
            table[value] = type(entry)(entry.value, text, entry.color)

    # -- parameters and actions ----------------------------------------------
    def set_param(self, name: str, value) -> Aspect:
        if self.params.get(name) == value:
            return Aspect.NOTHING
        self.params[name] = value
        if name == "value" and self.session is not None:
            # A different label: show its name, not the last one's.
            key = self.target()
            entry = self.session.painter.tables.get(key or "", {}).get(int(value))
            self.params["label_name"] = entry.name if entry is not None else ""
        if name == "label_name" and self.target() is not None:
            self._name_label(self.target())  # type: ignore[arg-type]
        return Aspect.SLICES | Aspect.GRAPH

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        from fastfuncstuff.viewer.vocab import PaintUndo

        if self.session is None:
            return Aspect.NOTHING
        if name == "new":
            self._new_layer()
            return Aspect.LAYERS | Aspect.SLICES
        if name == "undo":
            # Directly, not via PAINT_UNDO: this action is the recorded command.
            key = self.target()
            if key is None or not self.session.painter.undo(key):
                return Aspect.NOTHING
            return PaintUndo.aspects
        if name == "save":
            self._save()
            return Aspect.NOTHING
        return super().action(name, progress)

    def _save(self) -> Path:
        assert self.session is not None
        key = self.target()
        if key is None:
            raise ValueError("nothing painted yet")
        layer = self.session.state.layers.get(key)
        chosen = str(self.params.get("save_to") or "").strip()
        if chosen:
            out = Path(chosen).expanduser()
        else:
            base = self.session.state.layers.base
            folder = (
                Path(base.path).parent
                if base is not None and not str(base.path).startswith("<")
                else Path.cwd()
            )
            out = folder / f"{layer.name.replace(' ', '_')}.nii.gz"
        like = str(self.params.get("save_like") or OWN_GRID)
        path = self.session.painter.save(
            key, out, None if like == OWN_GRID else like.split(" ", 1)[0]
        )
        self._message = f"saved {path}"
        return path

    def status(self) -> str:
        key = self.target()
        if key is None:
            where = "the first stroke makes a new layer"
        else:
            counts = self.session.painter.counts(key) if self.session is not None else {}
            where = ", ".join(f"{v}: {n} vox" for v, n in sorted(counts.items())) or "empty"
        return (
            f"paint: label {int(self.params['value'])}, r {float(self.params['radius']):g} mm "
            f"{self.params['brush']}  ·  {where}  ·  left paint, right erase, ctrl+wheel size"
            + (f"  ·  {self._message}" if self._message else "")
        )


__all__ = ["PaintMode"]
