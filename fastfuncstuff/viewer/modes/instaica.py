"""InstaICA: decompose the run you are looking at into independent components.

InstaPCA's loop -- look, label signal or noise, next, SAVE NOISE -- on spatial
ICA instead of PCA: the run reduced to ``# dims`` leading components exactly as
InstaPCA does it, then rotated to the most non-Gaussian spatial maps, MELODIC's
recipe (see ``viewer/instaica.py``: the rotation needs no second pass over the
data, so it costs milliseconds on top of the PCA). An IC that is a vessel, an
edge or a slice artefact tends to be one component here where PCA spreads it
over several -- which is why ICA is the usual tool for labelling noise.

Three maps per component: the correlation of each voxel with the IC's time
course (amplitude-free, as InstaPCA's), its amplitude in percent signal, and the
z-scored spatial map MELODIC draws. Labelled noise goes to ``<run>_ica_noise.1D``
and to Denoise's "ICA noise", exactly as a loaded MELODIC folder's would.

Reviewing a folder made elsewhere -- MELODIC, ffs_ica -- is the ICA button in the
window bar, which opens the folder reviewer.
"""

from __future__ import annotations

from fastfuncstuff.viewer import instapca as engine
from fastfuncstuff.viewer.modes.base import HALF, Control, IntControl, ProgressFn, mode
from fastfuncstuff.viewer.modes.instapca import InstaPCAMode


@mode
class InstaICAMode(InstaPCAMode):
    name = "instaica"
    label = "InstaICA"
    tag = "IICA"
    component_word = "IC"
    ortvec_suffix = "ica_noise"
    map_choices = ("correlation", "amplitude", "z")

    def controls(self) -> tuple[Control, ...]:
        out = []
        for c in super().controls():
            if c.name == "n_components" and isinstance(c, IntControl):
                c = IntControl(
                    name=c.name,
                    label="# dims",
                    lo=2,
                    hi=c.hi,
                    default=30,
                    span=HALF,
                    help="Dimensions kept before the ICA rotation -- MELODIC's model "
                    "order. Each is one IC. More splits noise finer; too many splits "
                    "real networks.",
                )
            out.append(c)
        out.append(
            IntControl(
                name="seed",
                label="seed",
                lo=0,
                hi=999,
                default=0,
                span=HALF,
                newline=True,
                help="FastICA's starting rotation. A component that changes with the "
                "seed is not a stable one.",
            )
        )
        return tuple(out)

    def preparation_params(self) -> frozenset[str]:
        return super().preparation_params() | {"seed"}

    def _identity_extra(self) -> tuple:
        return (int(self.params.get("seed") or 0),)

    def _post(
        self, result: engine.Decomposition, progress: ProgressFn | None
    ) -> engine.Decomposition:
        from fastfuncstuff.viewer.instaica import rotate_to_ica

        if progress is not None:
            progress(0.97, "ica rotation")
        device = self.session.store.device if self.session is not None else None
        return rotate_to_ica(result, seed=int(self.params.get("seed") or 0), device=device)


__all__ = ["InstaICAMode"]
