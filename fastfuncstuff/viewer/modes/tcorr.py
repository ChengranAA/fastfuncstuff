"""tCorr: voxel-to-voxel correlation between datasets, as a live overlay.

Add two or more 4-D files on one grid -- two echoes of a run, two runs of a task,
a series before and after a correction -- press GO, and each voxel is coloured by
how well its time course in one file agrees with the same voxel in another.
``ffs_util_tcorr`` is the same computation from the command line; both go through
:func:`fastfuncstuff.stats.paired_correlation.paired_voxel_correlation`.

GO is the only thing that reads files. The arrays it loads are kept, so moving the
detrend or the blur re-correlates from memory -- a pass over the data on the
device -- and picking another pair only swaps which map is shown.

For eyeballing, not scoring: blur raises these correlations, and between echoes
motion is shared, so removing it can lower them (see the tool's help).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.viewer.commands import Aspect
from fastfuncstuff.viewer.modes.base import (
    FULL,
    HALF,
    ActionControl,
    ChoiceControl,
    ComputedOverlay,
    Control,
    IntControl,
    Mode,
    OptionalFloatControl,
    OverlayKind,
    PathListControl,
    ProgressFn,
    correlation_stat,
    mode,
)

#: Frames blurred per device pass, so a long run never has to fit on the card whole.
_BLUR_BLOCK = 32


@mode
class TCorrMode(Mode):
    name = "tcorr"
    label = "tCorr"
    tag = "TCORR"
    overlay_kind = OverlayKind.CORRELATION
    input_kind = "none"

    def __init__(self) -> None:
        self._request: tuple[str, ...] = ()  # the files GO asked for
        self._arrays: dict[str, np.ndarray] = {}  # loaded from disk, kept across GOs
        self._affine: np.ndarray | None = None
        self._zooms = (1.0, 1.0, 1.0)
        self._maps: torch.Tensor | None = None  # (nx, ny, nz, P), CPU
        self._pairs: list[tuple[int, int]] = []
        self._n_time = 0
        self._message = ""
        super().__init__()

    def controls(self) -> tuple[Control, ...]:
        return (
            PathListControl(
                name="files",
                label="files",
                filter="NIfTI (*.nii *.nii.gz *.nii.zst);;All (*)",
                help="Two or more 4-D datasets on one grid with the same frame count. "
                "Nothing is read until GO; untick one to leave it out of the next GO.",
            ),
            IntControl(
                name="polort",
                label="detrend",
                lo=-1,
                hi=6,
                default=2,
                span=HALF,
                newline=True,
                help="Legendre order removed from every voxel first; -1 for none. "
                "Shared drift correlates two series whatever their content.",
            ),
            OptionalFloatControl(
                name="blur",
                label="blur",
                on_value=4.0,
                lo=0.0,
                hi=12.0,
                default=0.0,
                step=0.5,
                unit=" mm",
                span=HALF,
                help="Spatial smoothing (FWHM) before correlating. Raises r by "
                "itself -- it averages independent noise away.",
            ),
            ChoiceControl(
                name="pairs",
                label="pairs",
                choices=("all", "adjacent", "first"),
                default="all",
                style="radio",
                newline=True,
                help="Every pair, consecutive files only, or file 1 against each other.",
            ),
            ChoiceControl(
                name="pair",
                label="show",
                choices=self._pair_choices(),
                default=self._pair_choices()[0],
                span=FULL,
                newline=True,
                help="Which pair's map is drawn. Switching does not recompute.",
            ),
        )

    def _pair_choices(self) -> tuple[str, ...]:
        return tuple(f"{a + 1}-{b + 1}" for a, b in self._pairs) or ("--",)

    def actions(self) -> tuple[ActionControl, ...]:
        return (
            ActionControl(
                name="go",
                label="go",
                help="Read the ticked files from disk and correlate them.",
            ),
            *super().actions(),
        )

    def preparation_params(self) -> frozenset[str]:
        # Files are not here: reading them is GO's job, so ticking one in does
        # not start a multi-gigabyte load behind your back. Detrend and blur only
        # re-correlate what is already in memory.
        return frozenset({"polort", "blur", "pairs"})

    def detach(self) -> None:
        """Drop the loaded files; they can be gigabytes. The map stays in the stack."""
        self._arrays = {}
        self._maps = None
        self._dirty = True
        super().detach()

    def action(self, name: str, progress: ProgressFn | None = None) -> Aspect:
        if name == "go":
            self._request = tuple(PathListControl.enabled(self.params.get("files")))
            # A fresh read: the files may have been rewritten since the last GO.
            self._arrays = {}
            self.invalidate()
            return self.refresh(progress)
        return super().action(name, progress)

    # -- the slow half -------------------------------------------------
    def _fail(self, why: str) -> bool:
        self._message = why
        self._maps = None
        self._dirty = False  # nothing to retry until GO or a parameter changes
        return False

    def prepare(self, progress: ProgressFn | None = None) -> bool:
        if not self._dirty and self._maps is not None:
            return True
        if not self._request:
            self._message = "add two or more 4-D files and press GO"
            return False
        if len(self._request) < 2:
            return self._fail("tick at least two files")

        def step(fraction: float, message: str) -> None:
            if progress is not None:
                progress(fraction, message)

        from fastfuncstuff.io.afni import load_nifti
        from fastfuncstuff.stats.paired_correlation import pair_list, paired_voxel_correlation
        from fastfuncstuff.viewer.tools.smooth import voxel_sizes

        n = len(self._request)
        for k, path in enumerate(self._request):
            if path in self._arrays:
                continue
            step(0.4 * k / n, f"reading {Path(path).name}")
            try:
                img = load_nifti(path)
            except (OSError, ValueError) as exc:
                return self._fail(f"cannot read {Path(path).name}: {exc}")
            arr = np.asarray(img.dataobj, dtype=np.float32)
            if arr.ndim != 4 or arr.shape[3] < 3:
                return self._fail(f"{Path(path).name} is not a 4-D series")
            self._arrays[path] = arr
            if k == 0:
                self._affine = np.asarray(img.affine, dtype=float)
                self._zooms = voxel_sizes(self._affine)
        series = [self._arrays[p] for p in self._request]
        if any(s.shape != series[0].shape for s in series):
            shapes = ", ".join(str(s.shape) for s in series)
            return self._fail(f"files differ in grid or frame count: {shapes}")

        device = self.session.store.device if self.session is not None else torch.device("cpu")
        blur = float(self.params.get("blur") or 0.0)
        if blur > 0:
            series = [self._blurred(s, blur, device, step, k, n) for k, s in enumerate(series)]

        rule = {"all": "all", "adjacent": "adjacent", "first": "ref:1"}[
            str(self.params.get("pairs", "all"))
        ]
        self._n_time = int(series[0].shape[-1])
        self._pairs = pair_list(n, rule)
        step(0.8, f"correlating {len(self._pairs)} pair(s)")
        self._maps = paired_voxel_correlation(
            series,
            self._pairs,
            polort=int(self.params.get("polort", 2)),
            device=device,
            verbose=False,
        )
        # The pair list may have changed under the picker; keep a choice that
        # still exists, else show the first.
        if self.params.get("pair") not in self._pair_choices():
            self.params["pair"] = self._pair_choices()[0]
        self._message = ""
        self._dirty = False
        step(1.0, "ready")
        return True

    def _blurred(self, arr, fwhm, device, step, k, n) -> torch.Tensor:
        from fastfuncstuff.stats.smooth3d import fwhm_mm_to_sigma_vox, gaussian3d_batched

        sigma = fwhm_mm_to_sigma_vox(fwhm, self._zooms)
        n_t = arr.shape[3]
        step(0.4 + 0.4 * k / n, f"blur {fwhm:g} mm, file {k + 1}")
        if arr.flags.f_contiguous and not arr.flags.c_contiguous:
            # A nibabel array: a block of volumes is already a contiguous (t, z, y, x)
            # array, so blur it in that order and keep the result F-ordered -- the
            # layout paired_voxel_correlation streams fastest.  Moving t to the front
            # of an F-ordered array instead is a full transpose on the host.
            out_tzyx = torch.empty(arr.shape[::-1], dtype=torch.float32)
            src = torch.from_numpy(arr).permute(3, 2, 1, 0)
            for t0 in range(0, n_t, _BLUR_BLOCK):
                t1 = min(n_t, t0 + _BLUR_BLOCK)
                blurred = gaussian3d_batched(src[t0:t1].to(device), sigma[::-1])
                out_tzyx[t0:t1].copy_(blurred)
            return out_tzyx.permute(3, 2, 1, 0)
        out = torch.empty(arr.shape, dtype=torch.float32)
        for t0 in range(0, n_t, _BLUR_BLOCK):
            t1 = min(n_t, t0 + _BLUR_BLOCK)
            block = torch.from_numpy(np.ascontiguousarray(np.moveaxis(arr[..., t0:t1], 3, 0)))
            out[..., t0:t1] = gaussian3d_batched(block.to(device), sigma).permute(1, 2, 3, 0).cpu()
        return out

    # -- the fast half -------------------------------------------------
    def _shown(self) -> int:
        choices = self._pair_choices()
        pick = str(self.params.get("pair"))
        return choices.index(pick) if pick in choices else 0

    def compute(self) -> ComputedOverlay | None:
        if self._maps is None or self._affine is None:
            return None
        k = self._shown()
        return ComputedOverlay(
            values=self._maps[..., k].numpy(),
            affine=self._affine,
            name=self.output_name(),
            kind=OverlayKind.CORRELATION,
            colormap="redblue",
            display_range=(-1.0, 1.0),
            threshold=0.2,
            threshold_stat=correlation_stat(self._n_time, int(self.params.get("polort", 2))),
        )

    def status(self) -> str:
        if self._message:
            return f"tcorr: {self._message}"
        if self._preparing:
            return "tcorr: correlating…"
        if self._maps is None:
            return "tcorr: add files and press GO"
        names = [Path(p).name for p in self._request]
        a, b = self._pairs[self._shown()]
        return f"tcorr: {names[a]} ~ {names[b]}"
