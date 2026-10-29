"""Tiered residency: disk to CPU RAM to compute device.

CPU RAM is the residency tier and device memory is the working set. Several
datasets live in RAM at once; only whatever InstaCorr or InstaGLM is currently
touching gets promoted. That split is what lets a 24 GB card hold a working set
while system RAM holds the session.

The measurements this is built around, on an M4 Max:

* header-only peek -- microseconds, so a directory can be indexed eagerly
* first volume out of a 66 MB ``.nii.gz`` -- 2.45 ms, because volume 0 sits at
  the head of the stream
* full inflate -- 335 MB/s single-threaded, so a 2 GB dataset takes ~6 s
* RAM to device -- 10 GB/s

That last figure is the one that surprises people on Apple silicon: unified
memory removes the *capacity* split between host and device, not the copy. A
promotion still costs real bandwidth, so eviction policy matters on MPS exactly
as it does on CUDA.

The ~6 s inflate is also why loading is split in two. Volume 0 goes on screen
almost immediately and the rest arrives on a worker thread; ``zlib`` releases
the GIL while inflating, so that thread genuinely does not block the UI.

**Eviction needs something to evict back to.** A dataset read off disk can be
dropped for free, because the file is still there. One the viewer *made* -- a
mode's output, a motion-corrected run, a selection -- has no file behind it, so
dropping it destroys it. Those are parked in a spill directory first, as raw
``.npy``: no compression, because a student's laptop is the machine that both
runs out of RAM and has the slow disk, and paying gzip on the way out would
turn a stall into a freeze. Reads come back memory-mapped, so scrubbing a
spilled run pages in one volume at a time instead of inflating the whole thing.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import count
from pathlib import Path

import numpy as np
import torch

from fastfuncstuff.io.dsetinfo import DatasetInfo, iter_volumes, read_info, read_volume
from fastfuncstuff.memory import get_available_memory


def _c_order(arr: np.ndarray) -> np.ndarray:
    """``arr`` in C order, transposed by torch's threaded copy when it is not.

    NumPy's copy is single-threaded: 1.1 s for a 340 MB F-ordered run on six
    cores, against 0.27 s here -- most of the wait after the gzip itself.
    """
    if arr.flags.c_contiguous:
        return arr
    if not arr.flags.writeable:
        return np.ascontiguousarray(arr)  # torch will not wrap read-only memory
    return torch.from_numpy(arr).contiguous().numpy()


class Tier(IntEnum):
    """How far along the disk-to-device path a dataset currently is.

    Ordered so that ``>=`` is a meaningful "at least this resident" test.
    """

    ABSENT = 0
    HEADER = 1  # DatasetInfo only
    PREVIEW = 2  # first volume in RAM
    RAM = 3  # whole 4-D array in RAM
    DEVICE = 4  # promoted to the compute device


@dataclass
class Resident:
    """One dataset's residency record."""

    key: str
    path: Path
    info: DatasetInfo
    preview: np.ndarray | None = None
    array: np.ndarray | None = None
    tensor: torch.Tensor | None = None
    #: Monotonic counter, not a clock: promotion order is all the LRU needs and
    #: a counter cannot go backwards when the system clock does.
    used_at: int = 0
    error: BaseException | None = None
    #: Where this dataset's array was parked when RAM ran short. Only ever set
    #: for a memory-backed dataset; a file-backed one already has its file.
    spill: Path | None = None
    #: A dataset painted on demand rather than held: volume ``i`` is
    #: ``provider(i)``. A surface result is one -- its sub-bricks live on the
    #: vertices and are painted into the ribbon only when drawn, so a 300-frame
    #: series costs one painted volume, not a 4-D copy on the grid.
    provider: Callable[[int], np.ndarray] | None = field(default=None, repr=False)
    _future: Future[np.ndarray] | None = field(default=None, repr=False)
    #: While a NIfTI inflates: the volumes so far, ``(t, k, j, i)``, of which
    #: the first ``ready`` are complete. Lets a time step read a volume the
    #: loader has already decoded instead of decoding the file again.
    partial: np.ndarray | None = field(default=None, repr=False)
    ready: int = 0
    #: Unique per record. A mode re-adopts new values under the same key on
    #: every click, so a consumer caching by key alone draws the first map
    #: forever; keyed on this, it sees each adopt as new data.
    generation: int = field(default_factory=count().__next__)

    @property
    def memory_backed(self) -> bool:
        """Whether this dataset exists only because the viewer made it.

        ``path`` is a label for these, not a location, so it is the one thing
        that must never be handed to a reader.
        """
        return self.info.storage == "MEMORY"

    @property
    def tier(self) -> Tier:
        if self.tensor is not None:
            return Tier.DEVICE
        if self.array is not None:
            return Tier.RAM
        if self.preview is not None:
            return Tier.PREVIEW
        return Tier.HEADER

    @property
    def pending(self) -> bool:
        """Whether a background inflate is actually in flight.

        Not the same as "no array yet": a 3-D dataset is complete the moment its
        single volume is previewed and never schedules a load, so testing for a
        missing array would leave it reading as loading forever.
        """
        return self._future is not None

    @property
    def ram_bytes(self) -> int:
        return 0 if self.array is None else int(self.array.nbytes)

    @property
    def device_bytes(self) -> int:
        if self.tensor is None:
            return 0
        return int(self.tensor.numel() * self.tensor.element_size())

    @property
    def full_bytes(self) -> int:
        """What a full float32 residency would cost, from the header alone."""
        nx, ny, nz, nv = self.info.shape
        return int(nx) * int(ny) * int(nz) * max(int(nv), 1) * 4


@dataclass
class Working:
    """A mode's prepared array: always in RAM, mirrored on the device while it fits.

    The host copy is what makes the mirror cheap to give up. Rebuilding a
    prepared array (mask, blur, detrend) is the slow half of a mode; with the
    result kept in RAM, losing the device copy costs one upload -- 64 ms for
    0.9 GB from pinned memory -- not a re-preparation.
    """

    key: str
    host: torch.Tensor
    tensor: torch.Tensor | None = None
    used_at: int = 0

    @property
    def device_bytes(self) -> int:
        if self.tensor is None:
            return 0
        return int(self.tensor.numel() * self.tensor.element_size())


#: Default share of the card's *total* memory the viewer may hold. Of total,
#: not of free: a budget re-read from free memory shrinks as the viewer fills
#: it and settled near a third of the card for simply looking at runs.
DEFAULT_DEVICE_SHARE = 0.25

#: Driver-free memory below which :meth:`VolumeStore.relieve` starts handing
#: device copies back, as a share of the card. Something else -- a fit, as a
#: rule -- is growing, and the viewer is the one that can afford to wait.
RELIEF_SHARE = 0.10

#: Every live store. One process can run several controllers, each with its own
#: store, and they share one card -- so the budget, the free-memory check and
#: the LRU are all taken across stores, never per store.
_STORES: weakref.WeakSet[VolumeStore] = weakref.WeakSet()

#: One recency clock for all stores, so their LRU orders interleave. A counter,
#: not a clock: it cannot go backwards when the system clock does.
_CLOCK = count(1)

#: ``-vram_budget``, when given: overrides the default share for every store.
_BUDGET: int | None = None


def configure_device_budget(budget: int | None) -> None:
    """Set (or clear) the process-wide device budget in bytes."""
    global _BUDGET
    _BUDGET = budget


def apply_vram_budget(text: str | None, device: torch.device) -> None:
    """``-vram_budget`` from the command line: unset keeps the default share."""
    configure_device_budget(None if not text else parse_budget(text, device_total(device)))


def device_total(device: torch.device) -> int:
    """The compute device's total memory in bytes; 0 for the CPU."""
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[1])
    if device.type == "mps":
        return int(torch.mps.recommended_max_memory())
    return 0


def parse_budget(text: str, total: int) -> int:
    """``-vram_budget`` as bytes: ``6`` / ``6G`` / ``512M`` absolute, ``30%`` of ``total``."""
    s = str(text).strip().upper().removesuffix("B")
    try:
        if s.endswith("%"):
            if total <= 0:
                raise ValueError("a percentage needs a GPU to be a percentage of")
            return int(total * float(s[:-1]) / 100.0)
        scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}.get(s[-1:], 0)
        return int(float(s[:-1] if scale else s) * (scale or 2**30))
    except ValueError as exc:
        raise ValueError(f"cannot read {text!r} as a memory budget (try 6G or 25%)") from exc


class VolumeStore:
    """Holds datasets across the residency tiers, evicting by least-recent use.

    Thread-safe: the UI thread reads previews and dispatches promotions while a
    worker inflates. Only the bookkeeping is locked -- the inflate itself runs
    outside the lock so a slow load never blocks a crosshair move.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        ram_budget: int | None = None,
        device_budget: int | None = None,
        max_workers: int = 2,
        zstd_threads: int | None = None,
        spill_dir: str | Path | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self._ram_budget = ram_budget
        self._device_budget = device_budget
        self._zstd_threads = zstd_threads
        self._items: dict[str, Resident] = {}
        self._work: dict[str, Working] = {}
        self._lock = threading.RLock()
        #: Signalled each time an inflating dataset gains a volume, and when it ends.
        self._progress = threading.Condition(self._lock)
        # Made on first spill, not on construction: a session that never runs
        # short of RAM should leave nothing behind in the temp directory. A
        # caller-supplied directory is used as-is and never removed.
        self._spill_dir: Path | None = Path(spill_dir).expanduser() if spill_dir else None
        self._owns_spill_dir = spill_dir is None
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ffs-viewer-load"
        )
        _STORES.add(self)

    # -- budgets -------------------------------------------------------
    #
    # RAM goes through memory.py. The device budget is a fixed share of the
    # card (DEFAULT_DEVICE_SHARE) because the viewer shares it with fits, and
    # every promotion also checks the driver's free memory so a budget that
    # was right at startup cannot push a card that has since filled.

    @property
    def ram_budget(self) -> int:
        if self._ram_budget is not None:
            return self._ram_budget
        return get_available_memory(torch.device("cpu"))

    @property
    def device_budget(self) -> int:
        if self._device_budget is not None:
            return self._device_budget
        if self.device.type == "cpu":
            return self.ram_budget
        if _BUDGET is not None:
            return _BUDGET
        return int(device_total(self.device) * DEFAULT_DEVICE_SHARE)

    def ram_used(self) -> int:
        with self._lock:
            return sum(r.ram_bytes for r in self._items.values())

    def device_used(self) -> int:
        with self._lock:
            return sum(r.device_bytes for r in self._items.values()) + sum(
                w.device_bytes for w in self._work.values()
            )

    def _peers(self) -> list[VolumeStore]:
        """Every live store computing on this store's device, this one included."""
        return [s for s in list(_STORES) if s.device == self.device] or [self]

    def device_used_all(self) -> int:
        """What every store on this device holds there -- what the budget bounds."""
        return sum(s.device_used() for s in self._peers())

    def _driver_free(self) -> int | None:
        """Free memory as the CUDA driver sees it, other processes included."""
        if self.device.type != "cuda":
            return None
        return int(torch.cuda.mem_get_info(self.device)[0])

    # -- opening -------------------------------------------------------
    def open(self, path: str | Path, key: str | None = None) -> Resident:
        """Read the header and register the dataset. Cheap enough to do eagerly."""
        p = Path(path)
        k = key or str(p)
        with self._lock:
            existing = self._items.get(k)
            if existing is not None:
                return existing
        info = read_info(p)
        if not info.exists:
            raise FileNotFoundError(f"no such dataset: {p}")
        res = Resident(key=k, path=p, info=info)
        with self._lock:
            self._items[k] = res
        return res

    def adopt(self, key: str, array: np.ndarray, *, name: str = "") -> Resident:
        """Register an in-memory volume as if it had been loaded.

        Modes compute overlays rather than reading them, but everything
        downstream -- slicing, colour mapping, the value readout -- addresses
        data through the store. Adopting keeps that one path instead of
        teaching each consumer about a second kind of layer.
        """
        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., None]
        nx, ny, nz, nv = arr.shape
        info = DatasetInfo(
            path=Path(name or key),
            iname=name or key,
            exists=True,
            storage="MEMORY",
            shape=(int(nx), int(ny), int(nz), int(nv)),
        )
        res = Resident(key=key, path=Path(name or key), info=info, array=arr)
        with self._lock:
            self._items[key] = res
            self._touch(res)
        return res

    def adopt_lazy(
        self,
        key: str,
        shape: tuple[int, int, int],
        n_volumes: int,
        provider: Callable[[int], np.ndarray],
        *,
        name: str = "",
    ) -> Resident:
        """Register a dataset whose volumes are made on request (``provider(i)``).

        Everything that reads through the store -- previews, slicing, readouts --
        sees an ordinary dataset; a full 4-D array is only ever built if something
        asks for all of it (:meth:`ensure_ram`).
        """
        nx, ny, nz = (int(v) for v in shape)
        info = DatasetInfo(
            path=Path(name or key),
            iname=name or key,
            exists=True,
            storage="MEMORY",
            shape=(nx, ny, nz, max(int(n_volumes), 1)),
        )
        res = Resident(key=key, path=Path(name or key), info=info, provider=provider)
        with self._lock:
            self._items[key] = res
            self._touch(res)
        return res

    # -- spill ---------------------------------------------------------
    def spill_dir(self) -> Path:
        """The directory parked arrays go to, made on demand."""
        with self._lock:
            if self._spill_dir is None:
                self._spill_dir = Path(tempfile.mkdtemp(prefix="ffs-viewer-"))
            self._spill_dir.mkdir(parents=True, exist_ok=True)
            return self._spill_dir

    def _write_spill(self, key: str, array: np.ndarray) -> Path:
        """Park one array, atomically enough that a crash cannot half-write it."""
        # Keys are minted (``D1``, ``A_ICORR``) so they are already filename-safe,
        # but a mode is free to name its output, so do not trust that.
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        final = self.spill_dir() / f"{stem}.npy"
        scratch = final.with_suffix(".npy.part")
        # Through a file object, not a path: np.save silently appends ".npy" to
        # a path that does not already end in it, so the rename below would
        # look for a file that was never written.
        with open(scratch, "wb") as fh:
            np.save(fh, array, allow_pickle=False)
        scratch.replace(final)
        return final

    def _drop_spill(self, res: Resident) -> None:
        if res.spill is None:
            return
        try:
            res.spill.unlink(missing_ok=True)
        except OSError:
            pass  # a temp file we could not remove is not worth failing a close over
        res.spill = None

    def get(self, key: str) -> Resident:
        with self._lock:
            try:
                return self._items[key]
            except KeyError:
                raise KeyError(f"dataset {key!r} is not open") from None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._items)

    def close(self, key: str) -> None:
        # Demote first: dropping the record alone frees the tensor into the
        # caching allocator, which keeps it reserved -- removing a layer then
        # showed no change in nvidia-smi.
        self.demote(key)
        with self._lock:
            res = self._items.pop(key, None)
        if res is not None:
            self._drop_spill(res)

    # -- preview -------------------------------------------------------
    def preview(self, key: str, index: int = 0) -> np.ndarray:
        """One volume, read only as far into the file as it sits.

        This is the "something on screen now" path. It deliberately does not
        promote the dataset or count against a budget: it is one volume, and
        making it evict a resident working set would be a bad trade.
        """
        res = self.get(key)
        if index == 0 and res.preview is not None:
            return res.preview
        if res.provider is not None:
            vol = np.ascontiguousarray(res.provider(int(index)), dtype=np.float32)
            if index == 0:
                with self._lock:
                    res.preview = vol
            return vol
        vol = None
        if res.partial is not None and 0 <= index < res.info.n_volumes:
            vol = self._from_inflate(res, int(index))
        if vol is None and res.spill is not None:
            # Memory-mapped, so stepping through a spilled run touches one
            # volume's worth of pages rather than reading the whole array back
            # in to throw all but one slab away.
            mapped = np.load(res.spill, mmap_mode="r", allow_pickle=False)
            vol = np.ascontiguousarray(mapped[..., index], dtype=np.float32)
        elif vol is None:
            vol, _ = read_volume(res.path, index)
        if index == 0:
            with self._lock:
                res.preview = vol
        return vol

    # -- full residency ------------------------------------------------
    def load_async(
        self, key: str, *, on_done: Callable[[str], None] | None = None
    ) -> Future[np.ndarray]:
        """Start (or join) a background inflate to RAM.

        Repeated calls share one future, so a UI that asks on every repaint does
        not start a second inflate of the same file.
        """
        res = self.get(key)
        with self._lock:
            if res.array is not None:
                fut: Future[np.ndarray] = Future()
                fut.set_result(res.array)
                return fut
            if res._future is not None:
                return res._future
            fut = self._pool.submit(self._inflate, key)
            res._future = fut

        if on_done is not None:
            fut.add_done_callback(lambda _f, k=key: on_done(k))
        return fut

    def _inflate(self, key: str) -> np.ndarray:
        res = self.get(key)
        try:
            if res.provider is not None:
                n = res.info.n_volumes
                arr = np.stack([res.provider(i) for i in range(n)], axis=3).astype(np.float32)
                with self._lock:
                    res.array = arr
                    res._future = None
                    self._touch(res)
                self._enforce_ram_budget(protect=key)
                return arr
            if res.spill is not None:
                arr = np.ascontiguousarray(np.load(res.spill, allow_pickle=False), dtype=np.float32)
                with self._lock:
                    res.array = arr
                    res._future = None
                    self._touch(res)
                self._enforce_ram_budget(protect=key)
                return arr

            if res.info.is_nifti and res.info.selector is None:
                arr = self._inflate_progressively(res)
            else:
                from fastfuncstuff.io.afni import load_nifti

                img = load_nifti(res.path, zstd_threads=self._zstd_threads)
                arr = np.asanyarray(img.dataobj, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[..., None]
            # NIfTI arrives Fortran-ordered. Normalizing to C order once here
            # costs one copy inside a load the user is already waiting on, and
            # makes every later (voxels, time) reshape a free view. Left as-is,
            # that copy was being paid again on every mode preparation --
            # measured at 1.9 GB of copying per InstaCorr parameter change on a
            # 1.4 GB dataset, for data that was already resident.
            arr = _c_order(arr)
        except BaseException as exc:  # surfaced via Resident.error, not swallowed
            with self._progress:
                res.error = exc
                res._future = None
                res.partial = None
                self._progress.notify_all()
            raise
        with self._progress:
            res.array = arr
            res._future = None
            res.partial = None
            self._touch(res)
            self._progress.notify_all()
        self._enforce_ram_budget(protect=key)
        return arr

    def _inflate_progressively(self, res: Resident) -> np.ndarray:
        """Decode a NIfTI volume by volume, publishing each as it lands.

        The same one pass over the file as a whole-file load, and the same
        speed, but a time step during the load no longer decodes the file a
        second time on the GUI thread to reach its volume -- 400-600 ms a step
        on a 250 MB run, during which nothing could be clicked.
        """
        nx, ny, nz, nv = res.info.shape
        buf = np.empty((nv, nz, ny, nx), np.float32)
        with self._lock:
            res.partial, res.ready = buf, 0
        volumes = iter_volumes(res.path, into=buf, zstd_threads=self._zstd_threads)
        for i, _ in enumerate(volumes):
            with self._progress:
                res.ready = i + 1
                self._progress.notify_all()
        return buf.transpose(3, 2, 1, 0)

    def _from_inflate(self, res: Resident, index: int) -> np.ndarray | None:
        """Volume ``index`` from an inflate under way, waiting for the loader if needed.

        Waiting is never slower than reading it ourselves: the loader is
        decoding the same stream from the same start, and is already ahead.
        ``None`` when no progressive inflate is running (or it failed).
        """
        with self._progress:
            while res.partial is not None and res.ready <= index:
                self._progress.wait()
            if res.partial is not None:
                return np.array(res.partial[index].T)
            array = res.array
        if array is not None and res.error is None:
            return np.ascontiguousarray(array[..., index])
        return None

    def ensure_ram(self, key: str) -> np.ndarray:
        """Block until the dataset is fully resident in RAM."""
        res = self.get(key)
        if res.array is not None:
            with self._lock:
                self._touch(res)
            return res.array
        return self.load_async(key).result()

    def promote(self, key: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Move a dataset onto the compute device, evicting others if needed.

        Returns a ``(nx, ny, nz, nv)`` tensor. Promotion is a real copy even on
        unified memory, so callers should hold the result rather than re-promote
        per frame.
        """
        res = self.get(key)
        if res.tensor is not None and res.tensor.dtype == dtype:
            with self._lock:
                self._touch(res)
            return res.tensor

        arr = self.ensure_ram(key)
        need = arr.size * torch.empty((), dtype=dtype).element_size()
        if not self._make_device_room(need, protect=key):
            raise MemoryError(f"no device room for {key!r} ({need / 2**30:.2f} GB)")
        tensor = torch.from_numpy(arr).to(device=self.device, dtype=dtype)
        with self._lock:
            res.tensor = tensor
            self._touch(res)
        return tensor

    def demote(self, key: str) -> None:
        """Drop the device copy, keeping RAM residency."""
        with self._lock:
            res = self._items.get(key)
            if res is None:
                return
            res.tensor = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def release(self, key: str) -> None:
        """Drop both device and RAM copies, keeping the header and preview.

        Free for a dataset read off disk -- the file is still there to read
        again. A dataset the viewer *made* has no such file, so it is written
        to the spill directory first. Without that step the LRU silently
        destroys exactly the results a session exists to produce, and the next
        access fails on a path that never existed.
        """
        self.demote(key)
        with self._lock:
            res = self._items.get(key)
            if res is None or res.array is None:
                return
            # A painted-on-demand dataset is free to drop, like a file: it repaints.
            keep = res.memory_backed and res.spill is None and res.provider is None
            array = res.array if keep else None
        if array is not None:
            try:
                spill = self._write_spill(key, array)
            except (OSError, ValueError):
                # Overshooting the RAM budget is recoverable; losing the only
                # copy of a result is not. Keep it resident and let the
                # eviction pass move on to a dataset that has a file.
                return
            with self._lock:
                res.spill = spill
        with self._lock:
            res.array = None

    # -- working arrays --------------------------------------------------
    def hold(self, key: str, value: torch.Tensor) -> torch.Tensor:
        """Keep a mode's prepared array: in RAM always, on the device if it fits.

        ``value`` may be on either side. Returns the copy to compute with now --
        the device mirror when there is one. Replaces anything held under
        ``key``. On CUDA the caching allocator is emptied afterwards, which is
        what hands a preparation's temporaries (several times the result)
        back to the card instead of leaving them reserved.
        """
        self.drop_working(key, empty_cache=False)
        on_device = value.device.type != "cpu"
        host = value.detach()
        if on_device:
            host = torch.empty(
                value.shape, dtype=value.dtype, pin_memory=self.device.type == "cuda"
            )
            host.copy_(value)
        work = Working(key=key, host=host)
        with self._lock:
            self._work[key] = work
            self._touch(work)
        size = work.host.numel() * work.host.element_size()
        if on_device and self._make_device_room(size, protect=key, allocated=True):
            work.tensor = value.detach()
        del value
        self._empty_cache()
        return work.tensor if work.tensor is not None else work.host

    def working(self, key: str, *, on_device: bool = False) -> torch.Tensor:
        """A held array: the device mirror if there is one, else the RAM copy.

        ``on_device`` asks for the mirror to be rebuilt when it was evicted --
        for work that is slow on the CPU (a GLM fit is 60x). It is still a
        request: with no room the RAM copy comes back and the caller computes
        on the CPU.
        """
        with self._lock:
            work = self._work.get(key)
            if work is None:
                raise KeyError(f"nothing held as {key!r}")
            self._touch(work)
            if work.tensor is not None or not on_device or self.device.type == "cpu":
                return work.tensor if work.tensor is not None else work.host
        need = work.host.numel() * work.host.element_size()
        if not self._make_device_room(need, protect=key):
            return work.host
        tensor = work.host.to(self.device, non_blocking=work.host.is_pinned())
        with self._lock:
            work.tensor = tensor
        return tensor

    def ram_copy(self, key: str) -> torch.Tensor:
        """A held array's RAM copy -- safe to keep a reference to.

        Keeping the *mirror* would pin it on the card past any eviction; the RAM
        copy exists for as long as the hold does anyway.
        """
        with self._lock:
            return self._work[key].host

    def mirrored(self, key: str) -> bool:
        with self._lock:
            work = self._work.get(key)
            return work is not None and work.tensor is not None

    def holds(self, key: str) -> bool:
        with self._lock:
            return key in self._work

    def drop_working(self, key: str, *, empty_cache: bool = True) -> None:
        with self._lock:
            work = self._work.pop(key, None)
        if work is not None and work.tensor is not None and empty_cache:
            work.tensor = None
            self._empty_cache()

    def _unmirror(self, key: str) -> None:
        with self._lock:
            work = self._work.get(key)
            if work is not None:
                work.tensor = None
        self._empty_cache()

    def _empty_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def relieve(self) -> int:
        """Give device copies back while the card is short; returns bytes released.

        For when something else -- a fit -- started after the viewer filled its
        budget. Everything released has a RAM copy, so the cost is a re-upload
        on next use, never lost work. A no-op off CUDA, where there is no driver
        figure to react to.
        """
        free = self._driver_free()
        if free is None:
            return 0
        floor = int(device_total(self.device) * RELIEF_SHARE)
        released = 0
        for store, kind, key, size in self._device_lru():
            if free + released >= floor:
                break
            store._evict(kind, key)
            released += size
        return released

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        _STORES.discard(self)
        with self._lock:
            self._work.clear()
            spill_dir = self._spill_dir if self._owns_spill_dir else None
            self._spill_dir = None
            for res in self._items.values():
                res.spill = None
        if spill_dir is not None:
            shutil.rmtree(spill_dir, ignore_errors=True)

    # -- eviction ------------------------------------------------------
    def _touch(self, res: Resident | Working) -> None:
        res.used_at = next(_CLOCK)

    def _lru_order(self, predicate: Callable[[Resident], bool]) -> list[Resident]:
        with self._lock:
            return sorted(
                (r for r in self._items.values() if predicate(r)),
                key=lambda r: r.used_at,
            )

    def _device_lru(self) -> list[tuple[VolumeStore, str, str, int]]:
        """Every device copy on this device, least recent first: (store, kind, key, bytes).

        Datasets and working arrays, in every store, share one recency order:
        whichever was touched last is what the person is looking at, whatever
        kind it is and whichever tab it is in.
        """
        entries = []
        for store in self._peers():
            with store._lock:
                entries += [
                    (r.used_at, id(store), store, "dataset", r.key, r.device_bytes)
                    for r in store._items.values()
                    if r.tensor is not None
                ] + [
                    (w.used_at, id(store), store, "working", w.key, w.device_bytes)
                    for w in store._work.values()
                    if w.tensor is not None
                ]
        entries.sort(key=lambda e: (e[0], e[1]))
        return [(store, kind, key, size) for _t, _i, store, kind, key, size in entries]

    def _evict(self, kind: str, key: str) -> None:
        if kind == "dataset":
            self.demote(key)
        else:
            self._unmirror(key)

    def _fits(self, need: int, allocated: bool) -> bool:
        if self.device_used_all() + need > self.device_budget:
            return False
        free = self._driver_free()
        # Leave the relief floor free too, or the next relieve() would hand
        # straight back what this is about to take. An ``allocated`` array is
        # already out of the driver's free figure, so it is not charged twice.
        take = 0 if allocated else need
        return free is None or take + int(device_total(self.device) * RELIEF_SHARE) <= free

    def _make_device_room(self, need: int, *, protect: str, allocated: bool = False) -> bool:
        """Evict least-recent device copies until ``need`` fits; whether it does."""
        for store, kind, key, _size in self._device_lru():
            if self._fits(need, allocated):
                return True
            if not (store is self and key == protect):
                store._evict(kind, key)
        return self._fits(need, allocated)

    def _enforce_ram_budget(self, *, protect: str) -> None:
        budget = self.ram_budget
        for res in self._lru_order(lambda r: r.array is not None):
            if self.ram_used() <= budget:
                return
            if res.key == protect or res.tensor is not None:
                continue  # never evict what the device is currently working on
            self.release(res.key)
