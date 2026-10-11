"""Parallel gzip decoding for nibabel reads of ``.nii.gz``, via rapidgzip.

Almost every dataset arrives as ``.nii.gz``, and zlib inflates on one core:
4.4 s for a 566 MB run that rapidgzip decodes on six cores in 1.1 s, with no
index file and no re-compression -- it works on any gzip anyone wrote.

It goes in through nibabel's documented extension point,
``ImageOpener.compress_ext_map``, rather than as a separate eager reader, so a
loaded image stays lazy: a header-only load still reads only the header, and
``dataobj[..., i]`` still reads only as far as volume ``i``. Two things that
point needs help with:

* rapidgzip spins up threads and reads ahead before the first byte comes back,
  which made a header-only load 48 -> 134 ms. :class:`_TieredGzip` serves reads
  near the start of the file from nibabel's own gzip reader and switches to
  rapidgzip only when a read reaches past :data:`_SHALLOW_BYTES`.
* nibabel memory-maps any file object it does not recognise as compressed, and
  numpy's memmap first seeks to the end to size the file -- a full decode thrown
  away. The class is added to nibabel's tuple of compressed file types.

A rapidgzip file still open at interpreter shutdown aborts the process from its
worker threads, so every one opened here is closed at exit.

Without rapidgzip installed, :func:`install` does nothing and nibabel reads as
it always has.
"""

from __future__ import annotations

import atexit
import io
import weakref
from typing import Any

#: A read whose end lies within this many decompressed bytes of the start is
#: served by the single-threaded reader: headers, extensions and the first few
#: volumes, where thread start-up and read-ahead cost more than they save.
_SHALLOW_BYTES = 16 * 1024 * 1024

_OPEN: weakref.WeakSet[_TieredGzip] = weakref.WeakSet()
_installed_threads: int | None = None


class _TieredGzip(io.RawIOBase):
    """A read-only gzip file that is single-threaded until a read goes deep."""

    def __init__(self, filename: str, threads: int, shallow_open: Any) -> None:
        super().__init__()
        self.name = filename
        self._threads = threads
        self._shallow_open = shallow_open
        self._shallow: Any = None
        self._deep: Any = None
        self._pos = 0
        _OPEN.add(self)

    def _reader(self, end: int) -> Any:
        if self._deep is not None:
            return self._deep
        if end <= _SHALLOW_BYTES:
            if self._shallow is None:
                self._shallow = self._shallow_open(self.name)
            return self._shallow
        import rapidgzip  # ty: ignore[unresolved-import]  # compiled, no stubs

        self._deep = rapidgzip.open(self.name, parallelization=self._threads)
        if self._shallow is not None:
            self._shallow.close()
            self._shallow = None
        return self._deep

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        raise io.UnsupportedOperation("a decompressing reader has no file descriptor")

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = int(offset)
        elif whence == io.SEEK_CUR:
            self._pos += int(offset)
        else:
            # Only the deep reader knows the decompressed size without a
            # single-threaded pass over the whole file.
            reader = self._reader(_SHALLOW_BYTES + 1)
            self._pos = reader.seek(offset, whence)
        return self._pos

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer).cast("B")
        reader = self._reader(self._pos + len(view))
        reader.seek(self._pos)
        got = 0
        # Either reader may return short; nibabel treats a short readinto as a
        # truncated file, so fill the buffer unless the stream really ended.
        while got < len(view):
            n = reader.readinto(view[got:])
            if not n:
                break
            got += n
        self._pos += got
        return got

    def close(self) -> None:
        for reader in (self._shallow, self._deep):
            if reader is not None:
                reader.close()
        self._shallow = self._deep = None
        super().close()


def _close_all() -> None:
    for f in list(_OPEN):
        f.close()


def install(threads: int) -> bool:
    """Route nibabel's ``.gz`` image reads through rapidgzip; idempotent.

    Returns whether rapidgzip is in use. ``threads`` is the decode
    parallelism; a later call with a different count updates it.
    """
    global _installed_threads
    try:
        import rapidgzip  # noqa: F401  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    from nibabel import _compression, openers, volumeutils

    threads = max(1, int(threads))
    if _installed_threads is None:
        writer, arg_names = openers.Opener.compress_ext_map[".gz"]

        def gz_open(filename: str, mode: str = "rb", *args: Any, **kwargs: Any) -> Any:
            if mode not in ("rb", "r"):
                return writer(filename, mode, *args, **kwargs)
            assert _installed_threads is not None
            return _TieredGzip(
                filename,
                _installed_threads,
                lambda name: writer(name, "rb", keep_open=kwargs.get("keep_open", False)),
            )

        openers.ImageOpener.compress_ext_map[".gz"] = (gz_open, arg_names)
        # volumeutils copied the tuple at import, so both bindings need it.
        for module in (_compression, volumeutils):
            module.COMPRESSED_FILE_LIKES += (_TieredGzip,)
        atexit.register(_close_all)
    _installed_threads = threads
    return True


__all__ = ["install"]
