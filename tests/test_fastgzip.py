"""rapidgzip behind nibabel's .gz opener: same bytes, still lazy, still writable."""

from __future__ import annotations

import io

import nibabel as nib
import numpy as np
import pytest

from fastfuncstuff.io import fastgzip
from fastfuncstuff.io.afni import load_nifti

pytest.importorskip("rapidgzip")


@pytest.fixture
def shallow(monkeypatch):
    """A switch-over small enough that a 1 MB fixture crosses it."""
    monkeypatch.setattr(fastgzip, "_SHALLOW_BYTES", 64 * 1024)
    assert fastgzip.install(2)


@pytest.fixture
def scaled_run(tmp_path):
    """int16 with a slope and intercept, so nibabel's scaling is exercised too."""
    rng = np.random.default_rng(0)
    data = rng.integers(-3000, 3000, size=(20, 18, 12, 40), dtype=np.int16)
    img = nib.Nifti1Image(data, np.diag([2.0, 2.0, 3.0, 1.0]))
    img.header.set_slope_inter(0.5, 10.0)
    path = tmp_path / "run.nii.gz"
    nib.save(img, str(path))
    return path


def open_plain(name):
    """nibabel's own .gz reader, which the installed opener replaces."""
    from nibabel._compression import gzip_open

    return gzip_open(name, "rb")


def _plain(path):
    """What stock nibabel reads, with our opener taken out of the way."""
    with open_plain(str(path)) as f:
        return nib.Nifti1Image.from_stream(io.BytesIO(f.read()))


def test_full_and_partial_reads_match_stock_nibabel(scaled_run, shallow):
    want = _plain(scaled_run)
    got = load_nifti(scaled_run)
    np.testing.assert_array_equal(np.asanyarray(got.dataobj), np.asanyarray(want.dataobj))
    # The last volume is past the switch-over, the first is before it.
    np.testing.assert_array_equal(got.dataobj[..., -1], want.dataobj[..., -1])
    np.testing.assert_array_equal(got.dataobj[..., 0], want.dataobj[..., 0])


def test_shallow_reads_never_start_the_parallel_reader(scaled_run, shallow):
    """Headers and early volumes are where thread start-up costs more than it saves."""
    f = fastgzip._TieredGzip(str(scaled_run), 2, open_plain)
    head = f.read(352)
    assert len(head) == 352 and f._deep is None
    f.seek(fastgzip._SHALLOW_BYTES + 10)
    f.read(100)
    assert f._deep is not None and f._shallow is None
    assert f.tell() == fastgzip._SHALLOW_BYTES + 110
    f.close()


def test_seek_end_reports_the_decompressed_size(scaled_run, shallow):
    size = len(open_plain(str(scaled_run)).read())
    f = fastgzip._TieredGzip(str(scaled_run), 2, open_plain)
    assert f.seek(0, io.SEEK_END) == size
    f.close()


def test_nibabel_does_not_try_to_memory_map_it(scaled_run, shallow):
    """numpy's memmap seeks to the end first -- a whole decode, thrown away."""
    from nibabel.volumeutils import _is_compressed_fobj

    f = fastgzip._TieredGzip(str(scaled_run), 2, open_plain)
    assert _is_compressed_fobj(f)
    f.close()


def test_writing_gz_still_works_once_installed(tmp_path, shallow):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    path = tmp_path / "out.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    np.testing.assert_array_equal(np.asanyarray(load_nifti(path).dataobj), data)
