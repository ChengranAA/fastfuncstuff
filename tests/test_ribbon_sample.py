"""Merged two-hemisphere meshes: splitting a result back into .lh / .rh files."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fastfuncstuff.cli_utils import split_ribbon_outputs
from fastfuncstuff.io.gifti import load_gifti_data, mesh_fingerprint, save_gifti_surface
from fastfuncstuff.processing.ribbon_sample import (
    MERGED_HEMIS_KEY,
    default_master,
    split_merged_surface,
)


def _triangle(offset: float):
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32) + offset
    return vertices, np.array([[0, 1, 2]], np.int32)


@pytest.fixture
def merged_bucket(tmp_path):
    """A two-column bucket on a merged mesh of two 3-vertex hemispheres."""
    bounds = []
    at = 0
    for hemi, offset in (("lh", 0.0), ("rh", 10.0)):
        vertices, faces = _triangle(offset)
        path = tmp_path / f"geom.native.{hemi}.midthickness.surf.gii"
        save_gifti_surface(path, vertices, faces)
        bounds.append(
            {
                "hemi": hemi,
                "start": at,
                "stop": at + 3,
                "mesh_fingerprint": mesh_fingerprint(faces, 3),
                "geometry": path.name,
            }
        )
        at += 3

    import nibabel as nib

    values = np.arange(12, dtype=np.float32).reshape(6, 2)
    arrays = [
        nib.gifti.GiftiDataArray(
            np.ascontiguousarray(values[:, k]),
            datatype="NIFTI_TYPE_FLOAT32",
            meta={"Name": name},
        )
        for k, name in enumerate(("x", "r2"))
    ]
    meta = {MERGED_HEMIS_KEY: json.dumps(bounds), "mesh_fingerprint": "merged", "other": "kept"}
    path = tmp_path / "prf.func.gii"
    nib.save(nib.gifti.GiftiImage(darrays=arrays, meta=nib.gifti.GiftiMetaData(meta)), path)
    return path, bounds, values


def test_split_gives_each_hemisphere_its_own_rows_fingerprint_and_geometry(merged_bucket):
    """The merged file says where the hemispheres are, so the split needs nothing else."""
    path, bounds, values = merged_bucket

    written = split_merged_surface(path)

    assert [p.rsplit("/", 1)[-1] for p in written] == ["prf.lh.func.gii", "prf.rh.func.gii"]
    for part, out_path in zip(bounds, written, strict=True):
        data, meta = load_gifti_data(out_path)
        np.testing.assert_array_equal(data, values[part["start"] : part["stop"]])
        assert meta["mesh_fingerprint"] == part["mesh_fingerprint"]
        assert meta["geometry"] == part["geometry"]
        assert meta["hemisphere"] == part["hemi"]
        assert meta["other"] == "kept"
        assert MERGED_HEMIS_KEY not in meta


def test_split_keeps_the_bucket_labels(merged_bucket):
    """A pRF bucket is only readable by column name; the split must not drop them."""
    import nibabel as nib

    path, _, _ = merged_bucket

    written = split_merged_surface(path)

    image = nib.load(written[0])
    assert [dict(a.meta)["Name"] for a in image.darrays] == ["x", "r2"]


def test_split_refuses_a_file_that_is_not_on_a_merged_mesh(tmp_path):
    """Without the metadata there is no honest guess at where the midline is."""
    from fastfuncstuff.io.gifti import save_gifti_data

    path = tmp_path / "plain.shape.gii"
    save_gifti_data(path, np.zeros(6, np.float32), {"mesh_fingerprint": "whatever"})

    with pytest.raises(ValueError, match="merged mesh"):
        split_merged_surface(path)


def test_split_outputs_pass_leaves_unmerged_files_alone(merged_bucket, tmp_path):
    """The post-pass splits what landed on the merged mesh and nothing else."""
    from fastfuncstuff.io.gifti import save_gifti_data

    path, _, _ = merged_bucket
    plain = tmp_path / "prf_screen.shape.gii"
    save_gifti_data(plain, np.zeros(6, np.float32), {"mesh_fingerprint": "whatever"})

    written = split_ribbon_outputs(str(tmp_path / "prf"), verbose=False)

    assert sorted(p.rsplit("/", 1)[-1] for p in written) == ["prf.lh.func.gii", "prf.rh.func.gii"]
    assert not path.exists()  # replaced by its two halves
    assert plain.exists()


def test_default_master_names_the_candidates_it_looked_for(tmp_path):
    """A subject without an anatomy in surface space has to be told which one to use."""
    (tmp_path / "surf").mkdir()

    with pytest.raises(FileNotFoundError, match="SUMA/brain.nii.gz"):
        default_master(tmp_path)


def test_geometry_merge_offsets_the_faces_and_records_each_hemisphere(tmp_path):
    """The merged mesh is two disconnected components, in hemisphere order."""
    from fastfuncstuff.io.gifti import load_gifti_surface
    from fastfuncstuff.processing.ribbon_sample import _geometry

    for hemi, offset in (("lh", 0.0), ("rh", 10.0)):
        vertices, faces = _triangle(offset)
        save_gifti_surface(tmp_path / f"geom.native.{hemi}.midthickness.surf.gii", vertices, faces)
    merged_path = str(tmp_path / "geom.native.midthickness.surf.gii")

    bounds, sizes, fingerprint = _geometry(
        str(tmp_path / "geom"), "native", ("lh", "rh"), merged_path, tmp_path
    )

    assert sizes == [3, 3]
    assert [(b["hemi"], b["start"], b["stop"]) for b in bounds] == [("lh", 0, 3), ("rh", 3, 6)]
    vertices, faces, meta = load_gifti_surface(merged_path)
    assert len(vertices) == 6
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5]])
    assert meta["mesh_fingerprint"] == fingerprint


def test_reuse_requires_the_same_settings_and_a_source_no_newer(tmp_path):
    """Sampling is minutes per run, but only an identical sampling may be reused."""
    import json
    import os

    from fastfuncstuff.processing.ribbon_sample import _reusable

    source = tmp_path / "run.nii.gz"
    source.write_bytes(b"")
    merged = tmp_path / "run-01.native.func.gii"
    merged.write_bytes(b"")
    sidecar = tmp_path / "run-01.native.sample.json"
    settings = {"subject": "s", "mesh": "native", "hemis": ["lh"], "depths": [0.5], "sampling": "f"}
    sidecar.write_text(json.dumps({**settings, "source": str(source)}))

    assert _reusable(str(merged), sidecar, str(source), settings)
    assert not _reusable(str(merged), sidecar, str(source), {**settings, "depths": [0.1, 0.9]})

    # A source touched after the sampling is new data, whatever the settings say.
    os.utime(source, (merged.stat().st_mtime + 10, merged.stat().st_mtime + 10))
    assert not _reusable(str(merged), sidecar, str(source), settings)
