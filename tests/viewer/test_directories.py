"""Directories: the numbered list a session has read from, and the favourites."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")

CPU = torch.device("cpu")


def _vol(path, value=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((6, 6, 4), value, np.float32)
    data[0, 0, 0] = 0.5  # not a label volume
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    from fastfuncstuff.viewer import favorites

    return favorites.config_path()


def test_favourites_toggle_round_trip_and_keep_other_keys(tmp_path, config):
    from fastfuncstuff.viewer import favorites

    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"other": 1}))
    a, b = tmp_path / "a", tmp_path / "b"
    assert favorites.toggle(a) and favorites.toggle(b)
    assert favorites.load() == [a.resolve(), b.resolve()]
    assert not favorites.toggle(a)
    assert favorites.load() == [b.resolve()]
    assert json.loads(config.read_text())["other"] == 1


def test_an_unreadable_favourites_file_is_set_aside_not_overwritten(config):
    from fastfuncstuff.viewer import favorites

    config.parent.mkdir(parents=True)
    config.write_text("{not json")
    assert favorites.load() == []
    assert config.with_suffix(".json.bad").read_text() == "{not json"


def test_files_are_numbered_by_the_directory_they_came_from(tmp_path):
    from fastfuncstuff.viewer.session import ViewerSession

    a = _vol(tmp_path / "one" / "anat.nii.gz")
    b = _vol(tmp_path / "two" / "stats.nii.gz")
    deep = _vol(tmp_path / "tree" / "sub" / "deep.nii.gz")
    loose = _vol(tmp_path / "elsewhere" / "loose.nii.gz")
    s = ViewerSession(device=CPU)
    try:
        s.read_directory(tmp_path / "one")
        s.read_directory(tmp_path / "two")
        s.read_directory(tmp_path / "tree", recursive=True)
        s.read_directory(tmp_path / "one")  # again: keeps its number
        assert list(s.directories) == [(tmp_path / d).resolve() for d in ("one", "two", "tree")]
        assert s.directories[(tmp_path / "tree").resolve()] is True
        assert [s.directory_number(p) for p in (a, b, deep)] == [1, 2, 3]
        assert s.directory_number(loose) is None
        s.load(str(loose))  # loaded by path: its directory joins the list
        assert s.directory_number(loose) == 4
    finally:
        s.close()


def test_going_back_to_a_directory_reads_only_what_changed(tmp_path, monkeypatch):
    from fastfuncstuff.viewer import catalog
    from fastfuncstuff.viewer.session import ViewerSession

    _vol(tmp_path / "one" / "a.nii.gz")
    _vol(tmp_path / "one" / "b.nii.gz")
    _vol(tmp_path / "two" / "c.nii.gz")
    s = ViewerSession(device=CPU)
    try:
        s.read_directory(tmp_path / "one")
        s.read_directory(tmp_path / "two")
        _vol(tmp_path / "one" / "new.nii.gz")
        described = []
        real = catalog.describe

        def counting(paths):
            paths = list(paths)
            described.extend(paths)
            return real(paths)

        monkeypatch.setattr(catalog, "describe", counting)
        s.read_directory(tmp_path / "one")
        assert [p.name for p in described] == ["new.nii.gz"]
        assert [e.name for e in s.catalog] == ["a.nii.gz", "b.nii.gz", "new.nii.gz"]
    finally:
        s.close()


def test_the_window_lists_directories_switches_and_tags_layers(tmp_path, config):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtCore, QtWidgets

    from fastfuncstuff.viewer import favorites
    from fastfuncstuff.viewer.session import ViewerSession
    from fastfuncstuff.viewer.ui.window import ViewerWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    one, two = tmp_path / "one", tmp_path / "two"
    _vol(one / "anat.nii.gz")
    _vol(two / "stats.nii.gz", 2.0)
    fav = tmp_path / "fav"
    fav.mkdir()
    favorites.save([fav.resolve()])
    session = ViewerSession(device=CPU)
    w = ViewerWindow(session)
    try:
        w.read_directory(one)
        session.load(str(one / "anat.nii.gz"))
        w._sync_layer_list()
        # One directory: no numbers.
        assert w.layer_list.item(0).text().startswith("anat")
        w.read_directory(two)
        session.load(str(two / "stats.nii.gz"))
        w._sync_layer_list()
        texts = [w.layer_list.item(r).text() for r in range(w.layer_list.count())]
        assert texts[0].startswith("2  stats") and texts[1].startswith("1  anat")
        box = w.dir_box
        items = [box.itemText(r) for r in range(box.count())]
        assert items[0] == "1  one" and items[1] == "2  two" and items[-1] == "★  fav"
        assert box.currentText() == "2  two"
        assert box.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole) == str(one.resolve())
        # A deep path never sets the window's width.
        assert box.minimumSizeHint().width() < 300
        # A row of its own, not squeezed into the button bar.
        assert w.toolBarArea(box.parent().parent()) == QtCore.Qt.ToolBarArea.TopToolBarArea
        assert w.read_button.parent() is not box.parent().parent()
        # Back to the first: the catalog is its files again.
        w._pick_directory(0)
        assert session.catalog_dir.resolve() == one.resolve()
        assert [e.name for e in session.catalog] == ["anat.nii.gz"]
        # Favourite it from the star.
        assert not w.fav_button.isChecked()
        w._toggle_favorite()
        assert one.resolve() in favorites.load() and w.fav_button.isChecked()
        assert w.dir_box.itemText(0).endswith("★")
    finally:
        w.close()
        session.close()
        app.processEvents()
