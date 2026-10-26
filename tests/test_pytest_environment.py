"""The invocation-independent bits of the test environment itself."""

from __future__ import annotations

import sys
from pathlib import Path


def test_repo_root_is_on_sys_path_whatever_launched_pytest():
    """27 test modules import fixtures from each other as ``tests.…``.

    ``python -m pytest`` puts the working directory on ``sys.path`` and the
    ``pytest`` console script does not, so this only holds because pyproject sets
    ``pythonpath = ["."]``. Without it, `make test-fast` collected 22 errors that
    a `python -m pytest` run of the same files did not have.
    """
    root = Path(__file__).resolve().parent.parent

    assert any(Path(entry or ".").resolve() == root for entry in sys.path), (
        f"{root} is not on sys.path: pytest's pythonpath setting is gone"
    )
