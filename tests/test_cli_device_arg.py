"""Every CLI's ``-device`` goes through ``add_device_arg`` / ``setup_device``.

A hand-rolled ``-device`` (or ``torch.device(args.device)``) silently rejects
the documented ``cuda,N`` / ``cpu,N`` spec forms, so this is a static guard
plus a parse-time check that the registered flag takes them verbatim.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

CLI_DIR = Path(__file__).resolve().parents[1] / "fastfuncstuff" / "cli"

# autoproc: -device is the script-wide DEVICE default, validated by its own
# type= (_device_spec) and written into the emitted script, not resolved here.
# clustsim: add_device_arg is called first; the bare add_argument is only its
# unreachable ImportError fallback.
_ALLOW_HAND_ROLLED = {"autoproc.py", "clustsim.py"}


def _cli_sources():
    return sorted(CLI_DIR.glob("*.py"))


def test_no_torch_device_of_args_device():
    bad = [p.name for p in _cli_sources() if "torch.device(args.device)" in p.read_text()]
    assert bad == []


def test_no_hand_rolled_device_flag():
    pat = re.compile(r"""add_argument\(\s*["']-device["']""")
    bad = [
        p.name
        for p in _cli_sources()
        if p.name not in _ALLOW_HAND_ROLLED and pat.search(p.read_text())
    ]
    assert bad == []


def _device_modules():
    mods = []
    for p in _cli_sources():
        if p.name in _ALLOW_HAND_ROLLED:
            continue
        if "add_device_arg(" in p.read_text():
            mods.append(p.stem)
    return mods


class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


@pytest.mark.parametrize("name", _device_modules())
def test_device_flag_accepts_spec_forms(name, monkeypatch):
    from fastfuncstuff.cli_help import FfsArgumentParser

    def _grab(self, *a, **k):
        raise _Captured(self)

    for meth in ("parse_args", "parse_known_args"):
        monkeypatch.setattr(FfsArgumentParser, meth, _grab)

    mod = importlib.import_module(f"fastfuncstuff.cli.{name}")
    parser = None
    for builder in ("create_parser", "build_parser"):
        if hasattr(mod, builder):
            try:
                parser = getattr(mod, builder)()
            except _Captured as c:
                parser = c.parser
            break
    else:
        if hasattr(mod, "parse_args"):
            try:
                try:
                    mod.parse_args([])
                except TypeError:  # parse_args() reading sys.argv directly
                    mod.parse_args()
            except _Captured as c:
                parser = c.parser
    if parser is None:
        pytest.skip(f"{name}: no cheap parser builder")

    action = parser._option_string_actions["-device"]
    # Open-ended spec (so cuda,0 / cpu,4 pass through as raw strings): no type=
    # coercion and no closed choices=.
    assert action.type is None
    assert action.choices is None
    assert "cuda,0" in action.help and "cpu,8" in action.help
