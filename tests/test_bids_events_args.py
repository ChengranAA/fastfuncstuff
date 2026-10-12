"""Every CLI that reads BIDS -events exposes the same flags that shape the read.

-event_cols and the -event_filter_in/out row filters used to be copy-pasted per
tool, so a flag that worked in one CLI was missing from the next, and only two
tools had the filters at all. Each tool now registers them through
cli_utils.add_bids_events_args; this pins that surface.
"""

import argparse
import importlib

import pytest

from fastfuncstuff.cli_help import FfsArgumentParser
from fastfuncstuff.cli_utils import bids_event_kwargs, check_bids_event_args

TOOLS = [
    "bsds",
    "deconvolve",
    "denoisatorial",
    "denoise",
    "fitbasis",
    "hrfopt",
    "ica",
    "librarian",
    "locomoco",
    "nordic",
    "phasereg",
    "reml",
    "ridge",
    "util_eventcheck",
]

EVENT_FLAGS = {
    "-events": ("events", "+"),
    "-event_ignore": ("event_ignore", "+"),
    "-event_cols": ("event_cols", 3),
    "-event_filter_in": ("event_filter_in", "+"),
    "-event_filter_out": ("event_filter_out", "+"),
}


class _Captured(Exception):
    pass


def _parser_for(module_name: str, monkeypatch) -> FfsArgumentParser:
    """The tool's parser, however its module happens to expose it."""
    seen: list[FfsArgumentParser] = []

    def capture(self, *a, **k):
        seen.append(self)
        raise _Captured

    monkeypatch.setattr(FfsArgumentParser, "parse_known_args", capture)
    module = importlib.import_module(f"fastfuncstuff.cli.{module_name}")
    for factory in ("create_parser", "build_parser", "parse_args"):
        fn = getattr(module, factory, None)
        if fn is None:
            continue
        try:
            out = fn([]) if factory == "parse_args" and module_name == "nordic" else fn()
        except _Captured:
            break
        if isinstance(out, FfsArgumentParser):
            return out
    assert seen, f"no parser found for {module_name}"
    return seen[-1]


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_registers_every_event_flag_once(tool, monkeypatch):
    parser = _parser_for(tool, monkeypatch)
    # Registered spellings only: the twin is derived at parse time, and
    # registering both by hand leaks it into -help and completions.
    registered = {s for a in parser._actions for s in a.option_strings}
    for flag, (dest, nargs) in EVENT_FLAGS.items():
        assert flag in registered, f"{tool} lacks {flag}"
        if "_" in flag:
            assert flag.replace("_", "-") not in registered, f"{tool} hand-registers a twin"
        action = next(a for a in parser._actions if flag in a.option_strings)
        assert (action.dest, action.nargs) == (dest, nargs), (tool, flag)

    parser._ffs_apply_spelling_aliases()
    assert "-event-cols" in parser._option_string_actions


def _ns(**kw):
    base = dict(
        events=None, event_ignore=None, event_cols=None, event_filter_in=None, event_filter_out=None
    )
    base.update(kw)
    return argparse.Namespace(**base)


@pytest.mark.parametrize(
    "flag, value",
    [
        ("event_ignore", ["fixation"]),
        ("event_cols", ["onset", "duration", "stim"]),
        ("event_filter_out", [["hemifield", "right"]]),
    ],
)
def test_shaping_flags_without_events_exit(flag, value, capsys):
    with pytest.raises(SystemExit):
        check_bids_event_args(_ns(**{flag: value}))
    assert f"-{flag} requires -events" in capsys.readouterr().out


def test_filter_without_value_exits(capsys):
    with pytest.raises(SystemExit):
        check_bids_event_args(_ns(events=["a.tsv"], event_filter_in=[["hemifield"]]))
    assert "needs a COLUMN and at least one VALUE" in capsys.readouterr().out


def test_kwargs_carry_filters_to_every_reader():
    kw = bids_event_kwargs(
        _ns(
            events=["a.tsv"],
            event_cols=["t", "d", "stim"],
            event_filter_out=[["hemifield", "right", "upper"]],
        )
    )
    assert kw["event_cols"] == ("t", "d", "stim")
    assert kw["event_ignore"] is None
    (flt,) = kw["event_filters"]
    assert (flt.column, flt.values, flt.action) == ("hemifield", ["right", "upper"], "out")
