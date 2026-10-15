#!/usr/bin/env python
"""Run only the tests a change can reach.

    scripts/test_changed.py                 # working tree + staged + untracked vs HEAD
    scripts/test_changed.py --since main~5  # everything since a ref
    scripts/test_changed.py --list          # show the selection, run nothing

A source file selects the test files that import its dotted module name (or
``from <pkg> import <mod>``), plus ``tests/test_<mod>.py`` by name.  A changed test
file runs as itself.  Hub modules (utils, cli_utils, ...) are imported by most of
the suite, so a hub match is capped to name-matched tests: run the full tier
deliberately for those, not on every edit.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUB_CAP = 40  # more matching test files than this means "everything imports it"


def changed_files(since: str) -> list[str]:
    def git(*a: str) -> list[str]:
        out = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True)
        return [line for line in out.stdout.splitlines() if line]

    return sorted(
        set(git("diff", "--name-only", since))
        | set(git("diff", "--name-only", "--cached"))
        | set(git("ls-files", "--others", "--exclude-standard"))
    )


def select(files: list[str]) -> tuple[list[str], list[str]]:
    tests = sorted((ROOT / "tests").glob("test_*.py"))
    text = {t: t.read_text(errors="replace") for t in tests}
    picked: set[Path] = set()
    notes: list[str] = []
    for f in files:
        p = Path(f)
        if p.suffix != ".py":
            continue
        if p.parts[:1] == ("tests",):
            if (ROOT / f).exists() and p.name.startswith("test_"):
                picked.add(ROOT / f)
            continue
        if p.parts[:1] != ("fastfuncstuff",):
            continue
        mod = p.stem
        pkg = ".".join(p.with_suffix("").parts[:-1])
        dotted = ".".join(p.with_suffix("").parts)
        pat = re.compile(
            rf"{re.escape(dotted)}\b|from {re.escape(pkg)} import[^\n]*\b{re.escape(mod)}\b"
        )
        hits = {t for t in tests if pat.search(text[t])}
        named = {t for t in tests if t.stem == f"test_{mod}"}
        if len(hits) > HUB_CAP:
            notes.append(f"{f}: imported by {len(hits)} test files -- hub, only name-matched")
            hits = named
        picked |= hits | named
        if not hits | named:
            notes.append(f"{f}: no test imports it")
    return [str(t.relative_to(ROOT)) for t in sorted(picked)], notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", default="HEAD")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("pytest_args", nargs="*", help="extra args after --")
    args = ap.parse_args()

    files = changed_files(args.since)
    selected, notes = select(files)
    for n in notes:
        print("note:", n, file=sys.stderr)
    print(f"{len(files)} changed file(s) -> {len(selected)} test file(s)", file=sys.stderr)
    if args.list or not selected:
        print("\n".join(selected))
        return 0
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *args.pytest_args,
        *selected,
    ]
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
