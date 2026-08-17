"""Repository hygiene: tracked files must not cite gitignored paths.

`run/` (probe scratch) and `old_context/` (superseded notes) are gitignored,
so any citation of them dangles in a fresh clone — and one of them reached the
published docs site via a rendered docstring. Conclusions belong in the text;
probe paths do not. CLAUDE.md's description of `old_context/` itself is out of
scope here: it documents the directory, it does not cite a file in it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCANNED = ("src", "design", "docs", "tests", "README.md")

_IGNORED_PATH_RE = re.compile(r"old_context/|(?<![\w./])run/\w+/")


def _tracked_files() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", *SCANNED],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git unavailable")
    return [REPO / line for line in out.stdout.splitlines() if line]


def test_no_gitignored_paths_cited_in_tracked_files() -> None:
    """No tracked file under src/, design/, docs/, tests/ names run/ or old_context/."""
    guard = Path(__file__).resolve()
    offenders: list[str] = []
    for path in _tracked_files():
        if path.suffix not in (".py", ".md", ".toml", ".yml", ".yaml"):
            continue
        if path.resolve() == guard:  # this file names the banned patterns
            continue
        for lineno, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            if _IGNORED_PATH_RE.search(line):
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")

    assert not offenders, "gitignored paths cited:\n" + "\n".join(offenders)


def test_run_as_an_output_directory_is_allowed() -> None:
    """The docs use run/ as an example out_dir; only probe subpaths are banned."""
    assert not _IGNORED_PATH_RE.search('ta.plot(out_dir="run/plots")')
    assert not _IGNORED_PATH_RE.search("xdg-open run/index.html")
    assert _IGNORED_PATH_RE.search("see run/spillover_band/findings.md")
