"""Tests for `tipopac.summary.summarize_skydip_scans` and its formatter.

The fast tests exercise the formatter with hand-rolled SkydipScanInfo
records so they can run without CASA or the validation MS. The slow
tests (marked `slow`) require ``data/tip_test.ms``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tipopac.readers.base import SkydipScanInfo
from tipopac.summary import _format_skydip_table, summarize_skydip_scans

MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"


# ---------------------------------------------------------------------------
# Fast tests — no MS required
# ---------------------------------------------------------------------------


def _sample_rows() -> list[SkydipScanInfo]:
    # 2024-03-15 12:34:56 UTC in MJD seconds = (60384 * 86400) + 45296
    t0 = 60384.0 * 86400.0 + (12 * 3600 + 34 * 60 + 56)
    t1 = t0 + 3600.0
    return [
        SkydipScanInfo(
            scan_id=3,
            start_mjd_s=t0,
            spw_ids=(16, 17, 18, 19),
            bands=("K",),
        ),
        SkydipScanInfo(
            scan_id=12,
            start_mjd_s=t1,
            spw_ids=(0, 1, 2, 3, 8, 9, 10, 11),
            bands=("L", "S"),
        ),
    ]


def test_format_skydip_table_basic() -> None:
    rows = _sample_rows()
    text = _format_skydip_table(rows, Path("/data/foo.ms"), "MS")
    # Header line includes path, format, and count.
    assert "Skydip scans in /data/foo.ms" in text
    assert "(MS, 2 scans)" in text
    # Column headers and at least one data row are present.
    assert "Scan" in text and "Start (UTC)" in text and "Band" in text
    assert "2024-03-15 12:34:56 UTC" in text
    # Multi-band scan rendered with comma-join.
    assert "L,S" in text
    # SPW list is comma+space separated.
    assert "16, 17, 18, 19" in text
    # Trailing newline.
    assert text.endswith("\n")


def test_format_skydip_table_empty() -> None:
    text = _format_skydip_table([], Path("/data/foo.sdm"), "SDM")
    assert "(SDM, 0 scans)" in text
    assert "(no DO_SKYDIP scans found)" in text


def test_format_skydip_table_columns_aligned() -> None:
    rows = _sample_rows()
    text = _format_skydip_table(rows, Path("/data/foo.ms"), "MS")
    data_lines = [
        ln
        for ln in text.splitlines()
        if ln.startswith("  ") and not ln.lstrip().startswith("-")
    ]
    # 1 header row + 2 data rows; the leading fixed-width columns
    # (Scan, Start, Band) must start at the same offset on every line,
    # which we measure by the position of the SPW IDs column.
    assert len(data_lines) == 3
    spw_starts = [ln.index("SPW IDs") for ln in data_lines if "SPW IDs" in ln]
    assert len(spw_starts) == 1
    spw_col = spw_starts[0]
    for ln in data_lines[1:]:
        # data rows start their SPW column at the same offset.
        # Find the third double-space separator.
        sep_positions = [i for i in range(len(ln) - 1) if ln[i : i + 2] == "  "]
        # We rely on the row template "  scan  time  band  spws"; the
        # final separator before spws is the one whose position + 2 ==
        # spw_col on the header line.
        assert any(p + 2 == spw_col for p in sep_positions)


# ---------------------------------------------------------------------------
# Slow tests — require data/tip_test.ms
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_summarize_skydip_scans_ms_stdout(capsys: pytest.CaptureFixture) -> None:
    if not MS_PATH.exists():
        pytest.skip(f"tip_test.ms not found at {MS_PATH}")
    summarize_skydip_scans(MS_PATH)
    out = capsys.readouterr().out
    assert f"Skydip scans in {MS_PATH}" in out
    assert "(MS," in out
    assert "Scan" in out and "Start (UTC)" in out


@pytest.mark.slow
def test_summarize_skydip_scans_ms_file(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    if not MS_PATH.exists():
        pytest.skip(f"tip_test.ms not found at {MS_PATH}")
    out_path = tmp_path / "summary.txt"
    summarize_skydip_scans(MS_PATH, output=out_path)
    # Nothing went to stdout.
    assert capsys.readouterr().out == ""
    # File content matches a fresh stdout render.
    summarize_skydip_scans(MS_PATH)
    assert out_path.read_text() == capsys.readouterr().out
