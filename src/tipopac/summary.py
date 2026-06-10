"""Pretty-print a table of DO_SKYDIP scans in an MS or SDM.

Public entry point: :func:`summarize_skydip_scans`. Also runnable as
``uv run python -m tipopac.summary <path>``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from tipopac.readers import detect_reader
from tipopac.readers.base import SkydipScanInfo
from tipopac.timeutils import mjd_s_to_unix_s

__all__ = ["summarize_skydip_scans"]


_FORMAT_LABELS: dict[str, str] = {
    "MSReader": "MS",
    "SDMReader": "SDM",
}


def summarize_skydip_scans(
    path: str | Path,
    *,
    output: str | Path | None = None,
) -> None:
    """Print a table of DO_SKYDIP scans for the MS or SDM at `path`.

    For every scan with ``DO_SKYDIP`` intent the table lists scan id,
    start time (UTC), band, and SPW ids. The input format (MS vs SDM)
    is auto-detected.

    Parameters
    ----------
    path:
        MS or SDM directory.
    output:
        File to write to. ``None`` (default) prints to stdout; otherwise
        the table is written to this path, overwriting any existing
        contents.
    """
    p = Path(path)
    reader = detect_reader(p)
    rows = reader.list_skydip_scans(p)
    fmt_label = _FORMAT_LABELS.get(reader.__name__, reader.__name__)
    text = _format_skydip_table(rows, p, fmt_label)

    if output is None:
        sys.stdout.write(text)
    else:
        Path(output).write_text(text)


def _format_skydip_table(
    rows: list[SkydipScanInfo],
    path: Path,
    fmt_label: str,
) -> str:
    """Return the human-readable scan-summary text."""
    header = f"Skydip scans in {path}  ({fmt_label}, {len(rows)} scans)\n"

    if not rows:
        return header + "\n(no DO_SKYDIP scans found)\n"

    scan_strs = [str(r.scan_id) for r in rows]
    time_strs = [_format_mjd_utc(r.start_mjd_s) for r in rows]
    band_strs = [",".join(r.bands) for r in rows]
    spw_strs = [", ".join(str(s) for s in r.spw_ids) for r in rows]

    w_scan = max(len("Scan"), *(len(s) for s in scan_strs))
    w_time = max(len("Start (UTC)"), *(len(s) for s in time_strs))
    w_band = max(len("Band"), *(len(s) for s in band_strs))
    w_spws = max(len("SPW IDs"), *(len(s) for s in spw_strs))

    def row(scan: str, t: str, band: str, spws: str) -> str:
        # SPW IDs is the last column — no padding so we don't emit
        # trailing whitespace.
        return f"  {scan:>{w_scan}}  {t:<{w_time}}  {band:<{w_band}}  {spws}\n"

    lines = [header, "\n"]
    lines.append(row("Scan", "Start (UTC)", "Band", "SPW IDs"))
    lines.append(
        "  "
        + "-" * w_scan
        + "  "
        + "-" * w_time
        + "  "
        + "-" * w_band
        + "  "
        + "-" * w_spws
        + "\n"
    )
    for s, t, b, sp in zip(scan_strs, time_strs, band_strs, spw_strs, strict=True):
        lines.append(row(s, t, b, sp))
    return "".join(lines)


def _format_mjd_utc(mjd_s: float) -> str:
    """Format MJD seconds as ``YYYY-MM-DD HH:MM:SS UTC``."""
    unix_s = float(mjd_s_to_unix_s(mjd_s))
    return datetime.fromtimestamp(unix_s, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m tipopac.summary",
        description="Print a DO_SKYDIP scan summary for an MS or SDM.",
    )
    parser.add_argument("path", help="path to an MS or SDM directory")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="write the summary to this file instead of stdout",
    )
    args = parser.parse_args()
    summarize_skydip_scans(args.path, output=args.output)
