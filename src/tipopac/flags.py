"""Online and user-file flag application for tipopac (DESIGN.md §8).

Public entry point: `apply(ds, online, file)` — updates ds['flag'] in-place.

Application uses a single interval-overlap expression per flag command:
    (time_utc >= t_start) & (time_utc <= t_end)
broadcast over (scan, antenna, spw, polarization, time).  This replaces
v2.6's four-case interval expansion (task_tipopac.py:1116–1199).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

__all__ = ["apply"]

_log = logging.getLogger(__name__)

# Excluded REASON values from online FLAG_CMD (task_tipopac.py:886)
_REASON_EXCLUDE = frozenset({"ANTENNA_NOT_ON_SOURCE", "SHADOW", "CLIP_ZERO_ALL"})

# MJD epoch: 1858-11-17 00:00:00 UTC
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

# Regex for CASA FLAG_CMD COMMAND strings.
# Antenna field: 'ea14&&*' — extract the name before '&&' or end of quotes.
_CMD_RE = re.compile(
    r"antenna\s*=\s*'(?P<antenna>[^&']+)(?:&&[^']*)?"
    r".*?"
    r"timerange\s*=\s*'(?P<t0>[^~']+)~(?P<t1>[^']+)'",
    re.DOTALL,
)


def _ymd_to_mjd_sec(s: str) -> float:
    """Parse 'YYYY/MM/DD/HH:MM:SS[.fff]' → MJD-seconds (float64)."""
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%Y/%m/%d/%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(s, "%Y/%m/%d/%H:%M:%S")
    return (dt.replace(tzinfo=timezone.utc) - _MJD_EPOCH).total_seconds()


def _parse_command(cmd: str) -> tuple[str, float, float] | None:
    """Parse a FLAG_CMD COMMAND string → (antenna_name, t_start, t_end).

    Returns None if the string does not match the expected pattern or the
    time fields are unparseable.
    """
    m = _CMD_RE.search(cmd)
    if m is None:
        return None
    try:
        t_start = _ymd_to_mjd_sec(m.group("t0"))
        t_end = _ymd_to_mjd_sec(m.group("t1"))
    except ValueError:
        return None
    return m.group("antenna"), t_start, t_end


def _parse_user_line(line: str) -> tuple[str, str, float, float] | None:
    """Parse a user flag-file line → (antenna, spw, t_start, t_end).

    Fields antenna and spw default to '*' (all) when absent, empty, or '*'.
    Legacy '-1' from v2.6 flag files is also treated as 'all'.
    Returns None if no timerange is found or the times are unparseable.
    """
    tr_m = re.search(r"timerange\s*=\s*'(?P<t0>[^~']+)~(?P<t1>[^']+)'", line)
    if tr_m is None:
        return None
    try:
        t_start = _ymd_to_mjd_sec(tr_m.group("t0"))
        t_end = _ymd_to_mjd_sec(tr_m.group("t1"))
    except ValueError:
        return None

    ant_m = re.search(r"antenna\s*=\s*'?(?P<v>[^'\s]*)'?", line)
    spw_m = re.search(r"spw\s*=\s*'?(?P<v>[^'\s]*)'?", line)

    antenna = ant_m.group("v") if ant_m else "*"
    spw = spw_m.group("v") if spw_m else "*"

    # Treat empty or legacy '-1' as wildcard
    if antenna in ("", "-1"):
        antenna = "*"
    if spw in ("", "-1"):
        spw = "*"

    return antenna, spw, t_start, t_end


def _apply_interval(
    flag: np.ndarray,
    time_utc: np.ndarray,
    ant_names: np.ndarray,
    spw_ids: np.ndarray,
    antenna: str,
    spw: str,
    t_start: float,
    t_end: float,
) -> None:
    """OR a time-interval flag into `flag` for the selected (antenna, spw).

    `flag` is mutated in-place.  `antenna='*'` and `spw='*'` select all.
    Unknown antenna or spw names are silently skipped.
    """
    time_mask = (time_utc >= t_start) & (time_utc <= t_end)  # (n_scan, n_time)
    tm = time_mask[:, np.newaxis, np.newaxis, np.newaxis, :]  # → (n_scan,1,1,1,n_time)

    if antenna == "*":
        ant_sl: slice = slice(None)
    else:
        idx = np.where(ant_names == antenna)[0]
        if len(idx) == 0:
            return
        i = int(idx[0])
        ant_sl = slice(i, i + 1)

    if spw == "*":
        spw_sl: slice = slice(None)
    else:
        try:
            spw_int = int(spw)
        except ValueError:
            return
        idx = np.where(spw_ids == spw_int)[0]
        if len(idx) == 0:
            return
        i = int(idx[0])
        spw_sl = slice(i, i + 1)

    flag[:, ant_sl, spw_sl, :, :] |= tm


def apply(ds: xr.Dataset, online: bool, file: Path | None) -> xr.Dataset:
    """Apply online and user-file flags into ds['flag'] in-place.

    online=True reads FLAG_CMD from the MS at ds.attrs['source_path'].
    Silently skipped for SDM-format datasets (no FLAG_CMD subtable).
    file, if given, is a path to a text file with one flag command per line.
    Returns ds (same object, flag array mutated).
    """
    flag = ds["flag"].values  # (n_scan, n_ant, n_spw, n_pol, n_time)
    time_utc = ds["time_utc"].values
    ant_names = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw_ids = ds.coords["spw"].values

    if online:
        if ds.attrs.get("source_format") == "ms":
            _apply_online_flags(
                flag, time_utc, ant_names, spw_ids, ds.attrs["source_path"]
            )
        else:
            _log.debug("online=True ignored: source_format is not 'ms'")

    if file is not None:
        _apply_user_flags(flag, time_utc, ant_names, spw_ids, Path(file))

    return ds


def _apply_online_flags(
    flag: np.ndarray,
    time_utc: np.ndarray,
    ant_names: np.ndarray,
    spw_ids: np.ndarray,
    source_path: str,
) -> None:
    from casatools import table as _table

    flag_cmd_path = Path(source_path) / "FLAG_CMD"
    if not flag_cmd_path.exists():
        _log.warning(
            "FLAG_CMD subtable not found at %s — skipping online flags", flag_cmd_path
        )
        return

    tb = _table()
    try:
        tb.open(str(flag_cmd_path))
        if tb.nrows() == 0:
            _log.warning("FLAG_CMD subtable is empty — no online flags applied")
            tb.close()
            return
        exclude_tql = " and ".join(f"REASON!='{r}'" for r in sorted(_REASON_EXCLUDE))
        sub = tb.query(exclude_tql)
        commands = list(sub.getcol("COMMAND")) if sub.nrows() > 0 else []
        sub.close()
    finally:
        tb.close()

    n_applied = 0
    for cmd in commands:
        parsed = _parse_command(str(cmd))
        if parsed is None:
            continue
        antenna, t_start, t_end = parsed
        # Online flags apply to all spws (v2.6 does not filter by spw)
        _apply_interval(
            flag, time_utc, ant_names, spw_ids, antenna, "*", t_start, t_end
        )
        n_applied += 1
    _log.debug("Applied %d online flag commands", n_applied)


def _apply_user_flags(
    flag: np.ndarray,
    time_utc: np.ndarray,
    ant_names: np.ndarray,
    spw_ids: np.ndarray,
    file: Path,
) -> None:
    n_applied = 0
    for line in file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parsed = _parse_user_line(line)
        if parsed is None:
            _log.warning("Could not parse user flag line: %r", line)
            continue
        antenna, spw, t_start, t_end = parsed
        _apply_interval(
            flag, time_utc, ant_names, spw_ids, antenna, spw, t_start, t_end
        )
        n_applied += 1
    _log.debug("Applied %d user flag commands from %s", n_applied, file)
