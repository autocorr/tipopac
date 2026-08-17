"""Online and user-file flag application for tipopac (DESIGN.md §8).

Public entry point: `apply(ds, online, file)` — updates ds['flag'] in-place.

Application uses a single interval-overlap expression per flag command:
    (time_utc >= t_start) & (time_utc <= t_end)
broadcast over (scan, antenna, spw, polarization, time).  This replaces
v2.6's four-case interval expansion (task_tipopac.py:1116–1199).

Online flags go through `_apply_intervals`, which accumulates that same
expression at (scan, antenna, time) and broadcasts once for the whole
batch; user-file flags keep the per-command `_apply_interval` because
they may select an spw.

Both paths share order-insensitive field regexes: `timerange` is
required, an absent antenna field selects all antennas, and commands or
rows that do not parse are warned about with their count.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from tipopac._casa import import_casatools

__all__ = ["apply"]

_log = logging.getLogger(__name__)

# Excluded REASON values from online FLAG_CMD / SDM Flag.xml (task_tipopac.py:886)
_REASON_EXCLUDE = frozenset({"ANTENNA_NOT_ON_SOURCE", "SHADOW", "CLIP_ZERO_ALL"})

# MJD epoch: 1858-11-17 00:00:00 UTC
_MJD_EPOCH = datetime(1858, 11, 17, tzinfo=timezone.utc)

# Field regexes for CASA FLAG_CMD COMMAND strings and user flag-file lines.
_TIMERANGE_RE = re.compile(r"timerange\s*=\s*'(?P<t0>[^~']+)~(?P<t1>[^']+)'")
_ANTENNA_RE = re.compile(r"(?<![A-Za-z_])antenna\s*=\s*'?(?P<v>[^&'\s]*)")


def _ymd_to_mjd_sec(s: str) -> float:
    """Parse 'YYYY/MM/DD/HH:MM:SS[.fff]' → MJD-seconds (float64)."""
    s = s.strip()
    try:
        dt = datetime.strptime(s, "%Y/%m/%d/%H:%M:%S.%f")
    except ValueError:
        dt = datetime.strptime(s, "%Y/%m/%d/%H:%M:%S")
    return (dt.replace(tzinfo=timezone.utc) - _MJD_EPOCH).total_seconds()


def _antenna_field(text: str) -> str | None:
    """Extract the antenna name from a flag command; '*' means all antennas.

    Returns None for a malformed field — an antenna value that ran into the
    next field — so the caller drops the command instead of widening it.
    """
    m = _ANTENNA_RE.search(text)
    if m is None:
        return "*"
    value = m.group("v")
    if value in ("", "-1", "*"):
        return "*"
    if "=" in value:
        return None
    return value


def _parse_timerange(text: str) -> tuple[float, float] | None:
    """Extract (t_start, t_end) in MJD-seconds from a timerange field."""
    m = _TIMERANGE_RE.search(text)
    if m is None:
        return None
    try:
        return _ymd_to_mjd_sec(m.group("t0")), _ymd_to_mjd_sec(m.group("t1"))
    except ValueError:
        return None


def _parse_command(cmd: str) -> tuple[str, float, float] | None:
    """Parse a FLAG_CMD COMMAND string → (antenna_name, t_start, t_end).

    Field order is immaterial; an absent antenna field selects all antennas.
    Returns None if no timerange is found, the times are unparseable, or the
    antenna field is malformed — mode directives without a timerange are
    dropped by the first route.
    """
    tr = _parse_timerange(cmd)
    if tr is None:
        return None
    antenna = _antenna_field(cmd)
    if antenna is None:
        return None
    return antenna, tr[0], tr[1]


def _parse_user_line(line: str) -> tuple[str, str, float, float] | None:
    """Parse a user flag-file line → (antenna, spw, t_start, t_end).

    Fields antenna and spw default to '*' (all) when absent, empty, or '*'.
    Legacy '-1' from v2.6 flag files is also treated as 'all'.
    Returns None if no timerange is found, the times are unparseable, or the
    antenna field is malformed.
    """
    tr = _parse_timerange(line)
    if tr is None:
        return None

    antenna = _antenna_field(line)
    if antenna is None:
        return None

    spw_m = re.search(r"spw\s*=\s*'?(?P<v>[^'\s]*)'?", line)
    spw = spw_m.group("v") if spw_m else "*"

    # Treat empty or legacy '-1' as wildcard
    if spw in ("", "-1"):
        spw = "*"

    return antenna, spw, tr[0], tr[1]


def _apply_interval(
    ds: xr.Dataset,
    antenna: str,
    spw: str,
    t_start: float,
    t_end: float,
) -> None:
    """OR a time-interval flag into ``ds['flag']`` for the selected (antenna, spw).

    ``antenna='*'`` and ``spw='*'`` select all. Unknown antenna or spw names
    are silently skipped. The (scan, time) interval mask broadcasts over the
    antenna/spw/polarization dims by name.
    """
    mask = (ds["time_utc"] >= t_start) & (ds["time_utc"] <= t_end)  # (scan, time)

    if antenna != "*":
        ant_coord = ds["antenna"].astype(str)
        if antenna not in ant_coord.values:
            return
        mask = mask & (ant_coord == antenna)

    if spw != "*":
        try:
            spw_int = int(spw)
        except ValueError:
            return
        if spw_int not in ds["spw"].values:
            return
        mask = mask & (ds["spw"] == spw_int)

    ds["flag"] = ds["flag"] | mask


def _apply_intervals(
    ds: xr.Dataset, commands: Sequence[tuple[str, float, float]]
) -> None:
    """OR a batch of (antenna, t_start, t_end) intervals into ``ds['flag']``.

    Same interval-overlap expression as ``_apply_interval``, accumulated at
    (scan, antenna, time) and broadcast over spw/polarization once rather
    than once per command. Unknown antenna names are silently skipped.
    """
    utc = ds["time_utc"].values
    ant_names = np.asarray(ds["antenna"].values, dtype=str)
    ant_idx = {name: i for i, name in enumerate(ant_names)}

    acc = np.zeros((utc.shape[0], len(ant_names), utc.shape[1]), dtype=bool)
    for antenna, t_start, t_end in commands:
        hit = (utc >= t_start) & (utc <= t_end)
        if antenna == "*":
            acc |= hit[:, None, :]
            continue
        a = ant_idx.get(antenna)
        if a is None:
            continue
        acc[:, a, :] |= hit

    ds["flag"] = ds["flag"] | xr.DataArray(acc, dims=("scan", "antenna", "time"))


def _warn_dropped(dropped: Sequence[str], label: str, n_total: int) -> None:
    """Warn that online flag commands or rows were unusable, with examples."""
    if not dropped:
        return
    examples = ", ".join(repr(d[:120]) for d in dropped[:3])
    _log.warning(
        "Dropped %d of %d online %ss (no usable timerange or antenna); first: %s",
        len(dropped),
        n_total,
        label,
        examples,
    )


def _warn_global(commands: Sequence[tuple[str, float, float]]) -> None:
    """Warn that antenna-less online flag commands flag every antenna."""
    n = sum(1 for antenna, _, _ in commands if antenna == "*")
    if n:
        _log.warning(
            "%d online flag commands carry no antenna field — "
            "applying them to all antennas",
            n,
        )


def apply(ds: xr.Dataset, online: bool, file: Path | None) -> xr.Dataset:
    """Apply online and user-file flags into ds['flag'] in-place.

    online=True reads FLAG_CMD from an MS, or Flag.xml from an SDM, at
    ds.attrs['source_path'].
    file, if given, is a path to a text file with one flag command per line.
    Returns ds (same object, flag variable updated).
    """
    if online:
        fmt = ds.attrs.get("source_format")
        if fmt == "ms":
            _apply_online_flags_ms(ds, ds.attrs["source_path"])
        elif fmt == "sdm":
            _apply_online_flags_sdm(ds, ds.attrs["source_path"])
        else:
            _log.debug("online=True ignored: unknown source_format %r", fmt)

    if file is not None:
        _apply_user_flags(ds, Path(file))

    return ds


def _apply_online_flags_ms(ds: xr.Dataset, source_path: str) -> None:
    _table = import_casatools().table

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

    # Online flags apply to all spws (v2.6 does not filter by spw)
    parsed: list[tuple[str, float, float]] = []
    dropped: list[str] = []
    for cmd in commands:
        p = _parse_command(str(cmd))
        if p is None:
            dropped.append(str(cmd))
        else:
            parsed.append(p)

    _apply_intervals(ds, parsed)
    _log.debug("Applied %d online flag commands", len(parsed))
    _warn_dropped(dropped, "FLAG_CMD command", len(commands))
    _warn_global(parsed)


def _parse_antenna_ids(field: str) -> list[str]:
    """Parse an ASDM antenna-id array '1 N Antenna_a Antenna_b ...' → id list."""
    parts = field.split()
    if len(parts) < 2:
        return []
    try:
        n = int(parts[1])
    except ValueError:
        return []
    return parts[2 : 2 + n]


def _apply_online_flags_sdm(ds: xr.Dataset, source_path: str) -> None:
    import sdmpy

    flag_path = Path(source_path) / "Flag.xml"
    if not flag_path.exists():
        _log.warning("Flag.xml not found at %s — skipping online flags", flag_path)
        return

    sdm = sdmpy.SDM(str(source_path), use_xsd=False, lazy=True)
    rows = list(sdm["Flag"])
    if not rows:
        _log.warning("Flag.xml is empty — no online flags applied")
        return

    ant_names = {str(a.antennaId): str(a.name) for a in sdm["Antenna"]}

    commands: list[tuple[str, float, float]] = []
    dropped: list[str] = []
    n_considered = 0
    for row in rows:
        if str(row.reason) in _REASON_EXCLUDE:
            continue
        n_considered += 1
        try:
            t_start = int(str(row.startTime)) / 1e9
            t_end = int(str(row.endTime)) / 1e9
        except ValueError:
            dropped.append(f"startTime={row.startTime} endTime={row.endTime}")
            continue
        names = [
            name
            for ant_id in _parse_antenna_ids(str(row.antennaId))
            if (name := ant_names.get(ant_id)) is not None
        ]
        if not names:
            dropped.append(f"antennaId={row.antennaId}")
            continue
        commands.extend((name, t_start, t_end) for name in names)

    _apply_intervals(ds, commands)
    _log.debug(
        "Applied %d online flag commands from %d Flag.xml rows",
        len(commands),
        n_considered - len(dropped),
    )
    _warn_dropped(dropped, "Flag.xml row", n_considered)


def _apply_user_flags(ds: xr.Dataset, file: Path) -> None:
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
        _apply_interval(ds, antenna, spw, t_start, t_end)
        n_applied += 1
    _log.debug("Applied %d user flag commands from %s", n_applied, file)
