"""Small time-conversion helpers shared across modules."""

from __future__ import annotations

from typing import overload

import numpy as np

# MJD value of the Unix epoch 1970-01-01 00:00:00 UTC.
MJD_UNIX_EPOCH: float = 40587.0


@overload
def mjd_s_to_unix_s(mjd_s: float) -> float: ...
@overload
def mjd_s_to_unix_s(mjd_s: np.ndarray) -> np.ndarray: ...
def mjd_s_to_unix_s(mjd_s: float | np.ndarray) -> float | np.ndarray:
    """Convert MJD seconds to Unix seconds."""
    return mjd_s - MJD_UNIX_EPOCH * 86400.0


def assign_groups(scan_time_start: np.ndarray, duration_s: float | None) -> np.ndarray:
    """Partition scans into time groups of at most `duration_s`.

    Greedy sequential windows in time order: open a group at the earliest
    ungrouped scan and admit scans while `t_start - group_start <=
    duration_s`. `duration_s=None` puts every scan in group 0.

    Returns the group index per scan, in the input's scan order.
    """
    n_scan = int(np.asarray(scan_time_start).size)
    groups = np.zeros(n_scan, dtype=np.int32)
    if duration_s is None or n_scan == 0:
        return groups
    if duration_s <= 0.0:
        raise ValueError(f"duration_s must be positive or None, got {duration_s!r}")

    t = np.asarray(scan_time_start, dtype=np.float64)
    order = np.argsort(t, kind="stable")
    group = 0
    window_start = t[order[0]]
    for i in order:
        if t[i] - window_start > duration_s:
            group += 1
            window_start = t[i]
        groups[i] = group
    return groups
