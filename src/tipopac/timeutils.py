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
