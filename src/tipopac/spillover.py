"""Stage-B spillover de-bias (DESIGN.md §6).

A constant, frequency-flat opacity offset δτ is present in the fitted zenith
opacity, likely ground pickup or an instrumental floor of some sort.
Subtracting it debiases ``tau_zenith`` and more effectively anchors PWV. This
is the localized swap-point for an alternative, future per-band η model.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = ["SPILLOVER_TAU_DEFAULT", "apply_spillover"]

# Campaign-measured offset in nepers (run/spillover/findings.md §8;
# HRRR-validated, C/X gives +0.0036–0.0044, am-floor-limited → central +0.0036).
SPILLOVER_TAU_DEFAULT: float = 0.0036


def apply_spillover(ds: xr.Dataset, spillover: float | None) -> None:
    """Subtract the spillover offset from ``tau_zenith`` and record it on ``ds``.

    Mutates ``ds`` in place. ``None`` or ``0`` is a no-op that records
    ``spillover_tau = 0.0``. Otherwise the constant is subtracted from finite
    ``tau_zenith`` cells only — NaN/unfit cells stay NaN — and no clamp is
    applied, so a negative value honestly signals over-subtraction (reachable
    only in opt-in low bands; default Ku/K/Ka/Q stay positive).
    """
    if not spillover:
        ds.attrs["spillover_tau"] = 0.0
        return
    tau = ds["tau_zenith"]
    ds["tau_zenith"] = xr.where(np.isfinite(tau), tau - spillover, tau)
    ds.attrs["spillover_tau"] = float(spillover)
