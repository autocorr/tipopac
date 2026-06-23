"""Physics primitives for tipopac (DESIGN.md §6.1).

Constants match v2.6 (task_tipopac.py:109-112).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Scalar-or-array type accepted by all public functions here.
_Numeric = float | np.ndarray

__all__ = [
    "k2nt",
    "predicted_tsys",
    "tsys_model",
    "weighted_mean_atm_T",
]

_H: float = 6.6261e-34  # J·s
_K: float = 1.3806e-23  # J/K


def k2nt(T_K: _Numeric, nu_Hz: _Numeric) -> _Numeric:
    """Nyquist-correct kinetic temperature to noise temperature.

    In the Rayleigh-Jeans limit (hν ≪ kT) this approaches T_K.
    """
    x = _H * nu_Hz / (_K * T_K)
    return T_K * x / (np.exp(x) - 1.0)


def tsys_model(
    z_deg: _Numeric,
    T0: float,
    tau0: float,
    Twmt: float,
) -> _Numeric:
    """Tipping-curve Tsys model: T0 + Twmt·(1 − exp(−τ₀/cos z)).

    All temperatures in noise K; z_deg in degrees.
    """
    return T0 + Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z_deg))))


def predicted_tsys(
    ds: xr.Dataset,
    z_deg: xr.DataArray | None = None,
) -> xr.DataArray:
    """xarray-aware Tsys reconstruction: ``(T0 + Twmt·(1−exp(−τ/cos z))) / c``.

    Uses fitted ``T0``, ``tau_zenith``, ``Twmt``, ``tcal_fit``, and ``tcal_ref``
    persisted on the dataset, with ``c = tcal_fit / tcal_ref`` (≡ 1 in
    ``tau_per_antenna`` mode where ``tcal_fit == tcal_ref``). With
    ``z_deg=None`` the per-sample ``ds["zenith_angle"]`` is used and the
    result has shape ``(scan, antenna, spw, polarization, time)``; pass a
    1-D DataArray (e.g. ``dims=("z",)``) for a dense-grid overlay.
    """
    if z_deg is None:
        z_deg = ds["zenith_angle"]
    c = ds["tcal_fit"] / ds["tcal_ref"]
    c = c.where(np.isfinite(c) & (c > 0), 1.0)
    pred = ds["T0"] + ds["Twmt"] * (
        1.0 - np.exp(-ds["tau_zenith"] / np.cos(np.deg2rad(z_deg)))
    )
    return pred / c


def weighted_mean_atm_T(T_surf_K: _Numeric) -> _Numeric:
    """Bevis (1992) empirical relation: T_atm = 70.2 + 0.72·T_surf (K)."""
    return 70.2 + 0.72 * T_surf_K
