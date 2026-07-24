"""Physics primitives for tipopac (DESIGN.md §6.1).

Constants match v2.6 (task_tipopac.py:109-112).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# Scalar-or-array type accepted by all public functions here. DataArray is
# included so callers can keep xarray's broadcast-by-dim-name semantics.
_Numeric = float | np.ndarray | xr.DataArray

__all__ = [
    "T_CMB",
    "k2nt",
    "predicted_tsys",
    "tsys_model",
    "weighted_mean_atm_T",
]

_H: float = 6.6261e-34  # J·s
_K: float = 1.3806e-23  # J/K

T_CMB: float = 2.725  # K (Fixsen 2009)


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
    Tcmb: float = 0.0,
    spillover: _Numeric = 0.0,
) -> _Numeric:
    """Tipping-curve Tsys model: T0 + Tcmb·e^{−τ₀/cos z} + Twmt·(1 − e^{−τ₀/cos z}).

    All temperatures in noise K; z_deg in degrees. ``Tcmb`` is the CMB
    radiation temperature ``k2nt(T_CMB, ν)``; ``Tcmb=0`` reproduces the
    pre-2026-07 model, which biased τ low by ~0.8% (run/cmb_term/findings.md).
    ``spillover`` is a pre-evaluated per-z ground-pickup term (0 disables it);
    see :func:`tipopac.spillover.spillover_tsys`.
    """
    transp = np.exp(-tau0 / np.cos(np.deg2rad(z_deg)))
    return T0 + Tcmb * transp + Twmt * (1.0 - transp) + spillover


def predicted_tsys(
    ds: xr.Dataset,
    z_deg: xr.DataArray | None = None,
) -> xr.DataArray:
    """xarray-aware Tsys reconstruction: ``(T0 + Tcmb·e^−τa + Twmt·(1−e^−τa) + spill) / c``.

    Uses fitted ``T0``, ``tau_zenith``, ``Twmt``, ``tcal_fit``, and ``tcal_ref``
    persisted on the dataset, with ``c = tcal_fit / tcal_ref`` (≡ 1 in
    ``tau_per_antenna`` mode where ``tcal_fit == tcal_ref``). With
    ``z_deg=None`` the per-sample ``ds["zenith_angle"]`` is used and the
    result has shape ``(scan, antenna, spw, polarization, time)``; pass a
    1-D DataArray (e.g. ``dims=("z",)``) for a dense-grid overlay.

    When the fit ran with the spillover forward model
    (``ds.attrs["spillover_model"]`` set), ``tau_zenith`` is already
    spillover-free, so the same ``η(ν)·Bg·airmass`` term is *added* to the
    model here (inside ``/c`` — ground pickup precedes the Tcal gain) to match
    the measured Tsys. ``Bg`` uses the per-scan mean surface temperature,
    ``ds["weather_T"].mean("time")``, which broadcasts cleanly over both the
    per-sample and dense-grid ``z_deg`` shapes.

    ``Tcmb`` is derived from the ``frequency`` coord rather than persisted:
    it is a pure function of ν, so storing it would be redundant state.
    """
    if z_deg is None:
        z_deg = ds["zenith_angle"]
    c = ds["tcal_fit"] / ds["tcal_ref"]
    c = c.where(np.isfinite(c) & (c > 0), 1.0)
    transp = np.exp(-ds["tau_zenith"] / np.cos(np.deg2rad(z_deg)))
    tcmb = k2nt(T_CMB, ds["frequency"])
    pred = ds["T0"] + tcmb * transp + ds["Twmt"] * (1.0 - transp)
    if ds.attrs.get("spillover_model"):
        from tipopac.spillover import spillover_tsys

        T_surf = ds["weather_T"].mean(dim="time")
        pred = pred + spillover_tsys(ds["frequency"], T_surf, z_deg)
    return pred / c


def weighted_mean_atm_T(T_surf_K: _Numeric) -> _Numeric:
    """Bevis (1992) empirical relation: T_atm = 70.2 + 0.72·T_surf (K)."""
    return 70.2 + 0.72 * T_surf_K
