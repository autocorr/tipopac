"""Physics primitives for tipopac (DESIGN.md §6.1).

Constants match v2.6 (task_tipopac.py:109-112).
"""

from __future__ import annotations

from typing import Union

import numpy as np

# Scalar-or-array type accepted by all public functions here.
_Numeric = Union[float, np.ndarray]

__all__ = [
    "airmass",
    "k2nt",
    "tsys_model",
    "weighted_mean_atm_T",
]

_H: float = 6.6261e-34  # J·s
_K: float = 1.3806e-23  # J/K


def k2nt(T_K: _Numeric, nu_Hz: float) -> _Numeric:
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


def airmass(zenith_angle_deg: _Numeric) -> _Numeric:
    """Flat-earth airmass: 1/cos(z). Matches v2.6 (no refraction correction)."""
    return 1.0 / np.cos(np.deg2rad(zenith_angle_deg))


def weighted_mean_atm_T(T_surf_K: _Numeric) -> _Numeric:
    """Bevis (1992) empirical relation: T_atm = 70.2 + 0.72·T_surf (K)."""
    return 70.2 + 0.72 * T_surf_K
