"""Instrumental spillover as a Tsys forward-model term (DESIGN.md §5, §6).

Ground emission entering the antenna sidelobes is *not* attenuated by the sky
column, so it adds a Tsys contribution ``∝ airmass`` that a naive opacity fit
partly absorbs into ``tau_zenith``. The physically-correct treatment carries it
inside the Tsys forward model as

    dTsys(ν, z) = η(ν) · k2nt(T_surf, ν) · airmass ,   airmass = 1/cos z

with ``η(ν)`` the *stored, sampling-independent* spillover efficiency
(dimensionless). δτ is then an emergent per-observation quantity, not a stored
constant — which is exactly why η, not δτ, is what we store. Injecting this term
into the Stage-A model makes ``tau_zenith`` spillover-free at the fit output, so
Stage B anchors PWV on it directly with no add-back.

η(ν) provenance: fitted-τ excess ``tau_zenith − am(hrrr_pwv)``, 1-GHz binned over
76 ``THIG0007_wide_CXUKAQ`` epochs, curvature-corrected to be sampling-
independent. See ``run/spillover_band/findings_roundtrip.md`` and
``run/spillover_band/eta_forward_model.json`` (the source of ``ETA_POLY_COEF``).

This module supersedes the retired flat post-hoc δτ de-bias.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tipopac.physics import k2nt

# Scalar-or-array type. DataArray is included so predicted_tsys keeps xarray's
# broadcast-by-dim-name semantics while the fit passes plain ndarrays.
_Numeric = float | np.ndarray | xr.DataArray

__all__ = [
    "ETA_MODEL_NAME",
    "ETA_POLY_COEF",
    "ETA_VALID_GHZ",
    "eta_of_nu",
    "spillover_tsys",
]

# Aggregate VLA spillover efficiency η(ν) = c2·ν² + c1·ν + c0 (ν in GHz,
# η dimensionless), from run/spillover_band/eta_forward_model.json
# ("eta_poly_coef_hi_to_lo"). η ≈ 0.47 % (4 GHz) → 0.14 % (50 GHz).
ETA_POLY_COEF: tuple[float, float, float] = (
    8.471277285232635e-07,
    -1.1846536203413985e-04,
    5.178032372034434e-03,
)
ETA_MODEL_NAME: str = "eta_poly_v1"

# Trusted frequency range, GHz. The JSON fit spans 4–50 GHz, but both edges
# are untrusted: C/X (<12 GHz) is diagnostic-only extrapolation (not in the
# default high-band set), and above ~45 GHz the binned δτ scatters to ~zero
# while the quadratic still gives η ≈ 0.14 %, over-correcting the Q-band top
# (findings_roundtrip.md). η is set to 0 outside this range.
ETA_VALID_GHZ: tuple[float, float] = (12.0, 45.0)


def eta_of_nu(freq_Hz: _Numeric) -> _Numeric:
    """Spillover efficiency η(ν) [dimensionless]; 0 outside ``ETA_VALID_GHZ``.

    Accepts a scalar/ndarray (fit path) or a ``DataArray`` (reconstruction);
    the boolean-mask multiply keeps xarray dim semantics intact where
    ``np.where`` would not.
    """
    nu_GHz = freq_Hz / 1e9
    c2, c1, c0 = ETA_POLY_COEF
    eta = c2 * nu_GHz**2 + c1 * nu_GHz + c0
    lo, hi = ETA_VALID_GHZ
    return eta * ((nu_GHz >= lo) & (nu_GHz <= hi))


def spillover_tsys(
    freq_Hz: _Numeric,
    T_surf_K: _Numeric,
    z_deg: _Numeric,
) -> _Numeric:
    """Forward-model spillover term ``η(ν)·k2nt(T_surf,ν)·(1/cos z)`` [noise K].

    Added to the Stage-A model before forming residuals; ground pickup precedes
    the Tcal gain, so in ``tcal_solve`` mode it sits inside ``(T0 + pred)/c``.
    Inputs may be ndarrays (fit) or ``DataArray``s (reconstruction), broadcast
    positionally or by dim name respectively.
    """
    eta = eta_of_nu(freq_Hz)
    Bg = k2nt(T_surf_K, freq_Hz)
    airmass = 1.0 / np.cos(np.deg2rad(z_deg))
    return eta * Bg * airmass
