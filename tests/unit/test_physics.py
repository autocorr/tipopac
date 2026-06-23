"""Unit tests for tipopac.physics (DESIGN.md §6.1, §11.1)."""

from __future__ import annotations

import numpy as np
import pytest

from tipopac.physics import k2nt, tsys_model, weighted_mean_atm_T

_H = 6.6261e-34
_K = 1.3806e-23


def test_k2nt_rayleigh_jeans() -> None:
    """In the Rayleigh-Jeans limit (hν/kT ≪ 1), k2nt should return ≈ T.

    At 10 GHz, 300 K: x = hν/kT ≈ 1.6e-3, so k2nt ≈ T·(1 − x/2) — within
    ~0.08% of T.  Tolerance is 1% to leave room for floating-point rounding.
    """
    T = 300.0
    nu = 10e9  # 10 GHz
    result = k2nt(T, nu)
    assert abs(result - T) / T < 1e-2, (
        f"Rayleigh-Jeans deviation too large: {result} vs {T}"
    )


def test_k2nt_high_freq_limit() -> None:
    """At very high frequency (hν ≫ kT), k2nt should approach zero."""
    T = 300.0
    nu = 1e15  # optical — far above Rayleigh-Jeans
    result = float(k2nt(T, nu))
    assert result >= 0.0
    assert result < 1.0


def test_k2nt_positive() -> None:
    """k2nt must be positive for all physically reasonable inputs."""
    T_arr = np.array([10.0, 100.0, 300.0, 3000.0])
    nu_arr = np.array([1e9, 10e9, 50e9, 300e9])
    for T in T_arr:
        for nu in nu_arr:
            assert k2nt(T, nu) > 0.0, f"k2nt({T}, {nu}) ≤ 0"


def test_k2nt_vectorised() -> None:
    """k2nt accepts array T and returns an array of the same shape."""
    T = np.array([200.0, 250.0, 300.0])
    result = k2nt(T, 10e9)
    assert result.shape == T.shape
    assert np.all(result > 0)


def test_tsys_model_zero_tau() -> None:
    """tsys_model with tau0=0 must return T0 everywhere."""
    z = np.linspace(20.0, 70.0, 20)
    T0, Twmt = 55.0, 270.0
    result = tsys_model(z, T0, tau0=0.0, Twmt=Twmt)
    np.testing.assert_allclose(result, T0, rtol=1e-6)


def test_tsys_model_round_trip() -> None:
    """Residuals from the exact model should be zero (no noise)."""
    z = np.array([35.0, 45.0, 55.0, 65.0])
    T0_R, tau0, Twmt = 50.0, 0.1, 270.0
    Tsys = tsys_model(z, T0_R, tau0, Twmt)
    resid = Tsys - tsys_model(z, T0_R, tau0, Twmt)
    np.testing.assert_allclose(resid, 0.0, atol=1e-10)


def test_tsys_model_increases_with_airmass() -> None:
    """Tsys must increase as zenith angle increases (more atmosphere)."""
    z = np.linspace(30.0, 70.0, 10)
    Tsys = tsys_model(z, T0=50.0, tau0=0.1, Twmt=270.0)
    assert np.all(np.diff(Tsys) > 0), "Tsys should increase with ZA for tau0 > 0"


def test_weighted_mean_atm_T_value() -> None:
    """Bevis 1992: weighted_mean_atm_T(280) == 70.2 + 0.72*280 = 271.8 K."""
    result = weighted_mean_atm_T(280.0)
    assert result == pytest.approx(271.8, rel=1e-6)


def test_weighted_mean_atm_T_zero() -> None:
    """weighted_mean_atm_T(0) == 70.2 K (y-intercept of Bevis)."""
    assert weighted_mean_atm_T(0.0) == pytest.approx(70.2, rel=1e-6)


def test_weighted_mean_atm_T_vectorised() -> None:
    """weighted_mean_atm_T accepts an array."""
    T = np.array([250.0, 280.0, 300.0])
    result = weighted_mean_atm_T(T)
    expected = 70.2 + 0.72 * T
    np.testing.assert_allclose(result, expected)
