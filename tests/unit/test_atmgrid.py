"""Unit tests for atmgrid.PwvGrid and the am-based builder.

The forward-model consistency check (`test_t_sky_at_zenith_equals_tb_z`) is
the most important here — if T_sky(airmass=1) computed from the grid does not
equal Tb_z from am at the same PWV, the entire Stage-2 fit is biased.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from tipopac.atmgrid import (
    PwvGrid,
    build_pwv_grid,
    pwv_mm_from_profile,
)


# ---------------------------------------------------------------------------
# PWV integral helper
# ---------------------------------------------------------------------------


def test_pwv_from_profile_roughly_matches_known_climatology() -> None:
    import amwrap

    clim = amwrap.Climatology("midlatitude_summer")
    pwv = pwv_mm_from_profile(clim.pressure, clim.mixing_ratio["h2o"])
    # AFGL midlatitude_summer has ~25–30 mm; allow a wide bracket since
    # this is a sanity check on the integral formula, not amwrap parity.
    assert 5.0 < pwv < 60.0, f"midlatitude_summer PWV out of plausible range: {pwv:.2f}"


def test_pwv_from_profile_independent_of_ordering() -> None:
    p = np.array([1000.0, 800.0, 500.0, 200.0]) * u.hPa
    vmr = np.array([1e-3, 5e-4, 1e-4, 1e-6])

    pwv_asc = pwv_mm_from_profile(p, vmr)
    pwv_desc = pwv_mm_from_profile(p[::-1], vmr[::-1])
    assert pwv_asc == pytest.approx(pwv_desc, rel=1e-12)


# ---------------------------------------------------------------------------
# PwvGrid: pure-python lookup tests (no am dependency)
# ---------------------------------------------------------------------------


_T_ATM_TOY: float = 270.0
_T_CMB_TOY: float = 2.725  # must match atmgrid._T_CMB


def _toy_grid() -> PwvGrid:
    """Analytic mock with am-style brightness (atmosphere + attenuated CMB).

    τ(PWV, ν) = PWV · (1 + 0.01·ν/1 GHz)
    Tb(PWV, ν) = T_atm · (1 − exp(−τ)) + T_cmb · exp(−τ)

    This matches what am.brightness_temperature actually returns (the
    atmosphere-only formulation would skip the second term).
    """
    pwv = np.linspace(1.0, 10.0, 10)
    freq = np.linspace(10e9, 30e9, 21)
    tau = pwv[:, None] * (1.0 + 0.01 * freq[None, :] / 1e9) * 0.01
    tb = _T_ATM_TOY * (1.0 - np.exp(-tau)) + _T_CMB_TOY * np.exp(-tau)
    return PwvGrid(pwv_mm=pwv, freq_Hz=freq, tau_z=tau, tb_z=tb)


def test_pwvgrid_lookup_on_grid_node() -> None:
    g = _toy_grid()
    tau, tmean = g.lookup(g.pwv_mm[3], np.array([g.freq_Hz[2], g.freq_Hz[5]]))
    np.testing.assert_allclose(tau, g.tau_z[3, [2, 5]])
    np.testing.assert_allclose(tmean, g.tmean[3, [2, 5]])


def test_pwvgrid_lookup_between_nodes_is_linear_mix() -> None:
    g = _toy_grid()
    p_lo, p_hi = g.pwv_mm[2], g.pwv_mm[3]
    p = 0.5 * (p_lo + p_hi)
    tau, _ = g.lookup(p, np.array([g.freq_Hz[7]]))
    expected = 0.5 * (g.tau_z[2, 7] + g.tau_z[3, 7])
    np.testing.assert_allclose(tau, [expected], rtol=1e-12)


def test_pwvgrid_lookup_clips_at_range() -> None:
    g = _toy_grid()
    tau_low, _ = g.lookup(0.1, g.freq_Hz)
    tau_min, _ = g.lookup(g.pwv_mm[0], g.freq_Hz)
    np.testing.assert_allclose(tau_low, tau_min)
    tau_high, _ = g.lookup(1000.0, g.freq_Hz)
    tau_max, _ = g.lookup(g.pwv_mm[-1], g.freq_Hz)
    np.testing.assert_allclose(tau_high, tau_max)


def test_pwvgrid_grad_matches_finite_difference() -> None:
    g = _toy_grid()
    p_lo, p_hi = g.pwv_mm[2], g.pwv_mm[3]
    p = 0.5 * (p_lo + p_hi)
    f = np.array([15e9, 22e9])
    _, _, dtau_dpwv, dtmean_dpwv = g.lookup_with_grad(p, f)

    delta = 1e-3
    tau_plus, tmean_plus = g.lookup(p + delta, f)
    tau_minus, tmean_minus = g.lookup(p - delta, f)
    fd_dtau = (tau_plus - tau_minus) / (2 * delta)
    fd_dtmean = (tmean_plus - tmean_minus) / (2 * delta)

    np.testing.assert_allclose(dtau_dpwv, fd_dtau, rtol=1e-9, atol=1e-15)
    np.testing.assert_allclose(dtmean_dpwv, fd_dtmean, rtol=1e-9, atol=1e-10)


def test_pwvgrid_grad_zero_at_clipped_edge() -> None:
    g = _toy_grid()
    _, _, dtau, dtmean = g.lookup_with_grad(0.0, g.freq_Hz)
    np.testing.assert_array_equal(dtau, np.zeros_like(dtau))
    np.testing.assert_array_equal(dtmean, np.zeros_like(dtmean))


def test_pwvgrid_tmean_finite_at_zero_tau() -> None:
    pwv = np.array([0.0, 1.0, 2.0])
    freq = np.array([10e9, 20e9])
    # τ = 0 row + finite rows
    tau = np.array([[0.0, 0.0], [0.05, 0.05], [0.10, 0.10]])
    # am-style Tb: atmosphere emission + attenuated CMB
    tb = _T_ATM_TOY * (1.0 - np.exp(-tau)) + _T_CMB_TOY * np.exp(-tau)
    g = PwvGrid(pwv_mm=pwv, freq_Hz=freq, tau_z=tau, tb_z=tb)
    assert np.all(np.isfinite(g.tmean))


# ---------------------------------------------------------------------------
# Forward-model consistency: T_sky(A=1) MUST equal Tb_z
# ---------------------------------------------------------------------------


def test_t_sky_at_zenith_equals_tb_z() -> None:
    """Critical: forward model at airmass=1 reproduces am's Tb_z exactly.

    am's Tb_z is total sky brightness (atmosphere + attenuated CMB).
    Stage A's T_mean is atmosphere-only, so the reconstruction adds the
    CMB term back: Tb_z = T_mean·(1−e^−τ) + T_cmb·e^−τ. If this drifts,
    every per_antenna_pwv fit is biased — the bias would look like a
    solver issue but it would be a coordinate mismatch between am's
    output and our reconstruction.
    """
    g = _toy_grid()
    airmass = 1.0
    f = np.array([12e9, 18e9, 25e9])
    for pwv_mm in g.pwv_mm[1:-1:2]:  # off-the-end PWV values
        tau_z, tmean = g.lookup(pwv_mm, f)
        t_sky = tmean * (1.0 - np.exp(-tau_z * airmass)) + _T_CMB_TOY * np.exp(
            -tau_z * airmass
        )
        # Reconstruct Tb_z at the same PWV by direct lookup.
        i = int(np.where(g.pwv_mm == pwv_mm)[0][0])
        tb_expected = np.interp(f, g.freq_Hz, g.tb_z[i, :])
        np.testing.assert_allclose(t_sky, tb_expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# am-backed builder (slow: needs the am binary)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_build_pwv_grid_with_afgl_profile() -> None:
    """Smoke test: build a small grid via am and verify it's sane."""
    import amwrap

    clim = amwrap.Climatology("midlatitude_winter")
    grid = build_pwv_grid(
        clim.pressure,
        clim.temperature,
        clim.mixing_ratio["h2o"],
        freq_min_Hz=18e9,
        freq_max_Hz=26e9,
        profile_source="afgl_midlatitude_winter",
        pwv_min_mm=1.0,
        pwv_max_mm=10.0,
        pwv_step_mm=1.0,  # coarse so the test runs fast
        n_workers=2,
    )
    assert grid.pwv_mm.size == 10
    assert grid.tau_z.shape == (10, grid.freq_Hz.size)
    # Monotone in PWV at any freq.
    for j in range(grid.freq_Hz.size):
        assert np.all(np.diff(grid.tau_z[:, j]) >= 0), (
            f"τ should be monotone increasing in PWV at freq idx {j}"
        )
    # Profile source is preserved.
    assert grid.profile_source == "afgl_midlatitude_winter"
    assert grid.pwv_unscaled_mm > 0
