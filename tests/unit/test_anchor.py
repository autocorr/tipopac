"""Unit tests for tipopac.anchor — Stage B PWV anchor + Stage A T_mean lookup.

See ``design/independent_tau_fit.md`` §2 for the architecture.
"""

from __future__ import annotations

import numpy as np
import pytest

from tipopac.anchor import anchor_pwv, compute_t_mean_grid
from tipopac.atmgrid import PwvGrid
from tipopac.physics import k2nt


# ---------------------------------------------------------------------------
# Toy grid: analytic τ(PWV, ν), Tb(PWV, ν) for stable test fixtures
# ---------------------------------------------------------------------------


def _toy_grid(pwv_unscaled_mm: float = 5.0) -> PwvGrid:
    """Analytic grid: τ ∝ PWV · (1 + 0.01·ν), Tb = 270·(1 − exp(−τ)).

    Linear-in-PWV so the bilinear interpolant is exact, which lets the
    Cramér–Rao σ_PWV self-consistency check pass tightly.
    """
    pwv = np.linspace(1.0, 10.0, 19)  # 0.5 mm step matches default
    freq = np.linspace(10e9, 30e9, 41)  # 0.5 GHz step
    tau = pwv[:, None] * (1.0 + 0.01 * freq[None, :] / 1e9) * 0.01
    tb = 270.0 * (1.0 - np.exp(-tau))
    return PwvGrid(
        pwv_mm=pwv,
        freq_Hz=freq,
        tau_z=tau,
        tb_z=tb,
        pwv_unscaled_mm=pwv_unscaled_mm,
    )


# ---------------------------------------------------------------------------
# anchor_pwv
# ---------------------------------------------------------------------------


def test_anchor_recovers_known_pwv_noiseless() -> None:
    """With τ_z synthesised at a known PWV, anchor recovers it tightly."""
    grid = _toy_grid()
    pwv_true = 4.3
    freqs = np.linspace(12e9, 28e9, 16)
    tau_true, _ = grid.lookup(pwv_true, freqs)

    n_scan, n_ant = 3, 4
    tau_z = np.tile(tau_true, (n_scan, n_ant, 1)).astype(np.float64)
    tau_err = np.full_like(tau_z, 1e-4)

    grids = {i: grid for i in range(n_scan)}
    pwv_ant, pwv_err_ant = anchor_pwv(tau_z, tau_err, grids, freqs)

    assert np.all(np.isfinite(pwv_ant))
    np.testing.assert_allclose(pwv_ant, pwv_true, atol=5e-3)
    # σ_PWV from Cramér–Rao should be small and finite
    assert np.all(pwv_err_ant > 0)
    assert np.all(pwv_err_ant < 0.05)


def test_anchor_sigma_pwv_matches_residual_scatter() -> None:
    """σ_PWV from Cramér–Rao should match the empirical scatter across realisations.

    Standard test for proper error propagation: an unbiased σ_PWV satisfies
    σ_empirical / σ_predicted ∈ 1 ± √(2/(n-1)) at the 1σ level.
    """
    grid = _toy_grid()
    pwv_true = 5.7
    freqs = np.linspace(12e9, 28e9, 16)
    tau_true, _ = grid.lookup(pwv_true, freqs)

    n_realisations = 80
    sigma_tau = 5e-4
    rng = np.random.default_rng(123)

    pwv_recovered = np.empty(n_realisations)
    pwv_err_reported = np.empty(n_realisations)
    for i in range(n_realisations):
        noise = rng.normal(0.0, sigma_tau, size=tau_true.shape)
        tau_z = (tau_true + noise)[None, None, :]
        tau_err = np.full_like(tau_z, sigma_tau)
        pwv_ant, pwv_err_ant = anchor_pwv(tau_z, tau_err, {0: grid}, freqs)
        pwv_recovered[i] = float(pwv_ant[0])
        pwv_err_reported[i] = float(pwv_err_ant[0])

    empirical = float(np.std(pwv_recovered, ddof=1))
    predicted = float(np.mean(pwv_err_reported))
    # 3× tolerance on the √(2/(n−1)) sampling bound (≈0.16) to keep CI green
    assert abs(empirical / predicted - 1.0) < 0.5, (
        f"empirical σ_PWV={empirical:.4f}, predicted σ_PWV={predicted:.4f}"
    )
    # And the recovered mean should not be biased away from the truth
    assert abs(np.mean(pwv_recovered) - pwv_true) < 3 * empirical / np.sqrt(
        n_realisations
    )


def test_anchor_nan_cells_skipped() -> None:
    """Non-finite τ_z cells contribute nothing; valid cells still drive the fit."""
    grid = _toy_grid()
    pwv_true = 3.1
    freqs = np.linspace(12e9, 28e9, 8)
    tau_true, _ = grid.lookup(pwv_true, freqs)

    tau_z = np.tile(tau_true, (1, 1, 1)).astype(np.float64)
    tau_err = np.full_like(tau_z, 5e-4)
    # NaN-out half the spws; fit should still recover PWV from the other half
    tau_z[0, 0, ::2] = np.nan

    pwv_ant, pwv_err_ant = anchor_pwv(tau_z, tau_err, {0: grid}, freqs)
    np.testing.assert_allclose(pwv_ant, pwv_true, atol=2e-2)
    assert np.isfinite(pwv_err_ant[0])


def test_anchor_zero_valid_cells_returns_nan() -> None:
    """Antenna with no finite (scan, spw) cells gets NaN PWV and σ_PWV."""
    grid = _toy_grid()
    freqs = np.linspace(12e9, 28e9, 8)
    tau_z = np.full((1, 1, len(freqs)), np.nan)
    tau_err = np.full_like(tau_z, np.nan)
    pwv_ant, pwv_err_ant = anchor_pwv(tau_z, tau_err, {0: grid}, freqs)
    assert np.isnan(pwv_ant[0])
    assert np.isnan(pwv_err_ant[0])


def test_anchor_missing_scan_grid_silently_skipped() -> None:
    """Scans absent from `grids` contribute nothing — the others still drive the fit."""
    grid = _toy_grid()
    pwv_true = 6.2
    freqs = np.linspace(12e9, 28e9, 12)
    tau_true, _ = grid.lookup(pwv_true, freqs)

    n_scan = 3
    tau_z = np.tile(tau_true, (n_scan, 1, 1)).astype(np.float64)
    tau_err = np.full_like(tau_z, 5e-4)
    # Only scan 1 has a grid; scans 0 and 2 are silently dropped.
    grids = {1: grid}
    pwv_ant, pwv_err_ant = anchor_pwv(tau_z, tau_err, grids, freqs)
    np.testing.assert_allclose(pwv_ant, pwv_true, atol=5e-3)
    assert np.isfinite(pwv_err_ant[0])


def test_anchor_shape_validation() -> None:
    """Shape mismatch between tau_z, tau_err, and freqs raises ValueError."""
    grid = _toy_grid()
    freqs = np.linspace(12e9, 28e9, 4)
    tau_z = np.zeros((1, 1, 4))
    tau_err = np.zeros((1, 1, 5))  # wrong shape
    with pytest.raises(ValueError, match="tau_err shape"):
        anchor_pwv(tau_z, tau_err, {0: grid}, freqs)

    tau_err = np.zeros_like(tau_z)
    bad_freqs = freqs[:3]
    with pytest.raises(ValueError, match="freqs_Hz shape"):
        anchor_pwv(tau_z, tau_err, {0: grid}, bad_freqs)


# ---------------------------------------------------------------------------
# compute_t_mean_grid
# ---------------------------------------------------------------------------


def test_compute_t_mean_grid_matches_lookup() -> None:
    """T_mean values must equal k2nt(grid.lookup(pwv_unscaled).tmean) per channel."""
    grid = _toy_grid(pwv_unscaled_mm=6.0)
    grids = {0: grid, 2: grid}
    freqs = np.linspace(12e9, 28e9, 5)
    t_mean = compute_t_mean_grid(grids, freqs, n_scan=3)

    assert t_mean.shape == (3, 5)
    # Scan 1 has no grid — row is NaN
    assert np.all(np.isnan(t_mean[1, :]))

    # Scans 0 and 2 share the same grid → identical T_mean rows
    np.testing.assert_allclose(t_mean[0, :], t_mean[2, :])

    # Spot-check: T_mean = k2nt(tmean_kinetic, ν) at pwv_unscaled
    _tau, tmean_K = grid.lookup(grid.pwv_unscaled_mm, freqs)
    expected = np.array(
        [float(k2nt(float(t), float(f))) for t, f in zip(tmean_K, freqs)]
    )
    np.testing.assert_allclose(t_mean[0, :], expected, rtol=1e-12)


def test_compute_t_mean_grid_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one PwvGrid"):
        compute_t_mean_grid({}, np.array([10e9]))


def test_compute_t_mean_grid_infers_n_scan() -> None:
    """Without n_scan, infer max(keys)+1 — leaves trailing missing scans absent."""
    grid = _toy_grid()
    freqs = np.array([15e9])
    t_mean = compute_t_mean_grid({0: grid, 2: grid}, freqs)
    assert t_mean.shape == (3, 1)
