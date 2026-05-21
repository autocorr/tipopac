"""Unit tests for tipopac.atmosphere (DESIGN.md §7, §11.1)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from tipopac import schema
from tipopac.atmosphere import _build_am_model, _tau_at_freqs, anchor, extrapolate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fitted_ds(
    tau0: float = 0.04,
    n_scan: int = 2,
    n_ant: int = 3,
    n_spw: int = 2,
    freqs_Hz: list[float] | None = None,
) -> xr.Dataset:
    """Return a minimal dataset with synthetic tau_zenith / tau_err."""
    if freqs_Hz is None:
        freqs_Hz = [22.2e9, 43.3e9]
    n_spw = len(freqs_Hz)
    rng = np.random.default_rng(42)

    tau = np.full((n_scan, n_ant, n_spw), tau0, dtype=np.float32)
    tau += rng.normal(0, 0.001, tau.shape).astype(np.float32)
    tau_err = np.full((n_scan, n_ant, n_spw), 0.005, dtype=np.float32)
    success = np.ones((n_scan, n_ant, n_spw), dtype=bool)

    n_time = 10
    return xr.Dataset(
        data_vars={
            "switched_diff": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32),
            ),
            "switched_sum": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32),
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.full((n_scan, n_ant, n_time), 45.0, dtype=np.float32),
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.full((n_ant, n_spw, 2), 5.0, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((n_scan, n_time), 280.0, dtype=np.float32),
            ),
            "weather_P": (
                ("scan", "time"),
                np.full((n_scan, n_time), 85000.0, dtype=np.float32),
            ),
            "weather_RH": (
                ("scan", "time"),
                np.full((n_scan, n_time), 0.3, dtype=np.float32),
            ),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=bool),
            ),
            "tau_zenith": (("scan", "antenna", "spw"), tau),
            "tau_err": (("scan", "antenna", "spw"), tau_err),
            "fit_success": (("scan", "antenna", "spw"), success),
            "fit_reason": (
                ("scan", "antenna", "spw"),
                np.full((n_scan, n_ant, n_spw), "ok", dtype=object),
            ),
        },
        coords={
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.arange(n_spw, dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), np.array(freqs_Hz, dtype=np.float64)),
            "bandwidth": (("spw",), np.full(n_spw, 2e9, dtype=np.float64)),
            "antenna_position": (
                ("antenna", "xyz"),
                np.zeros((n_ant, 3), dtype=np.float64),
            ),
            # MJD seconds for 2024-01-15T12:00:00 UTC (~5.13e9)
            "scan_time_start": (
                ("scan",),
                np.array([5131296000.0, 5131296120.0], dtype=np.float64)[:n_scan],
            ),
            "scan_time_end": (
                ("scan",),
                np.array([5131296090.0, 5131296210.0], dtype=np.float64)[:n_scan],
            ),
            "time_utc": (
                ("scan", "time"),
                np.tile(np.linspace(0, 90, n_time), (n_scan, 1)).astype(np.float64)
                + np.array([5131296000.0, 5131296120.0], dtype=np.float64)[
                    :n_scan, None
                ],
            ),
        },
        attrs={
            "source_path": "fake.ms",
            "source_format": "ms",
            "observatory": "VLA",
            "mode": "global_tau",
        },
    )


# ---------------------------------------------------------------------------
# anchor() — pure function tests
# ---------------------------------------------------------------------------


def test_anchor_recovers_known_scaling() -> None:
    """anchor() must recover the true PWV scaling to within 1% (DESIGN.md §11.1)."""
    import amwrap

    true_scaling = 1.3
    freqs_Hz = np.array([22.2e9, 43.3e9])

    clim = amwrap.Climatology("midlatitude_summer")
    model = _build_am_model(
        clim.pressure,
        clim.temperature,
        clim.mixing_ratio["h2o"],
        freqs_Hz.min() * 0.95,
        freqs_Hz.max() * 1.05,
    )

    # Generate synthetic τ_fit at the true scaling (no noise for a perfect test).
    tau_truth = _tau_at_freqs(model, freqs_Hz, true_scaling)
    # Small but nonzero error so the χ² denominator is valid.
    tau_err = np.full_like(tau_truth, 0.001)

    def tau_am_fn(scaling: float) -> np.ndarray:
        return _tau_at_freqs(model, freqs_Hz, scaling)

    recovered = anchor(tau_truth, tau_err, freqs_Hz, tau_am_fn)
    assert abs(recovered - true_scaling) / true_scaling < 0.01, (
        f"anchor recovered {recovered:.4f}, expected {true_scaling:.4f} (within 1%)"
    )


def test_anchor_with_noise_within_1pct() -> None:
    """anchor() should stay within 1% even with small noise on τ_fit."""
    import amwrap

    true_scaling = 0.8
    freqs_Hz = np.array([8.4e9, 22.2e9, 43.3e9])

    clim = amwrap.Climatology("midlatitude_summer")
    model = _build_am_model(
        clim.pressure,
        clim.temperature,
        clim.mixing_ratio["h2o"],
        freqs_Hz.min() * 0.95,
        freqs_Hz.max() * 1.05,
    )

    tau_truth = _tau_at_freqs(model, freqs_Hz, true_scaling)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 1e-4, tau_truth.shape)
    tau_obs = tau_truth + noise
    tau_err = np.full_like(tau_truth, 2e-4)

    def tau_am_fn(scaling: float) -> np.ndarray:
        return _tau_at_freqs(model, freqs_Hz, scaling)

    recovered = anchor(tau_obs, tau_err, freqs_Hz, tau_am_fn)
    assert abs(recovered - true_scaling) / true_scaling < 0.01


def test_anchor_bounds_respected() -> None:
    """anchor() must return a value in (0.1, 5.0) regardless of data."""
    # Garbage data that can't be fit — just check bounds.
    tau_obs = np.array([10.0, 10.0])  # unrealistically large
    tau_err = np.array([0.01, 0.01])
    freqs_Hz = np.array([22e9, 43e9])

    def tau_am_fn(scaling: float) -> np.ndarray:
        return np.array([scaling * 0.05, scaling * 0.03])

    result = anchor(tau_obs, tau_err, freqs_Hz, tau_am_fn)
    assert 0.1 <= result <= 5.0


# ---------------------------------------------------------------------------
# extrapolate() — integration with AFGL (no HTTP)
# ---------------------------------------------------------------------------


def test_extrapolate_afgl_populates_vars() -> None:
    """extrapolate() with AFGL source adds tau_extrapolated, am_freq_grid, am_tau."""
    ds = _make_fitted_ds(freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl")

    assert "tau_extrapolated" in ds
    assert "am_freq_grid" in ds
    assert "am_tau" in ds


def test_extrapolate_afgl_tau_extrapolated_shape() -> None:
    n_scan, n_spw = 2, 2
    ds = _make_fitted_ds(n_scan=n_scan, freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl")

    assert ds["tau_extrapolated"].dims == ("scan", "spw")
    assert ds["tau_extrapolated"].shape == (n_scan, n_spw)


def test_extrapolate_afgl_tau_extrapolated_positive() -> None:
    ds = _make_fitted_ds(freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl")
    assert (ds["tau_extrapolated"].values > 0).all()


def test_extrapolate_afgl_am_freq_grid_covers_spws() -> None:
    freqs_Hz = [22.2e9, 43.3e9]
    ds = _make_fitted_ds(freqs_Hz=freqs_Hz)
    extrapolate(ds, atm_profile_source="afgl")

    grid = ds["am_freq_grid"].values
    assert grid.min() < min(freqs_Hz)
    assert grid.max() > max(freqs_Hz)


def test_extrapolate_afgl_am_tau_same_length_as_grid() -> None:
    ds = _make_fitted_ds(freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl")
    assert ds["am_freq_grid"].shape == ds["am_tau"].shape


def test_extrapolate_afgl_attrs_set() -> None:
    ds = _make_fitted_ds(freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl", afgl_climatology="midlatitude_summer")

    assert ds.attrs["atm_profile_source"] == "afgl"
    assert ds.attrs["afgl_climatology"] == "midlatitude_summer"
    assert ds.attrs["pwv_scaling"] is not None
    assert ds.attrs["open_meteo_query"] is None


def test_extrapolate_afgl_pwv_scaling_in_bounds() -> None:
    ds = _make_fitted_ds(freqs_Hz=[22.2e9, 43.3e9])
    extrapolate(ds, atm_profile_source="afgl")
    assert 0.1 <= ds.attrs["pwv_scaling"] <= 5.0


def test_extrapolate_no_successful_fits_skips_gracefully() -> None:
    """If all fits failed, extrapolate() should not raise and should still set attrs."""
    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    ds["fit_success"].values[:] = False

    extrapolate(ds, atm_profile_source="afgl")

    # tau_extrapolated still written (from scaling=1.0 fallback)
    assert "tau_extrapolated" in ds
    # pwv_scaling is None since anchor couldn't run
    assert ds.attrs["pwv_scaling"] is None


# ---------------------------------------------------------------------------
# _fetch_open_meteo — monkeypatching test
# ---------------------------------------------------------------------------


def test_extrapolate_falls_back_to_afgl_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _fetch_open_meteo raises, extrapolate() falls back to AFGL."""
    import tipopac.atmosphere as atm_mod

    def _fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated network failure")

    monkeypatch.setattr(atm_mod, "_fetch_open_meteo", _fail)

    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    extrapolate(ds, atm_profile_source="open-meteo")

    assert ds.attrs["atm_profile_source"] == "afgl"
    assert "tau_extrapolated" in ds


# ---------------------------------------------------------------------------
# Slow tests — network-gated
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_fetch_open_meteo_live() -> None:
    """Live open-meteo call — shape and sign checks only."""
    import astropy.units as u

    from tipopac.atmosphere import _VLA_LAT, _VLA_LON, _fetch_open_meteo

    # A historical date guaranteed to be in archive
    # MJD seconds for 2024-01-15T12:00:00 UTC
    obs_time_mjd_s = (
        40587.0 + (datetime(2024, 1, 15, 12, tzinfo=timezone.utc).timestamp() / 86400.0)
    ) * 86400.0

    pressure, temperature, h2o_vmr, meta = _fetch_open_meteo(
        _VLA_LAT, _VLA_LON, obs_time_mjd_s
    )

    assert len(pressure) > 0
    assert (pressure.to(u.hPa).value > 0).all()
    assert (temperature.to(u.K).value > 0).all()
    assert (h2o_vmr.value >= 0).all()
    assert "endpoint" in meta
