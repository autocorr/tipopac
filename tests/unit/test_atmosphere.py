"""Unit tests for tipopac.atmosphere (DESIGN.md §7, §11.1)."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from tipopac import schema
from tipopac.atmosphere import _MJD_UNIX_EPOCH, _mjd_s_to_unix_s, attach_profile


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
            "exposure_time": (
                ("scan", "time"),
                np.full((n_scan, n_time), 1.0, dtype=np.float32),
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
                np.array([5212036800.0, 5212036920.0], dtype=np.float64)[:n_scan],
            ),
            "scan_time_end": (
                ("scan",),
                np.array([5212036890.0, 5212037010.0], dtype=np.float64)[:n_scan],
            ),
            "time_utc": (
                ("scan", "time"),
                np.tile(np.linspace(0, 90, n_time), (n_scan, 1)).astype(np.float64)
                + np.array([5212036800.0, 5212036920.0], dtype=np.float64)[
                    :n_scan, None
                ],
            ),
        },
        attrs={
            "source_path": "fake.ms",
            "source_format": "ms",
            "observatory": "VLA",
            "mode": "tcal_solve",
        },
    )


# ---------------------------------------------------------------------------
# _mjd_s_to_unix_s
# ---------------------------------------------------------------------------


def test_mjd_s_to_unix_s_at_unix_epoch() -> None:
    """MJD of the Unix epoch must map to Unix second 0."""
    assert _mjd_s_to_unix_s(_MJD_UNIX_EPOCH * 86400.0) == 0.0


def test_mjd_s_to_unix_s_one_day_after_epoch() -> None:
    """One MJD day after the Unix epoch is exactly 86400 Unix seconds."""
    assert _mjd_s_to_unix_s((_MJD_UNIX_EPOCH + 1.0) * 86400.0) == 86400.0


def test_mjd_s_to_unix_s_known_datetime() -> None:
    """Round-trip through datetime for a known observation timestamp."""
    # 5212036800.0 MJD-s is used in _make_fitted_ds with comment 2024-01-15T12:00:00 UTC.
    unix_s = _mjd_s_to_unix_s(5212036800.0)
    dt = datetime.fromtimestamp(unix_s, tz=timezone.utc)
    assert dt == datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_mjd_s_to_unix_s_array() -> None:
    """Vectorised call works element-wise on a numpy array."""
    mjd_s = np.array([_MJD_UNIX_EPOCH * 86400.0, (_MJD_UNIX_EPOCH + 1.0) * 86400.0])
    result = _mjd_s_to_unix_s(mjd_s)
    np.testing.assert_array_equal(result, [0.0, 86400.0])


# ---------------------------------------------------------------------------
# attach_profile() — AFGL path (no HTTP)
# ---------------------------------------------------------------------------


def test_attach_profile_afgl_writes_atm_vars() -> None:
    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    attach_profile(ds, source="afgl", afgl_climatology="midlatitude_summer")

    assert "atm_pressure" in ds.data_vars
    assert "atm_temperature" in ds.data_vars
    assert "atm_h2o_vmr" in ds.data_vars
    assert ds["atm_pressure"].dims == ("atm_level",)
    assert ds["atm_temperature"].dims == ("scan", "atm_level")
    assert ds.attrs["atm_profile_source"] == "afgl_midlatitude_summer"


def test_attach_profile_afgl_auto_picks_winter_in_winter() -> None:
    """auto climatology picks midlatitude_winter for a Jan observation."""
    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    # _make_fitted_ds uses MJD ~5.13e9 = 2024-01-15 — January.
    attach_profile(ds, source="afgl", afgl_climatology="auto")

    assert ds.attrs["atm_profile_source"] == "afgl_midlatitude_winter"


# ---------------------------------------------------------------------------
# _fetch_open_meteo — monkeypatching test
# ---------------------------------------------------------------------------


def test_attach_profile_falls_back_to_afgl_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _fetch_open_meteo raises, attach_profile() falls back to AFGL."""
    import tipopac.atmosphere as atm_mod

    def _fail(*args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated network failure")

    # Skip the retry-backoff sleep so the test stays fast.
    monkeypatch.setattr(atm_mod, "_fetch_open_meteo", _fail)
    monkeypatch.setattr(atm_mod.time, "sleep", lambda _s: None)

    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    attach_profile(ds, source="open-meteo", afgl_climatology="auto")

    # Jan obs → winter pick
    assert ds.attrs["atm_profile_source"] == "afgl_midlatitude_winter"
    assert "atm_pressure" in ds.data_vars


def test_attach_profile_pre_2021_skips_open_meteo_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observations before the gfs_hrrr archive cutoff bypass open-meteo entirely."""
    import tipopac.atmosphere as atm_mod

    def _fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("open-meteo must not be called for pre-2021 dates")

    monkeypatch.setattr(atm_mod, "_fetch_open_meteo", _fail_if_called)

    ds = _make_fitted_ds(freqs_Hz=[22.2e9])
    # MJD seconds for 2020-06-01T12:00:00 UTC — clearly before 2021-03-23.
    pre_cutoff_mjd_s = 5097729600.0
    n_scan = ds.sizes["scan"]
    ds = ds.assign_coords(
        scan_time_start=(
            ("scan",),
            np.array(
                [pre_cutoff_mjd_s + 120.0 * i for i in range(n_scan)], dtype=np.float64
            ),
        ),
        scan_time_end=(
            ("scan",),
            np.array(
                [pre_cutoff_mjd_s + 120.0 * i + 90.0 for i in range(n_scan)],
                dtype=np.float64,
            ),
        ),
    )

    attach_profile(ds, source="open-meteo", afgl_climatology="auto")

    # June obs → summer climatology
    assert ds.attrs["atm_profile_source"] == "afgl_midlatitude_summer"
    assert "open_meteo_query" not in ds.attrs
    assert "atm_pressure" in ds.data_vars


def test_attach_profile_open_meteo_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single open-meteo call regardless of scan count."""
    import astropy.units as u

    import tipopac.atmosphere as atm_mod

    call_count = {"n": 0}

    def _fake_fetch(lat, lon, date_start, date_end):
        call_count["n"] += 1
        # 2 hourly slices x 2 levels
        pressure = np.array([800.0, 500.0]) * u.hPa
        temperature = np.array([[280.0, 240.0], [281.0, 241.0]]) * u.K
        h2o_vmr = np.array([[1e-3, 1e-5], [1.1e-3, 1.05e-5]]) * u.dimensionless_unscaled
        hour_unix_s = np.array([0.0, 3600.0])
        return pressure, temperature, h2o_vmr, hour_unix_s, {"endpoint": "fake"}

    monkeypatch.setattr(atm_mod, "_fetch_open_meteo", _fake_fetch)

    ds = _make_fitted_ds(n_scan=2, freqs_Hz=[22.2e9])
    attach_profile(ds, source="open-meteo")

    assert call_count["n"] == 1
    assert ds.attrs["atm_profile_source"] == "open_meteo"
    assert "atm_pressure" in ds.data_vars


# ---------------------------------------------------------------------------
# Slow tests — network-gated
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_fetch_open_meteo_live() -> None:
    """Live open-meteo call — shape and sign checks only."""
    import astropy.units as u

    from tipopac.atmosphere import _VLA_LAT, _VLA_LON, _fetch_open_meteo

    # A historical date guaranteed to be in archive.
    date_str = datetime(2024, 1, 15, tzinfo=timezone.utc).strftime("%Y-%m-%d")

    pressure, temperature, h2o_vmr, hour_unix_s, meta = _fetch_open_meteo(
        _VLA_LAT, _VLA_LON, date_str, date_str
    )

    assert pressure.size > 0
    assert temperature.shape == (hour_unix_s.size, pressure.size)
    assert h2o_vmr.shape == temperature.shape
    assert (pressure.to(u.hPa).value > 0).all()
    assert (temperature.to(u.K).value > 0).all()
    assert (h2o_vmr.value >= 0).all()
    assert "endpoint" in meta
