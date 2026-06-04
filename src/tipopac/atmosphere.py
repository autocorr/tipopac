"""Atmospheric opacity model: am + open-meteo or AFGL fallback (DESIGN.md §7).

Public entry points
-------------------
``extrapolate(ds, *, atm_profile_source, afgl_climatology)``
    Anchors a single ``troposphere_h2o_scaling`` to the fitted τ values in
    ``ds`` and fills ``tau_extrapolated``, ``am_freq_grid``, ``am_tau``.

``fetch_profile(lat, lon, obs_time_mjd_s, *, source, afgl_climatology, …)``
    Returns ``(pressure, temperature, h2o_vmr, source_label)``. Used by
    ``extrapolate`` and by :func:`tipopac.atmgrid.build_pwv_grid`.

Testable pure function
----------------------
``anchor(tau_obs, tau_err, freqs_Hz, tau_am_fn)``
    Fits a single PWV scaling scalar; takes a callable so tests can drive it
    without a real amwrap.Model or HTTP call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import astropy.units as u
import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar

__all__ = ["anchor", "extrapolate", "fetch_profile"]


class _NoPressureLevelData(RuntimeError):
    """open-meteo response was structurally valid but carried no upper-air data.

    Raised when no pressure-level variables in the response have non-NaN
    values — usually because the requested date predates the gfs_hrrr
    pressure-level archive (≈ 2021-03-23). Deterministic, so the
    :func:`fetch_profile` retry loop should *not* retry on this; it bails
    straight to AFGL.
    """

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLA site constants
# ---------------------------------------------------------------------------

_VLA_LAT: float = 34.0784  # degrees N
_VLA_LON: float = -107.6177  # degrees E

# ---------------------------------------------------------------------------
# open-meteo configuration
# ---------------------------------------------------------------------------

_OM_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# Historical archive of past *forecast* runs (not ERA5 reanalysis). The
# regular `archive-api` ERA5 endpoint does not expose pressure-level
# (upper-air) variables — units come back as "undefined" and all values are
# null. Past forecasts from gfs_hrrr include pressure-level data for dates
# >= ~2021-03-23.
_OM_ARCHIVE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_OM_MODEL = "gfs_hrrr"
_OM_FORECAST_HORIZON_DAYS = 16  # beyond this age, use archive endpoint
_OM_TIMEOUT_S = 5.0

# Pressure levels requested from open-meteo (hPa, coarse grid is fine for am).
_OM_PRESSURE_LEVELS: list[int] = [
    1000,
    975,
    950,
    925,
    900,
    875,
    850,
    825,
    800,
    775,
    750,
    700,
    650,
    600,
    550,
    500,
    450,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    70,
    50,
    30,
    20,
    10,
]

# am grid step for the output (dense) curve; anchor runs over the same model.
_AM_FREQ_STEP_HZ = 50e6  # 50 MHz


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_profile(
    lat: float,
    lon: float,
    obs_time_mjd_s: float,
    *,
    source: str = "open-meteo",
    afgl_climatology: str = "midlatitude_summer",
    surface_pressure: u.Quantity | None = None,
    retry_delays_s: tuple[float, ...] = (5.0, 15.0, 45.0),
) -> tuple[u.Quantity, u.Quantity, u.Quantity, str, dict | None]:
    """Return ``(pressure, temperature, h2o_vmr, source_label, query_meta)``.

    ``source="open-meteo"`` attempts the network call with retry-and-backoff
    governed by ``retry_delays_s`` (length = number of attempts AFTER the
    first; defaults to 3 attempts total at 0, 5, 15+45 s offsets). On final
    failure, falls back to AFGL.

    ``source="afgl"`` skips the network call entirely. ``source_label`` in the
    returned tuple resolves the actual source used and is suitable for
    storing as ``pwv_profile_source`` per-scan provenance.

    ``surface_pressure`` (pressure :class:`~astropy.units.Quantity`) clips the
    profile so the lowest level corresponds to the antenna's surface — needed
    at high-elevation sites (VLA ≈ 794 hPa) so the integrated PWV and dry-air
    opacity reflect only the column above the antennas. Applied to both
    open-meteo and AFGL profiles. If the requested base is outside the
    profile's bounds the profile is returned unclipped (with a debug log).
    """
    if source not in ("open-meteo", "afgl"):
        raise ValueError(f"unknown atmospheric profile source: {source!r}")

    if afgl_climatology == "auto":
        afgl_climatology = _pick_climatology_for_date(obs_time_mjd_s)

    if source == "open-meteo":
        last_exc: Exception | None = None
        for attempt, delay in enumerate((0.0,) + retry_delays_s):
            if delay > 0:
                _log.info(
                    "open-meteo attempt %d/%d after %.0fs backoff",
                    attempt + 1,
                    len(retry_delays_s) + 1,
                    delay,
                )
                time.sleep(delay)
            try:
                p, t, h, meta = _fetch_open_meteo(
                    lat, lon, obs_time_mjd_s, surface_pressure=surface_pressure
                )
                return p, t, h, "open_meteo", meta
            except _NoPressureLevelData as exc:
                # Deterministic — retrying won't help. Bail to AFGL now.
                _log.warning(
                    "open-meteo has no upper-air data for this date "
                    "(%s); falling back to AFGL %r",
                    exc,
                    afgl_climatology,
                )
                last_exc = exc
                break
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "open-meteo attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )
        else:
            _log.warning(
                "open-meteo exhausted %d attempts; falling back to AFGL %r (last: %s)",
                len(retry_delays_s) + 1,
                afgl_climatology,
                last_exc,
            )

    pressure, temperature, h2o_vmr = _afgl_profile(
        afgl_climatology, surface_pressure=surface_pressure
    )
    return pressure, temperature, h2o_vmr, f"afgl_{afgl_climatology}", None


def _pick_climatology_for_date(obs_time_mjd_s: float) -> str:
    """Return midlatitude_summer / midlatitude_winter based on N-hemisphere month.

    Apr–Sep → midlatitude_summer; Oct–Mar → midlatitude_winter.
    VLA (34° N) is well into the northern mid-latitudes so this is unambiguous.
    """
    obs_dt = datetime.fromtimestamp(
        (obs_time_mjd_s / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
    return (
        "midlatitude_summer" if 4 <= obs_dt.month <= 9 else "midlatitude_winter"
    )


def extrapolate(
    ds: xr.Dataset,
    *,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "midlatitude_summer",
) -> None:
    """Anchor an am model to fitted τ values and extrapolate to all spws.

    Mutates *ds* in place.  Adds tau_extrapolated, am_freq_grid, am_tau to
    data_vars and writes atm_profile_source, afgl_climatology, pwv_scaling,
    open_meteo_query to attrs.

    Requires tau_zenith, tau_err, fit_success in ds.
    """
    freqs_Hz = ds.coords["frequency"].values  # (spw,)

    # Representative observation time: median of scan start times (MJD seconds).
    obs_time_mjd_s = float(np.nanmedian(ds.coords["scan_time_start"].values))

    open_meteo_query: dict | None = None

    # Surface pressure (median of WEATHER table samples) — clips both
    # open-meteo and AFGL profiles to the column above the antennas.
    surface_pressure: u.Quantity | None = None
    if "weather_P" in ds.data_vars:
        wp = ds["weather_P"].values
        wp_finite = wp[np.isfinite(wp)]
        if wp_finite.size:
            surface_pressure = (float(np.median(wp_finite)) / 100.0) * u.hPa

    if atm_profile_source == "open-meteo":
        try:
            pressure, temperature, h2o_vmr, open_meteo_query = _fetch_open_meteo(
                _VLA_LAT, _VLA_LON, obs_time_mjd_s,
                surface_pressure=surface_pressure,
            )
        except Exception:
            _log.warning(
                "open-meteo fetch failed; falling back to AFGL %r",
                afgl_climatology,
                exc_info=True,
            )
            atm_profile_source = "afgl"

    if atm_profile_source == "afgl":
        pressure, temperature, h2o_vmr = _afgl_profile(
            afgl_climatology, surface_pressure=surface_pressure
        )

    freq_min_Hz = float(freqs_Hz.min()) * 0.95
    freq_max_Hz = float(freqs_Hz.max()) * 1.05
    model = _build_am_model(pressure, temperature, h2o_vmr, freq_min_Hz, freq_max_Hz)

    # Collect valid (scan, antenna, spw) triples for the anchor cost function.
    tau_fit = ds["tau_zenith"].values  # (scan, antenna, spw)
    tau_err_vals = ds["tau_err"].values  # (scan, antenna, spw)
    success = ds["fit_success"].values  # (scan, antenna, spw)

    n_scan, n_ant, n_spw = tau_fit.shape
    freq_3d = np.broadcast_to(freqs_Hz[None, None, :], (n_scan, n_ant, n_spw))

    mask = (
        success & np.isfinite(tau_fit) & np.isfinite(tau_err_vals) & (tau_err_vals > 0)
    )
    tau_obs_flat = tau_fit[mask]
    tau_err_flat = tau_err_vals[mask]
    freqs_flat = freq_3d[mask]

    pwv_scaling: float | None = None

    if mask.sum() == 0:
        _log.warning("No successful fits available for am anchor; skipping.")
    else:

        def _tau_am(scaling: float) -> np.ndarray:
            return _tau_at_freqs(model, freqs_flat, scaling)

        pwv_scaling = anchor(tau_obs_flat, tau_err_flat, freqs_flat, _tau_am)
        _log.info(
            "am anchor: pwv_scaling=%.4f (source=%s, n_pts=%d)",
            pwv_scaling,
            atm_profile_source,
            int(mask.sum()),
        )

    # Final am run with anchored scaling (or scaling=1 if anchor failed).
    model.troposphere_h2o_scaling = pwv_scaling if pwv_scaling is not None else 1.0
    dense_df = model.run()
    am_freqs_Hz = dense_df["frequency"].values * 1e9  # GHz → Hz
    am_opacity = dense_df["opacity"].values

    # tau_extrapolated: am opacity interpolated to each spw centre frequency.
    tau_ext = np.interp(freqs_Hz, am_freqs_Hz, am_opacity).astype(np.float32)
    tau_extrapolated = np.tile(tau_ext, (n_scan, 1))  # (scan, spw)

    ds["tau_extrapolated"] = (("scan", "spw"), tau_extrapolated)
    ds["am_freq_grid"] = (("frequency_dense",), am_freqs_Hz)
    ds["am_tau"] = (("frequency_dense",), am_opacity)

    ds.attrs.update(
        atm_profile_source=atm_profile_source,
        afgl_climatology=afgl_climatology,
        pwv_scaling=pwv_scaling,
        open_meteo_query=open_meteo_query,
    )


def anchor(
    tau_obs: np.ndarray,
    tau_err: np.ndarray,
    freqs_Hz: np.ndarray,
    tau_am_fn: Callable[[float], np.ndarray],
) -> float:
    """Fit a single PWV scaling scalar against observed zenith opacities.

    Args:
        tau_obs:   Flat array of observed τ values (nepers).
        tau_err:   Matching τ uncertainties (nepers, positive).
        freqs_Hz:  Matching frequencies (Hz).
        tau_am_fn: Callable (scaling) → τ_am at freqs_Hz; must accept a float
                   and return an ndarray of the same shape as tau_obs.

    Returns:
        Fitted pwv_scaling scalar (dimensionless).
    """

    def _cost(scaling: float) -> float:
        residuals = (tau_obs - tau_am_fn(scaling)) / tau_err
        return float(np.dot(residuals, residuals))

    result = minimize_scalar(_cost, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_open_meteo(
    lat: float,
    lon: float,
    obs_time_mjd_s: float,
    *,
    surface_pressure: u.Quantity | None = None,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, dict]:
    """Fetch a vertical atmospheric profile from open-meteo.

    Returns (pressure, temperature, h2o_vmr, query_meta).
    Raises on HTTP error or timeout (caller falls back to AFGL).

    If ``surface_pressure`` is given, the returned profile is clipped (and
    bottom values linearly interpolated) so the lowest level corresponds to
    the site's actual surface — see :func:`amwrap.interp_by_pressure`.
    """
    import openmeteo_requests

    # MJD seconds → UTC datetime (MJD epoch = 1858-11-17; Unix epoch MJD = 40587 days)
    obs_dt = datetime.fromtimestamp(
        (obs_time_mjd_s / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
    date_str = obs_dt.strftime("%Y-%m-%d")

    now = datetime.now(tz=timezone.utc)
    age_days = (now - obs_dt).total_seconds() / 86400.0
    url = _OM_ARCHIVE_URL if age_days > _OM_FORECAST_HORIZON_DAYS else _OM_FORECAST_URL

    # The SDK's `var.PressureLevel()` returns 0 for every variable in the
    # response (at least with the current openmeteo_sdk wire format), so we
    # cannot key by it. Instead pair variables by index — they come back in
    # the same order we requested, alternating (T, RH) per level.
    hourly_pairs: list[tuple[int, str]] = []
    for p in _OM_PRESSURE_LEVELS:
        hourly_pairs.append((p, "temperature"))
        hourly_pairs.append((p, "relative_humidity"))
    hourly_vars = [f"{v}_{p}hPa" for p, v in hourly_pairs]
    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "start_date": date_str,
        "end_date": date_str,
        "models": _OM_MODEL,
        "timezone": "UTC",
    }
    query_meta = {
        "lat": lat,
        "lon": lon,
        "date": date_str,
        "endpoint": url,
        "model": _OM_MODEL,
    }

    client = openmeteo_requests.Client()
    responses = client.weather_api(url, params=params, timeout=_OM_TIMEOUT_S)
    hourly: Any = responses[0].Hourly()
    if hourly is None:
        raise RuntimeError("open-meteo response contained no hourly data")

    n_vars = hourly.VariablesLength()
    if n_vars != len(hourly_pairs):
        raise RuntimeError(
            f"open-meteo returned {n_vars} variables, expected {len(hourly_pairs)}"
        )

    # Find the hourly index closest to the observation time.
    times = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
    t_idx = int(np.argmin(np.abs(times - obs_dt.timestamp())))

    temp_by_level: dict[int, float] = {}
    rh_by_level: dict[int, float] = {}
    for i, (p_level, vname) in enumerate(hourly_pairs):
        var: Any = hourly.Variables(i)
        if var is None:
            continue
        vals = var.ValuesAsNumpy()
        if t_idx >= vals.size:
            continue
        val = float(vals[t_idx])
        if not np.isfinite(val):
            continue
        if vname == "temperature":
            temp_by_level[p_level] = val
        elif vname == "relative_humidity":
            rh_by_level[p_level] = val

    # Build ordered arrays (surface → top of atmosphere).
    # Open-meteo omits pressure levels below the surface (e.g. 1000 hPa at
    # high-altitude sites like the VLA at 2115 m); skip missing levels silently.
    valid_levels = [p for p in _OM_PRESSURE_LEVELS if p in temp_by_level and p in rh_by_level]
    if not valid_levels:
        raise _NoPressureLevelData(
            f"no upper-air data from {_OM_MODEL} for {date_str} (endpoint={url})"
        )
    pressure_hPa = np.array(valid_levels, dtype=np.float64)
    temperature_C = np.array(
        [temp_by_level[p] for p in valid_levels], dtype=np.float64
    )
    rh_frac = np.array(
        [rh_by_level[p] / 100.0 for p in valid_levels], dtype=np.float64
    )

    pressure_q = pressure_hPa * u.hPa
    temperature_q = (temperature_C + 273.15) * u.K
    rh_q = rh_frac * u.dimensionless_unscaled

    import amwrap as _amwrap

    h2o_vmr = _amwrap.mixing_ratio_from_relative_humidity(
        pressure_q, temperature_q, rh_q
    )

    if surface_pressure is not None:
        try:
            pressure_clipped = _amwrap.interp_by_pressure(
                pressure_q, pressure_q, surface_pressure
            )
            temperature_clipped = _amwrap.interp_by_pressure(
                temperature_q, pressure_q, surface_pressure
            )
            h2o_vmr_clipped = _amwrap.interp_by_pressure(
                h2o_vmr, pressure_q, surface_pressure
            )
            pressure_q = pressure_clipped
            temperature_q = temperature_clipped
            h2o_vmr = h2o_vmr_clipped
            query_meta["surface_pressure_hPa"] = float(
                surface_pressure.to(u.hPa).value
            )
        except ValueError as exc:
            # surface_pressure outside the open-meteo profile bounds — return
            # the un-clipped profile (the open-meteo response is already
            # truncated to its station elevation).
            _log.debug(
                "open-meteo profile not clipped: surface_pressure %.1f hPa "
                "outside response bounds [%.1f, %.1f] hPa (%s)",
                surface_pressure.to(u.hPa).value,
                pressure_q.to(u.hPa).value.min(),
                pressure_q.to(u.hPa).value.max(),
                exc,
            )

    return pressure_q, temperature_q, h2o_vmr, query_meta


def _afgl_profile(
    name: str,
    *,
    surface_pressure: u.Quantity | None = None,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Return (pressure, temperature, h2o_vmr) from an AFGL climatology.

    When ``surface_pressure`` is given, the climatology is clipped via
    :class:`amwrap.Climatology`'s ``pressure_base`` so the lowest level
    corresponds to the site's surface. AFGL profiles start at sea-level
    (~1018 hPa); without this clip a high-elevation site like the VLA
    (~794 hPa) gets ~3 mm of phantom sub-surface water vapour and a
    correspondingly inflated dry-air column.
    """
    import amwrap as _amwrap

    clim = _amwrap.Climatology(name, pressure_base=surface_pressure)
    return clim.pressure, clim.temperature, clim.mixing_ratio["h2o"]


def _build_am_model(
    pressure: u.Quantity,
    temperature: u.Quantity,
    h2o_vmr: u.Quantity,
    freq_min_Hz: float,
    freq_max_Hz: float,
) -> Any:
    """Construct an amwrap.Model spanning [freq_min_Hz, freq_max_Hz]."""
    import amwrap as _amwrap

    return _amwrap.Model(
        pressure=pressure,
        temperature=temperature,
        mixing_ratio={"h2o": h2o_vmr},
        freq_min=freq_min_Hz * u.Hz,
        freq_max=freq_max_Hz * u.Hz,
        freq_step=_AM_FREQ_STEP_HZ * u.Hz,
    )


def _tau_at_freqs(model: Any, freqs_Hz: np.ndarray, scaling: float) -> np.ndarray:
    """Run am model with *scaling* and interpolate τ to *freqs_Hz*."""
    model.troposphere_h2o_scaling = scaling
    df = model.run()
    am_freqs = df["frequency"].values * 1e9  # GHz → Hz
    am_opacity = df["opacity"].values
    return np.interp(freqs_Hz, am_freqs, am_opacity)
