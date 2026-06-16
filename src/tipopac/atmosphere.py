"""Atmospheric opacity model: am + open-meteo or AFGL fallback (DESIGN.md §7).

Public entry point
------------------
``attach_profile(ds, *, source, afgl_climatology, …)``
    The single network-touching stage. Runs open-meteo once (full hourly
    grid for the observation date range) or builds an AFGL profile, picks
    the closest hour per scan, clips at each scan's own surface pressure,
    and writes ``atm_pressure``, ``atm_temperature``, ``atm_h2o_vmr``, and
    ``surface_pressure_hPa(scan,)`` to *ds*. Provenance lands in
    ``ds.attrs`` (``atm_profile_source``, ``open_meteo_query``).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import astropy.units as u
import numpy as np
import xarray as xr

from tipopac.timeutils import mjd_s_to_unix_s as _mjd_s_to_unix_s

__all__ = ["attach_profile"]


class _NoPressureLevelData(RuntimeError):
    """open-meteo response was structurally valid but carried no upper-air data.

    Raised when no pressure-level variables in the response have non-NaN
    values — usually because the requested date predates the gfs_hrrr
    pressure-level archive (≈ 2021-03-23). Deterministic, so the
    :func:`attach_profile` retry loop should *not* retry on this; it bails
    straight to AFGL.
    """


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VLA site constants
# ---------------------------------------------------------------------------

_VLA_LAT: float = 34.0784  # degrees N
_VLA_LON: float = -107.6177  # degrees E
# Fallback when no scan has a finite weather_P sample (~794 hPa for the VLA
# site at ~2115 m altitude).
_VLA_DEFAULT_SURFACE_P_hPa: float = 794.0

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
# Earliest date with pressure-level data in the gfs_hrrr historical-forecast
# archive (empirical lower bound, no authoritative reference). Observations
# starting before this are routed to AFGL without an HTTP round-trip.
_OM_HRRR_ARCHIVE_START = "2021-03-23"

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

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attach_profile(
    ds: xr.Dataset,
    *,
    source: str = "open-meteo",
    afgl_climatology: str = "auto",
    retry_delays_s: tuple[float, ...] = (5.0, 15.0, 45.0),
) -> None:
    """Fetch the atmospheric profile once and attach it to *ds*.

    Adds data vars ``atm_pressure(scan, atm_level)``, ``atm_temperature(
    scan, atm_level)``, ``atm_h2o_vmr(scan, atm_level)``,
    ``surface_pressure_hPa(scan,)`` (omitted when no scan has finite
    weather_P) and attrs ``atm_profile_source``, ``open_meteo_query``.

    Behaviour:

    * ``source="open-meteo"`` issues **one** HTTP call covering the full
      observation date range. The closest hourly slice is picked per
      scan. Retries with backoff governed by ``retry_delays_s`` (default
      4 attempts at 0, 5, 15+45 s offsets) on transient failures; on
      deterministic failure (``_NoPressureLevelData``) bails to AFGL
      immediately. If the observation starts before
      ``_OM_HRRR_ARCHIVE_START`` (the gfs_hrrr pressure-level archive
      lower bound), the network call is skipped and AFGL is used.
    * ``source="afgl"`` skips the network call entirely.
    * ``afgl_climatology="auto"`` resolves to summer/winter from the
      observation's median month.

    The surface clip is applied per scan: index 0 of ``atm_level`` is
    each scan's own surface, increasing index moves up in altitude.
    Scans whose clip lands one level shorter than the longest get
    trailing NaN padding. Scans without a finite ``surface_pressure_hPa``
    fall back to the cross-scan median; if no scan has WEATHER data, to
    the VLA site default (~794 hPa).
    """
    if source not in ("open-meteo", "afgl"):
        raise ValueError(f"unknown atmospheric profile source: {source!r}")

    scan_starts = ds.coords["scan_time_start"].values  # (scan,), MJD seconds
    scan_ends = ds.coords["scan_time_end"].values

    median_obs_time_mjd_s = float(np.nanmedian(scan_starts))
    if afgl_climatology == "auto":
        afgl_climatology = _pick_climatology_for_date(median_obs_time_mjd_s)

    surface_pressures_hPa = _compute_surface_pressure(ds)
    per_scan_surface_hPa = _resolve_per_scan_surface_hPa(surface_pressures_hPa)

    open_meteo_query: dict | None = None
    used_source: str | None = None
    pressure_per_scan: u.Quantity | None = None
    temperature_per_scan: u.Quantity | None = None
    h2o_vmr_per_scan: u.Quantity | None = None

    if source == "open-meteo":
        date_start, date_end = _utc_date_range(scan_starts, scan_ends)
        if date_start < _OM_HRRR_ARCHIVE_START:
            _log.warning(
                "observation start %s predates open-meteo gfs_hrrr "
                "pressure-level archive (%s); using AFGL %r without HTTP request",
                date_start,
                _OM_HRRR_ARCHIVE_START,
                afgl_climatology,
            )
        else:
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
                    p_grid, t_grid, h_grid, hour_unix_s, meta = _fetch_open_meteo(
                        _VLA_LAT, _VLA_LON, date_start, date_end
                    )
                    (
                        pressure_per_scan,
                        temperature_per_scan,
                        h2o_vmr_per_scan,
                    ) = _pick_hourly_per_scan_and_clip(
                        p_grid,
                        t_grid,
                        h_grid,
                        hour_unix_s,
                        scan_starts,
                        per_scan_surface_hPa,
                    )
                    open_meteo_query = meta
                    used_source = "open_meteo"
                    break
                except _NoPressureLevelData as exc:
                    _log.warning(
                        "open-meteo has no upper-air data for this date range "
                        "(%s); falling back to AFGL %r",
                        exc,
                        afgl_climatology,
                    )
                    last_exc = exc
                    break
                except Exception as exc:
                    last_exc = exc
                    _log.warning("open-meteo attempt %d failed: %s", attempt + 1, exc)
            else:
                _log.warning(
                    "open-meteo exhausted %d attempts; falling back to AFGL %r (last: %s)",
                    len(retry_delays_s) + 1,
                    afgl_climatology,
                    last_exc,
                )

    if used_source is None:
        (
            pressure_per_scan,
            temperature_per_scan,
            h2o_vmr_per_scan,
        ) = _afgl_profile_per_scan(afgl_climatology, per_scan_surface_hPa)
        used_source = f"afgl_{afgl_climatology}"

    assert pressure_per_scan is not None
    assert temperature_per_scan is not None
    assert h2o_vmr_per_scan is not None

    ds["atm_pressure"] = (
        ("scan", "atm_level"),
        pressure_per_scan.to(u.Pa).value.astype(np.float64),
    )
    ds["atm_temperature"] = (
        ("scan", "atm_level"),
        temperature_per_scan.to(u.K).value.astype(np.float32),
    )
    ds["atm_h2o_vmr"] = (
        ("scan", "atm_level"),
        np.asarray(h2o_vmr_per_scan.value, dtype=np.float32),
    )
    ds.attrs["atm_profile_source"] = used_source
    if open_meteo_query is not None:
        ds.attrs["open_meteo_query"] = open_meteo_query
    if np.any(np.isfinite(surface_pressures_hPa)):
        ds["surface_pressure_hPa"] = (("scan",), surface_pressures_hPa)


def _compute_surface_pressure(ds: xr.Dataset) -> np.ndarray:
    """Return the per-scan median weather_P sample in hPa (NaN where missing)."""
    n_scan = int(ds.sizes["scan"])
    per_scan_hPa = np.full(n_scan, np.nan, dtype=np.float64)
    if "weather_P" not in ds.data_vars:
        return per_scan_hPa
    weather_P_Pa = ds["weather_P"].values  # (scan, time), Pa
    for i in range(n_scan):
        samples = weather_P_Pa[i][np.isfinite(weather_P_Pa[i])]
        if samples.size:
            per_scan_hPa[i] = float(np.median(samples)) / 100.0
    return per_scan_hPa


def _utc_date_range(
    scan_starts_mjd_s: np.ndarray, scan_ends_mjd_s: np.ndarray
) -> tuple[str, str]:
    """UTC date span covering all scans, as ``("YYYY-MM-DD", "YYYY-MM-DD")``."""
    t_min = float(np.nanmin(scan_starts_mjd_s))
    t_max = float(np.nanmax(scan_ends_mjd_s))
    dt_min = datetime.fromtimestamp(_mjd_s_to_unix_s(t_min), tz=timezone.utc)
    dt_max = datetime.fromtimestamp(_mjd_s_to_unix_s(t_max), tz=timezone.utc)
    return dt_min.strftime("%Y-%m-%d"), dt_max.strftime("%Y-%m-%d")


def _pick_hourly_per_scan_and_clip(
    p_grid: u.Quantity,
    t_grid: u.Quantity,
    h_grid: u.Quantity,
    hour_unix_s: np.ndarray,
    scan_starts_mjd_s: np.ndarray,
    surface_pressures_hPa: np.ndarray,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Pick closest hourly slice per scan, then apply each scan's own surface clip.

    ``p_grid`` is (n_level,); ``t_grid``, ``h_grid`` are (n_hour, n_level).
    ``surface_pressures_hPa`` is (n_scan,) of finite floats (NaN already
    resolved by the caller). Returns three per-scan arrays NaN-padded at the
    trailing edge to a common ``n_level'`` (= max clipped length); index 0 is
    each scan's own surface, increasing index moves up in altitude.
    """
    scan_unix_s = _mjd_s_to_unix_s(scan_starts_mjd_s)
    diff = np.abs(scan_unix_s[:, None] - hour_unix_s[None, :])
    hour_idx = np.argmin(diff, axis=1)  # (n_scan,)

    n_scan = scan_unix_s.size
    t_K = t_grid.to(u.K).value  # (n_hour, n_level)
    h_vmr = np.asarray(h_grid.value)  # (n_hour, n_level)

    p_rows: list[np.ndarray] = []
    t_rows: list[np.ndarray] = []
    h_rows: list[np.ndarray] = []
    for i in range(n_scan):
        p_q_i = float(surface_pressures_hPa[i]) * u.hPa
        t_full = t_K[hour_idx[i]] * u.K
        h_full = h_vmr[hour_idx[i]] * u.dimensionless_unscaled
        p_rows.append(
            _clip_or_fallback(p_grid, p_grid, p_q_i, "pressure").to(u.Pa).value
        )
        t_rows.append(
            _clip_or_fallback(t_full, p_grid, p_q_i, "temperature").to(u.K).value
        )
        h_rows.append(
            np.asarray(_clip_or_fallback(h_full, p_grid, p_q_i, "h2o_vmr").value)
        )

    return (
        _pad_rows_to_max(p_rows) * u.Pa,
        _pad_rows_to_max(t_rows) * u.K,
        _pad_rows_to_max(h_rows) * u.dimensionless_unscaled,
    )


def _clip_or_fallback(
    values: u.Quantity,
    pressure: u.Quantity,
    pressure_base: u.Quantity,
    label: str,
) -> u.Quantity:
    """``amwrap.interp_by_pressure`` with graceful out-of-range fallback.

    Returns the unclipped values when ``pressure_base`` is outside the data
    range — matches the prior single-clip behaviour at atmosphere.py:350-358.
    """
    import amwrap as _amwrap

    try:
        return _amwrap.interp_by_pressure(values, pressure, pressure_base)
    except ValueError as exc:
        _log.debug(
            "%s profile not clipped: surface_pressure %.1f hPa outside "
            "bounds [%.1f, %.1f] hPa (%s)",
            label,
            pressure_base.to(u.hPa).value,
            pressure.to(u.hPa).value.min(),
            pressure.to(u.hPa).value.max(),
            exc,
        )
        return values


def _pad_rows_to_max(rows: list[np.ndarray]) -> np.ndarray:
    """Stack 1-D rows into ``(n_row, max_len)`` with trailing NaN padding."""
    L = max(r.size for r in rows)
    out = np.full((len(rows), L), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        out[i, : r.size] = r
    return out


def _resolve_per_scan_surface_hPa(
    surface_pressures_hPa: np.ndarray,
) -> np.ndarray:
    """Return finite per-scan surface pressures in hPa.

    NaN entries fall back to the cross-scan median; if no scan has a finite
    sample, all fall back to the VLA site default.
    """
    out = np.asarray(surface_pressures_hPa, dtype=np.float64).copy()
    finite = out[np.isfinite(out)]
    if finite.size == 0:
        out[:] = _VLA_DEFAULT_SURFACE_P_hPa
    else:
        out[~np.isfinite(out)] = float(np.median(finite))
    return out


def _pick_climatology_for_date(obs_time_mjd_s: float) -> str:
    """Return midlatitude_summer / midlatitude_winter based on N-hemisphere month.

    Apr–Sep → midlatitude_summer; Oct–Mar → midlatitude_winter.
    VLA (34° N) is well into the northern mid-latitudes so this is unambiguous.
    """
    obs_dt = datetime.fromtimestamp(_mjd_s_to_unix_s(obs_time_mjd_s), tz=timezone.utc)
    return "midlatitude_summer" if 4 <= obs_dt.month <= 9 else "midlatitude_winter"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _fetch_open_meteo(
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
) -> tuple[u.Quantity, u.Quantity, u.Quantity, np.ndarray, dict]:
    """Fetch the full hourly vertical-profile grid from open-meteo.

    Returns
    -------
    pressure : Quantity, shape (n_level,)
        Pressure levels (hPa) actually returned for this location — open-
        meteo omits levels below the station surface (e.g. 1000 hPa at
        the VLA's ~2115 m altitude).
    temperature : Quantity, shape (n_hour, n_level)
        Hourly temperature profiles (K).
    h2o_vmr : Quantity, shape (n_hour, n_level)
        Hourly H₂O volume mixing ratio (dimensionless).
    hour_unix_s : ndarray, shape (n_hour,)
        UTC Unix-second timestamps for each hourly profile; used by the
        caller to pick the closest hour per scan.
    query_meta : dict
        ``(lat, lon, date_start, date_end, endpoint, model)`` — provenance
        recorded on the dataset.

    Raises on HTTP error / timeout (caller falls back to AFGL).
    """
    import openmeteo_requests

    end_dt = datetime.strptime(date_end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    age_days = (now - end_dt).total_seconds() / 86400.0
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
        "start_date": date_start,
        "end_date": date_end,
        "models": _OM_MODEL,
        "timezone": "UTC",
    }
    query_meta = {
        "lat": lat,
        "lon": lon,
        "date_start": date_start,
        "date_end": date_end,
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

    hour_unix_s = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval()).astype(
        np.float64
    )
    n_hour = hour_unix_s.size

    # Collect per-level arrays of shape (n_hour,). Levels are filtered to
    # those that returned all-finite data across the full hourly window.
    temp_by_level: dict[int, np.ndarray] = {}
    rh_by_level: dict[int, np.ndarray] = {}
    for i, (p_level, vname) in enumerate(hourly_pairs):
        var: Any = hourly.Variables(i)
        if var is None:
            continue
        vals = var.ValuesAsNumpy()
        if vals.size < n_hour:
            continue
        vals = vals[:n_hour]
        if not np.all(np.isfinite(vals)):
            continue
        if vname == "temperature":
            temp_by_level[p_level] = vals.astype(np.float64)
        elif vname == "relative_humidity":
            rh_by_level[p_level] = vals.astype(np.float64)

    valid_levels = [
        p for p in _OM_PRESSURE_LEVELS if p in temp_by_level and p in rh_by_level
    ]
    if not valid_levels:
        raise _NoPressureLevelData(
            f"no upper-air data from {_OM_MODEL} for {date_start}..{date_end} "
            f"(endpoint={url})"
        )

    n_level = len(valid_levels)
    pressure_hPa = np.array(valid_levels, dtype=np.float64)
    temperature_C = np.empty((n_hour, n_level), dtype=np.float64)
    rh_frac = np.empty((n_hour, n_level), dtype=np.float64)
    for j, p in enumerate(valid_levels):
        temperature_C[:, j] = temp_by_level[p]
        rh_frac[:, j] = rh_by_level[p] / 100.0

    pressure_q = pressure_hPa * u.hPa
    temperature_q = (temperature_C + 273.15) * u.K
    rh_q = rh_frac * u.dimensionless_unscaled

    import amwrap as _amwrap

    # mixing_ratio_from_relative_humidity broadcasts pressure (n_level,)
    # against T/RH (n_hour, n_level) — astropy handles this elementwise.
    h2o_vmr = _amwrap.mixing_ratio_from_relative_humidity(
        pressure_q[np.newaxis, :], temperature_q, rh_q
    )

    return pressure_q, temperature_q, h2o_vmr, hour_unix_s, query_meta


def _afgl_profile_per_scan(
    name: str,
    surface_pressures_hPa: np.ndarray,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Return per-scan ``(pressure, temperature, h2o_vmr)`` from an AFGL climatology.

    Each scan's profile is clipped at its own surface pressure (finite floats
    only — caller resolves NaN beforehand). AFGL profiles run from sea-level
    (~1018 hPa) up, so the clip drops sub-surface levels and interpolates a
    new lowest level at each scan's surface — without it a high-elevation site
    like the VLA (~794 hPa) gets several mm of phantom sub-surface H₂O.

    Returns three ``(n_scan, n_level)`` Quantities, NaN-padded at the trailing
    end when per-scan clip lengths differ.
    """
    import amwrap as _amwrap

    clim = _amwrap.Climatology(name)
    p_full, t_full, h_full = clim.pressure, clim.temperature, clim.mixing_ratio["h2o"]

    p_rows: list[np.ndarray] = []
    t_rows: list[np.ndarray] = []
    h_rows: list[np.ndarray] = []
    for p_hPa in surface_pressures_hPa:
        p_base = float(p_hPa) * u.hPa
        p_rows.append(
            _clip_or_fallback(p_full, p_full, p_base, "pressure").to(u.Pa).value
        )
        t_rows.append(
            _clip_or_fallback(t_full, p_full, p_base, "temperature").to(u.K).value
        )
        h_rows.append(
            np.asarray(_clip_or_fallback(h_full, p_full, p_base, "h2o_vmr").value)
        )

    return (
        _pad_rows_to_max(p_rows) * u.Pa,
        _pad_rows_to_max(t_rows) * u.K,
        _pad_rows_to_max(h_rows) * u.dimensionless_unscaled,
    )
