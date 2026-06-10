"""Atmospheric opacity model: am + open-meteo or AFGL fallback (DESIGN.md §7).

Public entry point
------------------
``attach_profile(ds, *, source, afgl_climatology, …)``
    The single network-touching stage. Runs open-meteo once (full hourly
    grid for the observation date range) or builds an AFGL profile, picks
    the closest hour per scan, clips to the median surface pressure, and
    writes ``atm_pressure``, ``atm_temperature``, ``atm_h2o_vmr`` to *ds*.
    Provenance lands in ``ds.attrs`` (``atm_profile_source``,
    ``open_meteo_query``, ``surface_pressure_hPa``).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import astropy.units as u
import numpy as np
import xarray as xr

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

    Adds data vars ``atm_pressure(atm_level)``, ``atm_temperature(scan,
    atm_level)``, ``atm_h2o_vmr(scan, atm_level)`` and attrs
    ``atm_profile_source``, ``open_meteo_query``, ``surface_pressure_hPa``.

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

    The surface-pressure clip uses the median of per-scan WEATHER-table
    samples. Per-scan surface variation is <2 hPa at the VLA, well below
    am modeling precision; collapsing to median keeps the ``atm_level``
    dim constant across scans.
    """
    if source not in ("open-meteo", "afgl"):
        raise ValueError(f"unknown atmospheric profile source: {source!r}")

    scan_starts = ds.coords["scan_time_start"].values  # (scan,), MJD seconds
    scan_ends = ds.coords["scan_time_end"].values

    median_obs_time_mjd_s = float(np.nanmedian(scan_starts))
    if afgl_climatology == "auto":
        afgl_climatology = _pick_climatology_for_date(median_obs_time_mjd_s)

    # Per-scan median surface pressure → median across scans for the clip.
    surface_pressure, surface_pressures_hPa = _compute_surface_pressure(ds)

    open_meteo_query: dict | None = None
    used_source: str | None = None
    pressure_q: u.Quantity | None = None
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
                        pressure_q,
                        temperature_per_scan,
                        h2o_vmr_per_scan,
                    ) = _pick_hourly_per_scan_and_clip(
                        p_grid,
                        t_grid,
                        h_grid,
                        hour_unix_s,
                        scan_starts,
                        surface_pressure,
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
        # source="afgl" path or open-meteo failure
        afgl_p, afgl_t, afgl_h = _afgl_profile(
            afgl_climatology, surface_pressure=surface_pressure
        )
        pressure_q = afgl_p
        n_scan = ds.sizes["scan"]
        # Broadcast the (constant) AFGL profile to every scan.
        temperature_per_scan = (
            np.broadcast_to(afgl_t.to(u.K).value, (n_scan, afgl_t.size)).copy() * u.K
        )
        h2o_vmr_per_scan = (
            np.broadcast_to(
                np.asarray(afgl_h, dtype=np.float64), (n_scan, afgl_h.size)
            ).copy()
            * u.dimensionless_unscaled
        )
        used_source = f"afgl_{afgl_climatology}"

    assert pressure_q is not None
    assert temperature_per_scan is not None
    assert h2o_vmr_per_scan is not None

    ds["atm_pressure"] = (
        ("atm_level",),
        pressure_q.to(u.Pa).value.astype(np.float64),
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
    if surface_pressures_hPa:
        ds.attrs["surface_pressure_hPa"] = surface_pressures_hPa


def _compute_surface_pressure(
    ds: xr.Dataset,
) -> tuple[u.Quantity | None, dict[int, float]]:
    """Return ``(median_surface_pressure_quantity, per_scan_hPa_dict)``.

    The Quantity is the median across scans of each scan's median weather_P
    sample, used for the single profile clip. The dict is provenance only.
    """
    if "weather_P" not in ds.data_vars:
        return None, {}
    weather_P_Pa = ds["weather_P"].values  # (scan, time), Pa
    scan_ids = ds.coords["scan"].values
    surface_pressures_hPa: dict[int, float] = {}
    per_scan_medians: list[float] = []
    for i, scan_id in enumerate(scan_ids):
        samples = weather_P_Pa[i][np.isfinite(weather_P_Pa[i])]
        if samples.size:
            p_surf_hPa = float(np.median(samples)) / 100.0
            surface_pressures_hPa[int(scan_id)] = p_surf_hPa
            per_scan_medians.append(p_surf_hPa)
    if not per_scan_medians:
        return None, {}
    p_med_hPa = float(np.median(per_scan_medians))
    return p_med_hPa * u.hPa, surface_pressures_hPa


def _utc_date_range(
    scan_starts_mjd_s: np.ndarray, scan_ends_mjd_s: np.ndarray
) -> tuple[str, str]:
    """UTC date span covering all scans, as ``("YYYY-MM-DD", "YYYY-MM-DD")``."""
    t_min = float(np.nanmin(scan_starts_mjd_s))
    t_max = float(np.nanmax(scan_ends_mjd_s))
    dt_min = datetime.fromtimestamp(
        (t_min / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
    dt_max = datetime.fromtimestamp(
        (t_max / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
    return dt_min.strftime("%Y-%m-%d"), dt_max.strftime("%Y-%m-%d")


def _pick_hourly_per_scan_and_clip(
    p_grid: u.Quantity,
    t_grid: u.Quantity,
    h_grid: u.Quantity,
    hour_unix_s: np.ndarray,
    scan_starts_mjd_s: np.ndarray,
    surface_pressure: u.Quantity | None,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Pick closest hourly slice per scan, then apply a single surface clip.

    ``p_grid`` is (n_level,); ``t_grid``, ``h_grid`` are (n_hour, n_level).
    Returns ``(pressure_clipped (n_level',), temperature (n_scan, n_level'),
    h2o_vmr (n_scan, n_level'))``.
    """
    import amwrap as _amwrap

    # MJD seconds → Unix seconds.
    scan_unix_s = (scan_starts_mjd_s / 86400.0 - 40587.0) * 86400.0
    # For each scan, find closest hour. (n_scan, n_hour) abs-diff matrix.
    diff = np.abs(scan_unix_s[:, None] - hour_unix_s[None, :])
    hour_idx = np.argmin(diff, axis=1)  # (n_scan,)

    n_scan = scan_unix_s.size
    t_K = t_grid.to(u.K).value  # (n_hour, n_level)
    h_vmr = np.asarray(h_grid.value)  # (n_hour, n_level)

    temperature_per_scan_K = t_K[hour_idx, :]  # (n_scan, n_level)
    h2o_per_scan = h_vmr[hour_idx, :]

    pressure_q: u.Quantity = p_grid
    if surface_pressure is not None:
        try:
            pressure_clipped = _amwrap.interp_by_pressure(
                p_grid, p_grid, surface_pressure
            )
            n_keep = pressure_clipped.size

            # Apply same clip to each scan's T and VMR.
            t_clipped = np.empty((n_scan, n_keep), dtype=np.float64)
            h_clipped = np.empty((n_scan, n_keep), dtype=np.float64)
            for i in range(n_scan):
                t_q = temperature_per_scan_K[i] * u.K
                h_q = h2o_per_scan[i] * u.dimensionless_unscaled
                t_clipped[i] = (
                    _amwrap.interp_by_pressure(t_q, p_grid, surface_pressure)
                    .to(u.K)
                    .value
                )
                h_clipped[i] = np.asarray(
                    _amwrap.interp_by_pressure(h_q, p_grid, surface_pressure).value
                )
            pressure_q = pressure_clipped
            temperature_per_scan_K = t_clipped
            h2o_per_scan = h_clipped
        except ValueError as exc:
            _log.debug(
                "open-meteo profile not clipped: surface_pressure %.1f hPa "
                "outside response bounds [%.1f, %.1f] hPa (%s)",
                surface_pressure.to(u.hPa).value,
                p_grid.to(u.hPa).value.min(),
                p_grid.to(u.hPa).value.max(),
                exc,
            )

    return (
        pressure_q,
        temperature_per_scan_K * u.K,
        h2o_per_scan * u.dimensionless_unscaled,
    )


def _pick_climatology_for_date(obs_time_mjd_s: float) -> str:
    """Return midlatitude_summer / midlatitude_winter based on N-hemisphere month.

    Apr–Sep → midlatitude_summer; Oct–Mar → midlatitude_winter.
    VLA (34° N) is well into the northern mid-latitudes so this is unambiguous.
    """
    obs_dt = datetime.fromtimestamp(
        (obs_time_mjd_s / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
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
