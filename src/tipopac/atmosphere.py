"""Atmospheric opacity model: am + open-meteo or AFGL fallback (DESIGN.md §7).

Public entry points
-------------------
``extrapolate(ds, *, atm_profile_source, afgl_climatology)``
    Stage-1 / legacy path. Anchors a single ``troposphere_h2o_scaling`` to
    the fitted τ values in ``ds`` and fills ``tau_extrapolated``,
    ``am_freq_grid``, ``am_tau``. Will be reworked in Stage 2 (task #15) to
    consume :class:`tipopac.atmgrid.PwvGrid` directly.

``fetch_profile(lat, lon, obs_time_mjd_s, *, source, afgl_climatology, …)``
    Returns ``(pressure, temperature, h2o_vmr, source_label)``. Used by both
    ``extrapolate`` and Stage-2 :func:`tipopac.atmgrid.build_pwv_grid`.

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
_OM_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
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
    """
    if source not in ("open-meteo", "afgl"):
        raise ValueError(f"unknown atmospheric profile source: {source!r}")

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
                p, t, h, meta = _fetch_open_meteo(lat, lon, obs_time_mjd_s)
                return p, t, h, "open_meteo", meta
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "open-meteo attempt %d failed: %s",
                    attempt + 1,
                    exc,
                )
        _log.warning(
            "open-meteo exhausted %d attempts; falling back to AFGL %r (last: %s)",
            len(retry_delays_s) + 1,
            afgl_climatology,
            last_exc,
        )

    pressure, temperature, h2o_vmr = _afgl_profile(afgl_climatology)
    return pressure, temperature, h2o_vmr, f"afgl_{afgl_climatology}", None


def extrapolate(
    ds: xr.Dataset,
    *,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "midlatitude_summer",
    grids: dict | None = None,
) -> None:
    """Anchor an am model to fitted τ values and extrapolate to all spws.

    Mutates *ds* in place.  Adds tau_extrapolated, am_freq_grid, am_tau to
    data_vars and writes atm_profile_source, afgl_climatology, pwv_scaling,
    open_meteo_query to attrs.

    Requires tau_zenith, tau_err, fit_success in ds.

    When called with ``grids={scan_id: PwvGrid, ...}`` after a Stage-2 fit
    (``mode ∈ {per_antenna_pwv, shared_pwv, tcal_solve}``), uses the
    precomputed PwvGrid directly: no scalar anchor fit, no extra am call.
    ``tau_extrapolated`` is filled per-scan at the fitted ``pwv_scan_median``.
    """
    if grids is not None and ds.attrs.get("mode") in (
        "per_antenna_pwv",
        "shared_pwv",
        "tcal_solve",
    ):
        _extrapolate_from_grids(ds, grids)
        return

    freqs_Hz = ds.coords["frequency"].values  # (spw,)

    # Representative observation time: median of scan start times (MJD seconds).
    obs_time_mjd_s = float(np.nanmedian(ds.coords["scan_time_start"].values))

    open_meteo_query: dict | None = None

    if atm_profile_source == "open-meteo":
        try:
            pressure, temperature, h2o_vmr, open_meteo_query = _fetch_open_meteo(
                _VLA_LAT, _VLA_LON, obs_time_mjd_s
            )
        except Exception:
            _log.warning(
                "open-meteo fetch failed; falling back to AFGL %r",
                afgl_climatology,
                exc_info=True,
            )
            atm_profile_source = "afgl"

    if atm_profile_source == "afgl":
        pressure, temperature, h2o_vmr = _afgl_profile(afgl_climatology)

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


def _extrapolate_from_grids(ds: xr.Dataset, grids: dict) -> None:
    """Stage-2 extrapolate: derive dense τ(ν) and per-spw τ from PwvGrid.

    No anchor fit — the PWV recovered by the fitter IS the anchor. For each
    scan, evaluates ``grid.lookup`` at ``pwv_scan_median[scan]`` and writes:

    - ``tau_extrapolated[scan, spw]``: τ at the observed spw centre freqs.
    - ``am_freq_grid``, ``am_tau``: dense curve from the first available scan's
      grid evaluated at the median consensus PWV (informational).

    ``ds.attrs["pwv_scaling"]`` is set to None (no anchor) so attr consumers
    know this came from the Stage-2 path.
    """
    freqs_Hz = ds.coords["frequency"].values
    scan_ids = ds.coords["scan"].values
    pwv_median = ds["pwv_scan_median"].values  # (scan,)

    n_scan = ds.sizes["scan"]
    tau_ext = np.full((n_scan, freqs_Hz.size), np.nan, dtype=np.float32)
    am_freq_first: np.ndarray | None = None
    am_tau_first: np.ndarray | None = None
    for i, scan_id in enumerate(scan_ids):
        sid = int(scan_id)
        grid = grids.get(sid)
        if grid is None or not np.isfinite(pwv_median[i]):
            continue
        tau_at_spw, _ = grid.lookup(float(pwv_median[i]), freqs_Hz)
        tau_ext[i, :] = tau_at_spw.astype(np.float32)
        if am_freq_first is None:
            am_freq_first = grid.freq_Hz.astype(np.float64)
            am_tau_first, _ = grid.lookup(float(pwv_median[i]), am_freq_first)

    ds["tau_extrapolated"] = (("scan", "spw"), tau_ext)
    if am_freq_first is not None and am_tau_first is not None:
        ds["am_freq_grid"] = (("frequency_dense",), am_freq_first)
        ds["am_tau"] = (("frequency_dense",), am_tau_first.astype(np.float64))
    ds.attrs["pwv_scaling"] = None
    ds.attrs["pwv_scan_medians_mm"] = {
        int(scan_ids[i]): float(pwv_median[i])
        for i in range(n_scan)
        if np.isfinite(pwv_median[i])
    }


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
) -> tuple[u.Quantity, u.Quantity, u.Quantity, dict]:
    """Fetch a vertical atmospheric profile from open-meteo.

    Returns (pressure, temperature, h2o_vmr, query_meta).
    Raises on HTTP error or timeout (caller falls back to AFGL).
    """
    import openmeteo_requests
    from openmeteo_sdk.Variable import Variable as OmVar

    # MJD seconds → UTC datetime (MJD epoch = 1858-11-17; Unix epoch MJD = 40587 days)
    obs_dt = datetime.fromtimestamp(
        (obs_time_mjd_s / 86400.0 - 40587.0) * 86400.0, tz=timezone.utc
    )
    date_str = obs_dt.strftime("%Y-%m-%d")

    now = datetime.now(tz=timezone.utc)
    age_days = (now - obs_dt).total_seconds() / 86400.0
    url = _OM_ARCHIVE_URL if age_days > _OM_FORECAST_HORIZON_DAYS else _OM_FORECAST_URL

    hourly_vars = [
        f"{v}_{p}hPa"
        for p in _OM_PRESSURE_LEVELS
        for v in ("temperature", "relative_humidity")
    ]
    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "UTC",
    }
    query_meta = {"lat": lat, "lon": lon, "date": date_str, "endpoint": url}

    client = openmeteo_requests.Client()
    responses = client.weather_api(url, params=params, timeout=_OM_TIMEOUT_S)
    hourly: Any = responses[0].Hourly()
    if hourly is None:
        raise RuntimeError("open-meteo response contained no hourly data")

    # Find the hourly index closest to the observation time.
    times = np.arange(hourly.Time(), hourly.TimeEnd(), hourly.Interval())
    t_idx = int(np.argmin(np.abs(times - obs_dt.timestamp())))

    temp_by_level: dict[int, float] = {}
    rh_by_level: dict[int, float] = {}
    for i in range(hourly.VariablesLength()):
        var: Any = hourly.Variables(i)
        if var is None:
            continue
        p_level = int(var.PressureLevel())
        if p_level not in _OM_PRESSURE_LEVELS:
            continue
        var_type = var.Variable()
        vals = var.ValuesAsNumpy()
        if var_type == OmVar.temperature:
            temp_by_level[p_level] = float(vals[t_idx])
        elif var_type == OmVar.relative_humidity:
            rh_by_level[p_level] = float(vals[t_idx])

    # Build ordered arrays (surface → top of atmosphere).
    # Open-meteo omits pressure levels below the surface (e.g. 1000 hPa at
    # high-altitude sites like the VLA at 2115 m); skip missing levels silently.
    valid_levels = [p for p in _OM_PRESSURE_LEVELS if p in temp_by_level and p in rh_by_level]
    if not valid_levels:
        raise RuntimeError("open-meteo returned no valid pressure levels")
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

    return pressure_q, temperature_q, h2o_vmr, query_meta


def _afgl_profile(
    name: str,
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """Return (pressure, temperature, h2o_vmr) from an AFGL climatology."""
    import amwrap as _amwrap

    clim = _amwrap.Climatology(name)
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
