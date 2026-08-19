"""MSReader — read a CASA Measurement Set into the canonical xarray.Dataset.

Reads DO_SKYDIP tipping scans from `casatools.table` and
`casatools.msmetadata`.  Returns a schema-valid `xr.Dataset` per design.md §4;
flag application (online FLAG_CMD) is deferred to `flags.py`.

Unit notes (confirmed against tip_test.ms):
  - WEATHER.PRESSURE:    stored in hPa despite QuantumUnits='Pa' → ×100
  - WEATHER.REL_HUMIDITY: stored in %  despite QuantumUnits='%'  → ÷100
  - WEATHER.TEMPERATURE: K, correct as stored
  - POINTING.ENCODER[1]: elevation in AZELGEO radians; zenith_angle = 90 − deg
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

from tipopac import schema
from tipopac._casa import import_casatools
from tipopac.bands import attach_selection_attrs, band_for_spw_name
from tipopac.readers.base import (
    SkydipScanInfo,
    _apply_selection,
    _drop_empty_scans,
    _map_ids,
    _nearest_idx,
    _scatter_syspower,
    _slot_indices,
    _slot_medians,
    build_canonical_dataset,
)


class MSReader:
    """Read a CASA MS into the canonical xr.Dataset.

    `scans` and `bands` filter the DO_SKYDIP set at read time so excluded
    data is never loaded. `scans=None` keeps all DO_SKYDIP scans;
    `bands=None` keeps only the high-frequency receivers (Ku, K, Ka, Q).
    """

    def __init__(
        self,
        path: Path,
        *,
        scans: Sequence[int] | None = None,
        bands: Sequence[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._scans_requested = scans
        self._bands_requested = bands

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        return (p / "table.dat").exists() and (p / "SYSPOWER").is_dir()

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        scans: Sequence[int] | None = None,
        bands: Sequence[str] | None = None,
    ) -> "MSReader":
        return cls(path, scans=scans, bands=bands)

    @classmethod
    def list_skydip_scans(cls, path: Path) -> list[SkydipScanInfo]:
        """Return scan-level metadata for every DO_SKYDIP scan in `path`.

        Lightweight: reads SPECTRAL_WINDOW and MS-metadata only — no
        pointing / weather / syspower / caldevice load. Used by
        ``tipopac.summary``.
        """
        p = Path(path)
        _, _, spw_bands = _read_spectral_window(p)
        scan_ids, scan_spws, scan_t_start, _ = _read_scan_meta(p)

        out: list[SkydipScanInfo] = []
        for sc in scan_ids:
            spws = tuple(scan_spws[sc])
            bands = tuple(sorted({str(spw_bands[s]) for s in spws}))
            out.append(
                SkydipScanInfo(
                    scan_id=sc,
                    start_mjd_s=scan_t_start[sc],
                    spw_ids=spws,
                    bands=bands,
                )
            )
        return out

    def read(self) -> xr.Dataset:
        path = self._path

        ant_names, ant_positions = _read_antenna(path)
        spw_freq, spw_bw, spw_bands = _read_spectral_window(path)
        scan_ids, scan_spws, scan_t_start, scan_t_end = _read_scan_meta(path)

        scan_ids, scan_spws, scan_t_start, scan_t_end, tip_spws = _apply_selection(
            scan_ids,
            scan_spws,
            scan_t_start,
            scan_t_end,
            spw_bands,
            self._scans_requested,
            self._bands_requested,
        )
        spw_to_idx = {s: i for i, s in enumerate(tip_spws)}

        tcal_ref = _read_caldevice(path, len(ant_names), tip_spws, spw_to_idx)
        point_t, point_za = _read_pointing(path, len(ant_names))
        wx_t, wx_T, wx_P, wx_RH = _read_weather(path)

        ds = _build_dataset(
            path=path,
            ant_names=ant_names,
            ant_positions=ant_positions,
            spw_freq=spw_freq,
            spw_bw=spw_bw,
            spw_bands=spw_bands,
            tip_spws=tip_spws,
            spw_to_idx=spw_to_idx,
            scan_ids=scan_ids,
            scan_spws=scan_spws,
            scan_t_start=scan_t_start,
            scan_t_end=scan_t_end,
            tcal_ref=tcal_ref,
            point_t=point_t,
            point_za=point_za,
            wx_t=wx_t,
            wx_T=wx_T,
            wx_P=wx_P,
            wx_RH=wx_RH,
            scans_requested=self._scans_requested,
        )

        attach_selection_attrs(ds, self._scans_requested, self._bands_requested)
        schema.validate(ds)
        return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_antenna(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (names, positions) where positions is (n_ant, 3) ITRF metres."""
    _table = import_casatools().table

    tb = _table()
    try:
        tb.open(str(path / "ANTENNA"))
        names = [str(n) for n in tb.getcol("NAME")]
        pos = tb.getcol("POSITION").T.copy()  # (3, n_ant) → (n_ant, 3)
    finally:
        tb.close()
    return names, pos


def _read_spectral_window(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ref_frequency, total_bandwidth, band_labels) for all SPWs."""
    _table = import_casatools().table

    tb = _table()
    try:
        tb.open(str(path / "SPECTRAL_WINDOW"))
        freq = tb.getcol("REF_FREQUENCY").copy()
        bw = tb.getcol("TOTAL_BANDWIDTH").copy()
        names = tb.getcol("NAME")
    finally:
        tb.close()
    bands = np.array(
        [band_for_spw_name(str(n)) for n in names],
        dtype="U4",
    )
    return freq, bw, bands


def _read_scan_meta(
    path: Path,
) -> tuple[list[int], dict[int, list[int]], dict[int, float], dict[int, float]]:
    """Return scan ids, per-scan SPW lists, and scan start/end times (MJD-sec)."""
    _msmd = import_casatools().msmetadata

    msmd = _msmd()
    try:
        msmd.open(str(path))
        scan_ids = sorted(int(s) for s in msmd.scansforintent("*DO_SKYDIP*"))
        scan_spws: dict[int, list[int]] = {}
        scan_t_start: dict[int, float] = {}
        scan_t_end: dict[int, float] = {}
        for sc in scan_ids:
            scan_spws[sc] = [int(s) for s in msmd.spwsforscan(sc)]
            times = msmd.timesforscan(sc)
            scan_t_start[sc] = float(times[0])
            scan_t_end[sc] = float(times[-1])
    finally:
        msmd.done()
    return scan_ids, scan_spws, scan_t_start, scan_t_end


def _read_caldevice(
    path: Path,
    n_ant: int,
    tip_spws: list[int],
    spw_to_idx: dict[int, int],
) -> np.ndarray:
    """Return tcal_ref (n_ant, n_spw, 2) float32 from CALDEVICE row 0 (noise tube).

    Missing (ant, spw) cells are filled by copying from the previous spw
    (matching v2.6's fallback at task_tipopac.py:1003–1007).
    """
    _table = import_casatools().table

    n_spw = len(tip_spws)
    out = np.full((n_ant, n_spw, 2), np.nan, dtype=np.float32)

    tb = _table()
    try:
        tb.open(str(path / "CALDEVICE"))
        nc = tb.getcol("NOISE_CAL")  # (n_load, n_pol, n_rows)
        ant_col = tb.getcol("ANTENNA_ID")
        spw_col = tb.getcol("SPECTRAL_WINDOW_ID")
    finally:
        tb.close()

    for row in range(len(ant_col)):
        a = int(ant_col[row])
        s = int(spw_col[row])
        if s in spw_to_idx and 0 <= a < n_ant:
            w = spw_to_idx[s]
            out[a, w, 0] = float(nc[0, 0, row])  # noise tube, R
            out[a, w, 1] = float(nc[0, 1, row])  # noise tube, L

    # fill NaN cells by propagating the previous spw (v2.6 fallback)
    for a in range(n_ant):
        for wi in range(1, n_spw):
            if np.isnan(out[a, wi, 0]):
                out[a, wi] = out[a, wi - 1]

    return out


def _read_pointing(
    path: Path,
    n_ant: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-antenna (times, zenith_angles_deg) lists.

    `times[a]` and `zenith_angles[a]` are 1-D float64 arrays sorted by time.
    """
    _table = import_casatools().table

    tb = _table()
    try:
        tb.open(str(path / "POINTING"))
        all_times = tb.getcol("TIME")  # MJD-sec, shape (n_rows,)
        all_enc = tb.getcol("ENCODER")  # (2, n_rows) radians AZELGEO
        all_ant = tb.getcol("ANTENNA_ID")  # (n_rows,)
    finally:
        tb.close()

    # elevation is ENCODER[1]; zenith_angle = 90 − deg(elevation)
    all_za = 90.0 - np.rad2deg(all_enc[1])

    point_t: list[np.ndarray] = []
    point_za: list[np.ndarray] = []
    for a in range(n_ant):
        mask = all_ant == a
        t_a = all_times[mask]
        za_a = all_za[mask]
        order = np.argsort(t_a)
        point_t.append(t_a[order])
        point_za.append(za_a[order])

    return point_t, point_za


def _read_weather(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, T_K, P_Pa, RH_frac) from the WEATHER subtable.

    Conversions applied:
      PRESSURE (stored hPa) → Pa  (×100)
      REL_HUMIDITY (stored %) → fraction (÷100)
    """
    _table = import_casatools().table

    tb = _table()
    try:
        tb.open(str(path / "WEATHER"))
        times = tb.getcol("TIME").copy()
        temp = tb.getcol("TEMPERATURE").copy().astype(np.float64)
        pres = tb.getcol("PRESSURE").copy().astype(np.float64)
        rh = tb.getcol("REL_HUMIDITY").copy().astype(np.float64)
        if "TEMPERATURE_FLAG" in tb.colnames():
            tflag = tb.getcol("TEMPERATURE_FLAG").astype(bool)
            temp[tflag] = np.nan
    finally:
        tb.close()

    order = np.argsort(times)
    return (
        times[order],
        temp[order],
        pres[order] * 100.0,  # hPa → Pa
        rh[order] / 100.0,  # % → fraction
    )


def _build_dataset(
    *,
    path: Path,
    ant_names: list[str],
    ant_positions: np.ndarray,
    spw_freq: np.ndarray,
    spw_bw: np.ndarray,
    spw_bands: np.ndarray,
    tip_spws: list[int],
    spw_to_idx: dict[int, int],
    scan_ids: list[int],
    scan_spws: dict[int, list[int]],
    scan_t_start: dict[int, float],
    scan_t_end: dict[int, float],
    tcal_ref: np.ndarray,
    point_t: list[np.ndarray],
    point_za: list[np.ndarray],
    wx_t: np.ndarray,
    wx_T: np.ndarray,
    wx_P: np.ndarray,
    wx_RH: np.ndarray,
    scans_requested: Sequence[int] | None,
) -> xr.Dataset:
    _table = import_casatools().table

    n_ant = len(ant_names)
    n_spw = len(tip_spws)

    # per-scan time axis: unique SYSPOWER timestamps across all antennas/spws
    # (shared dump cadence); n_time is the maximum over scans
    tb = _table()
    tb.open(str(path / "SYSPOWER"))

    scan_times: list[np.ndarray] = []
    scan_rows: dict[int, dict[str, np.ndarray]] = {}
    for sc in scan_ids:
        t_start, t_end = scan_t_start[sc], scan_t_end[sc]
        sub = tb.query(f"TIME>={t_start} && TIME<={t_end}")
        if sub.nrows() > 0:
            rows = {
                col: sub.getcol(col).copy()
                for col in (
                    "TIME",
                    "ANTENNA_ID",
                    "SPECTRAL_WINDOW_ID",
                    "SWITCHED_DIFF",
                    "SWITCHED_SUM",
                )
            }
            rows["INTERVAL"] = (
                sub.getcol("INTERVAL").copy()
                if "INTERVAL" in sub.colnames()
                else np.full(sub.nrows(), np.nan, dtype=np.float64)
            )
            scan_rows[sc] = rows
        sub.close()
        scan_times.append(
            np.unique(scan_rows[sc]["TIME"])
            if sc in scan_rows
            else np.array([], dtype=np.float64)
        )

    tb.close()
    scan_ids, scan_times = _drop_empty_scans(scan_ids, scan_times, scans_requested)
    n_scan = len(scan_ids)
    n_time = max(len(t) for t in scan_times)

    # allocate output arrays
    switched_diff = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    switched_sum = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    zenith_angle = np.full((n_scan, n_ant, n_time), np.nan, dtype=np.float32)
    weather_T = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_P = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_RH = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    exposure_time = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    # flag is True for NaN-pad and missing-spw positions
    flag = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=bool)
    time_utc = np.full((n_scan, n_time), np.nan, dtype=np.float64)
    scan_time_start_arr = np.full(n_scan, np.nan, dtype=np.float64)
    scan_time_end_arr = np.full(n_scan, np.nan, dtype=np.float64)

    for i, sc in enumerate(scan_ids):
        ts = scan_times[i]
        n_t = len(ts)

        scan_time_start_arr[i] = ts[0]
        scan_time_end_arr[i] = ts[-1]
        time_utc[i, :n_t] = ts

        sc_spw_set = set(scan_spws[sc])

        # --- weather (interpolated to SYSPOWER timestamps) ---
        weather_T[i, :n_t] = np.interp(ts, wx_t, wx_T).astype(np.float32)
        weather_P[i, :n_t] = np.interp(ts, wx_t, wx_P).astype(np.float32)
        weather_RH[i, :n_t] = np.interp(ts, wx_t, wx_RH).astype(np.float32)

        # --- zenith angle (nearest POINTING sample per antenna) ---
        for a in range(n_ant):
            if len(point_t[a]) == 0:
                continue
            idx = _nearest_idx(point_t[a], ts)
            zenith_angle[i, a, :n_t] = point_za[a][idx].astype(np.float32)

        # --- SYSPOWER per scan (read once above, all antennas and spws) ---
        rows = scan_rows[sc]
        sp_times = rows["TIME"]
        sp_ant = rows["ANTENNA_ID"]
        sp_spw = rows["SPECTRAL_WINDOW_ID"]
        sp_diff = rows["SWITCHED_DIFF"]  # (2, n_rows)
        sp_sum = rows["SWITCHED_SUM"]  # (2, n_rows)
        sp_interval = rows["INTERVAL"]

        slot_idx = _slot_indices(ts, sp_times)

        # exposure_time per scan-local time slot: take median across all
        # SYSPOWER rows at that timestamp (antennas+spws share the dump cycle)
        if np.isfinite(sp_interval).any():
            matched = slot_idx >= 0
            exposure_time[i, :n_t] = _slot_medians(
                slot_idx[matched], sp_interval[matched], n_t
            )
        # fallback: derive from time differences if INTERVAL missing or all-NaN
        if not np.isfinite(exposure_time[i, :n_t]).any() and n_t >= 2:
            dt = np.diff(ts)
            exposure_time[i, :n_t] = float(np.median(dt))

        ant_idx = sp_ant.astype(np.intp)
        spw_idx = _map_ids(sp_spw, lambda s: spw_to_idx.get(int(s), -1))
        in_scan = np.zeros(n_spw, dtype=bool)
        for s in sc_spw_set & spw_to_idx.keys():
            in_scan[spw_to_idx[s]] = True
        keep = (
            (ant_idx >= 0)
            & (ant_idx < n_ant)
            & (spw_idx >= 0)
            & (slot_idx >= 0)
            & in_scan[spw_idx]
        )
        ant_idx, spw_idx, slot_idx = ant_idx[keep], spw_idx[keep], slot_idx[keep]

        _scatter_syspower(
            scan=i,
            ant_idx=ant_idx,
            spw_idx=spw_idx,
            slot_idx=slot_idx,
            diff=sp_diff.T[keep],
            total=sp_sum.T[keep],
            switched_diff=switched_diff,
            switched_sum=switched_sum,
            flag=flag,
        )

    # pad positions must stay flagged regardless of data presence
    for i in range(n_scan):
        n_t = len(scan_times[i])
        if n_t < n_time:
            flag[i, :, :, :, n_t:] = True

    return build_canonical_dataset(
        path=path,
        source_format="ms",
        scan_ids=scan_ids,
        ant_names=ant_names,
        tip_spws=tip_spws,
        ant_positions=ant_positions,
        spw_freq=spw_freq,
        spw_bw=spw_bw,
        spw_bands=spw_bands,
        switched_diff=switched_diff,
        switched_sum=switched_sum,
        zenith_angle=zenith_angle,
        tcal_ref=tcal_ref,
        weather_T=weather_T,
        weather_P=weather_P,
        weather_RH=weather_RH,
        exposure_time=exposure_time,
        flag=flag,
        scan_time_start=scan_time_start_arr,
        scan_time_end=scan_time_end_arr,
        time_utc=time_utc,
    )
