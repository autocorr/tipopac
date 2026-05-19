"""MSReader — read a CASA Measurement Set into the canonical xarray.Dataset.

Reads DO_SKYDIP tipping scans from `casatools.table` and
`casatools.msmetadata`.  Returns a schema-valid `xr.Dataset` per DESIGN.md §5;
flag application (online FLAG_CMD) is deferred to `flags.py`.

Unit notes (confirmed against tip_test.ms):
  - WEATHER.PRESSURE:    stored in hPa despite QuantumUnits='Pa' → ×100
  - WEATHER.REL_HUMIDITY: stored in %  despite QuantumUnits='%'  → ÷100
  - WEATHER.TEMPERATURE: K, correct as stored
  - POINTING.ENCODER[1]: elevation in AZELGEO radians; zenith_angle = 90 − deg
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tipopac import schema


class MSReader:
    """Read a CASA MS into the canonical xr.Dataset."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @classmethod
    def supports(cls, path: Path) -> bool:
        p = Path(path)
        return (p / "table.dat").exists() and (p / "SYSPOWER").is_dir()

    @classmethod
    def from_path(cls, path: Path) -> "MSReader":
        return cls(path)

    def read(self) -> xr.Dataset:
        path = self._path

        ant_names, ant_positions = _read_antenna(path)
        spw_freq, spw_bw = _read_spectral_window(path)
        scan_ids, scan_spws, scan_t_start, scan_t_end = _read_scan_meta(path)

        tip_spws = sorted({s for spws in scan_spws.values() for s in spws})
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
        )

        schema.validate(ds)
        return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _read_antenna(path: Path) -> tuple[list[str], np.ndarray]:
    """Return (names, positions) where positions is (n_ant, 3) ITRF metres."""
    from casatools import table as _table

    tb = _table()
    try:
        tb.open(str(path / "ANTENNA"))
        names = [str(n) for n in tb.getcol("NAME")]
        pos = tb.getcol("POSITION").T.copy()  # (3, n_ant) → (n_ant, 3)
    finally:
        tb.close()
    return names, pos


def _read_spectral_window(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (ref_frequency, total_bandwidth) in Hz for all SPWs."""
    from casatools import table as _table

    tb = _table()
    try:
        tb.open(str(path / "SPECTRAL_WINDOW"))
        freq = tb.getcol("REF_FREQUENCY").copy()
        bw = tb.getcol("TOTAL_BANDWIDTH").copy()
    finally:
        tb.close()
    return freq, bw


def _read_scan_meta(
    path: Path,
) -> tuple[list[int], dict[int, list[int]], dict[int, float], dict[int, float]]:
    """Return scan ids, per-scan SPW lists, and scan start/end times (MJD-sec)."""
    from casatools import msmetadata as _msmd

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
    from casatools import table as _table

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
    from casatools import table as _table

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
    from casatools import table as _table

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


def _nearest_idx(ref_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Return indices into `ref_times` nearest to each value in `query_times`."""
    idx = np.searchsorted(ref_times, query_times)
    idx = np.clip(idx, 0, len(ref_times) - 1)
    left = np.clip(idx - 1, 0, len(ref_times) - 1)
    use_left = np.abs(query_times - ref_times[left]) < np.abs(
        query_times - ref_times[idx]
    )
    idx[use_left] = left[use_left]
    return idx


def _build_dataset(
    *,
    path: Path,
    ant_names: list[str],
    ant_positions: np.ndarray,
    spw_freq: np.ndarray,
    spw_bw: np.ndarray,
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
) -> xr.Dataset:
    from casatools import table as _table

    n_scan = len(scan_ids)
    n_ant = len(ant_names)
    n_spw = len(tip_spws)

    # determine n_time: read sample counts per scan then take the maximum
    tb = _table()
    tb.open(str(path / "SYSPOWER"))

    scan_times: list[np.ndarray] = []
    for sc in scan_ids:
        sc_spws = scan_spws[sc]
        t_start, t_end = scan_t_start[sc], scan_t_end[sc]
        spw0 = sc_spws[0]
        sub = tb.query(
            f"TIME>={t_start} && TIME<={t_end}"
            f" && ANTENNA_ID==0 && SPECTRAL_WINDOW_ID=={spw0}"
        )
        ts = (
            sub.getcol("TIME").copy()
            if sub.nrows() > 0
            else np.array([], dtype=np.float64)
        )
        sub.close()
        scan_times.append(np.sort(ts))

    n_time = max((len(t) for t in scan_times), default=1)

    # allocate output arrays
    switched_diff = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    switched_sum = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    zenith_angle = np.full((n_scan, n_ant, n_time), np.nan, dtype=np.float32)
    weather_T = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_P = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_RH = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    # flag is True for NaN-pad and missing-spw positions
    flag = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=bool)
    time_utc = np.full((n_scan, n_time), np.nan, dtype=np.float64)
    scan_time_start_arr = np.empty(n_scan, dtype=np.float64)
    scan_time_end_arr = np.empty(n_scan, dtype=np.float64)

    for i, sc in enumerate(scan_ids):
        ts = scan_times[i]
        n_t = len(ts)
        if n_t == 0:
            continue

        scan_time_start_arr[i] = ts[0]
        scan_time_end_arr[i] = ts[-1]
        time_utc[i, :n_t] = ts

        t_start, t_end = scan_t_start[sc], scan_t_end[sc]
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

        # --- SYSPOWER per scan (one query covers all antennas and spws) ---
        sub = tb.query(f"TIME>={t_start} && TIME<={t_end}")
        if sub.nrows() == 0:
            sub.close()
            continue

        sp_times = sub.getcol("TIME")
        sp_ant = sub.getcol("ANTENNA_ID")
        sp_spw = sub.getcol("SPECTRAL_WINDOW_ID")
        sp_diff = sub.getcol("SWITCHED_DIFF")  # (2, n_rows)
        sp_sum = sub.getcol("SWITCHED_SUM")  # (2, n_rows)
        sub.close()

        # build a time → scan-local index map for this scan
        t_to_idx: dict[float, int] = {float(t): j for j, t in enumerate(ts)}

        for row in range(len(sp_times)):
            a = int(sp_ant[row])
            s = int(sp_spw[row])
            if s not in spw_to_idx or s not in sc_spw_set:
                continue
            w = spw_to_idx[s]
            t_key = float(sp_times[row])
            j = t_to_idx.get(t_key)
            if j is None:
                continue
            switched_diff[i, a, w, 0, j] = sp_diff[0, row]
            switched_diff[i, a, w, 1, j] = sp_diff[1, row]
            switched_sum[i, a, w, 0, j] = sp_sum[0, row]
            switched_sum[i, a, w, 1, j] = sp_sum[1, row]
            # unflag every cell that got real data
            flag[i, a, w, :, j] = False

    tb.close()

    # pad positions must stay flagged regardless of data presence
    for i in range(n_scan):
        n_t = len(scan_times[i])
        if n_t < n_time:
            flag[i, :, :, :, n_t:] = True

    return xr.Dataset(
        data_vars={
            "switched_diff": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_diff,
            ),
            "switched_sum": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_sum,
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                zenith_angle,
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                tcal_ref,
            ),
            "weather_T": (("scan", "time"), weather_T),
            "weather_P": (("scan", "time"), weather_P),
            "weather_RH": (("scan", "time"), weather_RH),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                flag,
            ),
        },
        coords={
            "scan": np.array(scan_ids, dtype=np.intp),
            "antenna": ant_names,
            "spw": np.array(tip_spws, dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (
                ("spw",),
                spw_freq[tip_spws].astype(np.float64),
            ),
            "bandwidth": (
                ("spw",),
                spw_bw[tip_spws].astype(np.float64),
            ),
            "antenna_position": (
                ("antenna", "xyz"),
                ant_positions.astype(np.float64),
            ),
            "scan_time_start": (("scan",), scan_time_start_arr),
            "scan_time_end": (("scan",), scan_time_end_arr),
            "time_utc": (("scan", "time"), time_utc),
        },
        attrs={
            "source_path": str(path),
            "source_format": "ms",
            "observatory": "VLA",
        },
    )
