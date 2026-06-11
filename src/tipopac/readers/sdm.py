"""SDMReader — read a VLA/ALMA Science Data Model into the canonical xarray.Dataset.

Reads DO_SKYDIP tipping scans from `sdmpy` binary and XML tables.
Returns a schema-valid `xr.Dataset` per DESIGN.md §5; there is no SDM
equivalent of FLAG_CMD so the returned dataset has all flag cells False
(no online flags applied).

Unit notes (verified against tip_test.sdm):
  - SysPower.timeMid / Pointing.timeMid: ASDM nanoseconds → MJD-sec (÷1e9)
  - Weather.pressure:    already in Pa
  - Weather.relHumidity: stored in %  → fraction (÷100)
  - Weather.temperature: K, correct as stored
  - Pointing.encoder[row, 0, 1]: elevation in AZELGEO radians;
    zenith_angle = 90 − deg(elevation)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from tipopac import schema
from tipopac.bands import (
    attach_selection_attrs,
    band_for_spw_name,
    normalize_bands,
    select_spws_by_band,
    validate_scan_selection,
)
from tipopac.readers.base import SkydipScanInfo


class SDMReader:
    """Read a VLA/ALMA SDM into the canonical xr.Dataset.

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
        return (Path(path) / "ASDM.xml").exists()

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        scans: Sequence[int] | None = None,
        bands: Sequence[str] | None = None,
    ) -> "SDMReader":
        return cls(path, scans=scans, bands=bands)

    @classmethod
    def list_skydip_scans(cls, path: Path) -> list[SkydipScanInfo]:
        """Return scan-level metadata for every DO_SKYDIP scan in `path`.

        Lightweight: opens the SDM and reads Scan / SpectralWindow /
        SysPower (the same tables the full read uses to derive scan
        metadata), without any pointing / weather / caldevice load. Used
        by ``tipopac.summary``.
        """
        import sdmpy

        sdm = sdmpy.SDM(str(Path(path)), use_xsd=False)
        _, _, spw_bands, _ = _read_spectral_window(sdm)
        scan_ids, scan_spws, scan_t_start, _ = _read_scan_meta(sdm)

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
        import sdmpy

        sdm = sdmpy.SDM(str(self._path), use_xsd=False)

        ant_names, ant_positions, ant_id_to_idx = _read_antenna(sdm)
        spw_freq, spw_bw, spw_bands, spw_id_to_idx = _read_spectral_window(sdm)
        scan_ids, scan_spws, scan_t_start, scan_t_end = _read_scan_meta(sdm)

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

        tcal_ref = _read_caldevice(
            sdm, len(ant_names), ant_id_to_idx, tip_spws, spw_to_idx, spw_id_to_idx
        )
        point_t, point_za = _read_pointing(sdm, len(ant_names), ant_id_to_idx)
        wx_t, wx_T, wx_P, wx_RH = _read_weather(sdm)

        sp_data = sdm["SysPower"].data

        ds = _build_dataset(
            path=self._path,
            ant_names=ant_names,
            ant_positions=ant_positions,
            spw_freq=spw_freq,
            spw_bw=spw_bw,
            spw_bands=spw_bands,
            tip_spws=tip_spws,
            spw_to_idx=spw_to_idx,
            spw_id_to_idx=spw_id_to_idx,
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
            sp_data=sp_data,
            ant_id_to_idx=ant_id_to_idx,
        )

        attach_selection_attrs(ds, self._scans_requested, self._bands_requested)
        schema.validate(ds)
        return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _apply_selection(
    scan_ids: list[int],
    scan_spws: dict[int, list[int]],
    scan_t_start: dict[int, float],
    scan_t_end: dict[int, float],
    spw_bands: np.ndarray,
    scans_requested: Sequence[int] | None,
    bands_requested: Sequence[str] | None,
) -> tuple[
    list[int],
    dict[int, list[int]],
    dict[int, float],
    dict[int, float],
    list[int],
]:
    """Apply the user's scan + band selection to reader metadata.

    Mirrors the MS reader. Raises `ValueError` if the resulting set is
    empty.
    """
    scan_ids = validate_scan_selection(scans_requested, scan_ids)
    scan_spws = {sc: scan_spws[sc] for sc in scan_ids}
    scan_t_start = {sc: scan_t_start[sc] for sc in scan_ids}
    scan_t_end = {sc: scan_t_end[sc] for sc in scan_ids}

    tip_spws_all = sorted({s for spws in scan_spws.values() for s in spws})

    allowed_bands = normalize_bands(bands_requested)
    tip_spws = select_spws_by_band(tip_spws_all, spw_bands, allowed_bands)
    if not tip_spws:
        observed = sorted({str(spw_bands[s]) for s in tip_spws_all})
        raise ValueError(
            f"no SPWs match bands={list(allowed_bands)!r}; "
            f"observed bands in this dataset: {observed}"
        )

    keep_set = set(tip_spws)
    kept_scan_ids: list[int] = []
    for sc in scan_ids:
        scan_spws[sc] = [s for s in scan_spws[sc] if s in keep_set]
        if scan_spws[sc]:
            kept_scan_ids.append(sc)
    if not kept_scan_ids:
        raise ValueError(
            f"no scans retain any SPW after bands={list(allowed_bands)!r} "
            "filter; widen the band selection or pick different scans"
        )

    dropped = [sc for sc in scan_ids if sc not in kept_scan_ids]
    # When the user named scans explicitly, a band-filter drop of any of
    # those is a contract violation — match the "raise on miss" behavior
    # of `validate_scan_selection`.
    if scans_requested is not None and dropped:
        raise ValueError(
            f"requested scan(s) {dropped} have no SPWs in "
            f"bands={list(allowed_bands)!r}; either widen the band "
            "selection or drop these scans from the request"
        )
    for sc in dropped:
        scan_spws.pop(sc, None)
        scan_t_start.pop(sc, None)
        scan_t_end.pop(sc, None)

    return kept_scan_ids, scan_spws, scan_t_start, scan_t_end, tip_spws


def _read_antenna(
    sdm: Any,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Return (names, positions, ant_id_to_idx) from Antenna + Station tables.

    positions is (n_ant, 3) ITRF metres.
    ant_id_to_idx maps 'Antenna_N' → row index in Antenna table.
    """
    ant_rows = list(sdm["Antenna"])  # type: ignore[index]
    stations = {str(s.stationId): s for s in sdm["Station"]}  # type: ignore[index]

    names: list[str] = []
    positions: list[list[float]] = []
    ant_id_to_idx: dict[str, int] = {}

    for i, a in enumerate(ant_rows):
        ant_id = str(a.antennaId)
        ant_id_to_idx[ant_id] = i
        names.append(str(a.name))
        st = stations[str(a.stationId)]
        parts = str(st.position).split()
        # format: '1 3 X Y Z'
        positions.append([float(parts[2]), float(parts[3]), float(parts[4])])

    return names, np.array(positions, dtype=np.float64), ant_id_to_idx


def _read_spectral_window(
    sdm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Return (ref_frequency, total_bandwidth, band_labels, spw_id_to_idx).

    spw_id_to_idx maps 'SpectralWindow_N' → row index.
    """
    spw_rows = list(sdm["SpectralWindow"])  # type: ignore[index]
    freq = np.array([float(s.refFreq) for s in spw_rows], dtype=np.float64)
    bw = np.array([float(s.totBandwidth) for s in spw_rows], dtype=np.float64)
    bands = np.array(
        [band_for_spw_name(str(s.name)) for s in spw_rows],
        dtype="U4",
    )
    spw_id_to_idx = {str(s.spectralWindowId): i for i, s in enumerate(spw_rows)}
    return freq, bw, bands, spw_id_to_idx


def _read_scan_meta(
    sdm: Any,
) -> tuple[list[int], dict[int, list[int]], dict[int, float], dict[int, float]]:
    """Return (scan_ids, scan_spws, scan_t_start, scan_t_end) for DO_SKYDIP scans.

    Times are MJD-seconds.  Per-scan SPW lists are integer indices derived
    from the SysPower binary table contents for that scan's time window.
    """
    sp_data = sdm["SysPower"].data  # type: ignore[index]
    scan_rows = list(sdm["Scan"])  # type: ignore[index]

    skydip_rows = [r for r in scan_rows if "DO_SKYDIP" in str(r.scanIntent)]

    scan_ids: list[int] = []
    scan_spws: dict[int, list[int]] = {}
    scan_t_start: dict[int, float] = {}
    scan_t_end: dict[int, float] = {}

    for r in sorted(skydip_rows, key=lambda x: int(str(x.scanNumber))):
        sc = int(str(r.scanNumber))
        t0_ns = int(str(r.startTime))
        t1_ns = int(str(r.endTime))

        mask = (sp_data["timeMid"] >= t0_ns) & (sp_data["timeMid"] <= t1_ns)
        spw_ids_in_scan = np.unique(sp_data["spectralWindowId"][mask])
        spw_ints = sorted(int(s.split("_")[1]) for s in spw_ids_in_scan)

        if not spw_ints:
            continue

        scan_ids.append(sc)
        scan_spws[sc] = spw_ints
        scan_t_start[sc] = t0_ns / 1e9
        scan_t_end[sc] = t1_ns / 1e9

    return scan_ids, scan_spws, scan_t_start, scan_t_end


def _read_caldevice(
    sdm: Any,
    n_ant: int,
    ant_id_to_idx: dict[str, int],
    tip_spws: list[int],
    spw_to_idx: dict[int, int],
    spw_id_to_idx: dict[str, int],
) -> np.ndarray:
    """Return tcal_ref (n_ant, n_spw, 2) float32 from CalDevice (noise tube).

    Missing (ant, spw) cells are filled by copying from the previous spw,
    matching v2.6's fallback at task_tipopac.py:1003–1007.
    """
    n_spw = len(tip_spws)
    out = np.full((n_ant, n_spw, 2), np.nan, dtype=np.float32)

    tip_spw_set = set(tip_spws)

    for row in sdm["CalDevice"]:  # type: ignore[index]
        ant_id = str(row.antennaId)
        spw_id = str(row.spectralWindowId)

        a = ant_id_to_idx.get(ant_id)
        spw_int = spw_id_to_idx.get(spw_id)
        if a is None or spw_int is None or spw_int not in tip_spw_set:
            continue

        w = spw_to_idx[spw_int]
        # coupledNoiseCal is 2D: [numReceptor][numCalload], format '2 R C v00 v01 v10 v11'
        # Element [receptor, calload=0] is the noise-tube value for that receptor.
        parts = str(row.coupledNoiseCal).split()
        ncols = int(parts[2])  # number of cal loads
        out[a, w, 0] = float(parts[3])  # receptor R, noise tube (load 0)
        out[a, w, 1] = float(parts[3 + ncols])  # receptor L, noise tube (load 0)

    # fill NaN cells by propagating the previous spw (v2.6 fallback)
    for a in range(n_ant):
        for wi in range(1, n_spw):
            if np.isnan(out[a, wi, 0]):
                out[a, wi] = out[a, wi - 1]

    return out


def _read_pointing(
    sdm: Any,
    n_ant: int,
    ant_id_to_idx: dict[str, int],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return per-antenna (times_MJD_sec, zenith_angles_deg) lists.

    Each list entry is a 1-D float64 array sorted by time.
    encoder[row, 0, 1] is the AZELGEO elevation in radians.
    """
    pt_data = sdm["Pointing"].data  # type: ignore[index]

    times_ns = pt_data["timeMid"]
    encoders = pt_data["encoder"]  # (n_rows, 1, 2)
    ant_ids = pt_data["antennaId"]

    # elevation is encoder[:, 0, 1]; zenith_angle = 90 − deg(el)
    el_rad = encoders[:, 0, 1]
    za_all = 90.0 - np.rad2deg(el_rad)
    t_all_s = times_ns.astype(np.float64) / 1e9

    point_t: list[np.ndarray] = []
    point_za: list[np.ndarray] = []
    for _ in range(n_ant):
        point_t.append(np.empty(0, dtype=np.float64))
        point_za.append(np.empty(0, dtype=np.float64))

    for ant_id, idx in ant_id_to_idx.items():
        mask = ant_ids == ant_id
        t_a = t_all_s[mask]
        za_a = za_all[mask]
        order = np.argsort(t_a)
        point_t[idx] = t_a[order]
        point_za[idx] = za_a[order]

    return point_t, point_za


def _read_weather(
    sdm: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times_MJD_sec, T_K, P_Pa, RH_frac) from Weather table.

    Only rows from the WX monitor station (Station_0) are used.
    timeInterval format: 'startTime_ns duration_ns'.
    """
    wx_t: list[float] = []
    wx_T: list[float] = []
    wx_P: list[float] = []
    wx_RH: list[float] = []

    for row in sdm["Weather"]:  # type: ignore[index]
        # filter to WX station only
        if str(row.stationId) != "Station_0":
            continue

        parts = str(row.timeInterval).split()
        t_start_ns = int(parts[0])
        duration_ns = int(parts[1])
        t_mid_s = (t_start_ns + duration_ns // 2) / 1e9

        wx_t.append(t_mid_s)
        wx_T.append(float(row.temperature))
        wx_P.append(float(row.pressure))
        wx_RH.append(float(row.relHumidity) / 100.0)

    t_arr = np.array(wx_t, dtype=np.float64)
    order = np.argsort(t_arr)
    return (
        t_arr[order],
        np.array(wx_T, dtype=np.float64)[order],
        np.array(wx_P, dtype=np.float64)[order],
        np.array(wx_RH, dtype=np.float64)[order],
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
    spw_bands: np.ndarray,
    tip_spws: list[int],
    spw_to_idx: dict[int, int],
    spw_id_to_idx: dict[str, int],
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
    sp_data: np.ndarray,
    ant_id_to_idx: dict[str, int],
) -> xr.Dataset:
    n_scan = len(scan_ids)
    n_ant = len(ant_names)
    n_spw = len(tip_spws)

    # build reverse maps for binary table lookups
    idx_to_ant_id = {v: k for k, v in ant_id_to_idx.items()}
    # reverse: spw integer index → 'SpectralWindow_N'
    int_to_spw_id: dict[int, str] = {v: k for k, v in spw_id_to_idx.items()}

    # determine per-scan sample times using Antenna_0 + first scan SPW as reference
    scan_times: list[np.ndarray] = []
    for sc in scan_ids:
        t0_ns = int(scan_t_start[sc] * 1e9)
        t1_ns = int(scan_t_end[sc] * 1e9)
        ref_ant_id = idx_to_ant_id[0]
        ref_spw_id = int_to_spw_id[scan_spws[sc][0]]
        mask = (
            (sp_data["timeMid"] >= t0_ns)
            & (sp_data["timeMid"] <= t1_ns)
            & (sp_data["antennaId"] == ref_ant_id)
            & (sp_data["spectralWindowId"] == ref_spw_id)
        )
        ts_ns = np.sort(np.unique(sp_data["timeMid"][mask]))
        scan_times.append(ts_ns.astype(np.float64) / 1e9)

    n_time = max((len(t) for t in scan_times), default=1)

    # allocate output arrays
    switched_diff = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    switched_sum = np.full((n_scan, n_ant, n_spw, 2, n_time), np.nan, dtype=np.float32)
    zenith_angle = np.full((n_scan, n_ant, n_time), np.nan, dtype=np.float32)
    weather_T = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_P = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    weather_RH = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    exposure_time = np.full((n_scan, n_time), np.nan, dtype=np.float32)
    flag = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=bool)

    sp_field_names = set(sp_data.dtype.names or ())
    sp_interval_field: str | None = None
    for cand in ("interval", "duration", "integrationTime"):
        if cand in sp_field_names:
            sp_interval_field = cand
            break
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

        sc_spw_set = set(scan_spws[sc])

        # --- weather (interpolated to SysPower timestamps) ---
        if len(wx_t) > 0:
            weather_T[i, :n_t] = np.interp(ts, wx_t, wx_T).astype(np.float32)
            weather_P[i, :n_t] = np.interp(ts, wx_t, wx_P).astype(np.float32)
            weather_RH[i, :n_t] = np.interp(ts, wx_t, wx_RH).astype(np.float32)

        # --- zenith angle (nearest Pointing sample per antenna) ---
        for a in range(n_ant):
            if len(point_t[a]) == 0:
                continue
            idx = _nearest_idx(point_t[a], ts)
            zenith_angle[i, a, :n_t] = point_za[a][idx].astype(np.float32)

        # --- SysPower: one scan-wide mask then iterate rows ---
        t0_ns = int(scan_t_start[sc] * 1e9)
        t1_ns = int(scan_t_end[sc] * 1e9)
        sc_mask = (sp_data["timeMid"] >= t0_ns) & (sp_data["timeMid"] <= t1_ns)
        sc_sp = sp_data[sc_mask]

        if sc_sp.shape[0] == 0:
            continue

        # build time → scan-local index map
        t_to_j: dict[float, int] = {float(t): j for j, t in enumerate(ts)}

        sp_times = sc_sp["timeMid"].astype(np.float64) / 1e9
        sp_ant_ids = sc_sp["antennaId"]
        sp_spw_ids = sc_sp["spectralWindowId"]
        sp_diff = sc_sp["switchedPowerDifference"]  # (n_rows, 2)
        sp_sum = sc_sp["switchedPowerSum"]  # (n_rows, 2)

        # exposure per scan-local time slot: SysPower interval is in ns
        if sp_interval_field is not None:
            sp_dur_s = sc_sp[sp_interval_field].astype(np.float64) / 1e9
            for j_t, t_val in enumerate(ts):
                mask_t = np.isclose(sp_times, t_val)
                if mask_t.any():
                    exposure_time[i, j_t] = float(np.nanmedian(sp_dur_s[mask_t]))
        if not np.isfinite(exposure_time[i, :n_t]).any() and n_t >= 2:
            dt = np.diff(ts)
            exposure_time[i, :n_t] = float(np.median(dt))

        for row in range(sc_sp.shape[0]):
            a = ant_id_to_idx.get(str(sp_ant_ids[row]))
            spw_int = spw_id_to_idx.get(str(sp_spw_ids[row]))
            if a is None or spw_int is None or spw_int not in sc_spw_set:
                continue
            if spw_int not in spw_to_idx:
                continue
            w = spw_to_idx[spw_int]
            j = t_to_j.get(float(sp_times[row]))
            if j is None:
                continue
            switched_diff[i, a, w, 0, j] = sp_diff[row, 0]
            switched_diff[i, a, w, 1, j] = sp_diff[row, 1]
            switched_sum[i, a, w, 0, j] = sp_sum[row, 0]
            switched_sum[i, a, w, 1, j] = sp_sum[row, 1]
            flag[i, a, w, :, j] = False

    # pad positions must stay flagged
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
            "zenith_angle": (("scan", "antenna", "time"), zenith_angle),
            "tcal_ref": (("antenna", "spw", "polarization"), tcal_ref),
            "weather_T": (("scan", "time"), weather_T),
            "weather_P": (("scan", "time"), weather_P),
            "weather_RH": (("scan", "time"), weather_RH),
            "exposure_time": (("scan", "time"), exposure_time),
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
            "frequency": (("spw",), spw_freq[tip_spws].astype(np.float64)),
            "bandwidth": (("spw",), spw_bw[tip_spws].astype(np.float64)),
            "band": (("spw",), spw_bands[tip_spws].astype("U4")),
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
            "source_format": "sdm",
            "observatory": "VLA",
        },
    )
