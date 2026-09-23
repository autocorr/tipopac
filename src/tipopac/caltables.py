"""CASA-format caltable writers for tipopac (design.md §9.2).

Public entry points:
  `write_opacity(ds, path)` — write a TOpac calibration table.
  `write_tcal(ds, path)`    — write a CALDEVICE-clone Tcal table.

Both require fit results to be present in `ds` (call `fit_dataset` first).
`write_tcal` additionally requires `tcal_fit`; cells with no fitted value fall
back to `tcal_ref`, since CALDEVICE has no FLAG column to mark them.

MS input clones the on-disk schema from the MS; SDM input builds it from the
frozen descriptions in `_caltable_schema` and synthesizes the subtables from
`caltable_meta`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from tipopac import caltable_meta
from tipopac._caltable_schema import TOPAC_INFO, table_desc
from tipopac._casa import import_casatools
from tipopac.schema import antenna_weighted_tau

__all__ = ["write_opacity", "write_tcal"]

_log = logging.getLogger(__name__)

_CAL_LOAD_NAMES = np.array([["NOISE_TUBE_LOAD"], ["SOLAR_FILTER"]])


def _require_vars(ds: xr.Dataset, names: tuple[str, ...], caller: str) -> None:
    """Raise before any table is created if *ds* lacks a required var."""
    missing = [v for v in names if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{caller} requires {missing!r} on the dataset; run fit first")


def _is_sdm(ds: xr.Dataset) -> bool:
    """Whether the dataset was read from an SDM rather than an MS."""
    return ds.attrs.get("source_format") == "sdm"


def _create_table(path: Path, desc: dict[str, Any], info: dict[str, str] | None) -> Any:
    """Create an empty casacore table at *path* and return it open."""
    casatools = import_casatools()
    tb = casatools.table()
    if not tb.create(str(path), tabledesc=desc):
        raise RuntimeError(f"could not create table at {path}")
    if info is not None:
        tb.putinfo(info)
    return tb


def _put_rows(tb: Any, rows: Sequence[dict[str, Any]]) -> None:
    """Append *rows* to an open table, one `putcell` per column."""
    if not rows:
        return
    start = tb.nrows()
    tb.addrows(len(rows))
    for k, row in enumerate(rows, start=start):
        for col, val in row.items():
            tb.putcell(col, k, val)


def _antenna_rows(meta: caltable_meta.CaltableMeta) -> list[dict[str, Any]]:
    """ANTENNA subtable rows, one per antenna in source order."""
    return [
        {
            "NAME": str(meta.ant_name[a]),
            "STATION": str(meta.ant_station[a]),
            "TYPE": caltable_meta.ANTENNA_TYPE,
            "MOUNT": caltable_meta.ANTENNA_MOUNT,
            "POSITION": meta.ant_position[a],
            "OFFSET": meta.ant_offset[a],
            "DISH_DIAMETER": float(meta.ant_dish_diameter[a]),
            "FLAG_ROW": False,
        }
        for a in range(meta.n_antenna)
    ]


def _spectral_window_rows(meta: caltable_meta.CaltableMeta) -> list[dict[str, Any]]:
    """SPECTRAL_WINDOW rows; the row index is the spw id.

    The channel axis is collapsed to the single band-centre channel that
    `createcaltable` writes for a TOpac solution, not the source's spectral
    channels, since `FPARAM` carries one opacity per spw.
    """
    rows = []
    for s in range(meta.n_spw):
        bandwidth = float(meta.spw_total_bandwidth[s])
        centre = float(meta.spw_ref_frequency[s]) + bandwidth / 2.0
        rows.append(
            {
                "NAME": str(meta.spw_name[s]),
                "NUM_CHAN": 1,
                "REF_FREQUENCY": float(meta.spw_ref_frequency[s]),
                "CHAN_FREQ": np.array([centre]),
                "CHAN_WIDTH": np.array([bandwidth]),
                "EFFECTIVE_BW": np.array([bandwidth]),
                "RESOLUTION": np.array([bandwidth]),
                "TOTAL_BANDWIDTH": bandwidth,
                "MEAS_FREQ_REF": caltable_meta.MEAS_FREQ_REF,
                "NET_SIDEBAND": int(meta.spw_net_sideband[s]),
                "FREQ_GROUP": 0,
                "FREQ_GROUP_NAME": "",
                "IF_CONV_CHAIN": 0,
                "FLAG_ROW": False,
            }
        )
    return rows


def _field_rows() -> list[dict[str, Any]]:
    """One placeholder FIELD row; every main-table row carries `FIELD_ID = -1`."""
    zero = np.zeros((1, 2), dtype=np.float64)
    return [
        {
            "NAME": "",
            "CODE": "",
            "SOURCE_ID": -1,
            "NUM_POLY": 0,
            "TIME": 0.0,
            "DELAY_DIR": zero,
            "PHASE_DIR": zero,
            "REFERENCE_DIR": zero,
            "FLAG_ROW": False,
        }
    ]


def _observation_rows(
    ds: xr.Dataset, meta: caltable_meta.CaltableMeta
) -> list[dict[str, Any]]:
    """One OBSERVATION row spanning the scans the caltable covers."""
    t0 = float(np.nanmin(ds.coords["scan_time_start"].values))
    t1 = float(np.nanmax(ds.coords["scan_time_end"].values))
    return [
        {
            "TELESCOPE_NAME": meta.telescope or str(ds.attrs.get("observatory", "")),
            "OBSERVER": "",
            "PROJECT": "",
            "TIME_RANGE": np.array([t0, t1], dtype=np.float64),
            "RELEASE_DATE": 0.0,
            "SCHEDULE_TYPE": "",
            "FLAG_ROW": False,
        }
    ]


def _create_topac(path: Path, ds: xr.Dataset, meta: caltable_meta.CaltableMeta) -> Any:
    """Create an empty TOpac table with its five subtables and keywords."""
    tb = _create_table(path, table_desc("MAIN"), TOPAC_INFO)
    subtables = {
        "ANTENNA": _antenna_rows(meta),
        "SPECTRAL_WINDOW": _spectral_window_rows(meta),
        "FIELD": _field_rows(),
        "OBSERVATION": _observation_rows(ds, meta),
        "HISTORY": [],
    }
    for name, rows in subtables.items():
        sub_path = path / name
        sub = _create_table(sub_path, table_desc(name), None)
        _put_rows(sub, rows)
        sub.flush()
        sub.close()
        tb.putkeyword(name, f"Table: {sub_path}")

    tb.putkeyword("MSName", meta.source_name)
    tb.putkeyword("VisCal", "TOpac")
    tb.putkeyword("ParType", "Float")
    tb.putkeyword("PolBasis", "unknown")
    tb.putkeyword("CASA_Version", import_casatools().version_string())
    return tb


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def write_opacity(ds: xr.Dataset, path: str | Path) -> None:
    """Write a CASA TOpac opacity calibration table from *ds*.

    Requires: tau_zenith, tau_err, fit_success in ds.data_vars.
    Schema comes from the template MS, or is synthesized for SDM input.
    """
    _require_vars(ds, ("tau_zenith", "tau_err", "fit_success"), "write_opacity")

    source = ds.attrs["source_path"]
    rows = _build_opacity_rows(ds)

    if _is_sdm(ds):
        meta = caltable_meta.from_sdm(source)
        tb = _create_topac(Path(path), ds, meta)
        _put_rows(tb, rows)
        tb.flush()
        tb.close()
    else:
        casatools = import_casatools()
        cb = casatools.calibrater()
        cb.open(source, False, False, False)
        cb.createcaltable(str(path), "Real", "TOpac", True)
        cb.close()

        tb = casatools.table()
        tb.open(str(path), nomodify=False)
        _put_rows(tb, rows)
        tb.flush()
        tb.close()

    _log.info("wrote TOpac caltable (%d rows) to %s", len(rows), path)


def write_tcal(ds: xr.Dataset, path: str | Path) -> None:
    """Write a CALDEVICE-clone Tcal calibration table from *ds*.

    Requires: tcal_fit and tcal_ref in ds.data_vars.
    Schema is copied from the MS CALDEVICE, or synthesized for SDM input.
    """
    _require_vars(ds, ("tcal_fit", "tcal_ref"), "write_tcal")

    rows = _build_tcal_rows(ds)

    if _is_sdm(ds):
        tb = _create_table(Path(path), table_desc("CALDEVICE"), None)
    else:
        casatools = import_casatools()
        msname = ds.attrs["source_path"]
        tb = casatools.table()
        tb.open(f"{msname}/CALDEVICE")
        newtab = tb.copy(
            str(path), deep=True, valuecopy=True, norows=True, returnobject=True
        )
        tb.close()
        newtab.close()
        tb.open(str(path), nomodify=False)

    _put_rows(tb, rows)
    tb.flush()
    tb.close()

    _log.info("wrote CALDEVICE Tcal table (%d rows) to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Private row-building helpers (pure Python — testable without CASA)
# ---------------------------------------------------------------------------


def _iter_cells(ds: xr.Dataset) -> Iterator[tuple[int, int, int, int, int, float]]:
    """Yield ``(i, scan_num, a, s, spw_id, midtime)`` over (scan, antenna, spw).

    CALDEVICE order — kept for v2.6 output-format parity (design.md §9.2).
    """
    scan_vals = ds.coords["scan"].values
    spw_vals = ds.coords["spw"].values
    t_start = ds.coords["scan_time_start"].values
    t_end = ds.coords["scan_time_end"].values
    n_ant = ds.sizes["antenna"]

    for i, (scan_num, t0, t1) in enumerate(zip(scan_vals, t_start, t_end)):
        midtime = float((t0 + t1) / 2.0)
        for a in range(n_ant):
            for s, spw_id in enumerate(spw_vals):
                yield i, int(scan_num), a, s, int(spw_id), midtime


def _iter_opacity_cells(
    ds: xr.Dataset,
) -> Iterator[tuple[int, int, int, int, int, float]]:
    """Yield ``(i, scan_num, a, s, spw_id, midtime)`` over (scan, spw, antenna).

    spw slow, antenna fast within a scan — the ordering `gencal` produces for
    ``caltype='opac'``, where spw is the enumerated selection axis.
    """
    scan_vals = ds.coords["scan"].values
    spw_vals = ds.coords["spw"].values
    t_start = ds.coords["scan_time_start"].values
    t_end = ds.coords["scan_time_end"].values
    n_ant = ds.sizes["antenna"]

    for i, (scan_num, t0, t1) in enumerate(zip(scan_vals, t_start, t_end)):
        midtime = float((t0 + t1) / 2.0)
        for s, spw_id in enumerate(spw_vals):
            for a in range(n_ant):
                yield i, int(scan_num), a, s, int(spw_id), midtime


def _build_opacity_rows(ds: xr.Dataset) -> list[dict[str, Any]]:
    """Return one TOpac row dict per (scan, spw, antenna) in that order.

    ``FPARAM`` is the antenna-weighted mean τ, so every antenna in a
    ``(scan, spw)`` carries the same value. Opacity is a property of the sky,
    not of an antenna, and per-antenna tipping τ is noisy; the per-antenna
    values stay on the dataset and in the plots. `FLAG` is therefore set per
    ``(scan, spw)`` — flagging one antenna's row would make `applycal` drop
    that antenna's data outright.

    Only ``fit_success`` cells contribute. `measured_opacity_table` reduces the
    same way but over every cell carrying a value, so its τ also reflects
    ``poorly_identified`` fits; what gets applied to data stays stricter.
    """
    ok = ds["fit_success"]
    tau_mean, err_mean = antenna_weighted_tau(
        ds.assign(
            tau_zenith=ds["tau_zenith"].where(ok),
            tau_err=ds["tau_err"].where(ok),
        )
    )
    tau = tau_mean.values  # (scan, spw)
    err = err_mean.values  # (scan, spw)

    rows: list[dict[str, Any]] = []
    for i, scan_num, a, s, spw_id, midtime in _iter_opacity_cells(ds):
        ok = bool(np.isfinite(tau[i, s]) and np.isfinite(err[i, s]))
        tau_val = float(tau[i, s]) if ok else 0.0
        err_val = float(err[i, s]) if ok else 0.0
        snr_val = float(abs(tau_val) / err_val) if (ok and err_val > 0.0) else 1.0
        rows.append(
            {
                "TIME": midtime,
                "FIELD_ID": -1,
                "SPECTRAL_WINDOW_ID": spw_id,
                "ANTENNA1": a,
                "ANTENNA2": -1,
                "SCAN_NUMBER": scan_num,
                "FPARAM": np.array([[tau_val]]),
                "PARAMERR": np.array([[err_val]]),
                "FLAG": np.array([[not ok]], dtype=bool),
                "SNR": np.array([[snr_val]]),
            }
        )
    return rows


def _build_tcal_rows(ds: xr.Dataset) -> list[dict[str, Any]]:
    """Return one CALDEVICE row dict per (scan, antenna, spw) in that order."""
    # tcal_fit: (scan, antenna, spw, polarization) with polarization = [R, L]
    tcal = ds["tcal_fit"].values
    ref = ds["tcal_ref"].values  # (antenna, spw, polarization)
    tcal = np.where(np.isfinite(tcal), tcal, ref[None, ...])

    rows: list[dict[str, Any]] = []
    for i, _scan_num, a, s, spw_id, midtime in _iter_cells(ds):
        tcal_R = float(tcal[i, a, s, 0])
        tcal_L = float(tcal[i, a, s, 1])
        # Row 0: fitted noise-tube Tcal values; row 1: solar-filter slot (zeroed).
        noise_cal = np.array([[tcal_R, tcal_L], [0.0, 0.0]])
        rows.append(
            {
                "ANTENNA_ID": a,
                "SPECTRAL_WINDOW_ID": spw_id,
                "TIME": midtime,
                "NUM_CAL_LOAD": 2,
                "CAL_LOAD_NAMES": _CAL_LOAD_NAMES,
                "NUM_RECEPTOR": 2,
                "NOISE_CAL": noise_cal,
            }
        )
    return rows
