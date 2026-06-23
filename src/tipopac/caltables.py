"""CASA-format caltable writers for tipopac (DESIGN.md §9.2).

Public entry points:
  `write_opacity(ds, path)` — write a TOpac calibration table.
  `write_tcal(ds, path)`    — write a CALDEVICE-clone Tcal table.

Both require fit results to be present in `ds` (call `fit_dataset` first).
`write_tcal` additionally requires `tcal_fit` (mode='tcal_solve').
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

__all__ = ["write_opacity", "write_tcal"]

_log = logging.getLogger(__name__)

_CAL_LOAD_NAMES = np.array([["NOISE_TUBE_LOAD"], ["SOLAR_FILTER"]])


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def write_opacity(ds: xr.Dataset, path: str | Path) -> None:
    """Write a CASA TOpac opacity calibration table from *ds*.

    Requires: tau_zenith, tau_err, fit_success in ds.data_vars.
    Uses ds.attrs["source_path"] as the template MS for schema creation.
    """
    import casatools

    msname = ds.attrs["source_path"]
    path = str(path)

    cb = casatools.calibrater()
    cb.open(msname, False, False, False)
    cb.createcaltable(path, "Real", "TOpac", True)
    cb.close()

    rows = _build_opacity_rows(ds)

    tb = casatools.table()
    tb.open(path, nomodify=False)
    tb.addrows(len(rows))
    for k, row in enumerate(rows):
        for col, val in row.items():
            tb.putcell(col, k, val)
    tb.flush()
    tb.close()

    _log.info("wrote TOpac caltable (%d rows) to %s", len(rows), path)


def write_tcal(ds: xr.Dataset, path: str | Path) -> None:
    """Write a CALDEVICE-clone Tcal calibration table from *ds*.

    Requires: tcal_fit in ds.data_vars (only populated by mode='tcal_solve').
    Uses ds.attrs["source_path"] to copy the CALDEVICE schema.
    """
    if ds.attrs.get("mode") != "tcal_solve":
        raise ValueError(
            f"write_tcal requires mode='tcal_solve'; dataset has mode={ds.attrs.get('mode')!r}"
        )

    import casatools

    msname = ds.attrs["source_path"]
    path = str(path)

    tb = casatools.table()
    tb.open(f"{msname}/CALDEVICE")
    newtab = tb.copy(path, deep=True, valuecopy=True, norows=True, returnobject=True)
    tb.close()
    newtab.close()

    rows = _build_tcal_rows(ds)

    tb.open(path, nomodify=False)
    tb.addrows(len(rows))
    for k, row in enumerate(rows):
        for col, val in row.items():
            tb.putcell(col, k, val)
    tb.flush()
    tb.close()

    _log.info("wrote CALDEVICE Tcal table (%d rows) to %s", len(rows), path)


# ---------------------------------------------------------------------------
# Private row-building helpers (pure Python — testable without CASA)
# ---------------------------------------------------------------------------


def _iter_cells(ds: xr.Dataset) -> Iterator[tuple[int, int, int, int, int, float]]:
    """Yield ``(i, scan_num, a, s, spw_id, midtime)`` over (scan, antenna, spw)."""
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


def _build_opacity_rows(ds: xr.Dataset) -> list[dict[str, Any]]:
    """Return one TOpac row dict per (scan, antenna, spw) in that order."""
    tau = ds["tau_zenith"].values  # (scan, antenna, spw)
    tau_err = ds["tau_err"].values  # (scan, antenna, spw)
    success = ds["fit_success"].values  # (scan, antenna, spw)

    rows: list[dict[str, Any]] = []
    for i, scan_num, a, s, spw_id, midtime in _iter_cells(ds):
        ok = bool(success[i, a, s])
        tau_val = float(tau[i, a, s]) if ok else 0.0
        err_val = float(tau_err[i, a, s]) if ok else 0.0
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
