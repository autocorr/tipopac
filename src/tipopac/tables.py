"""Row builders for the tabular opacity outputs (TSV files and weblog pages).

Both consumers — ``api._write_tsv`` and the ``plot`` HTML table pages —
build their rows here so the two renderings can never disagree.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tipopac.atmgrid import GRID_FREQ_MAX_HZ, GRID_FREQ_MIN_HZ
from tipopac.schema import antenna_weighted_tau

__all__ = ["am_tau_1d", "measured_opacity_table", "model_opacity_table"]

ModelRow = tuple[float, float]
MeasuredRow = tuple[int, int, str, float, float, float, float]

MODEL_COLUMNS: tuple[str, ...] = ("frequency_Hz", "tau_model")
MEASURED_COLUMNS: tuple[str, ...] = (
    "scan",
    "spw",
    "band",
    "frequency_Hz",
    "tau_measured",
    "tau_err",
    "tau_model",
)


def model_opacity_table(
    ds: xr.Dataset,
) -> tuple[tuple[str, ...], list[ModelRow]]:
    """Model τ(ν) on the uniform am grid, sliced to 1–51 GHz."""
    if "am_freq_grid" not in ds.data_vars or "am_tau" not in ds.data_vars:
        return MODEL_COLUMNS, []

    freq_Hz = np.asarray(ds["am_freq_grid"].values, dtype=np.float64)
    tau = am_tau_1d(ds)
    keep = (freq_Hz >= GRID_FREQ_MIN_HZ - 1.0) & (freq_Hz <= GRID_FREQ_MAX_HZ + 1.0)
    return MODEL_COLUMNS, [
        (float(nu), float(t)) for nu, t in zip(freq_Hz[keep], tau[keep], strict=True)
    ]


def measured_opacity_table(
    ds: xr.Dataset,
) -> tuple[tuple[str, ...], list[MeasuredRow]]:
    """Fitted and model τ at the spw centre frequencies, ascending in frequency.

    One row per ``(scan, spw)`` that has any successful fit. ``tau_measured``
    is the 1/σ² antenna-weighted mean of ``tau_zenith`` (the reduction the
    τ-vs-frequency plot draws) and ``tau_err`` its propagated error.
    ``tau_model`` is interpolated from ``am_tau``, which is sampled at the
    run's median fitted PWV — it is a run-level curve, so scans sharing an
    spw get the same value.
    """
    if "tau_zenith" not in ds.data_vars or "tau_err" not in ds.data_vars:
        return MEASURED_COLUMNS, []

    freq_Hz = np.asarray(ds["frequency"].values, dtype=np.float64)
    band = np.asarray(ds["band"].values, dtype=str)
    tau_mean, err_mean = antenna_weighted_tau(ds)

    model = _model_at(ds, freq_Hz)

    rows: list[MeasuredRow] = []
    for i_spw in np.argsort(freq_Hz, kind="stable"):
        for i_scan, scan in enumerate(ds["scan"].values):
            tau_val = float(tau_mean.values[i_scan, i_spw])
            if not np.isfinite(tau_val):
                continue
            rows.append(
                (
                    int(scan),
                    int(ds["spw"].values[i_spw]),
                    str(band[i_spw]),
                    float(freq_Hz[i_spw]),
                    tau_val,
                    float(err_mean.values[i_scan, i_spw]),
                    float(model[i_spw]),
                )
            )
    return MEASURED_COLUMNS, rows


def am_tau_1d(ds: xr.Dataset) -> np.ndarray:
    """The single group's am τ(ν) curve as a 1-D array.

    ``am_tau`` carries a ``group`` axis; the table and plot consumers each
    render one group at a time and must be handed a
    :func:`tipopac.schema.select_group` slice.
    """
    da = ds["am_tau"]
    if da.sizes.get("group", 1) != 1:
        raise ValueError(
            f"am_tau has {da.sizes['group']} groups; pass a select_group() slice"
        )
    return np.asarray(da.values, dtype=np.float64).reshape(-1)


def _model_at(ds: xr.Dataset, freq_Hz: np.ndarray) -> np.ndarray:
    """Model τ interpolated onto ``freq_Hz``; NaN when no am curve is present."""
    if "am_freq_grid" not in ds.data_vars or "am_tau" not in ds.data_vars:
        return np.full(freq_Hz.size, np.nan)
    grid = np.asarray(ds["am_freq_grid"].values, dtype=np.float64)
    tau = am_tau_1d(ds)
    model = np.interp(freq_Hz, grid, tau)
    return np.where((freq_Hz < grid[0]) | (freq_Hz > grid[-1]), np.nan, model)
