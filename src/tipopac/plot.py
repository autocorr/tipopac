"""Per-(scan, antenna, spw) tipping-curve diagnostic plots (DESIGN.md §9.3).

Public entry point:
  ``plot_dataset(ds, out_dir)``
  Writes one PNG per successful (scan, antenna, spw) fit under ``out_dir``.

Uses the matplotlib ``Figure`` class directly (no pyplot state machine) so
no display is required and no global backend change is made.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from tipopac.physics import k2nt, tsys_model, weighted_mean_atm_T

__all__ = ["plot_dataset"]

_log = logging.getLogger(__name__)

# Dense ZA grid for smooth model curves.
_Z_GRID: np.ndarray = np.linspace(30.0, 90.0, 300)


def plot_dataset(ds: xr.Dataset, out_dir: Path | str) -> None:
    """Write one PNG per successful (scan, antenna, spw) fit under ``out_dir``.

    Requires in *ds*: Tsys, tau_zenith, tau_err, T0, tcal_fit, tcal_ref,
    fit_success, zenith_angle, weather_T, flag.

    When ``tau_extrapolated`` is present an am cross-check panel is added
    below the main panel (DESIGN.md §9.3).

    The output directory is created with ``parents=True, exist_ok=True``.
    Skips cells where ``fit_success`` is False.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_success = ds["fit_success"].values  # (scan, ant, spw)
    if not fit_success.any():
        _log.warning("plot_dataset: no successful fits; nothing to plot")
        return

    scan_ids = ds.coords["scan"].values
    ant_names = ds.coords["antenna"].values
    spw_ids = ds.coords["spw"].values
    freq_vals = ds.coords["frequency"].values  # Hz (spw,)
    has_am = "tau_extrapolated" in ds.data_vars

    for i_scan in range(ds.sizes["scan"]):
        for i_ant in range(ds.sizes["antenna"]):
            for i_spw in range(ds.sizes["spw"]):
                if not fit_success[i_scan, i_ant, i_spw]:
                    continue
                _plot_cell(
                    ds,
                    i_scan,
                    i_ant,
                    i_spw,
                    scan_ids,
                    ant_names,
                    spw_ids,
                    freq_vals,
                    has_am,
                    out_dir,
                )


def _plot_cell(
    ds: xr.Dataset,
    i_scan: int,
    i_ant: int,
    i_spw: int,
    scan_ids: np.ndarray,
    ant_names: np.ndarray,
    spw_ids: np.ndarray,
    freq_vals: np.ndarray,
    has_am: bool,
    out_dir: Path,
) -> None:
    scan_id = int(scan_ids[i_scan])
    ant_name = str(ant_names[i_ant])
    spw_id = int(spw_ids[i_spw])
    freq_Hz = float(freq_vals[i_spw])

    # Observed quantities
    za = ds["zenith_angle"].values[i_scan, i_ant, :]           # (time,)
    tsys_R = ds["Tsys"].values[i_scan, i_ant, i_spw, 0, :]    # (time,)
    tsys_L = ds["Tsys"].values[i_scan, i_ant, i_spw, 1, :]    # (time,)
    flag = ds["flag"].values[i_scan, i_ant, i_spw, :, :]      # (pol, time)
    good = ~(flag[0] | flag[1]) & np.isfinite(tsys_R) & np.isfinite(tsys_L)

    # Fitted parameters
    tau0 = float(ds["tau_zenith"].values[i_scan, i_ant, i_spw])
    tau_err_val = float(ds["tau_err"].values[i_scan, i_ant, i_spw])
    T0_R = float(ds["T0"].values[i_scan, i_ant, i_spw, 0])
    T0_L = float(ds["T0"].values[i_scan, i_ant, i_spw, 1])
    tcal_fit_R = float(ds["tcal_fit"].values[i_scan, i_ant, i_spw, 0])
    tcal_fit_L = float(ds["tcal_fit"].values[i_scan, i_ant, i_spw, 1])
    tcal_ref_R = float(ds["tcal_ref"].values[i_ant, i_spw, 0])
    tcal_ref_L = float(ds["tcal_ref"].values[i_ant, i_spw, 1])

    # Tcal correction multipliers (=1 unless tcal_solve mode)
    c_R = tcal_fit_R / tcal_ref_R if tcal_ref_R > 0 else 1.0
    c_L = tcal_fit_L / tcal_ref_L if tcal_ref_L > 0 else 1.0

    # Mean-atmosphere noise temperature for this scan / spw
    T_surf_mean = float(np.nanmean(ds["weather_T"].values[i_scan, :]))
    Twmt = float(k2nt(weighted_mean_atm_T(T_surf_mean), freq_Hz))

    # Smooth fitted model curves (Tsys vs ZA)
    fit_R = tsys_model(_Z_GRID, T0_R, tau0, Twmt) / c_R
    fit_L = tsys_model(_Z_GRID, T0_L, tau0, Twmt) / c_L

    n_panels = 2 if has_am else 1
    fig = Figure(figsize=(7, 4 * n_panels))

    if has_am:
        ax_top, ax_bot = fig.subplots(2, 1)
    else:
        ax_top = fig.subplots()
        ax_bot = None

    # --- top panel: observed Tsys + fitted curve ---
    ax_top.scatter(za[good], tsys_R[good], color="steelblue", s=14, label="R pol", zorder=3)
    ax_top.scatter(za[good], tsys_L[good], color="seagreen", s=14, label="L pol", zorder=3)
    ax_top.plot(_Z_GRID, fit_R, color="tomato", lw=1.5, label="fit")
    ax_top.plot(_Z_GRID, fit_L, color="tomato", lw=1.5)
    ax_top.set_xlabel("Zenith angle (deg)")
    ax_top.set_ylabel("Tsys (K)")
    ax_top.set_xlim(30, 90)
    ax_top.legend(loc="upper left", fontsize=8)
    ax_top.set_title(
        f"{ant_name}  spw {spw_id}  scan {scan_id}\n"
        f"τ = {tau0:.3f} ± {tau_err_val:.3f}  |  "
        f"T0_R = {T0_R:.1f} K  T0_L = {T0_L:.1f} K",
        fontsize=9,
    )

    # --- bottom panel: am cross-check ---
    if ax_bot is not None:
        tau_am = float(ds["tau_extrapolated"].values[i_scan, i_spw])
        am_R = tsys_model(_Z_GRID, T0_R, tau_am, Twmt) / c_R
        am_L = tsys_model(_Z_GRID, T0_L, tau_am, Twmt) / c_L
        ax_bot.scatter(za[good], tsys_R[good], color="steelblue", s=14, label="R pol", zorder=3)
        ax_bot.scatter(za[good], tsys_L[good], color="seagreen", s=14, label="L pol", zorder=3)
        ax_bot.plot(_Z_GRID, am_R, color="darkorange", lw=1.5, label="am model")
        ax_bot.plot(_Z_GRID, am_L, color="darkorange", lw=1.5)
        ax_bot.set_xlabel("Zenith angle (deg)")
        ax_bot.set_ylabel("Tsys (K)")
        ax_bot.set_xlim(30, 90)
        ax_bot.legend(loc="upper left", fontsize=8)
        ax_bot.set_title(f"am cross-check  (τ_am = {tau_am:.3f})", fontsize=9)

    fig.tight_layout()
    fname = out_dir / f"tippingcurve_spw_{spw_id}_{ant_name}_scan_{scan_id}.png"
    fig.savefig(fname, dpi=100)
    _log.debug("wrote %s", fname)
