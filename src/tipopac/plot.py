"""Per-(scan, antenna, spw) tipping-curve diagnostic plots (DESIGN.md §9.3).

Public entry point:
  ``plot_dataset(ds, out_dir)``
  Writes one PNG per successful (scan, antenna, spw) fit under ``out_dir``.

Uses the matplotlib ``Figure`` class directly (no pyplot state machine) so
no display is required and no global backend change is made.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import patheffects
from matplotlib.ticker import AutoMinorLocator

from tipopac.physics import k2nt, tsys_model, weighted_mean_atm_T

__all__ = ["plot_all_elevation_curves", "plot_elevation_curve"]

_log = logging.getLogger(__name__)

# Dense ZA grid for smooth model curves.
_Z_GRID: np.ndarray = np.linspace(30.0, 75.0, 300)

plt.rc("text", usetex=False)
plt.rc("font", size=10, family="cmu serif")
plt.rc("mathtext", fontset="cm")
plt.rc("xtick", direction="in", top=True)
plt.rc("ytick", direction="in", right=True)
plt.rc("axes", unicode_minus=False)
plt.ioff()

warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="This figure includes Axes that are not compatible with tight_layout.*",
)
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message="No artists with labels found to put in legend.*",
)


def savefig(
    path: Path, t_forecast=None, dpi=300, h_pad=0.3, w_pad=None, overwrite=True
):
    path.absolute().parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    if path.exists() and not overwrite:
        _log.warning(f"Figure exists, continuing: {path}")
    else:
        for ext in ("pdf", "png", "svgz"):
            filen = str(path) + f".{ext}"
            plt.savefig(filen, dpi=dpi)
        _log.info(f"Figure saved: {path}")
        plt.close("all")


def annotate_with_patheffects(
    ax,
    label,
    xy=(0.1, 0.1),
    xycoords="axes fraction",
    linewidth=2,
    color="black",
    foreground="white",
):
    anno = ax.annotate(label, xy=xy, xycoords=xycoords, color=color)
    anno.set_path_effects(
        [
            patheffects.withStroke(linewidth=linewidth, foreground=foreground),
        ]
    )
    return anno


def set_minor_ticks(ax, x=True, y=True, n_xticks=None, n_yticks=None):
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_xticks))
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_yticks))


def set_grid(ax):
    ax.grid(linestyle="dashed", color="0.3", linewidth=0.3)


def plot_all_elevation_curves(ds: xr.Dataset, out_dir: Path | str) -> None:
    """Write one PNG per successful (scan, antenna, spw) fit under ``out_dir``.

    Requires in *ds*: Tsys, tau_zenith, tau_err, T0, tcal_fit, tcal_ref,
    fit_success, zenith_angle, weather_T, flag.

    When ``tau_extrapolated`` is present an am cross-check panel is added
    below the main panel (DESIGN.md §9.3).

    The output directory is created with ``parents=True, exist_ok=True``.
    Skips cells where ``fit_success`` is False.
    """
    fit_success = ds["fit_success"].values  # (scan, ant, spw)
    if not fit_success.any():
        _log.warning("plot_dataset: no successful fits; nothing to plot")
        return

    scan_ids = ds.coords["scan"].values
    ant_names = ds.coords["antenna"].values
    spw_ids = ds.coords["spw"].values
    freq_vals = ds.coords["frequency"].values  # Hz (spw,)

    for i_scan in range(ds.sizes["scan"]):
        for i_ant in range(ds.sizes["antenna"]):
            for i_spw in range(ds.sizes["spw"]):
                if not fit_success[i_scan, i_ant, i_spw]:
                    continue
                plot_elevation_curve(
                    ds,
                    i_scan,
                    i_ant,
                    i_spw,
                    scan_ids,
                    ant_names,
                    spw_ids,
                    freq_vals,
                    out_dir,
                )


def plot_elevation_curve(
    ds: xr.Dataset,
    i_scan: int,
    i_ant: int,
    i_spw: int,
    scan_ids: np.ndarray,
    ant_names: np.ndarray,
    spw_ids: np.ndarray,
    freq_vals: np.ndarray,
    out_dir: Path,
) -> None:
    scan_id = int(scan_ids[i_scan])
    ant_name = str(ant_names[i_ant])
    spw_id = int(spw_ids[i_spw])
    freq_Hz = float(freq_vals[i_spw])

    # Observed quantities
    za = ds["zenith_angle"].values[i_scan, i_ant, :]  # (time,)
    tsys_R = ds["Tsys"].values[i_scan, i_ant, i_spw, 0, :]  # (time,)
    tsys_L = ds["Tsys"].values[i_scan, i_ant, i_spw, 1, :]  # (time,)
    flag = ds["flag"].values[i_scan, i_ant, i_spw, :, :]  # (pol, time)
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

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(za[good], tsys_R[good], color="firebrick", s=4, label="R pol", zorder=3)
    ax.scatter(za[good], tsys_L[good], color="dodgerblue", s=4, label="L pol", zorder=3)
    ax.plot(_Z_GRID, fit_R, color="firebrick", lw=1.5, alpha=0.75)
    ax.plot(_Z_GRID, fit_L, color="dodgerblue", lw=1.5, alpha=0.75)
    set_grid(ax)
    set_minor_ticks(ax)
    ax.set_xlabel(r"Zenith angle [deg]")
    ax.set_ylabel(r"$T_\mathrm{sys} [\mathrm{K}]$")
    ax.set_xlim(_Z_GRID.min(), _Z_GRID.max())
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(
        f"{ant_name}  spw {spw_id}  scan {scan_id} | "
        rf"$\tau = {tau0:.3f} \pm {tau_err_val:.3f}$ | "
        rf"$T_{{0, r}}$ = {T0_R:.1f} K  $T_{{0, l}}$ = {T0_L:.1f} K",
        fontsize=7,
    )

    out_path = out_dir / f"tippingcurve_spw_{spw_id}_{ant_name}_scan_{scan_id}"
    savefig(out_path)
