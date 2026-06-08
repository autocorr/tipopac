"""Diagnostic plots for tipopac (DESIGN §9.3).

Public surface:
  ``PlotData(ds)`` — wrapper around the canonical xarray.Dataset.
  ``PlotData.elevation_curve(scan, antenna, spw)`` — Tsys vs zenith angle.
  ``PlotData.tau_vs_frequency(scan)`` — fitted opacity vs frequency with am.
  ``PlotData.weather_panel()`` — surface T/P/RH across scans.
  ``PlotData.fit_success_heatmap()`` — pass/fail gestalt over all cells.
  ``PlotData.save_all(out_dir)`` — write every plot to disk as PDF+PNG+SVGZ.

Configures matplotlib for non-interactive use at import time; plot methods
return ``Figure`` objects so callers can save or display them as needed.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as _mp
import os as _os
import warnings
from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator

from tipopac import schema
from tipopac.physics import k2nt, tsys_model, weighted_mean_atm_T

__all__ = ["PlotData"]

_log = logging.getLogger(__name__)

# Dense ZA grid for smooth model curves.
_Z_GRID: np.ndarray = np.linspace(30.0, 75.0, 300)

plt.rc("text", usetex=False)
plt.rc("font", size=10, family="serif")
plt.rc("mathtext", fontset="dejavuserif")
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


def _save_figure(
    fig: Figure,
    path: Path,
    *,
    dpi: int = 300,
    h_pad: float = 0.3,
    w_pad: float | None = None,
    overwrite: bool = True,
) -> None:
    """Write *fig* to ``path.{pdf,png,svgz}`` then close it."""
    path.absolute().parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(h_pad=h_pad, w_pad=w_pad)
    skip = any(
        (path.with_suffix(f".{ext}").exists() and not overwrite)
        for ext in ("pdf", "png", "svgz")
    )
    if skip:
        _log.warning("Figure exists, skipping: %s", path)
    else:
        for ext in ("pdf", "png", "svgz"):
            fig.savefig(f"{path}.{ext}", dpi=dpi)
        _log.info("Figure saved: %s", path)
    plt.close(fig)


def _fix_minus_labels(ax, x=False, y=False):
    if x:
        ticks = ax.get_xticks().tolist()[1:-1]
        labels = ax.xaxis.get_ticklabels()[1:-1]
        ax.xaxis.set_ticks(ticks)
        ax.xaxis.set_ticklabels(
            [
                label.get_text().replace(r"\mathdefault", "")
                for label in labels
                if label.get_visible()
            ]
        )
    if y:
        ticks = ax.get_yticks().tolist()[1:-1]
        labels = ax.yaxis.get_ticklabels()[1:-1]
        ax.yaxis.set_ticks(ticks)
        ax.yaxis.set_ticklabels(
            [
                label.get_text().replace(r"\mathdefault", "")
                for label in labels
                if label.get_visible()
            ]
        )


def _annotate_with_patheffects(
    ax,
    label: str,
    xy: tuple[float, float] = (0.1, 0.1),
    xycoords: str = "axes fraction",
    linewidth: float = 2,
    color: str = "black",
    foreground: str = "white",
):
    anno = ax.annotate(label, xy=xy, xycoords=xycoords, color=color)
    anno.set_path_effects(
        [patheffects.withStroke(linewidth=linewidth, foreground=foreground)]
    )
    return anno


def _set_minor_ticks(
    ax, x: bool = True, y: bool = True, n_xticks=None, n_yticks=None
) -> None:
    if x:
        ax.xaxis.set_minor_locator(AutoMinorLocator(n_xticks))
    if y:
        ax.yaxis.set_minor_locator(AutoMinorLocator(n_yticks))


def _set_grid(ax) -> None:
    ax.grid(linestyle="dashed", color="0.3", linewidth=0.3)


def _get_fig(use_wide: bool = False) -> (Figure, plt.Axes):
    if use_wide:
        return plt.subplots(figsize=(8, 4.5))
    else:
        return plt.subplots(figsize=(4, 3.5))


# Matplotlib marker cycle used to distinguish scans when overlaying
# multiple scans on a single tcal_vs_frequency / c_vs_frequency figure.
_MARKER_CYCLE: tuple[str, ...] = ("o", "s", "^", "D", "v", "*", "X", "P")


def _scan_title(scans: list[int]) -> str:
    """Title prefix for one or more scan IDs: ``scan 3`` or ``scans 3, 5, 7``."""
    if len(scans) == 1:
        return f"scan {scans[0]}"
    return f"scans {', '.join(str(s) for s in scans)}"


# Matplotlib's pyplot figure registry (Gcf) is process-local, so each spawn
# worker has its own — plt.subplots / plt.close are safe across workers.
# Figure objects themselves never cross process boundaries: the worker
# builds the figure, saves it, and closes it; only the output stem (a str)
# is returned. The dataset is pickled once per worker via the pool
# initializer, not once per task.

_WORKER_PD: "PlotData | None" = None
_WORKER_OUT: Path | None = None


def _plot_pool_init(ds: xr.Dataset, out_dir: str) -> None:
    """Pool initializer: stash a worker-local PlotData and output dir."""
    global _WORKER_PD, _WORKER_OUT
    _WORKER_PD = PlotData(ds)
    _WORKER_OUT = Path(out_dir)


def _plot_worker(task: tuple[str, tuple, str]) -> str:
    """Pool task: render the named PlotData method and save."""
    method_name, args, stem = task
    pd = _WORKER_PD
    out = _WORKER_OUT
    if pd is None or out is None:
        raise RuntimeError("plot worker initializer did not run")
    fig = getattr(pd, method_name)(*args)
    _save_figure(fig, out / stem)
    return stem


class PlotData:
    """Wrap the canonical tipopac ``xr.Dataset`` for plotting.

    Each plot method returns a ``matplotlib.figure.Figure``; nothing is
    written to disk until :meth:`save_all` is called. The dataset is held
    by reference, not copied.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds.assign_coords(frequency_GHz=ds.frequency / 1e9)

    def validate_scans(self, scans) -> list[int]:
        if scans is None:
            return self.ds["scan"].values.tolist()
        else:
            return [int(s) for s in np.atleast_1d(scans)]

    def elevation_curve(self, scan: int, antenna: str, spw: int) -> Figure:
        """Tsys vs zenith angle for one ``(scan, antenna, spw)`` cell."""
        cell = self.ds.sel(scan=scan, antenna=antenna, spw=spw)

        za = cell["zenith_angle"].values
        tsys_R = cell["Tsys"].sel(polarization="R").values
        tsys_L = cell["Tsys"].sel(polarization="L").values
        flagged = cell["flag"].any(dim="polarization").values
        good = ~flagged & np.isfinite(tsys_R) & np.isfinite(tsys_L)

        tau0 = float(cell["tau_zenith"])
        tau_err_val = float(cell["tau_err"])
        T0_R = float(cell["T0"].sel(polarization="R"))
        T0_L = float(cell["T0"].sel(polarization="L"))
        tcal_fit_R = float(cell["tcal_fit"].sel(polarization="R"))
        tcal_fit_L = float(cell["tcal_fit"].sel(polarization="L"))
        tcal_ref_R = float(cell["tcal_ref"].sel(polarization="R"))
        tcal_ref_L = float(cell["tcal_ref"].sel(polarization="L"))

        c_R = tcal_fit_R / tcal_ref_R if tcal_ref_R > 0 else 1.0
        c_L = tcal_fit_L / tcal_ref_L if tcal_ref_L > 0 else 1.0

        freq_Hz = float(cell["frequency"])
        T_surf_mean = float(cell["weather_T"].mean(skipna=True))
        Twmt = float(k2nt(weighted_mean_atm_T(T_surf_mean), freq_Hz))

        fit_R = tsys_model(_Z_GRID, T0_R, tau0, Twmt) / c_R
        fit_L = tsys_model(_Z_GRID, T0_L, tau0, Twmt) / c_L

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.scatter(
            za[good], tsys_R[good], color="firebrick", s=4, label="R pol", zorder=3
        )
        ax.scatter(
            za[good], tsys_L[good], color="dodgerblue", s=4, label="L pol", zorder=3
        )
        ax.plot(_Z_GRID, fit_R, color="firebrick", lw=1.5, alpha=0.75)
        ax.plot(_Z_GRID, fit_L, color="dodgerblue", lw=1.5, alpha=0.75)
        _set_grid(ax)
        _set_minor_ticks(ax)
        ax.set_xlabel(r"Zenith angle [deg]")
        ax.set_ylabel(r"$T_\mathrm{sys} [\mathrm{K}]$")
        ax.set_xlim(_Z_GRID.min(), _Z_GRID.max())
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(
            f"{antenna}  spw {int(spw)}  scan {int(scan)} | "
            rf"$\tau = {tau0:.3f} \pm {tau_err_val:.3f}$ | "
            rf"$T_{{0, r}}$ = {T0_R:.1f} K  $T_{{0, l}}$ = {T0_L:.1f} K",
            fontsize=7,
        )
        return fig

    def tau_vs_frequency(self, scan_ids: int | list[int] | None = None) -> Figure:
        """
        Zenith opacity vs spw centre frequency for one or more scans.
        """
        scans = self.validate_scans(scan_ids)
        is_wide = len(scans) > 1
        ds = self.ds.sel(scan=scans)

        fig, ax = _get_fig(is_wide)

        ds.where(ds.fit_success).plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y="tau_zenith",
            marker=".",
            facecolor="0.5",
            edgecolor="none",
        )
        ds.where(~ds.fit_success).plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y="tau_zenith",
            marker=".",
            facecolor="orangered",
            edgecolor="none",
        )
        if "am_freq_grid" in ds.data_vars and "am_tau" in ds.data_vars:
            ax.plot(
                ds["am_freq_grid"].values / 1e9,
                ds["am_tau"].values,
                color="black",
                lw=1.5,
                label="am model",
                zorder=3,
            )
        weights = (1 / ds.tau_err**2).fillna(0)
        ds.tau_zenith.weighted(weights).mean(dim=["antenna"]).plot.scatter(
            ax=ax,
            x="frequency_GHz",
            marker=".",
            facecolor="firebrick",
            edgecolor="firebrick",
            linewidth=4,
            zorder=4,
        )

        ax.set_yscale("log")
        ax.set_ylim(ds["tau_zenith"].min() / 1.1, ds["tau_zenith"].max() * 1.1)
        _set_grid(ax)
        _set_minor_ticks(ax, y=False)
        ax.set_xlabel(r"Frequency [GHz]")
        ax.set_ylabel(r"$\tau_z$ [nepers]")
        return fig

    def tcal_vs_frequency(
        self, scan: int | list[int] | None = None, kind: str = "fit"
    ) -> Figure:
        """
        Fitted vs reference Tcal per polarization for one or more scans.
        Only meaningful for modes that solve for the Tcals.
        """
        scans = self.validate_scans(scan)
        is_wide = len(scans) > 1
        ds = self.ds.sel(scan=scans)

        fig, ax = _get_fig(is_wide)

        ds.plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y=f"tcal_{kind}",
            marker=".",
            facecolor="0.5",
            edgecolor="none",
            zorder=3,
        )
        ds.mean(dim=["polarization", "antenna"]).plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y=f"tcal_{kind}",
            marker=".",
            facecolor="firebrick",
            edgecolor="firebrick",
            linewidth=4,
            zorder=4,
        )

        _annotate_with_patheffects(ax, label=_scan_title(scans), xy=(0.02, 0.94))
        _set_grid(ax)
        _set_minor_ticks(ax)
        ax.set_xlabel(r"Frequency [GHz]")
        ax.set_ylabel(r"$T_\mathrm{cal}$ [K]")
        return fig

    def c_vs_frequency(self, scan: int | list[int] | None = None) -> Figure:
        """Tcal correction multiplier ``c = tcal_fit / tcal_ref`` vs frequency.

        ``c`` is the per-(antenna, spw, pol) correction the fit applies on
        top of the lab-measured ``tcal_ref`` (same factor used to scale the
        elevation-curve model in :meth:`elevation_curve`). Identically 1 in
        non-``tcal_solve`` modes by construction. ``scan`` accepts a single
        scan ID or a list — R/L pols by color (firebrick/dodgerblue),
        per-scan distinction by marker shape. Dashed reference line at
        ``c = 1``.
        """
        scans = self.validate_scans(scan)
        is_wide = len(scans) > 1
        ds = self.ds.sel(scan=scans)
        ds = ds.assign(c_ratio=(ds.tcal_fit / ds.tcal_ref))

        fig, ax = _get_fig(is_wide)
        fig, ax = plt.subplots(figsize=(5, 3.5))

        ax.axhline(1.0, color="0.5", lw=0.8, ls="--", zorder=1)
        ds.plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y="c_ratio",
            marker=".",
            facecolor="0.5",
            edgecolor="none",
            zorder=3,
        )
        ds.mean(dim=["polarization", "antenna"]).plot.scatter(
            ax=ax,
            x="frequency_GHz",
            y="c_ratio",
            marker=".",
            facecolor="firebrick",
            edgecolor="firebrick",
            linewidth=4,
            zorder=4,
        )

        _set_grid(ax)
        _set_minor_ticks(ax)
        ax.set_xlabel(r"Frequency [GHz]")
        ax.set_ylabel(r"$c = T_\mathrm{cal,fit} / T_\mathrm{cal,ref}$")
        ax.set_title(_scan_title(scans), fontsize=9)
        return fig

    def weather_panel(self) -> Figure:
        """Surface T, P, RH vs time across all scans (one figure)."""
        ds = self.ds

        fig, axes = plt.subplots(3, 1, figsize=(4, 5), sharex=True)

        ds.weather_T.plot.scatter(
            ax=axes[0],
            x="time_utc",
            add_legend=False,
        )
        ds.assign(weather_P_scaled=ds.weather_P / 100).plot.scatter(
            ax=axes[1],
            x="time_utc",
            y="weather_P_scaled",
            add_legend=False,
        )
        ds.assign(weather_RH_scaled=ds.weather_RH * 100).plot.scatter(
            ax=axes[2],
            x="time_utc",
            y="weather_RH_scaled",
            add_legend=False,
        )
        axes[0].set_ylabel("T [K]")
        axes[1].set_ylabel("P [hPa]")
        axes[2].set_ylabel("RH [%]")
        axes[2].set_xlabel("UTC Time")
        for ax in axes:
            _set_grid(ax)
            _set_minor_ticks(ax)
        return fig

    def fit_success_heatmap(self) -> Figure:
        """Pass/fail gestalt over every ``(scan, antenna, spw)`` cell.

        One subplot per scan in a ``ceil(n_scan / 3) × 3`` grid; each
        subplot is an ``(antenna × spw)`` imshow with success in green
        and failure in red.
        """
        ds = self.ds
        success = ds["fit_success"]  # (scan, ant, spw)
        n_scan = ds.sizes["scan"]
        n_col = min(3, n_scan)
        n_row = math.ceil(n_scan / n_col)
        ant_labels = [str(a) for a in ds.coords["antenna"].values]
        spw_labels = [str(int(s)) for s in ds.coords["spw"].values]

        fig, axes = plt.subplots(
            n_row,
            n_col,
            figsize=(n_col * 3.0, n_row * 2.6),
            squeeze=False,
        )
        cmap = ListedColormap(["firebrick", "forestgreen"])  # 0 fail, 1 ok
        for idx, scan_id in enumerate(ds.coords["scan"].values):
            r, c = divmod(idx, n_col)
            ax = axes[r, c]
            data = success.sel(scan=scan_id).astype(int).values  # (ant, spw)
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto", origin="lower")
            ax.set_xticks(range(len(spw_labels)))
            ax.set_xticklabels(spw_labels, fontsize=6)
            ax.set_yticks(range(len(ant_labels)))
            ax.set_yticklabels(ant_labels, fontsize=6)
            ax.set_title(f"scan {int(scan_id)}", fontsize=8)
            ax.set_xlabel("spw", fontsize=7)
        for j in range(n_scan, n_row * n_col):
            r, c = divmod(j, n_col)
            axes[r, c].axis("off")
        return fig

    def save_all(self, out_dir: str | Path, *, n_workers: int = 1) -> None:
        """Write every plot to ``out_dir`` as PDF + PNG + SVGZ.

        - One ``tippingcurve_spw_{spw}_{ant}_scan_{scan}`` per successful cell.
        - One ``tau_vs_frequency_scan_{scan}`` per scan with any successful fit.
        - One ``tcal_vs_frequency_scan_{scan}`` per scan, only when
          ``tcal_fit`` differs from ``tcal_ref`` (i.e. ``tcal_solve`` mode).
        - One ``c_vs_frequency_scan_{scan}`` per scan, under the same
          condition (correction multiplier ``tcal_fit / tcal_ref``).
        - One ``weather`` covering all scans.
        - One ``fit_success`` heatmap covering all (scan, ant, spw) cells.

        ``n_workers`` ≥ 2 dispatches via a :func:`multiprocessing.Pool` using
        the ``spawn`` start method (matches the fit-stage parallel pattern).
        Workers re-import this module and rebuild a fresh ``PlotData`` from a
        pickled copy of the dataset — pyplot's Gcf registry is process-local
        so ``plt.subplots`` / ``plt.close`` are safe to call concurrently.
        ``None`` or ``≤ 1`` runs serially.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        success = self.ds["fit_success"]
        if not bool(success.any()):
            _log.warning("save_all: no successful fits; weather/heatmap only")

        tasks = self._build_save_all_tasks(success)

        if n_workers <= 1 or len(tasks) <= 1:
            for method_name, args, stem in tasks:
                fig = getattr(self, method_name)(*args)
                _save_figure(fig, out / stem)
            return

        # Force Agg in the spawn workers' fresh interpreter; the parent's
        # already-loaded matplotlib is unaffected. Restored after the pool
        # tears down so we don't leak env mutation to the rest of the
        # process.
        prev_backend = _os.environ.get("MPLBACKEND")
        _os.environ["MPLBACKEND"] = "Agg"
        try:
            ctx = _mp.get_context("spawn")
            with ctx.Pool(
                processes=n_workers,
                initializer=_plot_pool_init,
                initargs=(self.ds, str(out)),
            ) as pool:
                chunksize = max(1, len(tasks) // (n_workers * 4))
                for _ in pool.imap_unordered(_plot_worker, tasks, chunksize=chunksize):
                    pass
        finally:
            if prev_backend is None:
                _os.environ.pop("MPLBACKEND", None)
            else:
                _os.environ["MPLBACKEND"] = prev_backend

    def _build_save_all_tasks(
        self, success: xr.DataArray
    ) -> list[tuple[str, tuple, str]]:
        """Return ``[(method_name, args, output_stem), ...]`` for save_all."""
        tasks: list[tuple[str, tuple, str]] = []

        # Per-cell elevation curves: stack the 3-D boolean to a 1-D MultiIndex,
        # then mask with the same array to recover only the True cells.
        cells = success.stack(cell=("scan", "antenna", "spw"))
        for scan_raw, ant_raw, spw_raw in cells.cell.values[cells.values]:
            scan_id, ant, spw_id = int(scan_raw), str(ant_raw), int(spw_raw)
            tasks.append(
                (
                    "elevation_curve",
                    (scan_id, ant, spw_id),
                    f"tippingcurve_spw_{spw_id}_{ant}_scan_{scan_id}",
                )
            )

        # Per-scan tasks: keep only scans with at least one successful cell.
        scans_with_fits = success.any(dim=("antenna", "spw"))
        scan_ids = [
            int(s) for s in success.coords["scan"].values[scans_with_fits.values]
        ]
        for scan_id in scan_ids:
            tasks.append(
                (
                    "tau_vs_frequency",
                    (scan_id,),
                    f"tau_vs_frequency_scan_{scan_id}",
                )
            )

        # Tcal plot only when fit and reference actually differ — in legacy
        # modes (tau_per_antenna, global_tau) tcal_fit is broadcast from
        # tcal_ref so the plot would be redundant.
        fit_b, ref_b = xr.broadcast(self.ds["tcal_fit"], self.ds["tcal_ref"])
        if not np.allclose(fit_b.values, ref_b.values, equal_nan=True):
            for scan_id in scan_ids:
                tasks.append(
                    (
                        "tcal_vs_frequency",
                        (scan_id,),
                        f"tcal_vs_frequency_scan_{scan_id}",
                    )
                )
                tasks.append(
                    (
                        "c_vs_frequency",
                        (scan_id,),
                        f"c_vs_frequency_scan_{scan_id}",
                    )
                )

        tasks.append(("weather_panel", (), "weather"))
        tasks.append(("fit_success_heatmap", (), "fit_success"))
        return tasks
