"""Interactive diagnostic plots for tipopac (DESIGN §9.3).

Each plot is a standalone vega-altair ``LayerChart`` that serialises to
one self-contained ``.html`` with hover tooltips disclosing the
``(scan, antenna, spw, polarization)`` identity of every point. Colour
encodes status (passed/flagged/weighted-mean), not identity.

Public surface:
  ``PlotData(ds)`` — wrapper around the canonical xarray.Dataset.
  ``PlotData.elevation_curve(scan, antenna, spw)``
  ``PlotData.tau_vs_frequency(scans=None)``
  ``PlotData.tcal_vs_frequency(scans=None, kind="fit")``
  ``PlotData.c_vs_frequency(scans=None)``
  ``PlotData.save_all(out_dir)`` — write every plot + ``index.html``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import xarray as xr

from tipopac.physics import k2nt, tsys_model, weighted_mean_atm_T

__all__ = [
    "CVsFrequency",
    "ElevationCurve",
    "Plot",
    "PlotData",
    "TauVsFrequency",
    "TcalVsFrequency",
]

_log = logging.getLogger(__name__)

# Embed all data inline in the HTML; default 5000-row cap is irrelevant
# for standalone diagnostic files that may have tens of thousands of
# tooltipped points.
alt.data_transformers.disable_max_rows()

# Dense ZA grid for smooth model curves in elevation_curve.
_Z_GRID: np.ndarray = np.linspace(30.0, 75.0, 300)


def _scan_title(scans: list[int]) -> str:
    """Title prefix: ``scan 3`` for one scan, ``scans 3, 5, 7`` for many."""
    if len(scans) == 1:
        return f"scan {scans[0]}"
    return f"scans {', '.join(str(s) for s in scans)}"


class Plot:
    """Base class. Subclasses implement :meth:`build`; :meth:`save` is shared."""

    # Colour palette — status, not identity (see module docstring).
    COLOR_GOOD = "gray"
    COLOR_FLAGGED = "orangered"
    COLOR_MEAN = "firebrick"
    COLOR_REF = "gray"
    COLOR_R_POL = "firebrick"
    COLOR_L_POL = "dodgerblue"
    COLOR_AM_MODEL = "black"

    WIDTH = 480
    HEIGHT = 320
    WIDTH_WIDE = 720
    POINT_SIZE = 16
    MEAN_POINT_SIZE = 64
    LINE_STROKE = 1.8

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    def build(self) -> alt.Chart | alt.LayerChart | alt.FacetChart:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Serialise the chart to ``path.html`` (standalone, inline data)."""
        path = Path(path).with_suffix(".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.build().save(str(path))
        _log.info("plot saved: %s", path)

    def _finalize(
        self,
        chart: alt.LayerChart | alt.FacetChart,
        *,
        title: str,
        width: int | None = None,
    ) -> alt.LayerChart | alt.FacetChart:
        return chart.properties(
            width=width if width is not None else self.WIDTH,
            height=self.HEIGHT,
            title=title,
        ).interactive()


def _validate_scans(ds: xr.Dataset, scans: int | list[int] | None) -> list[int]:
    if scans is None:
        return [int(s) for s in ds["scan"].values]
    return [int(s) for s in np.atleast_1d(scans)]


def _to_df(
    obj: xr.Dataset | xr.DataArray,
    *,
    name: str | None = None,
    dropna: str | list[str] | None = None,
) -> pd.DataFrame:
    """Convert an xarray object to a tidy DataFrame for Altair."""
    if isinstance(obj, xr.DataArray):
        obj = obj.to_dataset(name=name) if name is not None else obj.to_dataset()
    subset = dropna if dropna is not None else name
    if isinstance(subset, str):
        subset = [subset]
    return obj.to_dataframe().reset_index().dropna(subset=subset)


class _PerScanPlot(Plot):
    """Shared scaffolding for multi-scan plots."""

    def __init__(self, ds: xr.Dataset, scans: int | list[int] | None = None) -> None:
        super().__init__(ds)
        self.scans = _validate_scans(ds, scans)
        self.ds_sub = ds.sel(scan=self.scans)
        self.width = self.WIDTH_WIDE if len(self.scans) > 1 else self.WIDTH

    def _mean_layer(
        self,
        df: pd.DataFrame,
        *,
        value_col: str,
        y_title: str,
        value_fmt: str = ".4f",
    ) -> alt.Chart:
        return (
            alt.Chart(df)
            .mark_point(filled=True, size=self.MEAN_POINT_SIZE, color=self.COLOR_MEAN)
            .encode(
                x=alt.X("frequency_GHz:Q", title="Frequency [GHz]"),
                y=alt.Y(f"{value_col}:Q", title=y_title),
                tooltip=[
                    "scan:N",
                    "spw:N",
                    alt.Tooltip("frequency_GHz:Q", format=".3f"),
                    alt.Tooltip(f"{value_col}:Q", format=value_fmt),
                ],
            )
        )


class ElevationCurve(Plot):
    """Tsys vs zenith angle for one ``(scan, antenna, spw)`` cell.

    Two layered scatters (R/L polarisation) + two fitted model curves on a
    dense ZA grid. Hover tooltip carries (polarization, ZA, Tsys, UTC).
    """

    def __init__(self, ds: xr.Dataset, scan: int, antenna: str, spw: int) -> None:
        super().__init__(ds)
        self.scan = int(scan)
        self.antenna = str(antenna)
        self.spw = int(spw)

    def build(self) -> alt.LayerChart | alt.FacetChart:
        cell = self.ds.sel(scan=self.scan, antenna=self.antenna, spw=self.spw)

        good = ~cell["flag"].any(dim="polarization")
        tsys_masked = cell["Tsys"].where(good)
        df = _to_df(
            xr.Dataset({"Tsys": tsys_masked, "zenith_angle": cell["zenith_angle"]}),
            dropna=["Tsys", "zenith_angle"],
        )

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
        model_df = pd.DataFrame(
            {
                "zenith_angle": np.concatenate([_Z_GRID, _Z_GRID]),
                "Tsys": np.concatenate([fit_R, fit_L]),
                "polarization": ["R"] * _Z_GRID.size + ["L"] * _Z_GRID.size,
            }
        )

        pol_scale = alt.Scale(
            domain=["R", "L"], range=[self.COLOR_R_POL, self.COLOR_L_POL]
        )
        x_enc = alt.X(
            "zenith_angle:Q",
            title="Zenith angle [deg]",
            scale=alt.Scale(domain=[float(_Z_GRID.min()), float(_Z_GRID.max())]),
        )
        y_enc = alt.Y("Tsys:Q", title="T_sys [K]")

        scatter = (
            alt.Chart(df)
            .mark_point(filled=True, size=self.POINT_SIZE)
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color(
                    "polarization:N", scale=pol_scale, legend=alt.Legend(title=None)
                ),
                tooltip=[
                    "polarization:N",
                    alt.Tooltip("zenith_angle:Q", format=".2f"),
                    alt.Tooltip("Tsys:Q", format=".2f"),
                    alt.Tooltip("time_utc:Q", format=".1f"),
                ],
            )
        )
        model = (
            alt.Chart(model_df)
            .mark_line(strokeWidth=self.LINE_STROKE, opacity=0.75)
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color("polarization:N", scale=pol_scale, legend=None),
            )
        )

        title = (
            f"{self.antenna}  spw {self.spw}  scan {self.scan} | "
            f"τ = {tau0:.3f} ± {tau_err_val:.3f} | "
            f"T_0,R = {T0_R:.1f} K  T_0,L = {T0_L:.1f} K"
        )
        return self._finalize(scatter + model, title=title)


class TauVsFrequency(_PerScanPlot):
    """Zenith opacity vs spw centre frequency.

    Per-sample scatter (gray=passed, orangered=failed-fit) + antenna-weighted
    mean per spw (firebrick) + optional AM model line (black). Log y-axis.
    Hover discloses (scan, antenna, spw, frequency, τ, σ, fit_success).
    """

    def build(self) -> alt.LayerChart | alt.FacetChart:
        ds_sub = self.ds_sub
        y_title = "τ_z [nepers]"

        df = _to_df(
            ds_sub[["tau_zenith", "tau_err", "fit_success"]], dropna="tau_zenith"
        )

        # Weighted mean per spw across antennas. Keep scan in the dims so each
        # scan*spw combination shows as one mean point.
        weights = (1.0 / ds_sub["tau_err"] ** 2).fillna(0.0)
        mean_da = ds_sub["tau_zenith"].weighted(weights).mean(dim="antenna")
        mean_df = _to_df(mean_da, name="mean_tau")

        # Domain from the full tau spread (NaN-safe via xarray).
        tau_min = float(ds_sub["tau_zenith"].min(skipna=True))
        tau_max = float(ds_sub["tau_zenith"].max(skipna=True))
        # Guard against non-positive values that would break the log axis.
        tau_min = max(tau_min, 1e-4)
        y_domain = [tau_min / 1.1, tau_max * 1.1]

        x_enc = alt.X("frequency_GHz:Q", title="Frequency [GHz]")
        y_enc = alt.Y(
            "tau_zenith:Q",
            title=y_title,
            scale=alt.Scale(type="log", domain=y_domain),
        )

        status_scale = alt.Scale(
            domain=[True, False], range=[self.COLOR_GOOD, self.COLOR_FLAGGED]
        )
        samples = (
            alt.Chart(df)
            .mark_point(filled=True, size=self.POINT_SIZE)
            .encode(
                x=x_enc,
                y=y_enc,
                color=alt.Color("fit_success:N", scale=status_scale, legend=None),
                tooltip=[
                    "scan:N",
                    "antenna:N",
                    "spw:N",
                    alt.Tooltip("frequency_GHz:Q", format=".3f"),
                    alt.Tooltip("tau_zenith:Q", format=".4f"),
                    alt.Tooltip("tau_err:Q", format=".4f"),
                    "fit_success:N",
                ],
            )
        )

        mean = self._mean_layer(mean_df, value_col="mean_tau", y_title=y_title)

        layers: list[alt.Chart] = [samples, mean]
        if "am_freq_grid" in ds_sub.data_vars and "am_tau" in ds_sub.data_vars:
            am_df = pd.DataFrame(
                {
                    "frequency_GHz": ds_sub["am_freq_grid"].values / 1e9,
                    "am_tau": ds_sub["am_tau"].values,
                }
            )
            am_line = (
                alt.Chart(am_df, title="am model")
                .mark_line(color=self.COLOR_AM_MODEL, strokeWidth=self.LINE_STROKE)
                .encode(
                    x=x_enc,
                    y=alt.Y("am_tau:Q", title=y_title),
                    tooltip=[
                        alt.Tooltip("frequency_GHz:Q", format=".3f"),
                        alt.Tooltip("am_tau:Q", format=".4f"),
                    ],
                )
            )
            layers.append(am_line)

        return self._finalize(
            alt.layer(*layers), title=_scan_title(self.scans), width=self.width
        )


class TcalVsFrequency(_PerScanPlot):
    """Fitted Tcal vs frequency, with per-pol/antenna scatter + summary mean."""

    def __init__(
        self,
        ds: xr.Dataset,
        scans: int | list[int] | None = None,
        kind: str = "fit",
    ) -> None:
        super().__init__(ds, scans)
        self.kind = str(kind)

    def build(self) -> alt.LayerChart | alt.FacetChart:
        ds_sub = self.ds_sub
        y_title = "T_cal [K]"

        col = f"tcal_{self.kind}"
        df = _to_df(ds_sub[["tcal_fit", "tcal_ref"]], dropna=col)
        mean_da = ds_sub[col].mean(dim=["polarization", "antenna"])
        mean_df = _to_df(mean_da, name="mean_tcal")

        samples = (
            alt.Chart(df)
            .mark_point(filled=True, size=self.POINT_SIZE, color=self.COLOR_GOOD)
            .encode(
                x=alt.X("frequency_GHz:Q", title="Frequency [GHz]"),
                y=alt.Y(f"{col}:Q", title=y_title),
                tooltip=[
                    "scan:N",
                    "antenna:N",
                    "spw:N",
                    "polarization:N",
                    alt.Tooltip("frequency_GHz:Q", format=".3f"),
                    alt.Tooltip("tcal_fit:Q", format=".3f"),
                    alt.Tooltip("tcal_ref:Q", format=".3f"),
                ],
            )
        )
        mean = self._mean_layer(
            mean_df, value_col="mean_tcal", y_title=y_title, value_fmt=".3f"
        )

        return self._finalize(
            alt.layer(samples, mean),
            title=_scan_title(self.scans),
            width=self.width,
        )


class CVsFrequency(_PerScanPlot):
    """Tcal correction multiplier c = tcal_fit / tcal_ref vs frequency.

    Dashed reference line at c=1 + per-(antenna, spw, pol) gray scatter +
    polarisation/antenna-averaged firebrick scatter.
    """

    def build(self) -> alt.LayerChart | alt.FacetChart:
        ds_sub = self.ds_sub
        y_title = "c = T_cal,fit / T_cal,ref"

        c_da = ds_sub["tcal_fit"] / ds_sub["tcal_ref"]
        df = _to_df(c_da, name="c_ratio")
        mean_da = c_da.mean(dim=["polarization", "antenna"])
        mean_df = _to_df(mean_da, name="mean_c")

        ref = (
            alt.Chart(pd.DataFrame({"c": [1.0]}))
            .mark_rule(color=self.COLOR_REF, strokeDash=[4, 2], strokeWidth=0.8)
            .encode(y=alt.Y("c:Q", title=y_title))
        )
        samples = (
            alt.Chart(df)
            .mark_point(filled=True, size=self.POINT_SIZE, color=self.COLOR_GOOD)
            .encode(
                x=alt.X("frequency_GHz:Q", title="Frequency [GHz]"),
                y=alt.Y("c_ratio:Q", title=y_title),
                tooltip=[
                    "scan:N",
                    "antenna:N",
                    "spw:N",
                    "polarization:N",
                    alt.Tooltip("frequency_GHz:Q", format=".3f"),
                    alt.Tooltip("c_ratio:Q", format=".4f"),
                ],
            )
        )
        mean = self._mean_layer(mean_df, value_col="mean_c", y_title=y_title)

        return self._finalize(
            alt.layer(ref, samples, mean),
            title=_scan_title(self.scans),
            width=self.width,
        )


# Sections in the index page; insertion order = display order.
_INDEX_SECTIONS: tuple[tuple[str, str], ...] = (
    ("tippingcurve_", "Elevation curves"),
    ("tau_vs_frequency_", "τ vs frequency"),
    ("tcal_vs_frequency_", "T_cal vs frequency"),
    ("c_vs_frequency_", "c = T_cal,fit / T_cal,ref"),
)


def _write_index_html(out_dir: Path, entries: list[tuple[str, str]]) -> None:
    """Write ``out_dir/index.html`` linking every per-plot file grouped by section.

    ``entries`` is ``[(section_title, filename), ...]``.
    """
    by_section: dict[str, list[str]] = {title: [] for _, title in _INDEX_SECTIONS}
    for section, filename in entries:
        by_section.setdefault(section, []).append(filename)

    parts = [
        "<!doctype html>",
        '<html lang="en"><head>',
        '<meta charset="utf-8">',
        "<title>tipopac plots</title>",
        "<style>",
        "body { font-family: sans-serif; margin: 2em; max-width: 60em; }",
        "h1 { border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }",
        "h2 { margin-top: 1.5em; }",
        "ul { line-height: 1.5; }",
        "</style>",
        "</head><body>",
        "<h1>tipopac diagnostic plots</h1>",
    ]
    for section_title, files in by_section.items():
        if not files:
            continue
        parts.append(f"<h2>{section_title}</h2>")
        parts.append("<ul>")
        for fn in sorted(files):
            label = Path(fn).stem
            parts.append(f'<li><a href="{fn}">{label}</a></li>')
        parts.append("</ul>")
    parts.append("</body></html>")

    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


class PlotData:
    """Wrap the canonical tipopac dataset and dispatch the four plot types.

    Convenience methods (``elevation_curve`` etc.) return ``alt.LayerChart``
    objects so callers can inspect or render them; :meth:`save_all` writes
    every applicable plot to ``out_dir`` as ``.html`` plus a top-level
    ``index.html`` linking them.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds.assign_coords(frequency_GHz=ds.frequency / 1e9)

    def elevation_curve(
        self, scan: int, antenna: str, spw: int
    ) -> alt.LayerChart | alt.FacetChart:
        return ElevationCurve(self.ds, scan, antenna, spw).build()

    def tau_vs_frequency(
        self, scans: int | list[int] | None = None
    ) -> alt.LayerChart | alt.FacetChart:
        return TauVsFrequency(self.ds, scans).build()

    def tcal_vs_frequency(
        self, scans: int | list[int] | None = None, kind: str = "fit"
    ) -> alt.LayerChart | alt.FacetChart:
        return TcalVsFrequency(self.ds, scans, kind).build()

    def c_vs_frequency(
        self, scans: int | list[int] | None = None
    ) -> alt.LayerChart | alt.FacetChart:
        return CVsFrequency(self.ds, scans).build()

    def save_all(self, out_dir: str | Path) -> None:
        """Write every applicable plot to ``out_dir`` as ``.html``.

        - One ``tippingcurve_spw_{spw}_{ant}_scan_{scan}`` per successful cell.
        - One ``tau_vs_frequency_scan_{scan}`` per scan with any successful fit.
        - One ``tcal_vs_frequency_scan_{scan}`` per scan when ``tcal_fit`` and
          ``tcal_ref`` actually differ (i.e. ``independent_tau_solve`` mode).
        - One ``c_vs_frequency_scan_{scan}`` per scan under the same condition.
        - One ``index.html`` linking everything.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        success = self.ds["fit_success"]
        if not bool(success.any()):
            _log.warning(
                "save_all: no successful fits; only index.html will be written"
            )

        entries: list[tuple[str, str]] = []

        # Per-cell elevation curves.
        cells = success.stack(cell=("scan", "antenna", "spw"))
        for scan_raw, ant_raw, spw_raw in cells.cell.values[cells.values]:
            scan_id, ant, spw_id = int(scan_raw), str(ant_raw), int(spw_raw)
            stem = f"tippingcurve_spw_{spw_id}_{ant}_scan_{scan_id}"
            ElevationCurve(self.ds, scan_id, ant, spw_id).save(out / stem)
            entries.append(("Elevation curves", f"{stem}.html"))

        # Per-scan plots: only scans with at least one successful cell.
        scans_with_fits = success.any(dim=("antenna", "spw"))
        scan_ids = [
            int(s) for s in success.coords["scan"].values[scans_with_fits.values]
        ]
        for scan_id in scan_ids:
            stem = f"tau_vs_frequency_scan_{scan_id}"
            TauVsFrequency(self.ds, [scan_id]).save(out / stem)
            entries.append(("τ vs frequency", f"{stem}.html"))

        # Tcal / c plots only when fit and reference actually differ — in
        # tau_per_antenna mode tcal_fit broadcasts from tcal_ref so the
        # plots would be redundant.
        fit_b, ref_b = xr.broadcast(self.ds["tcal_fit"], self.ds["tcal_ref"])
        if not np.allclose(fit_b.values, ref_b.values, equal_nan=True):
            for scan_id in scan_ids:
                tcal_stem = f"tcal_vs_frequency_scan_{scan_id}"
                TcalVsFrequency(self.ds, [scan_id]).save(out / tcal_stem)
                entries.append(("T_cal vs frequency", f"{tcal_stem}.html"))

                c_stem = f"c_vs_frequency_scan_{scan_id}"
                CVsFrequency(self.ds, [scan_id]).save(out / c_stem)
                entries.append(("c = T_cal,fit / T_cal,ref", f"{c_stem}.html"))

        _write_index_html(out, entries)
