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
  ``PlotData.save_all(out_dir)`` — write every plot ``.html`` file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import xarray as xr

from tipopac.physics import predicted_tsys

__all__ = [
    "AtmosphericProfile",
    "CVsFrequency",
    "ElevationCurve",
    "FitQualityHeatmap",
    "Plot",
    "PlotData",
    "ResidualRmsHeatmap",
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

    WIDTH = 800
    HEIGHT = 600
    WIDTH_WIDE = 1200
    POINT_SIZE = 25
    MEAN_POINT_SIZE = 80
    LINE_STROKE = 2.0

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds

    def build(self) -> alt.Chart | alt.LayerChart | alt.FacetChart:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        """Serialise the chart to ``path.html`` (standalone, inline data)."""
        path = Path(path).with_suffix(".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.build().save(path)
        _log.info("plot saved: %s", path)

    def _finalize(
        self,
        chart: alt.LayerChart | alt.FacetChart,
        *,
        title: str,
        width: int | None = None,
        interactive: bool = True,
    ) -> alt.LayerChart | alt.FacetChart:
        chart = chart.properties(
            width=width if width is not None else self.WIDTH,
            height=self.HEIGHT,
            title=title,
        )
        return chart.interactive() if interactive else chart


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


class _QuantityVsFrequency(Plot):
    """Shared scaffolding for multi-scan plots."""

    def __init__(self, ds: xr.Dataset, scans: int | list[int] | None = None) -> None:
        super().__init__(ds)
        self.scans = _validate_scans(ds, scans)
        self.ds_sub = ds.sel(scan=self.scans)
        self.width = self.WIDTH_WIDE if len(self.scans) > 1 else self.WIDTH

    @property
    def freq_domain(self) -> tuple[float, float]:
        f_min = self.ds_sub["frequency_GHz"].min() - 2
        f_max = self.ds_sub["frequency_GHz"].max() + 2
        return float(f_min), float(f_max)

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

        z_grid = xr.DataArray(_Z_GRID, dims=("z",))
        pred = predicted_tsys(cell, z_deg=z_grid)
        fit_R = pred.sel(polarization="R").values
        fit_L = pred.sel(polarization="L").values
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
            scale=alt.Scale(domain=[30, 70], nice=False),
        )
        y_enc = alt.Y(
            "Tsys:Q",
            title="System Temperature [K]",
            scale=alt.Scale(
                domain=[df.Tsys.min() / 1.05, df.Tsys.max() * 1.05], nice=False
            ),
        )

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


class TauVsFrequency(_QuantityVsFrequency):
    """Zenith opacity vs spw centre frequency.

    Per-sample scatter (gray=passed, orangered=failed-fit) + antenna-weighted
    mean per spw (firebrick) + optional AM model line (black). Log y-axis.
    Hover discloses (scan, antenna, spw, frequency, τ, σ, fit_success).
    """

    def build(self) -> alt.LayerChart | alt.FacetChart:
        ds_sub = self.ds_sub
        y_title = "Zenith optical depth [nepers]"

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
        y_domain = [tau_min / 1.2, tau_max * 1.2]

        x_enc = alt.X(
            "frequency_GHz:Q",
            title="Frequency [GHz]",
            scale=alt.Scale(domain=self.freq_domain, nice=False),
        )
        y_enc = alt.Y(
            "tau_zenith:Q",
            title=y_title,
            scale=alt.Scale(type="log", domain=y_domain, nice=False),
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


class TcalVsFrequency(_QuantityVsFrequency):
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
        y_title = f"Calibration device temperature ({self.kind}) [K]"

        col = f"tcal_{self.kind}"
        if self.kind == "ref":
            # tcal_ref has no scan dim; pulling it alongside tcal_fit into a
            # joint Dataset broadcasts it N_scan-fold (~7× empirically) and
            # bloats the embedded data. Plot it from the bare DataArray.
            df = _to_df(ds_sub["tcal_ref"], name="tcal_ref")
            tooltip = [
                "antenna:N",
                "spw:N",
                "polarization:N",
                alt.Tooltip("frequency_GHz:Q", format=".3f"),
                alt.Tooltip("tcal_ref:Q", format=".3f"),
            ]
        else:
            df = _to_df(ds_sub[["tcal_fit", "tcal_ref"]], dropna=col)
            tooltip = [
                "scan:N",
                "antenna:N",
                "spw:N",
                "polarization:N",
                alt.Tooltip("frequency_GHz:Q", format=".3f"),
                alt.Tooltip("tcal_fit:Q", format=".3f"),
                alt.Tooltip("tcal_ref:Q", format=".3f"),
            ]
        mean_da = ds_sub[col].mean(dim=["polarization", "antenna"])
        mean_df = _to_df(mean_da, name="mean_tcal")

        samples = (
            alt.Chart(df)
            .mark_point(filled=True, size=self.POINT_SIZE, color=self.COLOR_GOOD)
            .encode(
                x=alt.X(
                    "frequency_GHz:Q",
                    title="Frequency [GHz]",
                    scale=alt.Scale(domain=self.freq_domain, nice=False),
                ),
                y=alt.Y(f"{col}:Q", title=y_title),
                tooltip=tooltip,
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


class CVsFrequency(_QuantityVsFrequency):
    """Tcal correction multiplier c = tcal_fit / tcal_ref vs frequency.

    Dashed reference line at c=1 + per-(antenna, spw, pol) gray scatter +
    polarisation/antenna-averaged firebrick scatter.
    """

    def build(self) -> alt.LayerChart | alt.FacetChart:
        ds_sub = self.ds_sub
        y_title = "Cal. device scaling (c = T_cal,fit / T_cal,ref)"

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
                x=alt.X(
                    "frequency_GHz:Q",
                    title="Frequency [GHz]",
                    scale=alt.Scale(domain=self.freq_domain, nice=False),
                ),
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


class AtmosphericProfile(Plot):
    """Vertical T and H₂O mixing-ratio profiles vs pressure.

    Pressure on a log y-axis (850 → 10 hPa, high pressure at the bottom).
    Temperature on a linear x-axis (bottom edge, firebrick) and H₂O
    volume mixing ratio on an independent log x-axis (top edge,
    dodgerblue). ``scan=None`` plots the across-scan mean; an int picks
    a single scan.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        scan: int | None = None,
        temperature_unit: str = "C",
    ) -> None:
        super().__init__(ds)
        if temperature_unit not in ("C", "K"):
            raise ValueError(
                f"temperature_unit must be 'C' or 'K', got {temperature_unit!r}"
            )
        self.scan = None if scan is None else int(scan)
        self.temperature_unit = temperature_unit

    def build(self) -> alt.LayerChart | alt.FacetChart:
        pressure_hPa = self.ds["atm_pressure"].values / 100.0
        if self.scan is None:
            temp_K = self.ds["atm_temperature"].mean(dim="scan").values
            vmr = self.ds["atm_h2o_vmr"].mean(dim="scan").values
        else:
            temp_K = self.ds["atm_temperature"].sel(scan=self.scan).values
            vmr = self.ds["atm_h2o_vmr"].sel(scan=self.scan).values

        temp_C = temp_K - 273.15
        temp_col = "temperature_C" if self.temperature_unit == "C" else "temperature_K"
        temp_values = temp_C if self.temperature_unit == "C" else temp_K
        temp_title = (
            "Temperature [°C]" if self.temperature_unit == "C" else "Temperature [K]"
        )

        # Floor VMR for the log axis — open-meteo + amwrap never emit zeros
        # in practice, but ``mark_line`` with a log scale silently drops any
        # non-positive sample, which would leave a misleading gap.
        df = pd.DataFrame(
            {
                "pressure_hPa": pressure_hPa,
                temp_col: temp_values,
                "h2o_vmr": np.clip(vmr, 1e-12, None),
            }
        )

        y_enc = alt.Y(
            "pressure_hPa:Q",
            title="Pressure [hPa]",
            scale=alt.Scale(type="log", domain=[10, 875], reverse=True, nice=False),
        )
        # Vega-Lite sorts ``mark_line`` points by the x channel by default,
        # which would draw the temperature and VMR lines in value order
        # rather than profile order. Force the trace to follow pressure.
        line_order = alt.Order("pressure_hPa:Q")
        # Pan/zoom selections: the two x scales are independent so each
        # layer carries its own x-scale binding; y is shared at the layer
        # composition. A single drag fires all three selections, so the
        # user pans both lines together on whichever axis they grab.
        t_pan = alt.selection_interval(bind="scales", encodings=["x"], name="t_pan")
        q_pan = alt.selection_interval(bind="scales", encodings=["x"], name="q_pan")
        y_pan = alt.selection_interval(bind="scales", encodings=["y"], name="y_pan")
        t_line = (
            alt.Chart(df)
            .mark_line(color=self.COLOR_R_POL, strokeWidth=self.LINE_STROKE)
            .encode(
                x=alt.X(
                    f"{temp_col}:Q",
                    title=temp_title,
                    scale=alt.Scale(domain=[-70, 40], nice=False),
                    axis=alt.Axis(
                        orient="bottom",
                        titleColor=self.COLOR_R_POL,
                        labelColor=self.COLOR_R_POL,
                    ),
                ),
                y=y_enc,
                order=line_order,
                tooltip=[
                    alt.Tooltip("pressure_hPa:Q", format=".1f"),
                    alt.Tooltip(f"{temp_col}:Q", format=".2f"),
                ],
            )
            .add_params(t_pan)
        )
        q_line = (
            alt.Chart(df)
            .mark_line(color=self.COLOR_L_POL, strokeWidth=self.LINE_STROKE)
            .encode(
                x=alt.X(
                    "h2o_vmr:Q",
                    title="H₂O volume mixing ratio",
                    scale=alt.Scale(type="log", domain=[2e-6, 0.01], nice=False),
                    axis=alt.Axis(
                        orient="top",
                        titleColor=self.COLOR_L_POL,
                        labelColor=self.COLOR_L_POL,
                    ),
                ),
                y=y_enc,
                order=line_order,
                tooltip=[
                    alt.Tooltip("pressure_hPa:Q", format=".1f"),
                    alt.Tooltip("h2o_vmr:Q", format=".2e"),
                ],
            )
            .add_params(q_pan)
        )

        if self.scan is None:
            n_scan = int(self.ds.sizes["scan"])
            title = f"mean profile across {n_scan} scan{'s' if n_scan != 1 else ''}"
        else:
            title = f"scan {self.scan}"
        source = self.ds.attrs.get("atm_profile_source")
        if source:
            title = f"{title} — {source}"

        chart = (
            alt.layer(t_line, q_line).resolve_scale(x="independent").add_params(y_pan)
        )
        return self._finalize(chart, title=title, interactive=False)


class _Heatmap(Plot):
    """Per-(antenna, spw) ``mark_rect`` heatmap, faceted by scan when many.

    Shared scaffolding for the categorical fit-quality and continuous
    residual-RMS heatmaps. Subclasses provide a metric DataArray, a
    column name, a colour encoding, and a tooltip entry; the base handles
    flag-fraction computation, dropping unobserved cells, facet layout,
    and ``resolve_scale(x="independent")`` so each scan sizes to its own
    observed spws (VLA scans observe disjoint spw blocks).
    """

    CELL_HEIGHT = 16
    CELL_WIDTH = 22

    def __init__(self, ds: xr.Dataset, scans: int | list[int] | None = None) -> None:
        super().__init__(ds)
        self.scans = _validate_scans(ds, scans)
        self.ds_sub = ds.sel(scan=self.scans)

    # Hooks ----------------------------------------------------------------
    def _metric_name(self) -> str:
        raise NotImplementedError

    def _metric_array(self) -> xr.DataArray:
        raise NotImplementedError

    def _color_encoding(self) -> alt.Color:
        raise NotImplementedError

    def _metric_tooltip(self) -> alt.Tooltip:
        raise NotImplementedError

    def _extra_tooltip(self) -> list[alt.Tooltip]:
        """Subclass hook for extra tooltip entries beyond the shared four."""
        return []

    def _extra_data_arrays(self) -> dict[str, xr.DataArray]:
        """Subclass hook for extra columns to merge into the DataFrame.

        Use this to surface columns referenced by ``_extra_tooltip`` (the
        flag-fraction and metric columns are always present).
        """
        return {}

    # ----------------------------------------------------------------------
    def _flag_fraction(self) -> xr.DataArray:
        # Both readers NaN-init switched_diff and only write observed cells,
        # so ``~isnan(switched_diff)`` masks out NaN time-pad AND missing-spw
        # cells in one shot. Cells with zero real samples drop out below.
        has_data = ~self.ds_sub["switched_diff"].isnull()
        denom = has_data.sum(dim=("polarization", "time"))
        flagged = (self.ds_sub["flag"] & has_data).sum(dim=("polarization", "time"))
        return (flagged / denom.where(denom > 0)).astype(np.float32)

    def build(self) -> alt.Chart | alt.FacetChart:
        metric_name = self._metric_name()
        plot_ds = xr.Dataset(
            {
                "flag_fraction": self._flag_fraction(),
                metric_name: self._metric_array(),
                **self._extra_data_arrays(),
            }
        )
        df = plot_ds.to_dataframe().reset_index()
        df = df[df["flag_fraction"].notna() & df[metric_name].notna()]

        tooltip: list = [
            "antenna:N",
            "spw:N",
            "scan:N",
            alt.Tooltip("flag_fraction:Q", format=".1%", title="Flagged fraction"),
            self._metric_tooltip(),
            *self._extra_tooltip(),
        ]

        chart_h = max(120, self.ds_sub.sizes["antenna"] * self.CELL_HEIGHT)
        # Width is sized from the busiest facet's spw count, not the global
        # spw axis. Without independent x scales the data would otherwise
        # crowd into a thin scan-specific strip and the rest go off-screen.
        max_spws_per_facet = int(df.groupby("scan")["spw"].nunique().max())
        facet_w = max(120, max_spws_per_facet * self.CELL_WIDTH)

        base = (
            alt.Chart(df)
            .mark_rect(stroke="white", strokeWidth=0.5)
            .encode(
                x=alt.X("spw:O", title="Spectral window"),
                y=alt.Y("antenna:N", title="Antenna"),
                color=self._color_encoding(),
                tooltip=tooltip,
            )
            .properties(width=facet_w, height=chart_h)
        )

        title = _scan_title(self.scans)
        if len(self.scans) == 1:
            return base.properties(title=title)
        return (
            base.facet(column=alt.Column("scan:N", title="Scan"))
            .resolve_scale(x="independent")
            .properties(title=title)
        )


class FitQualityHeatmap(_Heatmap):
    """Per-(antenna, spw) fit-quality heatmap, faceted by scan when many.

    Cell colour encodes ``fit_reason``; tooltip carries antenna, spw,
    scan, the fraction of flagged time/polarisation samples within the
    tipping scan (NaN time-pad excluded from the denominator), and the
    fit quality label.
    """

    # Categorical palette ordered best → worst so the legend ranks
    # failures intuitively. "ok" is a low-contrast grey so failures pop.
    _REASON_DOMAIN: tuple[str, ...] = (
        "ok",
        "poorly_identified",
        "high_chi2",
        "fit_failed",
        "too_few_samples",
    )
    _REASON_RANGE: tuple[str, ...] = (
        "lightgray",
        "khaki",
        "orange",
        "orangered",
        "firebrick",
    )

    def _metric_name(self) -> str:
        return "fit_reason"

    def _metric_array(self) -> xr.DataArray:
        return self.ds_sub["fit_reason"]

    def _color_encoding(self) -> alt.Color:
        return alt.Color(
            "fit_reason:N",
            scale=alt.Scale(
                domain=list(self._REASON_DOMAIN), range=list(self._REASON_RANGE)
            ),
            legend=alt.Legend(title="Fit quality"),
        )

    def _metric_tooltip(self) -> alt.Tooltip:
        return alt.Tooltip("fit_reason:N", title="Fit quality")


class ResidualRmsHeatmap(_Heatmap):
    """Per-(antenna, spw) Tsys-fit residual RMS in Kelvin, faceted by scan.

    The predicted Tsys curve is reconstructed from the persisted fit
    parameters (``T0``, ``tau_zenith``, ``Twmt``, ``tcal_fit/tcal_ref``)
    via :func:`tipopac.physics.predicted_tsys`. RMS is taken over
    ``(polarization, time)`` of the un-normalised Kelvin residual after
    masking ``flag``. Failed-fit cells have NaN parameters and drop out;
    use :class:`FitQualityHeatmap` to see *which* category they fell
    into.
    """

    def _metric_name(self) -> str:
        return "residual_rms_K"

    def _metric_array(self) -> xr.DataArray:
        pred = predicted_tsys(self.ds_sub)
        resid = (self.ds_sub["Tsys"] - pred).where(~self.ds_sub["flag"])
        return (resid**2).mean(dim=("polarization", "time")) ** 0.5

    def _color_encoding(self) -> alt.Color:
        return alt.Color(
            "residual_rms_K:Q",
            scale=alt.Scale(type="log", scheme="viridis"),
            legend=alt.Legend(title="Residual RMS [K]"),
        )

    def _metric_tooltip(self) -> alt.Tooltip:
        return alt.Tooltip("residual_rms_K:Q", format=".2f", title="Residual RMS [K]")

    def _extra_tooltip(self) -> list[alt.Tooltip]:
        return [alt.Tooltip("fit_reason:N", title="Fit quality")]

    def _extra_data_arrays(self) -> dict[str, xr.DataArray]:
        return {"fit_reason": self.ds_sub["fit_reason"]}


class PlotData:
    """Wrap the canonical tipopac dataset and dispatch the four plot types.

    Convenience methods (``elevation_curve`` etc.) return ``alt.LayerChart``
    objects so callers can inspect or render them; :meth:`save_all` writes
    every applicable plot to ``out_dir`` as ``.html``.
    """

    def __init__(self, ds: xr.Dataset) -> None:
        self.ds = ds.assign_coords(frequency_GHz=ds.frequency / 1e9)

    def elevation_curve(self, scan: int, antenna: str, spw: int) -> ElevationCurve:
        return ElevationCurve(self.ds, scan, antenna, spw)

    def tau_vs_frequency(self, scans: int | list[int] | None = None) -> TauVsFrequency:
        return TauVsFrequency(self.ds, scans)

    def tcal_vs_frequency(
        self, scans: int | list[int] | None = None, kind: str = "fit"
    ) -> TcalVsFrequency:
        return TcalVsFrequency(self.ds, scans, kind)

    def c_vs_frequency(self, scans: int | list[int] | None = None) -> CVsFrequency:
        return CVsFrequency(self.ds, scans)

    def atmospheric_profile(
        self, scan: int | None = None, temperature_unit: str = "C"
    ) -> AtmosphericProfile:
        return AtmosphericProfile(self.ds, scan, temperature_unit)

    def fit_quality_heatmap(
        self, scans: int | list[int] | None = None
    ) -> FitQualityHeatmap:
        return FitQualityHeatmap(self.ds, scans)

    def residual_rms_heatmap(
        self, scans: int | list[int] | None = None
    ) -> ResidualRmsHeatmap:
        return ResidualRmsHeatmap(self.ds, scans)

    def save_all(
        self, out_dir: str | Path = Path("."), plot_elev: bool = False
    ) -> None:
        """Write every applicable plot to ``out_dir`` as stand-alone ``.html``.

        - ``tippingcurve_spw_{spw}_{ant}_scan_{scan}`` per successful cell.
        - ``tau_vs_frequency`` over every scan.
        - ``tcal_ref_vs_frequency`` over every scan.
        - ``tcal_fit_vs_frequency`` and ``c_vs_frequency`` additionally when
          ``tcal_fit`` differs from ``tcal_ref`` (``independent_tau_solve`` mode).
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        success = self.ds["fit_success"]
        if not bool(success.any()):
            _log.warning("save_all: no successful fits; no plots will be written")

        # Per-cell elevation curves.
        if plot_elev:
            cells = success.stack(cell=("scan", "antenna", "spw"))
            for scan_raw, ant_raw, spw_raw in cells.cell.values[cells.values]:
                scan_id, ant, spw_id = int(scan_raw), str(ant_raw), int(spw_raw)
                stem = f"tippingcurve_spw_{spw_id}_{ant}_scan_{scan_id}"
                self.elevation_curve(scan_id, ant, spw_id).save(out / stem)

        # Parameter versus frequency plots.
        self.tau_vs_frequency().save(out / "tau_vs_frequency")
        self.tcal_vs_frequency(kind="ref").save(out / "tcal_ref_vs_frequency")

        # Fit-quality heatmap over every scan.
        if "fit_reason" in self.ds.data_vars:
            self.fit_quality_heatmap().save(out / "fit_quality_heatmap")

        # Residual-RMS heatmap. Requires the fitted parameter vars that
        # ``predicted_tsys`` reconstructs the model from.
        residual_rms_deps = ("T0", "tau_zenith", "Twmt", "tcal_fit", "tcal_ref", "Tsys")
        if all(v in self.ds.data_vars for v in residual_rms_deps):
            self.residual_rms_heatmap().save(out / "residual_rms_heatmap")

        # Atmospheric profile (mean across scans); skip when the optional
        # atm vars are not on the dataset.
        if "atm_pressure" in self.ds.data_vars:
            self.atmospheric_profile().save(out / "atmospheric_profile")

        # Only generate fitted Tcal and "c" plots when fit.
        if self.ds.attrs["mode"] == "independent_tau":
            pass
        else:
            self.tcal_vs_frequency(kind="fit").save(out / "tcal_fit_vs_frequency")
            self.c_vs_frequency().save(out / "c_vs_frequency")
