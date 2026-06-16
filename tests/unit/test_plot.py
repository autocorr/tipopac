"""Unit tests for tipopac.plot (DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import xarray as xr

from tipopac import schema
from tipopac.plot import PlotData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plot_ds(
    *,
    n_scan: int = 1,
    n_ant: int = 1,
    n_spw: int = 1,
    success: bool = True,
    with_am: bool = False,
    with_atm: bool = False,
    freq_Hz: float = 22.2e9,
    mode: str = "independent_tau_solve",
) -> xr.Dataset:
    """Minimal dataset ready for PlotData.

    ZA values span 35-80 deg; Tsys is synthetic but positive. When
    *with_am* is True, ``am_freq_grid`` and ``am_tau`` are populated so
    the am-overlay path runs. ``mode`` sets ``ds.attrs["mode"]`` —
    save_all dispatches Tcal-fit / c plots from it.
    """
    n_time = 5

    za = np.linspace(35.0, 80.0, n_time, dtype=np.float32)
    za_arr = np.broadcast_to(za, (n_scan, n_ant, n_time)).copy()

    tsys_val = 80.0
    tsys = np.full((n_scan, n_ant, n_spw, 2, n_time), tsys_val, dtype=np.float32)

    tau0 = 0.05
    tau_zenith = np.full((n_scan, n_ant, n_spw), tau0, dtype=np.float32)
    tau_err = np.full((n_scan, n_ant, n_spw), 0.002, dtype=np.float32)
    T0 = np.full((n_scan, n_ant, n_spw, 2), 50.0, dtype=np.float32)
    tcal_ref_val = 5.0
    tcal_ref = np.full((n_ant, n_spw, 2), tcal_ref_val, dtype=np.float32)
    tcal_fit = np.full((n_scan, n_ant, n_spw, 2), tcal_ref_val, dtype=np.float32)
    fit_success_arr = np.full((n_scan, n_ant, n_spw), success, dtype=bool)
    fit_reason = np.full(
        (n_scan, n_ant, n_spw), "ok" if success else "dz_too_small", dtype=object
    )

    freqs = np.linspace(freq_Hz, freq_Hz * 1.05, n_spw, dtype=np.float64)

    data_vars: dict = {
        "switched_diff": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32),
        ),
        "switched_sum": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.full((n_scan, n_ant, n_spw, 2, n_time), 2.0, dtype=np.float32),
        ),
        "zenith_angle": (("scan", "antenna", "time"), za_arr),
        "tcal_ref": (("antenna", "spw", "polarization"), tcal_ref),
        "weather_T": (
            ("scan", "time"),
            np.full((n_scan, n_time), 280.0, dtype=np.float32),
        ),
        "weather_P": (
            ("scan", "time"),
            np.full((n_scan, n_time), 85000.0, dtype=np.float32),
        ),
        "weather_RH": (
            ("scan", "time"),
            np.full((n_scan, n_time), 0.3, dtype=np.float32),
        ),
        "exposure_time": (
            ("scan", "time"),
            np.full((n_scan, n_time), 1.0, dtype=np.float32),
        ),
        "flag": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=bool),
        ),
        "Tsys": (("scan", "antenna", "spw", "polarization", "time"), tsys),
        "tau_zenith": (("scan", "antenna", "spw"), tau_zenith),
        "tau_err": (("scan", "antenna", "spw"), tau_err),
        "T0": (("scan", "antenna", "spw", "polarization"), T0),
        "tcal_fit": (("scan", "antenna", "spw", "polarization"), tcal_fit),
        "Twmt": (
            ("scan", "spw"),
            np.full((n_scan, n_spw), 270.0, dtype=np.float32),
        ),
        "fit_success": (("scan", "antenna", "spw"), fit_success_arr),
        "fit_reason": (("scan", "antenna", "spw"), fit_reason),
    }

    if with_am:
        am_freq_grid = np.linspace(
            freqs.min() * 0.95, freqs.max() * 1.05, 50, dtype=np.float64
        )
        data_vars["am_freq_grid"] = (("frequency_dense",), am_freq_grid)
        data_vars["am_tau"] = (
            ("frequency_dense",),
            np.full(am_freq_grid.size, tau0, dtype=np.float64),
        )

    if with_atm:
        # 10-level synthetic profile, 850 → 10 hPa. Stored in Pa (schema §5).
        atm_p_hPa = np.array(
            [850, 700, 500, 300, 200, 100, 50, 30, 20, 10], dtype=np.float64
        )
        atm_p_Pa = atm_p_hPa * 100.0
        atm_T = np.linspace(280.0, 210.0, atm_p_hPa.size, dtype=np.float32)
        atm_vmr = np.logspace(-3, -6, atm_p_hPa.size).astype(np.float32)
        data_vars["atm_pressure"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_p_Pa, (n_scan, atm_p_hPa.size)).copy(),
        )
        data_vars["atm_temperature"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_T, (n_scan, atm_p_hPa.size)).copy(),
        )
        data_vars["atm_h2o_vmr"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_vmr, (n_scan, atm_p_hPa.size)).copy(),
        )

    coords = {
        "scan": np.arange(1, n_scan + 1, dtype=np.intp),
        "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
        "spw": np.arange(n_spw, dtype=np.intp),
        "polarization": list(schema.POL_VALUES),
        "xyz": ["X", "Y", "Z"],
        "frequency": (("spw",), freqs),
        "bandwidth": (("spw",), np.full(n_spw, 2e9, dtype=np.float64)),
        "band": (("spw",), np.full(n_spw, "K", dtype="U4")),
        "antenna_position": (
            ("antenna", "xyz"),
            np.zeros((n_ant, 3), dtype=np.float64),
        ),
        "scan_time_start": (
            ("scan",),
            np.linspace(
                5131296000.0, 5131296000.0 + 120.0 * n_scan, n_scan, dtype=np.float64
            ),
        ),
        "scan_time_end": (
            ("scan",),
            np.linspace(
                5131296090.0, 5131296090.0 + 120.0 * n_scan, n_scan, dtype=np.float64
            ),
        ),
        "time_utc": (
            ("scan", "time"),
            np.tile(np.linspace(5131296000.0, 5131296090.0, n_time), (n_scan, 1)),
        ),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["mode"] = mode
    return ds


def _tooltip_fields(layer_spec: dict) -> list[str]:
    tt = layer_spec.get("encoding", {}).get("tooltip", [])
    return [item.get("field") for item in tt]


# ---------------------------------------------------------------------------
# Per-method tests (return Plot subclasses; .build() yields the alt chart)
# ---------------------------------------------------------------------------


def test_elevation_curve_returns_layerchart() -> None:
    ds = _make_plot_ds(success=True)
    chart = PlotData(ds).elevation_curve(scan=1, antenna="ea01", spw=0).build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    assert len(spec["layer"]) == 2  # scatter + model line
    marks = {layer["mark"]["type"] for layer in spec["layer"]}
    assert marks == {"point", "line"}


def test_elevation_curve_tooltip_has_polarization_and_tsys() -> None:
    ds = _make_plot_ds(success=True)
    spec = PlotData(ds).elevation_curve(scan=1, antenna="ea01", spw=0).build().to_dict()
    scatter_layer = next(
        layer for layer in spec["layer"] if layer["mark"]["type"] == "point"
    )
    fields = _tooltip_fields(scatter_layer)
    assert "polarization" in fields
    assert "Tsys" in fields
    assert "zenith_angle" in fields


def test_tau_vs_frequency_with_am_overlay() -> None:
    ds = _make_plot_ds(n_spw=4, success=True, with_am=True)
    chart = PlotData(ds).tau_vs_frequency(scans=1).build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    # samples + weighted mean + am line
    assert len(spec["layer"]) == 3
    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert "line" in marks  # am model line layer


def test_tau_vs_frequency_without_am() -> None:
    ds = _make_plot_ds(n_spw=4, success=True, with_am=False)
    spec = PlotData(ds).tau_vs_frequency(scans=1).build().to_dict()
    assert len(spec["layer"]) == 2  # samples + weighted mean only
    marks = [layer["mark"]["type"] for layer in spec["layer"]]
    assert "line" not in marks


def test_tau_vs_frequency_uses_log_scale() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    spec = PlotData(ds).tau_vs_frequency(scans=1).build().to_dict()
    samples = spec["layer"][0]
    y_enc = samples["encoding"]["y"]
    assert y_enc["scale"]["type"] == "log"


def test_tau_vs_frequency_tooltip_carries_identity() -> None:
    ds = _make_plot_ds(n_ant=2, n_spw=3, success=True)
    spec = PlotData(ds).tau_vs_frequency(scans=1).build().to_dict()
    samples = spec["layer"][0]
    fields = _tooltip_fields(samples)
    for required in ("scan", "antenna", "spw", "frequency_GHz", "tau_zenith"):
        assert required in fields


def test_tcal_vs_frequency_returns_layerchart() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    chart = PlotData(ds).tcal_vs_frequency(scans=1).build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    # per-(antenna, spw, pol) scatter + polarization/antenna-averaged mean
    assert len(spec["layer"]) == 2


def test_tcal_vs_frequency_tooltip_has_polarization() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    spec = PlotData(ds).tcal_vs_frequency(scans=1).build().to_dict()
    samples = spec["layer"][0]
    fields = _tooltip_fields(samples)
    for required in ("scan", "antenna", "spw", "polarization", "tcal_fit"):
        assert required in fields


def test_tcal_ref_vs_frequency_does_not_replicate_over_scans() -> None:
    """tcal_ref has no scan dim — samples must not be N_scan× duplicated."""
    n_scan, n_ant, n_spw, n_pol = 5, 3, 4, 2
    ds = _make_plot_ds(n_scan=n_scan, n_ant=n_ant, n_spw=n_spw, success=True)
    spec = PlotData(ds).tcal_vs_frequency(kind="ref").build().to_dict()
    samples = spec["layer"][0]
    rows = spec["datasets"][samples["data"]["name"]]
    assert len(rows) == n_ant * n_spw * n_pol


def test_tcal_ref_vs_frequency_tooltip_omits_scan_and_fit() -> None:
    """ref-mode tooltip carries only ref-relevant fields (no scan, no tcal_fit)."""
    ds = _make_plot_ds(n_spw=3, success=True)
    spec = PlotData(ds).tcal_vs_frequency(kind="ref").build().to_dict()
    samples = spec["layer"][0]
    fields = _tooltip_fields(samples)
    assert "scan" not in fields
    assert "tcal_fit" not in fields
    for required in ("antenna", "spw", "polarization", "tcal_ref"):
        assert required in fields


def test_c_vs_frequency_returns_layerchart() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    ds["tcal_fit"].values *= 1.1
    chart = PlotData(ds).c_vs_frequency(scans=1).build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    # ref rule + per-cell scatter + averaged mean
    assert len(spec["layer"]) == 3
    rule_layer = spec["layer"][0]
    assert rule_layer["mark"]["type"] == "rule"


def test_c_vs_frequency_tooltip_carries_c_ratio() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    ds["tcal_fit"].values *= 1.1
    spec = PlotData(ds).c_vs_frequency(scans=1).build().to_dict()
    samples = spec["layer"][1]
    fields = _tooltip_fields(samples)
    assert "c_ratio" in fields
    assert "polarization" in fields


def test_tau_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=3, n_spw=2, success=True, with_am=True)
    chart = PlotData(ds).tau_vs_frequency(scans=[1, 2, 3]).build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    assert len(spec["layer"]) == 3  # samples + mean + am line


def test_tau_vs_frequency_single_scan_via_list() -> None:
    ds = _make_plot_ds(success=True)
    chart = PlotData(ds).tau_vs_frequency(scans=[1]).build()
    assert isinstance(chart, alt.LayerChart)


def test_tcal_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=2, success=True)
    ds["tcal_fit"].values *= 1.1
    chart = PlotData(ds).tcal_vs_frequency(scans=[1, 2]).build()
    assert isinstance(chart, alt.LayerChart)


def test_c_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=2, success=True)
    ds["tcal_fit"].values *= 1.05
    chart = PlotData(ds).c_vs_frequency(scans=[1, 2]).build()
    assert isinstance(chart, alt.LayerChart)


def test_atmospheric_profile_returns_layerchart() -> None:
    ds = _make_plot_ds(n_scan=2, success=True, with_atm=True)
    chart = PlotData(ds).atmospheric_profile().build()
    assert isinstance(chart, alt.LayerChart)
    spec = chart.to_dict()
    # T line + mixing-ratio line.
    assert len(spec["layer"]) == 2
    marks = {layer["mark"]["type"] for layer in spec["layer"]}
    assert marks == {"line"}
    # x-scales must be independent so the two measures get separate axes.
    assert spec["resolve"]["scale"]["x"] == "independent"


def test_atmospheric_profile_axes_and_scales() -> None:
    ds = _make_plot_ds(success=True, with_atm=True)
    spec = PlotData(ds).atmospheric_profile().build().to_dict()
    orients: set[str] = set()
    x_scale_types: set[str] = set()
    for layer in spec["layer"]:
        x_enc = layer["encoding"]["x"]
        orients.add(x_enc["axis"]["orient"])
        x_scale_types.add(x_enc["scale"].get("type", "linear"))
        y_scale = layer["encoding"]["y"]["scale"]
        assert y_scale["type"] == "log"
        # Domain runs from ~10 hPa at top to ≥850 hPa at bottom (reversed
        # so high pressure sits at the bottom of the chart).
        assert y_scale["domain"][0] == 10
        assert y_scale["domain"][1] >= 850
        assert y_scale["reverse"] is True
    assert orients == {"bottom", "top"}
    assert x_scale_types == {"linear", "log"}


def _profile_rows(spec: dict) -> list[dict]:
    """Extract the (deduped) row list from an AtmosphericProfile spec."""
    return spec["datasets"][spec["data"]["name"]]


def test_atmospheric_profile_kelvin_unit() -> None:
    ds = _make_plot_ds(success=True, with_atm=True)
    spec = PlotData(ds).atmospheric_profile(temperature_unit="K").build().to_dict()
    t_layer = next(
        layer
        for layer in spec["layer"]
        if layer["encoding"]["x"]["axis"]["orient"] == "bottom"
    )
    assert t_layer["encoding"]["x"]["field"] == "temperature_K"
    rows = _profile_rows(spec)
    # Synthetic profile uses 280 → 210 K; values must be in Kelvin range.
    assert max(row["temperature_K"] for row in rows) > 250


def test_atmospheric_profile_per_scan_selection() -> None:
    ds = _make_plot_ds(n_scan=3, success=True, with_atm=True)
    # Make scan 2 distinctive so we can verify selection.
    ds["atm_temperature"].values[1, :] = 250.0
    spec = PlotData(ds).atmospheric_profile(scan=2).build().to_dict()
    rows = _profile_rows(spec)
    # Scan 2 was set uniformly to 250 K → -23.15 °C.
    assert all(abs(row["temperature_C"] - (250.0 - 273.15)) < 1e-3 for row in rows)


def test_atmospheric_profile_rejects_bad_unit() -> None:
    import pytest

    ds = _make_plot_ds(success=True, with_atm=True)
    with pytest.raises(ValueError, match="temperature_unit"):
        PlotData(ds).atmospheric_profile(temperature_unit="F")


def test_atmospheric_profile_line_order_is_pressure() -> None:
    """Both lines must trace in pressure order, not in T/VMR order.

    Vega-Lite's default ``mark_line`` sort is by x; without an explicit
    ``order`` encoding a non-monotonic T or VMR profile renders as a
    zigzag instead of a smooth atmospheric trace.
    """
    ds = _make_plot_ds(success=True, with_atm=True)
    spec = PlotData(ds).atmospheric_profile().build().to_dict()
    for layer in spec["layer"]:
        order = layer["encoding"].get("order")
        assert order is not None
        assert order["field"] == "pressure_hPa"


def test_atmospheric_profile_pan_binds_all_axes() -> None:
    """Pan/zoom must cover both x axes and the shared y axis.

    The two x scales are independent, so each layer carries its own
    x-scale binding; y is bound once at the composition level. A drag
    fires all three so the user can pan both lines together on whichever
    axis they grab.
    """
    ds = _make_plot_ds(success=True, with_atm=True)
    spec = PlotData(ds).atmospheric_profile().build().to_dict()
    # Altair hoists layer-level params to the top of the layered spec.
    # Asserting by explicit name confirms none were deduplicated away
    # (the failure mode when autogen names collide).
    scale_params = {
        p["name"]: p["select"]["encodings"]
        for p in spec.get("params", [])
        if p.get("bind") == "scales"
    }
    assert scale_params == {"t_pan": ["x"], "q_pan": ["x"], "y_pan": ["y"]}


# ---------------------------------------------------------------------------
# fit_quality_heatmap
# ---------------------------------------------------------------------------


def test_fit_quality_heatmap_single_scan_is_rect_chart() -> None:
    ds = _make_plot_ds(n_ant=3, n_spw=2, success=True)
    chart = PlotData(ds).fit_quality_heatmap(scans=1).build()
    spec = chart.to_dict()
    # Single scan → no facet wrapper; a plain Chart with mark.rect.
    assert spec["mark"]["type"] == "rect"
    assert spec["encoding"]["x"]["field"] == "spw"
    assert spec["encoding"]["y"]["field"] == "antenna"


def test_fit_quality_heatmap_multi_scan_facets_by_scan() -> None:
    ds = _make_plot_ds(n_scan=3, n_ant=2, n_spw=2, success=True)
    chart = PlotData(ds).fit_quality_heatmap().build()
    spec = chart.to_dict()
    # Facet column carries scan; the inner spec is the rect chart.
    assert spec["facet"]["column"]["field"] == "scan"
    assert spec["spec"]["mark"]["type"] == "rect"


def test_fit_quality_heatmap_tooltip_carries_required_fields() -> None:
    ds = _make_plot_ds(n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).fit_quality_heatmap(scans=1).build().to_dict()
    fields = [item.get("field") for item in spec["encoding"]["tooltip"]]
    for required in ("antenna", "spw", "scan", "flag_fraction", "fit_reason"):
        assert required in fields


def test_fit_quality_heatmap_colors_by_fit_reason() -> None:
    ds = _make_plot_ds(n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).fit_quality_heatmap(scans=1).build().to_dict()
    color = spec["encoding"]["color"]
    assert color["field"] == "fit_reason"
    # Domain locks the categorical order so the legend renders consistently.
    assert "ok" in color["scale"]["domain"]
    assert "fit_failed" in color["scale"]["domain"]


def test_fit_quality_heatmap_flag_fraction_excludes_nan_pad() -> None:
    """Denominator must be n_real_samples, not n_time × n_pol.

    Both readers NaN-init switched_diff and only write observed cells,
    so the real-sample mask is `~isnan(switched_diff)`. A scan with 2 of
    5 time slots holding data and one (pol, time) cell flagged within
    the data region should report 1/(2*2)=0.25, not 1/(5*2)=0.10.
    """
    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=1, success=True)
    # Mark the last 3 time samples as NaN-pad (no input data, flag=True).
    ds["switched_diff"].values[0, 0, 0, :, 2:] = np.nan
    ds["flag"].values[0, 0, 0, :, 2:] = True
    # One genuine flag in the real region: (pol=0, time=1).
    ds["flag"].values[0, 0, 0, 0, 1] = True
    spec = PlotData(ds).fit_quality_heatmap(scans=1).build().to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    [row] = rows
    assert abs(row["flag_fraction"] - 0.25) < 1e-6


def test_fit_quality_heatmap_drops_missing_spw_cells() -> None:
    """Cells the scan never observed must not render.

    Readers leave ``switched_diff`` all-NaN with ``flag=True`` for
    (scan, antenna, spw) cells the scan didn't observe (one band per
    scan on the VLA, ~108 spws). Without the drop, those cells
    dominate the heatmap as fully-flagged ``too_few_samples``.
    """
    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=2, success=True)
    # spw=1 was never observed: no data, flag fully True.
    ds["switched_diff"].values[0, 0, 1, :, :] = np.nan
    ds["flag"].values[0, 0, 1, :, :] = True
    spec = PlotData(ds).fit_quality_heatmap(scans=1).build().to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    assert {row["spw"] for row in rows} == {0}


def test_fit_quality_heatmap_facet_x_scale_is_independent() -> None:
    """VLA scans observe disjoint spw blocks (one band per scan).

    Without independent x scales, every facet inherits the global spw
    domain and the data crowds into a thin scan-specific strip — cells
    outside the first facet end up at off-screen x positions. The fix
    is ``resolve_scale(x='independent')`` so each facet sizes its own
    x-axis to its actually-observed spws.
    """
    ds = _make_plot_ds(n_scan=3, n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).fit_quality_heatmap().build().to_dict()
    assert spec["resolve"]["scale"]["x"] == "independent"


def test_fit_quality_heatmap_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=3, n_ant=2, n_spw=2, success=True)
    chart = PlotData(ds).fit_quality_heatmap(scans=[1, 3]).build()
    spec = chart.to_dict()
    assert spec["facet"]["column"]["field"] == "scan"
    scans_in_data = {row["scan"] for row in spec["datasets"][spec["data"]["name"]]}
    assert scans_in_data == {1, 3}


# ---------------------------------------------------------------------------
# predicted_tsys helper + residual_rms_heatmap
# ---------------------------------------------------------------------------


def test_predicted_tsys_default_uses_zenith_angle_shape() -> None:
    from tipopac.physics import predicted_tsys

    ds = _make_plot_ds(n_scan=2, n_ant=3, n_spw=2, success=True)
    pred = predicted_tsys(ds)
    assert pred.dims == ("scan", "antenna", "spw", "polarization", "time")
    assert pred.shape == (2, 3, 2, 2, 5)


def test_predicted_tsys_dense_grid_overlay_matches_tsys_model() -> None:
    """Dense-grid call must equal tsys_model evaluated point-wise."""
    from tipopac.physics import predicted_tsys, tsys_model

    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=1, success=True)
    cell = ds.sel(scan=1, antenna="ea01", spw=0)
    z_grid = xr.DataArray(np.linspace(35.0, 75.0, 9), dims=("z",))
    pred = predicted_tsys(cell, z_deg=z_grid).sel(polarization="R").values
    expected = tsys_model(
        z_grid.values,
        float(cell["T0"].sel(polarization="R")),
        float(cell["tau_zenith"]),
        float(cell["Twmt"]),
    )
    # tcal_fit == tcal_ref in the fixture, so c == 1 and pred == tsys_model.
    np.testing.assert_allclose(pred, expected, rtol=1e-5)


def test_predicted_tsys_divides_by_c_in_tcal_solve_mode() -> None:
    """When tcal_fit/tcal_ref != 1, the prediction must divide by c."""
    from tipopac.physics import predicted_tsys, tsys_model

    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=1, success=True)
    ds["tcal_fit"].values[:] *= 1.2  # c = 1.2 for both pols
    cell = ds.sel(scan=1, antenna="ea01", spw=0)
    z_grid = xr.DataArray(np.linspace(35.0, 75.0, 5), dims=("z",))
    pred = predicted_tsys(cell, z_deg=z_grid).sel(polarization="R").values
    base = tsys_model(
        z_grid.values,
        float(cell["T0"].sel(polarization="R")),
        float(cell["tau_zenith"]),
        float(cell["Twmt"]),
    )
    np.testing.assert_allclose(pred, base / 1.2, rtol=1e-5)


def test_residual_rms_heatmap_zero_when_data_equals_model() -> None:
    """If Tsys is set exactly to the model, the heatmap RMS is 0."""
    from tipopac.physics import predicted_tsys

    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=1, success=True)
    ds["Tsys"].values[:] = predicted_tsys(ds).values
    spec = PlotData(ds).residual_rms_heatmap(scans=1).build().to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    [row] = rows
    assert row["residual_rms_K"] == 0.0


def test_residual_rms_heatmap_matches_closed_form_offset() -> None:
    """Constant +2 K offset on Tsys yields RMS = 2 K everywhere."""
    from tipopac.physics import predicted_tsys

    ds = _make_plot_ds(n_scan=1, n_ant=1, n_spw=1, success=True)
    ds["Tsys"].values[:] = predicted_tsys(ds).values + 2.0
    spec = PlotData(ds).residual_rms_heatmap(scans=1).build().to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    [row] = rows
    assert abs(row["residual_rms_K"] - 2.0) < 1e-4


def test_residual_rms_heatmap_drops_failed_fit_cells() -> None:
    """NaN fit params ⇒ NaN predicted Tsys ⇒ NaN RMS ⇒ row dropped."""
    ds = _make_plot_ds(n_scan=1, n_ant=2, n_spw=1, success=True)
    # ea02 has no valid fit: NaN out T0 and tau_zenith.
    ds["T0"].values[0, 1, 0, :] = np.nan
    ds["tau_zenith"].values[0, 1, 0] = np.nan
    ds["fit_reason"].values[0, 1, 0] = "fit_failed"
    spec = PlotData(ds).residual_rms_heatmap(scans=1).build().to_dict()
    rows = spec["datasets"][spec["data"]["name"]]
    antennas = {row["antenna"] for row in rows}
    assert antennas == {"ea01"}


def test_residual_rms_heatmap_color_is_continuous_log() -> None:
    ds = _make_plot_ds(n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).residual_rms_heatmap(scans=1).build().to_dict()
    color = spec["encoding"]["color"]
    assert color["field"] == "residual_rms_K"
    assert color["type"] == "quantitative"
    assert color["scale"]["type"] == "log"


def test_residual_rms_heatmap_tooltip_carries_fit_reason() -> None:
    """fit_reason is the categorical diagnostic alongside the continuous RMS."""
    ds = _make_plot_ds(n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).residual_rms_heatmap(scans=1).build().to_dict()
    fields = [item.get("field") for item in spec["encoding"]["tooltip"]]
    for required in (
        "antenna",
        "spw",
        "scan",
        "flag_fraction",
        "residual_rms_K",
        "fit_reason",
    ):
        assert required in fields


def test_residual_rms_heatmap_multi_scan_facets_independent_x() -> None:
    ds = _make_plot_ds(n_scan=3, n_ant=2, n_spw=2, success=True)
    spec = PlotData(ds).residual_rms_heatmap().build().to_dict()
    assert spec["facet"]["column"]["field"] == "scan"
    assert spec["resolve"]["scale"]["x"] == "independent"


def test_save_all_writes_residual_rms_heatmap(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=2, n_spw=2, success=True)
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "residual_rms_heatmap.html").exists()


# ---------------------------------------------------------------------------
# save_all integration tests (write plot files only; no index.html — that
# lives in tipopac.weblog and is exercised in test_weblog.py).
# ---------------------------------------------------------------------------


def test_save_all_writes_tipping_curves(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    htmls = list(tmp_path.glob("tippingcurve_*.html"))
    assert len(htmls) == 1


def test_save_all_filename_convention(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    [html] = list(tmp_path.glob("tippingcurve_*.html"))
    assert html.name == "tippingcurve_spw_0_ea01_scan_1.html"


def test_save_all_skips_failed_cells(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=False)
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    assert list(tmp_path.glob("tippingcurve_*.html")) == []
    # No index.html is written by save_all — that's weblog.build_weblog's job.
    assert not (tmp_path / "index.html").exists()


def test_save_all_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "new_subdir" / "plots"
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(out, plot_elev=True)
    assert out.is_dir()
    assert len(list(out.glob("tippingcurve_*.html"))) == 1


def test_save_all_with_am_overlay(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    assert len(list(tmp_path.glob("tippingcurve_*.html"))) == 1
    assert (tmp_path / "tau_vs_frequency.html").exists()


def test_save_all_multi_cell(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=3, n_spw=2, success=True)
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    assert len(list(tmp_path.glob("tippingcurve_*.html"))) == 2 * 3 * 2
    # tau_vs_frequency is a single aggregate file across all scans.
    assert (tmp_path / "tau_vs_frequency.html").exists()


def test_save_all_partial_success(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=1, n_ant=2, n_spw=1, success=True)
    ds["fit_success"].values[0, 1, 0] = False
    PlotData(ds).save_all(tmp_path, plot_elev=True)
    htmls = list(tmp_path.glob("tippingcurve_*.html"))
    assert len(htmls) == 1


def test_save_all_writes_only_html(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    PlotData(ds).save_all(tmp_path)
    # No matplotlib-era extensions should leak through.
    for ext in ("pdf", "png", "svgz"):
        assert list(tmp_path.glob(f"*.{ext}")) == []


def test_save_all_always_writes_tcal_ref(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "tcal_ref_vs_frequency.html").exists()


def test_save_all_skips_tcal_fit_and_c_in_independent_tau_mode(
    tmp_path: Path,
) -> None:
    ds = _make_plot_ds(n_spw=2, success=True, mode="independent_tau")
    PlotData(ds).save_all(tmp_path)
    assert not (tmp_path / "tcal_fit_vs_frequency.html").exists()
    assert not (tmp_path / "c_vs_frequency.html").exists()


def test_save_all_emits_tcal_fit_and_c_in_solve_mode(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_spw=2, success=True, mode="independent_tau_solve")
    ds["tcal_fit"].values *= 1.05
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "tcal_fit_vs_frequency.html").exists()
    assert (tmp_path / "c_vs_frequency.html").exists()


def test_save_all_writes_fit_quality_heatmap(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=2, n_spw=2, success=True)
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "fit_quality_heatmap.html").exists()


def test_save_all_writes_atmospheric_profile_when_present(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_atm=True)
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "atmospheric_profile.html").exists()


def test_save_all_skips_atmospheric_profile_when_absent(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path)
    assert not (tmp_path / "atmospheric_profile.html").exists()
