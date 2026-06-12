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

    coords = {
        "scan": np.arange(1, n_scan + 1, dtype=np.intp),
        "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
        "spw": np.arange(n_spw, dtype=np.intp),
        "polarization": list(schema.POL_VALUES),
        "xyz": ["X", "Y", "Z"],
        "frequency": (("spw",), freqs),
        "bandwidth": (("spw",), np.full(n_spw, 2e9, dtype=np.float64)),
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
    spec = (
        PlotData(ds).elevation_curve(scan=1, antenna="ea01", spw=0).build().to_dict()
    )
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
