"""Unit tests for tipopac.plot (DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

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
) -> xr.Dataset:
    """Minimal dataset ready for PlotData.

    ZA values span 35-80 deg; Tsys is synthetic but positive. When
    *with_am* is True, ``tau_extrapolated``, ``am_freq_grid``, and
    ``am_tau`` are populated so the am-overlay path runs.
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

    # spw centre frequencies spread around freq_Hz so tau_vs_frequency has
    # multiple x positions to plot.
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
        data_vars["tau_extrapolated"] = (
            ("scan", "spw"),
            np.full((n_scan, n_spw), tau0, dtype=np.float32),
        )
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

    return xr.Dataset(data_vars=data_vars, coords=coords)


# ---------------------------------------------------------------------------
# save_all integration tests
# ---------------------------------------------------------------------------


def test_save_all_writes_tipping_curves(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path)
    pngs = list(tmp_path.glob("tippingcurve_*.png"))
    assert len(pngs) == 1


def test_save_all_filename_convention(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path)
    pngs = list(tmp_path.glob("tippingcurve_*.png"))
    assert pngs[0].name == "tippingcurve_spw_0_ea01_scan_1.png"


def test_save_all_skips_failed_cells(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=False)
    PlotData(ds).save_all(tmp_path)
    assert list(tmp_path.glob("tippingcurve_*.png")) == []
    assert list(tmp_path.glob("tau_vs_frequency_*.png")) == []
    # weather/heatmap still emitted
    assert (tmp_path / "weather.png").exists()
    assert (tmp_path / "fit_success.png").exists()


def test_save_all_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "new_subdir" / "plots"
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(out)
    assert out.is_dir()
    assert len(list(out.glob("tippingcurve_*.png"))) == 1


def test_save_all_with_am_overlay(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    PlotData(ds).save_all(tmp_path)
    assert len(list(tmp_path.glob("tippingcurve_*.png"))) == 1
    assert (tmp_path / "tau_vs_frequency_scan_1.png").exists()


def test_save_all_multi_cell(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=3, n_spw=2, success=True)
    PlotData(ds).save_all(tmp_path)
    assert len(list(tmp_path.glob("tippingcurve_*.png"))) == 2 * 3 * 2
    # one tau_vs_frequency per scan
    assert len(list(tmp_path.glob("tau_vs_frequency_*.png"))) == 2


def test_save_all_partial_success(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=1, n_ant=2, n_spw=1, success=True)
    ds["fit_success"].values[0, 1, 0] = False
    PlotData(ds).save_all(tmp_path)
    pngs = list(tmp_path.glob("tippingcurve_*.png"))
    assert len(pngs) == 1


def test_save_all_writes_all_output_formats(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    PlotData(ds).save_all(tmp_path)
    stem = "tippingcurve_spw_0_ea01_scan_1"
    for ext in ("pdf", "png", "svgz"):
        assert (tmp_path / f"{stem}.{ext}").exists()
    for stem in ("weather", "fit_success", "tau_vs_frequency_scan_1"):
        for ext in ("pdf", "png", "svgz"):
            assert (tmp_path / f"{stem}.{ext}").exists()


# ---------------------------------------------------------------------------
# Per-method tests (return Figure, no file I/O)
# ---------------------------------------------------------------------------


def test_elevation_curve_returns_figure() -> None:
    ds = _make_plot_ds(success=True)
    fig = PlotData(ds).elevation_curve(scan=1, antenna="ea01", spw=0)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_tau_vs_frequency_with_am_overlay() -> None:
    ds = _make_plot_ds(n_spw=4, success=True, with_am=True)
    fig = PlotData(ds).tau_vs_frequency(scan=1)
    assert isinstance(fig, Figure)
    [ax] = fig.axes
    # errorbar -> 1 Line2D point marker + 1 PathCollection for caps/bars is
    # backend-dependent; what we care about is that the am overlay added an
    # extra plain Line2D that scatter alone would not have.
    lines = [ln for ln in ax.lines if ln.get_label() == "am model"]
    assert len(lines) == 1


def test_tau_vs_frequency_without_am() -> None:
    ds = _make_plot_ds(n_spw=4, success=True, with_am=False)
    fig = PlotData(ds).tau_vs_frequency(scan=1)
    assert isinstance(fig, Figure)
    [ax] = fig.axes
    lines = [ln for ln in ax.lines if ln.get_label() == "am model"]
    assert lines == []


def test_tau_vs_frequency_uses_pwv_scaling_in_title() -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    ds.attrs["pwv_scaling"] = 0.85
    fig = PlotData(ds).tau_vs_frequency(scan=1)
    [ax] = fig.axes
    assert "0.85" in ax.get_title()


def test_tcal_vs_frequency_returns_figure() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    fig = PlotData(ds).tcal_vs_frequency(scan=1)
    assert isinstance(fig, Figure)
    [ax] = fig.axes
    # Two pols × {ref, fit} = 4 scatter PathCollections.
    assert len(ax.collections) == 4


def test_save_all_skips_tcal_when_identical_to_ref(tmp_path: Path) -> None:
    # _make_plot_ds sets tcal_fit == tcal_ref by default → no file.
    ds = _make_plot_ds(success=True)
    PlotData(ds).save_all(tmp_path)
    assert list(tmp_path.glob("tcal_vs_frequency_*.png")) == []


def test_save_all_emits_tcal_when_differs(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=3, success=True)
    # Perturb tcal_fit so it no longer matches the broadcast reference.
    ds["tcal_fit"].values *= 1.05
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "tcal_vs_frequency_scan_1.png").exists()
    assert (tmp_path / "tcal_vs_frequency_scan_2.png").exists()


def test_c_vs_frequency_returns_figure() -> None:
    ds = _make_plot_ds(n_spw=3, success=True)
    # Push tcal_fit away from tcal_ref so c != 1 and the scatter is visible.
    ds["tcal_fit"].values *= 1.1
    fig = PlotData(ds).c_vs_frequency(scan=1)
    assert isinstance(fig, Figure)
    [ax] = fig.axes
    # Two pols => 2 PathCollections for the scatter.
    assert len(ax.collections) == 2
    # axhline reference at c=1 produces one Line2D.
    assert any(line.get_ydata()[0] == 1.0 for line in ax.lines)


def test_save_all_emits_c_when_tcal_differs(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=2, success=True)
    ds["tcal_fit"].values *= 1.05
    PlotData(ds).save_all(tmp_path)
    assert (tmp_path / "c_vs_frequency_scan_1.png").exists()
    assert (tmp_path / "c_vs_frequency_scan_2.png").exists()


def test_save_all_skips_c_when_identical_to_ref(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)  # tcal_fit == tcal_ref by default
    PlotData(ds).save_all(tmp_path)
    assert list(tmp_path.glob("c_vs_frequency_*.png")) == []


def test_save_all_parallel_writes_all_plots(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=2, n_spw=2, success=True, with_am=True)
    PlotData(ds).save_all(tmp_path, n_workers=2)
    assert len(list(tmp_path.glob("tippingcurve_*.png"))) == 2 * 2 * 2
    assert len(list(tmp_path.glob("tau_vs_frequency_*.png"))) == 2
    assert (tmp_path / "weather.png").exists()
    assert (tmp_path / "fit_success.png").exists()


def test_save_all_parallel_matches_serial(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=2, n_spw=2, success=True, with_am=True)
    serial_dir = tmp_path / "serial"
    parallel_dir = tmp_path / "parallel"
    PlotData(ds).save_all(serial_dir)
    PlotData(ds).save_all(parallel_dir, n_workers=2)
    serial_files = sorted(p.name for p in serial_dir.iterdir())
    parallel_files = sorted(p.name for p in parallel_dir.iterdir())
    assert serial_files == parallel_files


def test_save_all_parallel_restores_mplbackend(tmp_path: Path) -> None:
    import os

    sentinel = os.environ.get("MPLBACKEND")
    try:
        os.environ["MPLBACKEND"] = "pdf"  # something other than Agg
        ds = _make_plot_ds(success=True)
        PlotData(ds).save_all(tmp_path, n_workers=2)
        assert os.environ["MPLBACKEND"] == "pdf"
    finally:
        if sentinel is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = sentinel


def test_tau_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=3, n_spw=2, success=True, with_am=True)
    fig = PlotData(ds).tau_vs_frequency(scan=[1, 2, 3])
    [ax] = fig.axes
    # Three scans -> three errorbar series (each as a "container" added to
    # ax.containers in modern matplotlib) plus one am-model Line2D.
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "scan 1" in labels
    assert "scan 2" in labels
    assert "scan 3" in labels
    assert "am model" in labels
    assert "scans 1, 2, 3" in ax.get_title()


def test_tau_vs_frequency_single_scan_via_list() -> None:
    ds = _make_plot_ds(success=True)
    fig = PlotData(ds).tau_vs_frequency(scan=[1])
    [ax] = fig.axes
    # Single-element list keeps the "fitted" label (not "scan 1").
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "fitted" in labels


def test_tcal_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=2, success=True)
    ds["tcal_fit"].values *= 1.1
    fig = PlotData(ds).tcal_vs_frequency(scan=[1, 2])
    [ax] = fig.axes
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # 2 ref series (R, L, scan-invariant) + 2 scans × 2 pols = 6 entries.
    assert "R ref" in labels
    assert "L ref" in labels
    assert "scan 1 R fit" in labels
    assert "scan 2 L fit" in labels
    assert "scans 1, 2" in ax.get_title()


def test_c_vs_frequency_accepts_scan_list() -> None:
    ds = _make_plot_ds(n_scan=2, n_spw=2, success=True)
    ds["tcal_fit"].values *= 1.05
    fig = PlotData(ds).c_vs_frequency(scan=[1, 2])
    [ax] = fig.axes
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "scan 1 R" in labels
    assert "scan 2 L" in labels


def test_weather_panel_basic() -> None:
    ds = _make_plot_ds(n_scan=2, success=True)
    fig = PlotData(ds).weather_panel()
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3


def test_fit_success_heatmap_basic() -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=3, n_spw=2, success=True)
    fig = PlotData(ds).fit_success_heatmap()
    assert isinstance(fig, Figure)
    # Two scans -> grid is (1 row, 2 cols); 2 active subplots, no hidden.
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.images]
    assert len(visible) == 2


def test_fit_success_heatmap_hides_unused_subplots() -> None:
    # 4 scans -> grid is (2 rows, 3 cols) = 6 tiles, 2 unused/hidden.
    ds = _make_plot_ds(n_scan=4, n_ant=2, n_spw=1, success=True)
    fig = PlotData(ds).fit_success_heatmap()
    with_imgs = [ax for ax in fig.axes if ax.images]
    assert len(with_imgs) == 4
