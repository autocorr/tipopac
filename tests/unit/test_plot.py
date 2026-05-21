"""Unit tests for tipopac.plot (DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tipopac import schema
from tipopac.plot import plot_dataset


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
    """Minimal dataset ready for plot_dataset.

    ZA values span 35–80 deg; Tsys is synthetic but positive.
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

    coords = {
        "scan": np.arange(1, n_scan + 1, dtype=np.intp),
        "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
        "spw": np.arange(n_spw, dtype=np.intp),
        "polarization": list(schema.POL_VALUES),
        "xyz": ["X", "Y", "Z"],
        "frequency": (("spw",), np.full(n_spw, freq_Hz, dtype=np.float64)),
        "bandwidth": (("spw",), np.full(n_spw, 2e9, dtype=np.float64)),
        "antenna_position": (
            ("antenna", "xyz"),
            np.zeros((n_ant, 3), dtype=np.float64),
        ),
        "scan_time_start": (
            ("scan",),
            np.linspace(5131296000.0, 5131296000.0 + 120.0 * n_scan, n_scan, dtype=np.float64),
        ),
        "scan_time_end": (
            ("scan",),
            np.linspace(5131296090.0, 5131296090.0 + 120.0 * n_scan, n_scan, dtype=np.float64),
        ),
        "time_utc": (
            ("scan", "time"),
            np.tile(np.linspace(5131296000.0, 5131296090.0, n_time), (n_scan, 1)),
        ),
    }

    return xr.Dataset(data_vars=data_vars, coords=coords)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plot_writes_png(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1


def test_plot_filename_convention(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True)
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert pngs[0].name == "tippingcurve_spw_0_ea01_scan_1.png"


def test_plot_skips_failures(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=False)
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 0


def test_plot_creates_output_dir(tmp_path: Path) -> None:
    out = tmp_path / "new_subdir" / "plots"
    ds = _make_plot_ds(success=True)
    plot_dataset(ds, out)
    assert out.is_dir()
    assert len(list(out.glob("*.png"))) == 1


def test_plot_with_am_overlay(tmp_path: Path) -> None:
    ds = _make_plot_ds(success=True, with_am=True)
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1


def test_plot_multi_cell(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=2, n_ant=3, n_spw=2, success=True)
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 2 * 3 * 2


def test_plot_partial_success(tmp_path: Path) -> None:
    ds = _make_plot_ds(n_scan=1, n_ant=2, n_spw=1, success=True)
    # Flag one antenna as failed after construction
    ds["fit_success"].values[0, 1, 0] = False
    plot_dataset(ds, tmp_path)
    pngs = list(tmp_path.glob("*.png"))
    assert len(pngs) == 1
