"""Unit tests for the opacity row builders in ``tipopac.tables``."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tipopac.tables import measured_opacity_table, model_opacity_table


def _dataset(*, with_am: bool = True) -> xr.Dataset:
    """Two scans × two antennas × three spws, spws *not* in frequency order."""
    freq = np.array([30e9, 20e9, 25e9], dtype=np.float64)
    tau = np.array(
        [
            [[0.10, 0.20, 0.30], [0.12, 0.22, 0.32]],
            [[0.11, np.nan, 0.31], [0.13, np.nan, 0.33]],
        ],
        dtype=np.float32,
    )
    err = np.full_like(tau, 0.01)
    data_vars: dict = {
        "tau_zenith": (("scan", "antenna", "spw"), tau),
        "tau_err": (("scan", "antenna", "spw"), err),
    }
    if with_am:
        grid = np.arange(1e9, 51e9 + 1.0, 100e6)
        data_vars["am_freq_grid"] = (("frequency_dense",), grid)
        data_vars["am_tau"] = (("group", "frequency_dense"), (grid / 1e12)[None, :])
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "scan": np.array([4, 10], dtype=np.intp),
            "antenna": np.array(["ea01", "ea02"], dtype="U4"),
            "spw": np.array([5, 6, 7], dtype=np.intp),
            "frequency": (("spw",), freq),
            "band": (("spw",), np.array(["Ka", "K", "K"], dtype="U4")),
        },
    )


def test_model_opacity_table_slices_to_band_range() -> None:
    """Rows outside 1–51 GHz are dropped; the rest keep grid order."""
    ds = xr.Dataset(
        {
            "am_freq_grid": (
                ("frequency_dense",),
                np.array([0.9e9, 1.0e9, 25e9, 51.0e9, 52.5e9]),
            ),
            "am_tau": (
                ("group", "frequency_dense"),
                np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            ),
        }
    )

    columns, rows = model_opacity_table(ds)

    assert columns == ("frequency_Hz", "tau_model")
    assert rows == [(1.0e9, 0.2), (25e9, 0.3), (51.0e9, 0.4)]


def test_model_opacity_table_without_am_curve() -> None:
    """No am curve → header only, no rows."""
    _, rows = model_opacity_table(_dataset(with_am=False))
    assert rows == []


def test_measured_opacity_table_is_frequency_ordered() -> None:
    """Rows ascend in frequency even though the spw axis is id-ordered."""
    columns, rows = measured_opacity_table(_dataset())

    assert columns == (
        "scan",
        "spw",
        "band",
        "frequency_Hz",
        "tau_measured",
        "tau_err",
        "tau_model",
    )
    freqs = [row[3] for row in rows]
    assert freqs == sorted(freqs)
    # scan 10 has no fit at 20 GHz (spw 6) → 5 rows, not 6.
    assert len(rows) == 5
    assert [(row[0], row[1]) for row in rows] == [
        (4, 6),
        (4, 7),
        (10, 7),
        (4, 5),
        (10, 5),
    ]


def test_measured_opacity_table_weighted_mean_and_model() -> None:
    """τ is the 1/σ² antenna-weighted mean; τ_model is interpolated from am."""
    _, rows = measured_opacity_table(_dataset())
    scan4_20ghz = next(r for r in rows if r[0] == 4 and r[3] == 20e9)

    # Equal errors → plain mean of 0.20 and 0.22, error 0.01/sqrt(2).
    assert scan4_20ghz[4] == pytest.approx(0.21, rel=1e-6)
    assert scan4_20ghz[5] == pytest.approx(0.01 / np.sqrt(2.0), rel=1e-6)
    assert scan4_20ghz[6] == pytest.approx(20e9 / 1e12, rel=1e-6)


def test_measured_opacity_table_without_am_curve() -> None:
    """No am curve → rows still written, with NaN model τ."""
    _, rows = measured_opacity_table(_dataset(with_am=False))
    assert len(rows) == 5
    assert all(np.isnan(row[6]) for row in rows)


def test_measured_opacity_table_without_fits() -> None:
    """No fitted τ on the dataset → header only, no rows."""
    ds = _dataset().drop_vars(["tau_zenith", "tau_err"])
    _, rows = measured_opacity_table(ds)
    assert rows == []
