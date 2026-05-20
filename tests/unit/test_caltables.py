"""Unit tests for tipopac.caltables (DESIGN.md §9.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from pathlib import Path

from tipopac import physics, schema
from tipopac.caltables import (
    _build_opacity_rows,
    _build_tcal_rows,
    write_opacity,
    write_tcal,
)
from tipopac.fit import fit_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tip_ds(
    tau0: float = 0.04,
    freq_Hz: float = 10e9,
    n_time: int = 30,
    n_scan: int = 2,
    n_ant: int = 3,
    n_spw: int = 2,
    *,
    rng: np.random.Generator | None = None,
) -> xr.Dataset:
    if rng is None:
        rng = np.random.default_rng(0)

    T_surf = 280.0
    Twmt = float(physics.k2nt(physics.weighted_mean_atm_T(T_surf), freq_Hz))
    z = np.linspace(35.0, 65.0, n_time)
    tcal = 5.0
    T0 = 50.0

    switched_diff = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    switched_sum = np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            for i_w in range(n_spw):
                tsys = physics.tsys_model(z, T0, tau0, Twmt) + rng.normal(
                    0.0, 0.3, n_time
                )
                switched_sum[i_sc, i_a, i_w, 0, :] = (2.0 * tsys / tcal).astype(
                    np.float32
                )
                switched_sum[i_sc, i_a, i_w, 1, :] = (2.0 * tsys / tcal).astype(
                    np.float32
                )

    zenith_arr = np.zeros((n_scan, n_ant, n_time), dtype=np.float32)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            zenith_arr[i_sc, i_a, :] = z.astype(np.float32)

    return xr.Dataset(
        data_vars={
            "switched_diff": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_diff,
            ),
            "switched_sum": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_sum,
            ),
            "zenith_angle": (("scan", "antenna", "time"), zenith_arr),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.full((n_ant, n_spw, 2), tcal, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((n_scan, n_time), T_surf, dtype=np.float32),
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
        },
        coords={
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.array(list(range(n_spw)), dtype=np.intp),
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
                np.arange(n_scan, dtype=np.float64) * 120.0,
            ),
            "scan_time_end": (
                ("scan",),
                np.arange(n_scan, dtype=np.float64) * 120.0 + 90.0,
            ),
            "time_utc": (
                ("scan", "time"),
                np.tile(np.arange(n_time, dtype=np.float64), (n_scan, 1))
                + np.arange(n_scan, dtype=np.float64)[:, None] * 120.0,
            ),
        },
        attrs={"source_path": "fake.ms", "source_format": "ms", "observatory": "VLA"},
    )


def _make_fitted_ds(**kwargs: object) -> xr.Dataset:
    ds = _make_tip_ds(**kwargs)  # type: ignore[arg-type]
    fit_dataset(ds, "tau_per_antenna")
    return ds


def _make_tcalsolve_ds(**kwargs: object) -> xr.Dataset:
    ds = _make_tip_ds(**kwargs)  # type: ignore[arg-type]
    fit_dataset(ds, "tcal_solve")
    return ds


# ---------------------------------------------------------------------------
# _build_opacity_rows
# ---------------------------------------------------------------------------


def test_build_opacity_rows_count() -> None:
    n_scan, n_ant, n_spw = 2, 3, 2
    ds = _make_fitted_ds(n_scan=n_scan, n_ant=n_ant, n_spw=n_spw)
    rows = _build_opacity_rows(ds)
    assert len(rows) == n_scan * n_ant * n_spw


def test_build_opacity_rows_ordering() -> None:
    n_scan, n_ant, n_spw = 2, 3, 2
    ds = _make_fitted_ds(n_scan=n_scan, n_ant=n_ant, n_spw=n_spw)
    rows = _build_opacity_rows(ds)
    k = 0
    for i in range(n_scan):
        for a in range(n_ant):
            for s in range(n_spw):
                row = rows[k]
                assert row["SCAN_NUMBER"] == int(ds.coords["scan"].values[i])
                assert row["ANTENNA1"] == a
                assert row["SPECTRAL_WINDOW_ID"] == int(ds.coords["spw"].values[s])
                k += 1


def test_build_opacity_rows_time() -> None:
    ds = _make_fitted_ds(n_scan=2, n_ant=1, n_spw=1)
    rows = _build_opacity_rows(ds)
    for i in range(2):
        expected = float(
            (
                ds.coords["scan_time_start"].values[i]
                + ds.coords["scan_time_end"].values[i]
            )
            / 2.0
        )
        assert math.isclose(rows[i]["TIME"], expected)


def test_build_opacity_rows_successful_scan() -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_opacity_rows(ds)[0]
    # Good data → unflagged with positive tau and SNR
    assert not row["FLAG"][0, 0]
    assert row["FPARAM"][0, 0] > 0.0
    assert row["SNR"][0, 0] > 0.0
    assert row["PARAMERR"][0, 0] > 0.0


def test_build_opacity_rows_failed_scan() -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    # Force all fits to fail by overwriting fit_success
    ds["fit_success"].values[:] = False
    row = _build_opacity_rows(ds)[0]
    assert row["FLAG"][0, 0]
    assert row["FPARAM"][0, 0] == 0.0
    assert row["PARAMERR"][0, 0] == 0.0
    assert row["SNR"][0, 0] == 1.0


def test_build_opacity_rows_array_shapes() -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_opacity_rows(ds)[0]
    assert row["FPARAM"].shape == (1, 1)
    assert row["PARAMERR"].shape == (1, 1)
    assert row["FLAG"].shape == (1, 1)
    assert row["SNR"].shape == (1, 1)


def test_build_opacity_rows_fixed_fields() -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_opacity_rows(ds)[0]
    assert row["FIELD_ID"] == -1
    assert row["ANTENNA2"] == -1


# ---------------------------------------------------------------------------
# _build_tcal_rows
# ---------------------------------------------------------------------------


def test_build_tcal_rows_count() -> None:
    n_scan, n_ant, n_spw = 2, 3, 2
    ds = _make_tcalsolve_ds(n_scan=n_scan, n_ant=n_ant, n_spw=n_spw)
    rows = _build_tcal_rows(ds)
    assert len(rows) == n_scan * n_ant * n_spw


def test_build_tcal_rows_noise_cal_shape() -> None:
    ds = _make_tcalsolve_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_tcal_rows(ds)[0]
    assert row["NOISE_CAL"].shape == (2, 2)


def test_build_tcal_rows_zero_row1() -> None:
    ds = _make_tcalsolve_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_tcal_rows(ds)[0]
    np.testing.assert_array_equal(row["NOISE_CAL"][1], [0.0, 0.0])


def test_build_tcal_rows_noise_cal_values() -> None:
    ds = _make_tcalsolve_ds(n_scan=1, n_ant=1, n_spw=1)
    rows = _build_tcal_rows(ds)
    tcal_R = float(ds["tcal_fit"].values[0, 0, 0, 0])
    tcal_L = float(ds["tcal_fit"].values[0, 0, 0, 1])
    np.testing.assert_allclose(rows[0]["NOISE_CAL"][0], [tcal_R, tcal_L])


def test_build_tcal_rows_cal_load_names() -> None:
    ds = _make_tcalsolve_ds(n_scan=1, n_ant=1, n_spw=1)
    row = _build_tcal_rows(ds)[0]
    assert row["NUM_CAL_LOAD"] == 2
    assert row["NUM_RECEPTOR"] == 2
    assert row["CAL_LOAD_NAMES"][0, 0] == "NOISE_TUBE_LOAD"
    assert row["CAL_LOAD_NAMES"][1, 0] == "SOLAR_FILTER"


# ---------------------------------------------------------------------------
# write_tcal guard
# ---------------------------------------------------------------------------


def test_write_tcal_requires_tcalsolve_mode(tmp_path: object) -> None:
    # tau_per_antenna also populates tcal_fit (with reference values), but the
    # guard must reject it — only tcal_solve fits are meaningful for a Tcal table.
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    assert ds.attrs.get("mode") == "tau_per_antenna"
    with pytest.raises(ValueError, match="tcal_solve"):
        write_tcal(ds, tmp_path / "t.cal")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Slow integration tests — require data/tip_test.ms
# ---------------------------------------------------------------------------

MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"


@pytest.mark.slow
def test_write_opacity_roundtrip(tmp_path: Path) -> None:
    """Write a real TOpac table and read it back via casatools.table."""
    import casatools

    from tipopac.readers.ms import MSReader

    assert MSReader.supports(MS_PATH), f"tip_test.ms not found at {MS_PATH}"
    ds = MSReader(MS_PATH).read()
    fit_dataset(ds, "global_tau")
    out = tmp_path / "test_opacity.cal"
    write_opacity(ds, out)

    tb = casatools.table()
    tb.open(str(out))
    nrows = tb.nrows()
    n_expected = ds.sizes["scan"] * ds.sizes["antenna"] * ds.sizes["spw"]
    assert nrows == n_expected

    # All rows have the correct column presence
    assert "FPARAM" in tb.colnames()
    assert "FLAG" in tb.colnames()
    assert "PARAMERR" in tb.colnames()
    assert "SNR" in tb.colnames()

    # At least one row should be unflagged (good data)
    flags = tb.getcol("FLAG")  # shape (1, 1, nrows) for TOpac
    assert not flags.all(), "Expected at least one unflagged row in a good MS"

    tb.close()
