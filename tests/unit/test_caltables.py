"""Unit tests for tipopac.caltables (design.md §9.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from pathlib import Path

from tipopac import caltables
from tipopac.caltables import (
    _build_opacity_rows,
    _build_tcal_rows,
    write_opacity,
    write_tcal,
)
from tipopac.fit import fit_dataset

from tests.factories import make_tipping_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tip_ds(**kwargs: object) -> xr.Dataset:
    defaults: dict[str, object] = {
        "T0_R": 50.0,
        "T0_L": 50.0,
        "seed": 0,
        "n_scan": 2,
        "n_ant": 3,
        "n_spw": 2,
        "with_cmb": False,
        "per_cell_noise": True,
        "attrs": {
            "source_path": "fake.ms",
            "source_format": "ms",
            "observatory": "VLA",
        },
    }
    return make_tipping_dataset(**{**defaults, **kwargs})  # type: ignore[arg-type]


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
    """scan slow, then spw, then antenna fastest — the gencal 'opac' order."""
    n_scan, n_ant, n_spw = 2, 3, 2
    ds = _make_fitted_ds(n_scan=n_scan, n_ant=n_ant, n_spw=n_spw)
    rows = _build_opacity_rows(ds)
    k = 0
    for i in range(n_scan):
        for s in range(n_spw):
            for a in range(n_ant):
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


def test_build_opacity_rows_tau_is_antenna_weighted_mean() -> None:
    """Every antenna in a (scan, spw) carries the same 1/σ²-weighted τ."""
    ds = _make_fitted_ds(n_scan=1, n_ant=2, n_spw=1)
    ds["tau_zenith"].values[0, :, 0] = [0.10, 0.20]
    ds["tau_err"].values[0, :, 0] = [0.01, 0.02]

    rows = _build_opacity_rows(ds)
    assert len(rows) == 2
    # w = [1e4, 2.5e3] → mean 0.12, σ = (1/1.25e4)**0.5
    for row in rows:
        assert math.isclose(row["FPARAM"][0, 0], 0.12, rel_tol=1e-6)
        assert math.isclose(row["PARAMERR"][0, 0], (1.0 / 1.25e4) ** 0.5, rel_tol=1e-6)
        assert not row["FLAG"][0, 0]


def test_build_opacity_rows_one_bad_antenna_does_not_flag_the_block() -> None:
    """A screened-out antenna still gets the block's τ — opacity is a sky property."""
    ds = _make_fitted_ds(n_scan=1, n_ant=2, n_spw=1)
    ds["fit_success"].values[0, 1, 0] = False

    rows = _build_opacity_rows(ds)
    assert [bool(r["FLAG"][0, 0]) for r in rows] == [False, False]
    assert rows[0]["FPARAM"][0, 0] == rows[1]["FPARAM"][0, 0]


def test_build_opacity_rows_excludes_unsuccessful_fits_from_the_mean() -> None:
    """Unlike measured_opacity_table, only fit_success cells reach the caltable."""
    ds = _make_fitted_ds(n_scan=1, n_ant=2, n_spw=1)
    ds["tau_zenith"].values[0, :, 0] = [0.10, 0.99]
    ds["tau_err"].values[0, :, 0] = [0.01, 0.01]
    ds["fit_success"].values[0, 1, 0] = False

    row = _build_opacity_rows(ds)[0]
    assert math.isclose(row["FPARAM"][0, 0], 0.10, rel_tol=1e-6)


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


def test_write_tcal_requires_tcal_vars(tmp_path: object) -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1).drop_vars("tcal_fit")
    with pytest.raises(ValueError, match="tcal_fit"):
        write_tcal(ds, tmp_path / "t.cal")  # type: ignore[arg-type]


def test_write_tcal_guard_ignores_the_mode_attr(tmp_path: object) -> None:
    # The old guard tested attrs["mode"] == "tcal_solve", but api.fit overwrites
    # that with the public label, so no API-produced dataset could ever pass.
    # Missing vars must now be the only path that raises.
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1)
    ds.attrs["mode"] = "independent_tau"
    with pytest.raises(ValueError, match="tcal_fit"):
        write_tcal(ds.drop_vars("tcal_fit"), tmp_path / "t.cal")  # type: ignore[arg-type]
    # With both vars present the guard passes and the row builder runs.
    assert len(_build_tcal_rows(ds)) == 1


def test_build_tcal_rows_falls_back_to_tcal_ref() -> None:
    ds = _make_tcalsolve_ds(n_scan=1, n_ant=1, n_spw=1)
    ds["tcal_fit"].values[0, 0, 0, :] = np.nan
    row = _build_tcal_rows(ds)[0]
    np.testing.assert_allclose(
        row["NOISE_CAL"][0], ds["tcal_ref"].values[0, 0, :], rtol=1e-6
    )


# ---------------------------------------------------------------------------
# write_opacity guard
# ---------------------------------------------------------------------------


def test_write_opacity_requires_fit_vars(tmp_path: Path) -> None:
    ds = _make_tip_ds(n_scan=1, n_ant=1, n_spw=1)
    with pytest.raises(ValueError) as exc:
        write_opacity(ds, tmp_path / "o.cal")
    for name in ("tau_zenith", "tau_err", "fit_success"):
        assert name in str(exc.value)


@pytest.mark.parametrize("var", ["tau_zenith", "tau_err", "fit_success"])
def test_write_opacity_requires_each_fit_var(tmp_path: Path, var: str) -> None:
    ds = _make_fitted_ds(n_scan=1, n_ant=1, n_spw=1).drop_vars(var)
    with pytest.raises(ValueError, match=var):
        write_opacity(ds, tmp_path / "o.cal")


def test_write_opacity_guard_precedes_casa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard runs before createcaltable, so no partial table is left behind."""

    def _boom() -> None:
        raise AssertionError("casatools imported before the guard raised")

    monkeypatch.setattr(caltables, "import_casatools", _boom)
    ds = _make_tip_ds(n_scan=1, n_ant=1, n_spw=1)
    with pytest.raises(ValueError, match="tau_zenith"):
        write_opacity(ds, tmp_path / "o.cal")


# ---------------------------------------------------------------------------
# Slow integration tests — require data/tip_test.ms
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_write_opacity_roundtrip(tmp_path: Path, ds_ms, n_workers) -> None:
    """Write a real TOpac table and read it back via casatools.table."""
    import casatools

    ds = ds_ms
    fit_dataset(ds, "tau_per_antenna", n_workers=n_workers)
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


@pytest.mark.slow
def test_write_tcal_roundtrip(tmp_path: Path, ds_ms, n_workers) -> None:
    """Write a real CALDEVICE Tcal table and read it back via casatools.table."""
    import casatools

    ds = ds_ms
    fit_dataset(ds, "tau_per_antenna", n_workers=n_workers)
    out = tmp_path / "test_tcal.cal"
    write_tcal(ds, out)

    tb = casatools.table()
    tb.open(str(out))
    assert tb.nrows() == ds.sizes["scan"] * ds.sizes["antenna"] * ds.sizes["spw"]
    for col in ("ANTENNA_ID", "SPECTRAL_WINDOW_ID", "TIME", "NOISE_CAL"):
        assert col in tb.colnames()

    noise_cal = tb.getcell("NOISE_CAL", 0)
    assert noise_cal.shape == (2, 2)
    # The tcal_ref fallback keeps every row finite and positive.
    assert np.all(np.isfinite(noise_cal))
    assert np.all(noise_cal[0] > 0.0)

    tb.close()
