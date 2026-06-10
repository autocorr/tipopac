"""Unit tests for `tipopac.schema` (DESIGN.md §5)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tipopac.schema import (
    POL_VALUES,
    SchemaError,
    apply_flags,
    validate,
)


def make_minimal_ds(
    n_scans: int = 2,
    n_ant: int = 3,
    n_spw: int = 2,
    n_time: int = 4,
    *,
    with_fit_results: bool = False,
) -> xr.Dataset:
    """Build a dataset conforming to the schema's input contract."""
    n_pol = len(POL_VALUES)
    full_shape = (n_scans, n_ant, n_spw, n_pol, n_time)
    full_dims = ("scan", "antenna", "spw", "polarization", "time")

    rng = np.random.default_rng(0)

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "switched_diff": (
            full_dims,
            rng.standard_normal(full_shape).astype(np.float32),
        ),
        "switched_sum": (
            full_dims,
            rng.standard_normal(full_shape).astype(np.float32),
        ),
        "zenith_angle": (
            ("scan", "antenna", "time"),
            rng.uniform(20.0, 70.0, (n_scans, n_ant, n_time)).astype(np.float32),
        ),
        "tcal_ref": (
            ("antenna", "spw", "polarization"),
            np.ones((n_ant, n_spw, n_pol), dtype=np.float32),
        ),
        "weather_T": (
            ("scan", "time"),
            np.full((n_scans, n_time), 280.0, dtype=np.float32),
        ),
        "weather_P": (
            ("scan", "time"),
            np.full((n_scans, n_time), 85000.0, dtype=np.float32),
        ),
        "weather_RH": (
            ("scan", "time"),
            np.full((n_scans, n_time), 0.3, dtype=np.float32),
        ),
        "exposure_time": (
            ("scan", "time"),
            np.full((n_scans, n_time), 1.0, dtype=np.float32),
        ),
        "flag": (full_dims, np.zeros(full_shape, dtype=np.bool_)),
    }

    from tipopac.bands import band_for_frequency

    freqs = np.linspace(1.0e9, 50.0e9, n_spw).astype(np.float64)
    coords: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "polarization": (("polarization",), np.array(POL_VALUES)),
        "frequency": (("spw",), freqs),
        "bandwidth": (
            ("spw",),
            np.full(n_spw, 2.0e9, dtype=np.float64),
        ),
        "band": (
            ("spw",),
            np.array([band_for_frequency(float(f)) for f in freqs], dtype="U4"),
        ),
        "antenna_position": (
            ("antenna", "xyz"),
            np.zeros((n_ant, 3), dtype=np.float64),
        ),
        "scan_time_start": (
            ("scan",),
            np.arange(n_scans, dtype=np.float64) * 3600.0,
        ),
        "scan_time_end": (
            ("scan",),
            np.arange(n_scans, dtype=np.float64) * 3600.0 + 60.0,
        ),
        "time_utc": (
            ("scan", "time"),
            np.tile(np.arange(n_time, dtype=np.float64), (n_scans, 1))
            + np.arange(n_scans, dtype=np.float64)[:, None] * 3600.0,
        ),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    if with_fit_results:
        ds["Tsys"] = (full_dims, np.full(full_shape, 100.0, dtype=np.float32))
        ds["tau_zenith"] = (
            ("scan", "antenna", "spw"),
            np.full((n_scans, n_ant, n_spw), 0.05, dtype=np.float32),
        )
        ds["tau_err"] = (
            ("scan", "antenna", "spw"),
            np.full((n_scans, n_ant, n_spw), 0.001, dtype=np.float32),
        )
        ds["T0"] = (
            ("scan", "antenna", "spw", "polarization"),
            np.full((n_scans, n_ant, n_spw, n_pol), 50.0, dtype=np.float32),
        )
        ds["tcal_fit"] = (
            ("scan", "antenna", "spw", "polarization"),
            np.full((n_scans, n_ant, n_spw, n_pol), 1.0, dtype=np.float32),
        )
        ds["fit_success"] = (
            ("scan", "antenna", "spw"),
            np.ones((n_scans, n_ant, n_spw), dtype=np.bool_),
        )
        ds["fit_reason"] = (
            ("scan", "antenna", "spw"),
            np.full((n_scans, n_ant, n_spw), "ok", dtype=object),
        )

    return ds


def test_validate_passes_on_minimal_ds() -> None:
    assert validate(make_minimal_ds()) is None


def test_validate_passes_with_fit_results() -> None:
    assert validate(make_minimal_ds(with_fit_results=True)) is None


def test_validate_rejects_wrong_dtype() -> None:
    ds = make_minimal_ds()
    ds["switched_diff"] = ds["switched_diff"].astype(np.float64)
    with pytest.raises(SchemaError, match="switched_diff"):
        validate(ds)


def test_validate_rejects_missing_var() -> None:
    ds = make_minimal_ds().drop_vars("tcal_ref")
    with pytest.raises(SchemaError, match="tcal_ref"):
        validate(ds)


def test_validate_rejects_wrong_dim_order() -> None:
    ds = make_minimal_ds()
    ds["switched_diff"] = ds["switched_diff"].transpose(
        "antenna", "scan", "spw", "polarization", "time"
    )
    with pytest.raises(SchemaError, match="switched_diff"):
        validate(ds)


def test_validate_rejects_bad_polarization() -> None:
    ds = make_minimal_ds()
    ds = ds.assign_coords(polarization=np.array(["X", "Y"]))
    with pytest.raises(SchemaError, match="polarization"):
        validate(ds)


def test_validate_rejects_optional_dtype_drift() -> None:
    ds = make_minimal_ds()
    ds["tau_zenith"] = (
        ("scan", "antenna", "spw"),
        np.zeros(
            (ds.sizes["scan"], ds.sizes["antenna"], ds.sizes["spw"]),
            dtype=np.float64,
        ),
    )
    with pytest.raises(SchemaError, match="tau_zenith"):
        validate(ds)


def test_apply_flags_full_rank() -> None:
    ds = make_minimal_ds()
    ds["flag"].values[0, 0, 0, 0, 0] = True
    out = apply_flags(ds, "switched_diff")
    assert out.shape == ds["switched_diff"].shape
    assert np.isnan(out.values[0, 0, 0, 0, 0])
    assert np.isfinite(out.values[0, 0, 0, 0, 1])
    assert np.isfinite(out.values[1, 0, 0, 0, 0])


def test_apply_flags_partial_rank() -> None:
    ds = make_minimal_ds()
    ds["flag"].values[0, 0, 0, 0, 2] = True
    out = apply_flags(ds, "weather_T")
    assert out.dims == ("scan", "time")
    assert out.shape == (ds.sizes["scan"], ds.sizes["time"])
    assert np.isnan(out.values[0, 2])
    assert np.isfinite(out.values[0, 0])
    assert np.isfinite(out.values[1, 2])


def test_apply_flags_partial_rank_any_semantics() -> None:
    ds = make_minimal_ds()
    n_ant, n_spw, n_pol = (
        ds.sizes["antenna"],
        ds.sizes["spw"],
        ds.sizes["polarization"],
    )
    # All cells flagged at (scan=0, time=1).
    ds["flag"].values[0, :, :, :, 1] = True
    # Exactly one cell flagged at (scan=1, time=3).
    ds["flag"].values[1, n_ant - 1, n_spw - 1, n_pol - 1, 3] = True
    out = apply_flags(ds, "weather_T")
    assert np.isnan(out.values[0, 1])
    assert np.isnan(out.values[1, 3])
    assert np.isfinite(out.values[0, 0])
    assert np.isfinite(out.values[1, 0])


def test_apply_flags_missing_var_raises() -> None:
    ds = make_minimal_ds()
    with pytest.raises(KeyError):
        apply_flags(ds, "no_such_var")
