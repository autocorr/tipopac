"""Canonical `xarray.Dataset` schema for tipopac.

The schema is defined in DESIGN.md §5. This module exposes the contract
(`INPUT_DATA_VARS`, `REQUIRED_COORDS`, `OPTIONAL_DATA_VARS`, `POL_VALUES`),
a validator (`validate`), and the flag-respecting projection helper
(`apply_flags`) used by every reduction over the time axis.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

__all__ = [
    "INPUT_DATA_VARS",
    "OPTIONAL_DATA_VARS",
    "POL_VALUES",
    "REQUIRED_COORDS",
    "SchemaError",
    "apply_flags",
    "validate",
]


class SchemaError(ValueError):
    """Raised when a dataset does not conform to the §5 contract."""


INPUT_DATA_VARS: dict[str, tuple[tuple[str, ...], np.dtype]] = {
    "switched_diff": (
        ("scan", "antenna", "spw", "polarization", "time"),
        np.dtype(np.float32),
    ),
    "switched_sum": (
        ("scan", "antenna", "spw", "polarization", "time"),
        np.dtype(np.float32),
    ),
    "zenith_angle": (("scan", "antenna", "time"), np.dtype(np.float32)),
    "tcal_ref": (("antenna", "spw", "polarization"), np.dtype(np.float32)),
    "weather_T": (("scan", "time"), np.dtype(np.float32)),
    "weather_P": (("scan", "time"), np.dtype(np.float32)),
    "weather_RH": (("scan", "time"), np.dtype(np.float32)),
    "exposure_time": (("scan", "time"), np.dtype(np.float32)),
    "flag": (
        ("scan", "antenna", "spw", "polarization", "time"),
        np.dtype(np.bool_),
    ),
}

REQUIRED_COORDS: dict[str, tuple[tuple[str, ...], np.dtype]] = {
    "frequency": (("spw",), np.dtype(np.float64)),
    "bandwidth": (("spw",), np.dtype(np.float64)),
    "antenna_position": (("antenna", "xyz"), np.dtype(np.float64)),
    "scan_time_start": (("scan",), np.dtype(np.float64)),
    "scan_time_end": (("scan",), np.dtype(np.float64)),
    "time_utc": (("scan", "time"), np.dtype(np.float64)),
}

OPTIONAL_DATA_VARS: dict[str, tuple[tuple[str, ...], np.dtype]] = {
    "Tsys": (
        ("scan", "antenna", "spw", "polarization", "time"),
        np.dtype(np.float32),
    ),
    "sigma_Tsys": (
        ("scan", "antenna", "spw", "polarization", "time"),
        np.dtype(np.float32),
    ),
    "tau_zenith": (("scan", "antenna", "spw"), np.dtype(np.float32)),
    "tau_err": (("scan", "antenna", "spw"), np.dtype(np.float32)),
    "T0": (("scan", "antenna", "spw", "polarization"), np.dtype(np.float32)),
    "tcal_fit": (
        ("scan", "antenna", "spw", "polarization"),
        np.dtype(np.float32),
    ),
    "fit_success": (("scan", "antenna", "spw"), np.dtype(np.bool_)),
    "fit_reason": (("scan", "antenna", "spw"), np.dtype("O")),
    "tau_extrapolated": (("scan", "spw"), np.dtype(np.float32)),
    "am_freq_grid": (("frequency_dense",), np.dtype(np.float64)),
    "am_tau": (("frequency_dense",), np.dtype(np.float64)),
    # Post-fit atmospheric anchor (see design/independent_tau_fit.md):
    # one PWV per antenna, fitted against τ_z(ν) from this antenna's
    # per-spw fits across all scans. Not yet written by any code path.
    "pwv": (("antenna",), np.dtype(np.float32)),
    "pwv_err": (("antenna",), np.dtype(np.float32)),
}

POL_VALUES: tuple[str, ...] = ("R", "L")

_REQUIRED_DIMS: tuple[str, ...] = (
    "scan",
    "antenna",
    "spw",
    "polarization",
    "time",
    "xyz",
)


def _dtype_matches(actual: np.dtype, expected: np.dtype) -> bool:
    if expected == np.dtype("O"):
        return actual == np.dtype("O")
    return np.issubdtype(actual, expected) and actual.itemsize == expected.itemsize


def _check_var(
    ds: xr.Dataset,
    name: str,
    expected_dims: tuple[str, ...],
    expected_dtype: np.dtype,
    *,
    kind: str,
) -> None:
    da = ds[name]
    if tuple(da.dims) != expected_dims:
        raise SchemaError(
            f"{kind} {name!r} has dims {tuple(da.dims)}, expected {expected_dims}"
        )
    if not _dtype_matches(da.dtype, expected_dtype):
        raise SchemaError(
            f"{kind} {name!r} has dtype {da.dtype}, expected {expected_dtype}"
        )


def validate(ds: xr.Dataset) -> None:
    """Assert that `ds` conforms to the canonical schema (DESIGN.md §5).

    Required dims, coords, and input data vars must be present with the
    listed dim signature and dtype. Optional vars (fit results, am output)
    are tolerated when absent but must match the contract when present.

    Raises `SchemaError` on the first mismatch.
    """
    for dim in _REQUIRED_DIMS:
        if dim not in ds.dims:
            raise SchemaError(f"missing required dim {dim!r}")
        if ds.sizes[dim] == 0:
            raise SchemaError(f"dim {dim!r} has length 0")

    if "polarization" not in ds.coords:
        raise SchemaError("missing required coord 'polarization'")
    pol_values = tuple(str(v) for v in ds.coords["polarization"].values.tolist())
    if pol_values != POL_VALUES:
        raise SchemaError(f"polarization coord is {pol_values}, expected {POL_VALUES}")

    for name, (dims, dtype) in REQUIRED_COORDS.items():
        if name not in ds.coords:
            raise SchemaError(f"missing required coord {name!r}")
        _check_var(ds, name, dims, dtype, kind="coord")

    for name, (dims, dtype) in INPUT_DATA_VARS.items():
        if name not in ds.data_vars:
            raise SchemaError(f"missing required data var {name!r}")
        _check_var(ds, name, dims, dtype, kind="data var")

    if ds["flag"].shape != ds["switched_diff"].shape:
        raise SchemaError(
            f"flag shape {ds['flag'].shape} does not match "
            f"switched_diff shape {ds['switched_diff'].shape}"
        )

    for name, (dims, dtype) in OPTIONAL_DATA_VARS.items():
        if name in ds.data_vars:
            _check_var(ds, name, dims, dtype, kind="optional data var")


def apply_flags(ds: xr.Dataset, var: str) -> xr.DataArray:
    """Return `ds[var]` with flagged (or NaN-pad-flagged) samples masked.

    For full-rank variables this is `ds[var].where(~ds.flag)`. For
    partial-rank variables (e.g. `weather_T(scan, time)`,
    `zenith_angle(scan, antenna, time)`) the flag array is first reduced
    over the dims the variable lacks via `.any` — a `(scan, time)` sample
    counts as flagged if any `(antenna, spw, polarization)` cell at that
    sample is flagged.

    Callers must use this helper instead of touching `ds[var]` directly
    for any reduction over the `time` axis (Tsys statistics, residual σ,
    σ-clip masking) — see DESIGN.md §5 "Representation choices".
    """
    da = ds[var]
    flag = ds["flag"]
    extra_dims = tuple(d for d in flag.dims if d not in da.dims)
    if extra_dims:
        flag = flag.any(dim=extra_dims)
    return da.where(~flag)
