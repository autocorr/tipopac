"""Unit tests for tipopac.fit — tau_per_antenna mode (DESIGN.md §6.3, §11.1)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tipopac import schema
from tipopac import physics
from tipopac.fit import fit_dataset


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------


def _make_tip_ds(
    T0_R: float = 50.0,
    T0_L: float = 48.0,
    tau0: float = 0.04,  # realistic for VLA L/C/X-band; 0.08 at 10 GHz exceeds stdTsys=5 K gate
    freq_Hz: float = 10e9,
    n_time: int = 30,
    noise_K: float = 0.3,
    *,
    rng: np.random.Generator | None = None,
    flat_za: bool = False,
    za_range: tuple[float, float] = (35.0, 65.0),
    n_scan: int = 1,
    n_ant: int = 1,
    n_spw: int = 1,
) -> xr.Dataset:
    """Build a minimal dataset with synthetic tipping data.

    switched_diff = 1.0, tcal_ref = 5.0 K, so
    Tsys = switched_sum / 2.0 * tcal_ref = switched_sum * 2.5.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    T_surf = 280.0  # K
    Twmt = float(physics.k2nt(physics.weighted_mean_atm_T(T_surf), freq_Hz))

    z = np.linspace(*za_range, n_time) if not flat_za else np.full(n_time, za_range[0])

    tsys_R = physics.tsys_model(z, T0_R, tau0, Twmt) + rng.normal(0.0, noise_K, n_time)
    tsys_L = physics.tsys_model(z, T0_L, tau0, Twmt) + rng.normal(0.0, noise_K, n_time)

    tcal = 5.0
    # Tsys = (switched_sum/2) / switched_diff * tcal_ref
    # With switched_diff=1, switched_sum = 2 * tsys / tcal
    switched_diff = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    switched_sum = np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            for i_w in range(n_spw):
                switched_sum[i_sc, i_a, i_w, 0, :] = (2.0 * tsys_R / tcal).astype(
                    np.float32
                )
                switched_sum[i_sc, i_a, i_w, 1, :] = (2.0 * tsys_L / tcal).astype(
                    np.float32
                )

    zenith_arr = np.zeros((n_scan, n_ant, n_time), dtype=np.float32)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            zenith_arr[i_sc, i_a, :] = z.astype(np.float32)

    ant_names = [f"ea{i + 1:02d}" for i in range(n_ant)]
    spw_ids = list(range(n_spw))

    ds = xr.Dataset(
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
            "antenna": ant_names,
            "spw": np.array(spw_ids, dtype=np.intp),
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
    )
    return ds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fit_tau_per_antenna_recovers_params() -> None:
    """Fit must recover tau0 within max(0.005, 0.05·tau_true) and T0 within 2 K.

    tau0=0.04 at 10 GHz: Tsys swing ~11 K, std ~3.4 K < 5 K gate (≤18 GHz).
    """
    tau_true = 0.04
    T0_R_true, T0_L_true = 50.0, 48.0
    ds = _make_tip_ds(T0_R=T0_R_true, T0_L=T0_L_true, tau0=tau_true, noise_K=0.3)

    fit_dataset(ds, mode="tau_per_antenna")

    assert bool(ds["fit_success"].values[0, 0, 0]), ds["fit_reason"].values[0, 0, 0]
    assert ds["fit_reason"].values[0, 0, 0] == "ok"

    tau_fit = float(ds["tau_zenith"].values[0, 0, 0])
    tol = max(0.005, 0.05 * tau_true)
    assert abs(tau_fit - tau_true) < tol, (
        f"tau recovered {tau_fit:.4f}, true {tau_true:.4f}, tol {tol:.4f}"
    )

    T0_R_fit = float(ds["T0"].values[0, 0, 0, 0])
    T0_L_fit = float(ds["T0"].values[0, 0, 0, 1])
    assert abs(T0_R_fit - T0_R_true) < 2.0, (
        f"T0_R recovered {T0_R_fit:.2f}, true {T0_R_true:.2f}"
    )
    assert abs(T0_L_fit - T0_L_true) < 2.0, (
        f"T0_L recovered {T0_L_fit:.2f}, true {T0_L_true:.2f}"
    )


def test_fit_tau_err_is_positive() -> None:
    """tau_err must be a small positive number for a clean fit."""
    ds = _make_tip_ds()
    fit_dataset(ds, mode="tau_per_antenna")
    tau_err = float(ds["tau_err"].values[0, 0, 0])
    assert tau_err > 0.0
    assert tau_err < 0.05  # sanity: uncertainty should be < 50% of tau


def test_fit_tau_per_antenna_stores_schema_vars() -> None:
    """After fitting, all optional schema vars must have correct dims/dtypes."""
    ds = _make_tip_ds()
    fit_dataset(ds, mode="tau_per_antenna")
    schema.validate(ds)  # raises SchemaError on mismatch


def test_fit_tcal_fit_equals_tcal_ref_in_tau_per_antenna() -> None:
    """In tau_per_antenna mode, tcal_fit must equal tcal_ref (no correction)."""
    ds = _make_tip_ds()
    fit_dataset(ds, mode="tau_per_antenna")
    np.testing.assert_array_equal(
        ds["tcal_fit"].values[0, 0, 0, :],
        ds["tcal_ref"].values[0, 0, :],
    )


def test_fit_dz_too_small() -> None:
    """All identical ZA → dz_too_small gate fires, fit_success=False."""
    ds = _make_tip_ds(flat_za=True)
    fit_dataset(ds, mode="tau_per_antenna")
    assert not bool(ds["fit_success"].values[0, 0, 0])
    assert ds["fit_reason"].values[0, 0, 0] == "dz_too_small"


def test_fit_too_few_samples() -> None:
    """Only 2 unflagged samples → too_few_samples, fit_success=False."""
    ds = _make_tip_ds(n_time=10)
    # Flag all but two samples
    ds["flag"].values[0, 0, 0, :, 2:] = True
    fit_dataset(ds, mode="tau_per_antenna")
    assert not bool(ds["fit_success"].values[0, 0, 0])
    assert ds["fit_reason"].values[0, 0, 0] == "too_few_samples"


def test_fit_resid_clip_removes_outlier() -> None:
    """A single large outlier is clipped and the fit still converges."""
    ds = _make_tip_ds(noise_K=0.0)  # noiseless baseline
    # Inject a 10 K outlier in R-pol at time index 5
    ds["switched_sum"].values[0, 0, 0, 0, 5] += 2.0 * 10.0 / 5.0  # +10 K in Tsys
    fit_dataset(ds, mode="tau_per_antenna")
    assert bool(ds["fit_success"].values[0, 0, 0]), (
        f"Expected success after clip, got: {ds['fit_reason'].values[0, 0, 0]}"
    )


def test_fit_invalid_mode_raises() -> None:
    """Unrecognised mode raises ValueError before touching the dataset."""
    ds = _make_tip_ds()
    with pytest.raises(ValueError, match="mode"):
        fit_dataset(ds, mode="banana")


def test_fit_global_tau_not_implemented() -> None:
    """global_tau raises NotImplementedError (milestone 5)."""
    ds = _make_tip_ds()
    with pytest.raises(NotImplementedError):
        fit_dataset(ds, mode="global_tau")


def test_fit_mode_stored_in_attrs() -> None:
    """ds.attrs['mode'] is set to the mode string after fitting."""
    ds = _make_tip_ds()
    fit_dataset(ds, mode="tau_per_antenna")
    assert ds.attrs["mode"] == "tau_per_antenna"


def test_fit_multi_scan_multi_ant() -> None:
    """fit_dataset handles multiple scans and antennas without error."""
    ds = _make_tip_ds(n_scan=2, n_ant=3, n_spw=2)
    fit_dataset(ds, mode="tau_per_antenna")
    assert ds["fit_success"].shape == (2, 3, 2)
    # All cells should succeed (clean synthetic data)
    assert ds["fit_success"].values.all()
