"""Unit tests for tipopac.fit — all three modes (DESIGN.md §6.3, §11.1)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tipopac import schema
from tipopac import physics
from tipopac.fit import fit_dataset


def _band_label(freq_Hz: float) -> str:
    """Stand-in band label for synthetic datasets — fit logic ignores it."""
    return "K"


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

    tsys_R_clean = physics.tsys_model(z, T0_R, tau0, Twmt)
    tsys_L_clean = physics.tsys_model(z, T0_L, tau0, Twmt)
    tsys_R = tsys_R_clean + rng.normal(0.0, noise_K, n_time)
    tsys_L = tsys_L_clean + rng.normal(0.0, noise_K, n_time)

    tcal = 5.0
    # Pick exposure_time so radiometer σ_Tsys ≈ max(noise_K, 0.01 K) at the
    # scan-mean Tsys. This keeps synthetic test data consistent with the
    # reader-derived σ that the fit consumes:
    #   σ = 2 · Tsys² / (Tcal · √(Δν·τ_int))  →  τ_int = 4·Tsys⁴ / (Tcal²·σ²·Δν)
    bandwidth_Hz = 2e9
    Tsys_typ = float(np.mean((tsys_R_clean + tsys_L_clean) / 2.0))
    sigma_eff = max(float(noise_K), 0.01)
    expo_s = float(4.0 * Tsys_typ**4 / (tcal**2 * sigma_eff**2 * bandwidth_Hz))
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
            "exposure_time": (
                ("scan", "time"),
                np.full((n_scan, n_time), expo_s, dtype=np.float32),
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
            "band": (
                ("spw",),
                np.array(
                    [_band_label(freq_Hz)] * n_spw,
                    dtype="U4",
                ),
            ),
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


def test_fit_zero_airmass_leverage_is_poorly_identified() -> None:
    """Flat ZA (no airmass leverage) → poorly_identified, not silent success.

    Replaces the legacy `dz_too_small` geometric gate with an identifiability
    signal derived from the fit covariance (σ_τ/τ > _TAU_REL_ERR_MAX). See
    design/model_refactor.md §1.3.
    """
    ds = _make_tip_ds(flat_za=True)
    fit_dataset(ds, mode="tau_per_antenna")
    assert not bool(ds["fit_success"].values[0, 0, 0])
    assert ds["fit_reason"].values[0, 0, 0] == "poorly_identified"


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


def test_fit_mode_stored_in_attrs() -> None:
    """ds.attrs['mode'] is set to the mode string after fitting."""
    ds = _make_tip_ds()
    fit_dataset(ds, mode="tau_per_antenna")
    assert ds.attrs["mode"] == "tau_per_antenna"


def _make_tcal_ds(
    tau0: float = 0.06,
    T0_R: float = 50.0,
    T0_L: float = 48.0,
    c_R: list[float] | None = None,
    c_L: list[float] | None = None,
    freq_Hz: float = 10e9,
    n_time: int = 30,
    noise_K: float = 0.3,
    rng: np.random.Generator | None = None,
) -> xr.Dataset:
    """Dataset with per-antenna Tcal correction factors applied to switched_sum.

    Simulates miscalibrated Tcal: Tsys_measured = Tsys_true / c_a.
    The tcal_solve fitter should recover c_a within 1%.
    """
    n_ant = len(c_R) if c_R is not None else 3
    if c_R is None:
        c_R = [1.0] * n_ant
    if c_L is None:
        c_L = [1.0] * n_ant
    if rng is None:
        rng = np.random.default_rng(99)

    T_surf = 280.0
    Twmt = float(physics.k2nt(physics.weighted_mean_atm_T(T_surf), freq_Hz))
    z = np.linspace(35.0, 65.0, n_time)
    tsys_R_clean = physics.tsys_model(z, T0_R, tau0, Twmt)
    tsys_L_clean = physics.tsys_model(z, T0_L, tau0, Twmt)
    tsys_R_true = tsys_R_clean + rng.normal(0, noise_K, n_time)
    tsys_L_true = tsys_L_clean + rng.normal(0, noise_K, n_time)

    tcal = 5.0
    bandwidth_Hz = 2e9
    Tsys_typ = float(np.mean((tsys_R_clean + tsys_L_clean) / 2.0))
    sigma_eff = max(float(noise_K), 0.01)
    # σ = 2·Tsys²/(Tcal·√(Δν·τ_int))  →  τ_int = 4·Tsys⁴/(Tcal²·σ²·Δν)
    expo_s = float(4.0 * Tsys_typ**4 / (tcal**2 * sigma_eff**2 * bandwidth_Hz))
    switched_diff = np.ones((1, n_ant, 1, 2, n_time), dtype=np.float32)
    switched_sum = np.zeros((1, n_ant, 1, 2, n_time), dtype=np.float32)
    for ia in range(n_ant):
        # Tsys_measured = Tsys_true / c → switched_sum = 2 * Tsys_true / (c * tcal_ref)
        switched_sum[0, ia, 0, 0, :] = (2.0 * tsys_R_true / (c_R[ia] * tcal)).astype(
            np.float32
        )
        switched_sum[0, ia, 0, 1, :] = (2.0 * tsys_L_true / (c_L[ia] * tcal)).astype(
            np.float32
        )

    ant_names = [f"ea{i + 1:02d}" for i in range(n_ant)]
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
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.tile(z.astype(np.float32), (1, n_ant, 1)),
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.full((n_ant, 1, 2), tcal, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((1, n_time), T_surf, dtype=np.float32),
            ),
            "weather_P": (
                ("scan", "time"),
                np.full((1, n_time), 85000.0, dtype=np.float32),
            ),
            "weather_RH": (
                ("scan", "time"),
                np.full((1, n_time), 0.3, dtype=np.float32),
            ),
            "exposure_time": (
                ("scan", "time"),
                np.full((1, n_time), expo_s, dtype=np.float32),
            ),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((1, n_ant, 1, 2, n_time), dtype=bool),
            ),
        },
        coords={
            "scan": np.array([1], dtype=np.intp),
            "antenna": ant_names,
            "spw": np.array([0], dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), np.array([freq_Hz])),
            "bandwidth": (("spw",), np.array([2e9])),
            "band": (("spw",), np.array([_band_label(freq_Hz)], dtype="U4")),
            "antenna_position": (
                ("antenna", "xyz"),
                np.zeros((n_ant, 3), dtype=np.float64),
            ),
            "scan_time_start": (("scan",), np.array([0.0])),
            "scan_time_end": (("scan",), np.array([90.0])),
            "time_utc": (
                ("scan", "time"),
                np.arange(n_time, dtype=np.float64)[np.newaxis, :],
            ),
        },
    )


def test_fit_tcal_solve_recovers_params() -> None:
    """tcal_solve: recover tau0 and per-antenna Tcal corrections within 1%.

    noise_K=0.002 is intentionally low: at ≥0.01 K the bounded optimizer finds a
    local minimum where all c values shift by a common factor α and tau scales with
    α — the (T0, c, tau)→(T0·α, c·α, tau·α) near-degeneracy that v2.6 escaped via
    multi-layer bound relaxation (DESIGN.md §12 deferred, §6.3 single-pass policy).
    """
    tau_true = 0.06
    c_R_true = [1.0, 1.05, 0.97]
    c_L_true = [1.0, 0.98, 1.03]

    ds = _make_tcal_ds(tau0=tau_true, c_R=c_R_true, c_L=c_L_true, noise_K=0.002)
    fit_dataset(ds, mode="tcal_solve")

    assert ds["fit_success"].values.all(), ds["fit_reason"].values

    tau_fits = ds["tau_zenith"].values[0, :, 0]
    assert np.all(np.isclose(tau_fits, tau_fits[0])), (
        "tau_zenith must be equal across antennas"
    )

    tol = max(0.005, 0.05 * tau_true)
    assert abs(float(tau_fits[0]) - tau_true) < tol, (
        f"tau recovered {tau_fits[0]:.4f}, true {tau_true:.4f}"
    )

    tcal_ref = float(ds["tcal_ref"].values[0, 0, 0])  # 5 K for all
    for ia, (cr, cl) in enumerate(zip(c_R_true, c_L_true)):
        tcal_fit_R = float(ds["tcal_fit"].values[0, ia, 0, 0])
        tcal_fit_L = float(ds["tcal_fit"].values[0, ia, 0, 1])
        assert abs(tcal_fit_R / (cr * tcal_ref) - 1.0) < 0.01, (
            f"ant {ia} R Tcal: fit={tcal_fit_R:.4f}, true={cr * tcal_ref:.4f}"
        )
        assert abs(tcal_fit_L / (cl * tcal_ref) - 1.0) < 0.01, (
            f"ant {ia} L Tcal: fit={tcal_fit_L:.4f}, true={cl * tcal_ref:.4f}"
        )


def test_fit_tcal_solve_schema_valid() -> None:
    """After tcal_solve fit, schema.validate passes."""
    ds = _make_tcal_ds()
    fit_dataset(ds, mode="tcal_solve")
    schema.validate(ds)


def test_fit_tcal_solve_forces_global_tau() -> None:
    """tcal_solve: tau_zenith is equal across all antennas (forces global tau)."""
    ds = _make_tcal_ds()
    fit_dataset(ds, mode="tcal_solve")
    tau_arr = ds["tau_zenith"].values[0, :, 0]
    assert np.all(np.isclose(tau_arr, tau_arr[0]))


def test_fit_multi_scan_multi_ant() -> None:
    """fit_dataset handles multiple scans and antennas without error."""
    ds = _make_tip_ds(n_scan=2, n_ant=3, n_spw=2)
    fit_dataset(ds, mode="tau_per_antenna")
    assert ds["fit_success"].shape == (2, 3, 2)
    # All cells should succeed (clean synthetic data)
    assert ds["fit_success"].values.all()


# ---------------------------------------------------------------------------
# Stage-A additions: t_mean override and process-pool dispatch
# ---------------------------------------------------------------------------


def test_fit_t_mean_override_matches_default_on_matching_data() -> None:
    """Passing an explicit `t_mean` equal to the Bevis form recovers the same fit.

    Synthetic data is built against `Twmt = k2nt(weighted_mean_atm_T(280 K), ν)`.
    Passing the same value as `t_mean` should produce a fit numerically
    indistinguishable from the no-override default — proves the override path
    threads through to `_screen_antenna` cleanly.
    """
    T_surf = 280.0
    freq_Hz = 10e9
    Twmt = float(physics.k2nt(physics.weighted_mean_atm_T(T_surf), freq_Hz))

    ds_default = _make_tip_ds(n_scan=2, n_ant=3, freq_Hz=freq_Hz)
    fit_dataset(ds_default, mode="tau_per_antenna")

    ds_override = _make_tip_ds(n_scan=2, n_ant=3, freq_Hz=freq_Hz)
    t_mean = np.full((ds_override.sizes["scan"], ds_override.sizes["spw"]), Twmt)
    fit_dataset(ds_override, mode="tau_per_antenna", t_mean=t_mean)

    np.testing.assert_allclose(
        ds_default["tau_zenith"].values,
        ds_override["tau_zenith"].values,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ds_default["T0"].values,
        ds_override["T0"].values,
        rtol=1e-10,
        atol=1e-10,
    )


def test_fit_t_mean_override_changes_tau_when_different() -> None:
    """A wrong T_mean biases τ — sanity check that the override is actually consumed."""
    ds_a = _make_tip_ds(n_ant=2)
    fit_dataset(ds_a, mode="tau_per_antenna")
    tau_a = float(ds_a["tau_zenith"].values[0, 0, 0])

    ds_b = _make_tip_ds(n_ant=2)
    # Halve T_mean → fit needs ~2× τ to reproduce the same Tsys swing.
    t_mean = np.full((ds_b.sizes["scan"], ds_b.sizes["spw"]), 50.0)
    fit_dataset(ds_b, mode="tau_per_antenna", t_mean=t_mean)
    tau_b = float(ds_b["tau_zenith"].values[0, 0, 0])

    assert abs(tau_a - tau_b) > 0.01, (
        f"tau_a={tau_a:.4f}, tau_b={tau_b:.4f} — override appears inert"
    )


def test_fit_t_mean_shape_validation() -> None:
    """Wrong-shaped `t_mean` raises ValueError before the fit runs."""
    ds = _make_tip_ds(n_scan=2, n_spw=3)
    bad = np.zeros((2, 4))  # n_spw=4, dataset has n_spw=3
    with pytest.raises(ValueError, match="t_mean shape"):
        fit_dataset(ds, mode="tau_per_antenna", t_mean=bad)


def test_fit_t_mean_nan_falls_back_to_bevis() -> None:
    """NaN cells in `t_mean` invoke the Bevis fallback for those cells.

    Result should equal the no-override fit when every cell is NaN.
    """
    ds_default = _make_tip_ds(n_ant=2)
    fit_dataset(ds_default, mode="tau_per_antenna")

    ds_nan = _make_tip_ds(n_ant=2)
    t_mean = np.full(
        (ds_nan.sizes["scan"], ds_nan.sizes["spw"]), np.nan, dtype=np.float64
    )
    fit_dataset(ds_nan, mode="tau_per_antenna", t_mean=t_mean)

    np.testing.assert_allclose(
        ds_default["tau_zenith"].values,
        ds_nan["tau_zenith"].values,
        rtol=1e-10,
        atol=1e-12,
    )


def test_fit_n_workers_pool_matches_serial_opacity() -> None:
    """Pool dispatch must produce identical results to the serial path (opacity)."""
    ds_serial = _make_tip_ds(n_scan=2, n_ant=3, n_spw=2)
    fit_dataset(ds_serial, mode="tau_per_antenna")

    ds_pool = _make_tip_ds(n_scan=2, n_ant=3, n_spw=2)
    fit_dataset(ds_pool, mode="tau_per_antenna", n_workers=2)

    np.testing.assert_array_equal(
        ds_serial["tau_zenith"].values,
        ds_pool["tau_zenith"].values,
    )
    np.testing.assert_array_equal(ds_serial["T0"].values, ds_pool["T0"].values)
    np.testing.assert_array_equal(
        ds_serial["fit_success"].values, ds_pool["fit_success"].values
    )


def test_fit_n_workers_pool_matches_serial_tcal() -> None:
    """Pool dispatch must produce identical results to the serial path (tcal_solve)."""
    ds_serial = _make_tcal_ds(noise_K=0.002)
    fit_dataset(ds_serial, mode="tcal_solve")

    ds_pool = _make_tcal_ds(noise_K=0.002)
    fit_dataset(ds_pool, mode="tcal_solve", n_workers=2)

    np.testing.assert_allclose(
        ds_serial["tau_zenith"].values,
        ds_pool["tau_zenith"].values,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ds_serial["tcal_fit"].values,
        ds_pool["tcal_fit"].values,
        rtol=1e-10,
        atol=1e-10,
    )
