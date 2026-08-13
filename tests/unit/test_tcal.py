"""Unit tests for tipopac.tcal — the Stage-C anchor-pinned Tcal estimator."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from scipy.optimize import least_squares

from tipopac import physics, schema
from tipopac.tcal import _anchor_fit, _wls, solve_tcal


# ---------------------------------------------------------------------------
# The closed form, on its own
# ---------------------------------------------------------------------------


def _pred(z: np.ndarray, tau: float, twmt: float, nu: float) -> np.ndarray:
    transp = np.exp(-tau / np.cos(np.deg2rad(z)))
    return float(physics.k2nt(physics.T_CMB, nu)) * transp + twmt * (1.0 - transp)


_Z = np.rad2deg(np.arccos(1.0 / np.linspace(1.02, 2.6, 24)))


@pytest.mark.parametrize("c_true", [0.75, 0.9, 1.0, 1.15, 1.3])
@pytest.mark.parametrize("tau", [0.02, 0.09, 0.25])
def test_wls_recovers_injected_c(c_true: float, tau: float) -> None:
    """A noiseless curve at known (T0, c, tau_am) returns c exactly."""
    pred = _pred(_Z, tau, 265.0, 22e9)
    y = (35.0 + pred) / c_true
    _, B, _ = _wls(pred, y, np.ones_like(y))
    assert abs(1.0 / B - c_true) < 1e-9


def test_wls_matches_nonlinear_two_parameter_refit() -> None:
    """Agrees with a scipy (T0, c) refit at fixed tau_am under plain L2."""
    rng = np.random.default_rng(7)
    pred = _pred(_Z, 0.09, 265.0, 22e9)
    worst = 0.0
    for _ in range(20):
        c_true = float(rng.uniform(0.8, 1.25))
        sig = np.full(pred.size, 0.4)
        y = (35.0 + pred) / c_true + rng.normal(0.0, sig)
        _, B, _ = _wls(pred, y, 1.0 / sig**2)
        res = least_squares(lambda p: (y - (p[0] + pred) / p[1]) / sig, x0=[30.0, 1.0])
        worst = max(worst, abs(1.0 / B - res.x[1]) / res.x[1])
    assert worst < 1e-8


def test_wls_var_b_matches_covariance() -> None:
    """var_B from the normal equations equals the (XᵀWX)⁻¹ diagonal."""
    pred = _pred(_Z, 0.09, 265.0, 22e9)
    sig = np.linspace(0.3, 0.7, pred.size)
    ivar = 1.0 / sig**2
    _, _, var_B = _wls(pred, 40.0 + pred, ivar)
    X = np.column_stack([np.ones_like(pred), pred])
    cov = np.linalg.inv(X.T @ (ivar[:, None] * X))
    assert np.isclose(var_B, cov[1, 1], rtol=1e-10)


def test_wls_singular_design_returns_nan() -> None:
    x = np.ones(5)
    A, B, var_B = _wls(x, x, np.ones(5))
    assert np.isnan(A) and np.isnan(B) and np.isnan(var_B)


def test_anchor_fit_rejects_outlier_on_joint_mask() -> None:
    """A 4σ outlier in R alone drops that time sample from both pols."""
    pred = _pred(_Z, 0.09, 265.0, 22e9)
    sig = np.full(pred.size, 0.2)
    ivar = 1.0 / sig**2
    y_R = (35.0 + pred) / 1.1
    y_L = (35.0 + pred) / 1.1
    y_R = y_R.copy()
    y_R[5] += 2.0  # 10σ in R only

    res = _anchor_fit(pred, y_R, y_L, ivar, ivar)
    assert res["n"] == pred.size - 1
    assert not res["keep"][5]
    # Both pols recover c despite the R-only outlier.
    assert np.allclose([1.0 / res["B"][0], 1.0 / res["B"][1]], 1.1, rtol=1e-9)


def test_anchor_fit_too_few_samples() -> None:
    pred = np.array([1.0, 2.0])
    ivar = np.ones(2)
    assert _anchor_fit(pred, pred, pred, ivar, ivar)["n"] == 0


# ---------------------------------------------------------------------------
# solve_tcal on a synthetic dataset
# ---------------------------------------------------------------------------


_TAU = 0.05
_NU = 22e9
_TWMT = 265.0
_TCAL_REF = 5.0


def _make_stage_c_ds(
    c_true: float = 1.2,
    n_scan: int = 2,
    n_ant: int = 3,
    n_spw: int = 2,
    n_time: int = 24,
    *,
    airmass_hi: float = 2.6,
    group_taus: tuple[float, ...] = (_TAU,),
) -> xr.Dataset:
    """A post-Stage-B dataset whose Tsys is exactly (T0 + pred)/c_true."""
    z = np.rad2deg(np.arccos(1.0 / np.linspace(1.02, airmass_hi, n_time)))
    pred = _pred(z, _TAU, _TWMT, _NU)
    tsys = (50.0 + pred) / c_true

    shape = (n_scan, n_ant, n_spw, 2, n_time)
    tsys_arr = np.broadcast_to(tsys, shape).astype(np.float32).copy()
    # A dense am grid, flat at each group's tau so interpolation is exact.
    grid = np.linspace(1e9, 51e9, 501)
    n_group = len(group_taus)
    am_tau = np.stack([np.full(grid.size, t) for t in group_taus])
    scan_group = (np.arange(n_scan) % n_group).astype(np.int32)

    return xr.Dataset(
        data_vars={
            "Tsys": (("scan", "antenna", "spw", "polarization", "time"), tsys_arr),
            "sigma_Tsys": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.full(shape, 0.2, dtype=np.float32),
            ),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros(shape, dtype=bool),
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.broadcast_to(z, (n_scan, n_ant, n_time)).astype(np.float32).copy(),
            ),
            "Twmt": (
                ("scan", "spw"),
                np.full((n_scan, n_spw), _TWMT, dtype=np.float32),
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.full((n_ant, n_spw, 2), _TCAL_REF, dtype=np.float32),
            ),
            "tcal_fit": (
                ("scan", "antenna", "spw", "polarization"),
                np.full((n_scan, n_ant, n_spw, 2), _TCAL_REF, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((n_scan, n_time), 280.0, dtype=np.float32),
            ),
            "am_freq_grid": (("frequency_dense",), grid),
            "am_tau": (("group", "frequency_dense"), am_tau),
        },
        coords={
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.arange(n_spw, dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "frequency": (("spw",), np.full(n_spw, _NU, dtype=np.float64)),
            "group": np.arange(n_group, dtype=np.int32),
            "scan_group": (("scan",), scan_group),
        },
        attrs={"mode": "independent_tau"},
    )


def test_solve_tcal_recovers_injected_c() -> None:
    ds = _make_stage_c_ds(c_true=1.2)
    solve_tcal(ds)
    np.testing.assert_allclose(ds["tcal_fit"].values, 1.2 * _TCAL_REF, rtol=1e-5)
    assert np.all(np.isfinite(ds["sigma_tcal"].values))
    assert np.all(ds["sigma_tcal"].values > 0.0)


def test_solve_tcal_writes_schema_conformant_vars() -> None:
    ds = _make_stage_c_ds()
    solve_tcal(ds)
    dims, dtype = schema.OPTIONAL_DATA_VARS["sigma_tcal"]
    assert ds["sigma_tcal"].dims == dims
    assert ds["sigma_tcal"].dtype == dtype
    assert ds["tcal_fit"].dims == schema.OPTIONAL_DATA_VARS["tcal_fit"][0]


def test_solve_tcal_leverage_gate_emits_nan() -> None:
    """A short tip (span below the floor) yields NaN in both outputs."""
    ds = _make_stage_c_ds(airmass_hi=1.1)
    solve_tcal(ds, min_airmass_span=0.3)
    assert np.all(np.isnan(ds["tcal_fit"].values))
    assert np.all(np.isnan(ds["sigma_tcal"].values))

    ds2 = _make_stage_c_ds(airmass_hi=1.1)
    solve_tcal(ds2, min_airmass_span=0.05)
    assert np.all(np.isfinite(ds2["tcal_fit"].values))


def test_solve_tcal_nan_where_too_few_samples() -> None:
    ds = _make_stage_c_ds()
    ds["flag"].values[0, 0, 0, :, :] = True
    solve_tcal(ds)
    assert np.all(np.isnan(ds["tcal_fit"].values[0, 0, 0, :]))
    assert np.all(np.isnan(ds["sigma_tcal"].values[0, 0, 0, :]))
    assert np.all(np.isfinite(ds["tcal_fit"].values[0, 1, 0, :]))


def test_solve_tcal_nan_where_anchor_missing() -> None:
    """An spw off the am grid gets no anchor and therefore no c."""
    ds = _make_stage_c_ds()
    ds.coords["frequency"] = (("spw",), np.array([_NU, 90e9]))
    solve_tcal(ds)
    assert np.all(np.isnan(ds["tcal_fit"].values[:, :, 1, :]))
    assert np.all(np.isfinite(ds["tcal_fit"].values[:, :, 0, :]))


def test_solve_tcal_uses_per_group_anchor() -> None:
    """Two groups with different am curves give different c."""
    ds = _make_stage_c_ds(n_scan=2, group_taus=(_TAU, 2.0 * _TAU))
    solve_tcal(ds)

    c0 = ds["tcal_fit"].values[0] / _TCAL_REF
    c1 = ds["tcal_fit"].values[1] / _TCAL_REF
    np.testing.assert_allclose(c0, 1.2, rtol=1e-5)
    assert not np.allclose(c1, 1.2, rtol=1e-3)


def test_solve_tcal_interpolates_the_anchor_at_each_spw_frequency() -> None:
    """A sloped am curve with two spws catches a wrong-frequency lookup.

    Every other Stage-C test uses a flat grid, where `_model_at` returns the
    right tau regardless of which frequency it looks up.
    """
    nu_lo, nu_hi = 20e9, 40e9
    tau_lo, tau_hi = 0.04, 0.12
    c_true = 1.15
    n_time = 24
    z = np.rad2deg(np.arccos(1.0 / np.linspace(1.02, 2.6, n_time)))

    ds = _make_stage_c_ds(c_true=c_true, n_scan=1, n_ant=1, n_spw=2, n_time=n_time)
    ds.coords["frequency"] = (("spw",), np.array([nu_lo, nu_hi]))

    # A grid linear in nu, so tau(nu_lo) = tau_lo and tau(nu_hi) = tau_hi.
    grid = np.linspace(1e9, 51e9, 501)
    slope = (tau_hi - tau_lo) / (nu_hi - nu_lo)
    ds["am_freq_grid"] = (("frequency_dense",), grid)
    ds["am_tau"] = (
        ("group", "frequency_dense"),
        (tau_lo + slope * (grid - nu_lo))[None, :],
    )

    # Build Tsys per spw at that spw's own tau and frequency.
    for i_spw, (nu, tau) in enumerate(((nu_lo, tau_lo), (nu_hi, tau_hi))):
        tsys = (50.0 + _pred(z, tau, _TWMT, nu)) / c_true
        ds["Tsys"].values[0, 0, i_spw, :, :] = tsys.astype(np.float32)

    solve_tcal(ds)
    np.testing.assert_allclose(
        ds["tcal_fit"].values[0, 0] / _TCAL_REF, c_true, rtol=1e-4
    )


def test_solve_tcal_requires_the_stage_b_anchor() -> None:
    ds = _make_stage_c_ds().drop_vars("am_tau")
    with pytest.raises(schema.SchemaError, match="am_tau"):
        solve_tcal(ds)


def test_solve_tcal_spillover_enters_the_numerator() -> None:
    """With the spillover attr set, c shifts — spill sits inside the /c term."""
    ds = _make_stage_c_ds()
    solve_tcal(ds)
    c_off = ds["tcal_fit"].values.copy()

    ds2 = _make_stage_c_ds()
    ds2.attrs["spillover_model"] = "eta_poly_v1"
    solve_tcal(ds2)
    assert not np.allclose(c_off, ds2["tcal_fit"].values, rtol=1e-4)
