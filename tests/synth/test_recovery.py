"""Stage-2 forward-model recovery tests with synthetic tipping data.

These tests construct synthetic Tsys time series from a real PwvGrid (built
via am, module-scoped fixture). They are the post-refactor primary
acceptance check — see `design/model_refactor.md` §3 and `design/initial_design.md`
§11.3.

For unit-test speed, the grid is built with a small frequency span and a coarse
PWV step. Recovery accuracy is asserted in *physical* terms (PWV mm), not vs
v2.6.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
import xarray as xr

from tipopac import schema
from tipopac.atmgrid import PwvGrid, build_pwv_grid
from tipopac.fit import fit_dataset


_FREQ_HZ = np.array([18.0e9, 22.0e9, 23.5e9, 25.0e9])  # K-band-ish, 4 spws
_BW_HZ = 2.0e9
_TCAL_K = 5.0


# ---------------------------------------------------------------------------
# Fixture: one real PwvGrid via am (slow), module-scoped.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_grid() -> PwvGrid:
    """Build a small but real PwvGrid via am over a midlatitude_winter profile."""
    import amwrap

    clim = amwrap.Climatology("midlatitude_winter")
    return build_pwv_grid(
        clim.pressure,
        clim.temperature,
        clim.mixing_ratio["h2o"],
        freq_min_Hz=17e9,
        freq_max_Hz=26e9,
        profile_source="afgl_midlatitude_winter",
        pwv_min_mm=1.0,
        pwv_max_mm=30.0,
        pwv_step_mm=1.0,
        n_workers=4,
    )


# ---------------------------------------------------------------------------
# Synthetic-data builder
# ---------------------------------------------------------------------------


def _make_pwv_synth_ds(
    *,
    grid: PwvGrid,
    pwv_true_mm: float,
    T0_R: np.ndarray | None = None,
    T0_L: np.ndarray | None = None,
    c_R: np.ndarray | None = None,
    c_L: np.ndarray | None = None,
    z_range: tuple[float, float] = (20.0, 70.0),
    n_time: int = 25,
    n_ant: int = 1,
    noise_K: float = 0.5,
    rng_seed: int = 0,
) -> xr.Dataset:
    """Build a synthetic dataset whose Tsys series follow the forward model.

    Returns an xr.Dataset that schema.validate accepts and that fit_dataset can
    run on. The "true" PWV and per-(antenna, spw, pol) T0 / c are injected
    deterministically; noise is Gaussian on Tsys.
    """
    rng = np.random.default_rng(rng_seed)
    n_spw = _FREQ_HZ.size
    n_pol = 2
    pol = list(schema.POL_VALUES)

    if T0_R is None:
        T0_R = np.full((n_ant, n_spw), 50.0)
    if T0_L is None:
        T0_L = np.full((n_ant, n_spw), 48.0)
    if c_R is None:
        c_R = np.ones((n_ant, n_spw))
    if c_L is None:
        c_L = np.ones((n_ant, n_spw))

    z = np.linspace(*z_range, n_time)
    airmass = 1.0 / np.cos(np.deg2rad(z))

    tau_z, tmean = grid.lookup(pwv_true_mm, _FREQ_HZ)
    t_sky = tmean[:, None] * (1.0 - np.exp(-tau_z[:, None] * airmass[None, :]))

    # Tsys per (ant, spw, pol, time)
    tsys = np.empty((n_ant, n_spw, n_pol, n_time))
    for a in range(n_ant):
        for k in range(n_spw):
            tsys[a, k, 0, :] = (T0_R[a, k] + t_sky[k, :]) / c_R[a, k]
            tsys[a, k, 1, :] = (T0_L[a, k] + t_sky[k, :]) / c_L[a, k]
    tsys += rng.normal(0.0, noise_K, tsys.shape)

    # switched_diff = 1; switched_sum = 2 · Tsys / Tcal_ref. Tsys = sum/2 · Tcal/diff.
    switched_diff = np.ones((1, n_ant, n_spw, n_pol, n_time), dtype=np.float32)
    switched_sum = np.zeros((1, n_ant, n_spw, n_pol, n_time), dtype=np.float32)
    switched_sum[0, :, :, :, :] = (2.0 * tsys / _TCAL_K).astype(np.float32)

    # Calibrate exposure_time so the radiometer σ_Tsys at typical Tsys equals
    # the injected noise_K — keeps the χ² of the fit ~1.
    sigma_eff = max(float(noise_K), 0.05)
    Tsys_typ = float(np.mean(tsys))
    expo_s = float(2.0 * Tsys_typ**4 / (_TCAL_K**2 * sigma_eff**2 * _BW_HZ))

    zenith_arr = np.tile(z.astype(np.float32), (1, n_ant, 1))

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
                np.full((n_ant, n_spw, n_pol), _TCAL_K, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((1, n_time), 280.0, dtype=np.float32),
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
                np.zeros((1, n_ant, n_spw, n_pol, n_time), dtype=bool),
            ),
        },
        coords={
            "scan": np.array([1], dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.arange(n_spw, dtype=np.intp),
            "polarization": pol,
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), _FREQ_HZ.astype(np.float64)),
            "bandwidth": (("spw",), np.full(n_spw, _BW_HZ, dtype=np.float64)),
            "antenna_position": (
                ("antenna", "xyz"),
                np.zeros((n_ant, 3), dtype=np.float64),
            ),
            "scan_time_start": (("scan",), np.array([0.0])),
            "scan_time_end": (("scan",), np.array([float(n_time)])),
            "time_utc": (
                ("scan", "time"),
                np.arange(n_time, dtype=np.float64)[np.newaxis, :],
            ),
        },
    )


# ---------------------------------------------------------------------------
# Recovery tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("pwv_true", [3.0, 7.5, 12.0, 18.3])
def test_per_antenna_pwv_recovers_injected_pwv(real_grid: PwvGrid, pwv_true: float) -> None:
    ds = _make_pwv_synth_ds(grid=real_grid, pwv_true_mm=pwv_true, rng_seed=int(pwv_true))
    fit_dataset(ds, mode="per_antenna_pwv", grids={1: real_grid})

    assert bool(ds["fit_success"].values[0, 0, 0]), str(ds["fit_reason"].values[0, 0, 0])

    pwv_hat = float(ds["pwv"].values[0, 0])
    pwv_err = float(ds["pwv_err"].values[0, 0])
    # Within max(1 mm, 3σ) — tolerance covers correlated curvature degeneracy
    # at low airmass.
    tol = max(1.0, 3.0 * pwv_err)
    assert abs(pwv_hat - pwv_true) < tol, (
        f"PWV recovery: true={pwv_true:.2f}, got={pwv_hat:.2f} ± {pwv_err:.2f}"
    )


@pytest.mark.slow
def test_per_antenna_pwv_recovers_T0(real_grid: PwvGrid) -> None:
    T0_R_true = np.array([[40.0, 45.0, 50.0, 55.0]])
    T0_L_true = np.array([[42.0, 47.0, 52.0, 57.0]])
    ds = _make_pwv_synth_ds(
        grid=real_grid,
        pwv_true_mm=8.0,
        T0_R=T0_R_true,
        T0_L=T0_L_true,
        noise_K=0.3,
    )
    fit_dataset(ds, mode="per_antenna_pwv", grids={1: real_grid})

    T0 = ds["T0"].values[0, 0, :, :]  # (n_spw, n_pol)
    # Within 3 K — at this noise level the per-(spw,pol) T0 σ is ~1 K.
    np.testing.assert_allclose(T0[:, 0], T0_R_true[0], atol=3.0)
    np.testing.assert_allclose(T0[:, 1], T0_L_true[0], atol=3.0)


@pytest.mark.slow
def test_tcal_solve_recovers_c(real_grid: PwvGrid) -> None:
    """tcal_solve in the Stage-2 framework recovers injected c factors."""
    n_spw = _FREQ_HZ.size
    c_R_true = np.array([[0.95, 1.05, 0.98, 1.02]])
    c_L_true = np.array([[1.10, 0.92, 1.07, 0.94]])
    ds = _make_pwv_synth_ds(
        grid=real_grid,
        pwv_true_mm=6.0,
        c_R=c_R_true,
        c_L=c_L_true,
        noise_K=0.3,
        n_time=40,  # more samples — c has weak leverage at low τ
    )
    fit_dataset(ds, mode="tcal_solve", grids={1: real_grid})

    assert bool(ds["fit_success"].values[0, 0, 0]), str(
        ds["fit_reason"].values[0, 0, 0]
    )
    tcal_fit = ds["tcal_fit"].values[0, 0, :, :]  # (n_spw, n_pol)
    c_R_fit = tcal_fit[:, 0] / _TCAL_K
    c_L_fit = tcal_fit[:, 1] / _TCAL_K
    np.testing.assert_allclose(c_R_fit, c_R_true[0], atol=0.05)
    np.testing.assert_allclose(c_L_fit, c_L_true[0], atol=0.05)


@pytest.mark.slow
def test_shared_pwv_freezes_to_scan_median(real_grid: PwvGrid) -> None:
    """shared_pwv: ds.pwv = pwv_scan_median AND forward model is consistent.

    Two things must hold:
      1. ds.pwv equals the consensus PWV for every fitted antenna.
      2. The T0 values stored alongside that PWV produce a forward model
         consistent with the data — that is, the second-pass LM actually
         *pinned* PWV when refitting T0, rather than letting it drift and
         then mislabelling the result. This is the regression check for the
         advisor-flagged shared_pwv inconsistency.
    """
    n_ant = 3
    ds = _make_pwv_synth_ds(
        grid=real_grid, pwv_true_mm=10.0, n_ant=n_ant, noise_K=0.5
    )
    fit_dataset(ds, mode="shared_pwv", grids={1: real_grid})

    median = float(ds["pwv_scan_median"].values[0])
    for a in range(n_ant):
        pv = float(ds["pwv"].values[0, a])
        if np.isfinite(pv):
            assert pv == pytest.approx(median, rel=0.0, abs=1e-6), (
                f"shared_pwv ant {a} drifted from median: {pv:.3f} vs {median:.3f}"
            )

    # Forward-model consistency: |Tsys - model| / σ < 5 for all unflagged samples.
    tau_z, tmean = real_grid.lookup(median, _FREQ_HZ)
    z = ds["zenith_angle"].values[0, 0, :]
    airmass = 1.0 / np.cos(np.deg2rad(z))
    t_sky = tmean[:, None] * (1.0 - np.exp(-tau_z[:, None] * airmass[None, :]))
    for a in range(n_ant):
        T0_R = ds["T0"].values[0, a, :, 0]
        T0_L = ds["T0"].values[0, a, :, 1]
        Tsys_meas = ds["Tsys"].values[0, a, :, :, :]  # (n_spw, n_pol, n_time)
        sigma = ds["sigma_Tsys"].values[0, a, :, :, :]
        flag = ds["flag"].values[0, a, :, :, :]
        pred_R = T0_R[:, None] + t_sky  # (n_spw, n_time)
        pred_L = T0_L[:, None] + t_sky
        pred = np.stack([pred_R, pred_L], axis=1)  # (n_spw, n_pol, n_time)
        z_score = np.abs(Tsys_meas - pred) / np.where(sigma > 0, sigma, 1.0)
        max_z = float(np.max(z_score[~flag]))
        assert max_z < 5.0, (
            f"ant {a}: |Tsys - forward(shared_pwv, T0)| / σ peaks at {max_z:.2f} — "
            "T0 was fit against a drifting PWV instead of the pinned median."
        )


@pytest.mark.slow
def test_pwv_outlier_flag_fires_for_biased_antenna(real_grid: PwvGrid) -> None:
    """Inject one antenna with PWV biased by 5 mm; outlier flag must fire."""
    n_ant = 5
    # All ants see the same Tsys (built from pwv_true=8 mm), but ant index 2's
    # data is built from pwv=15 mm — so its fit will recover ~15.
    ds_normal = _make_pwv_synth_ds(
        grid=real_grid, pwv_true_mm=8.0, n_ant=n_ant, noise_K=0.3
    )
    ds_biased = _make_pwv_synth_ds(
        grid=real_grid, pwv_true_mm=15.0, n_ant=1, noise_K=0.3, rng_seed=99
    )
    # Splice ant 2 of normal with ant 0 of biased.
    for v in ("switched_diff", "switched_sum", "zenith_angle", "flag"):
        if "antenna" in ds_normal[v].dims:
            arr = ds_normal[v].values.copy()
            ax = ds_normal[v].dims.index("antenna")
            biased_arr = ds_biased[v].values
            slc_n: list = [slice(None)] * arr.ndim
            slc_b: list = [slice(None)] * biased_arr.ndim
            slc_n[ax] = 2
            slc_b[ax] = 0
            arr[tuple(slc_n)] = biased_arr[tuple(slc_b)]
            ds_normal[v] = (ds_normal[v].dims, arr)

    fit_dataset(ds_normal, mode="per_antenna_pwv", grids={1: real_grid})

    outliers = ds_normal["pwv_outlier"].values[0, :]
    pwvs = ds_normal["pwv"].values[0, :]
    assert outliers[2], (
        f"ant 2 should be flagged outlier. PWVs={pwvs}, outliers={outliers}"
    )
    # No other antenna should be flagged.
    assert int(np.sum(outliers)) == 1, (
        f"only ant 2 should be flagged; got {int(np.sum(outliers))} outliers, "
        f"PWVs={pwvs}"
    )
