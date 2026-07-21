"""Unit tests for tipopac.spillover — Stage-B spillover de-bias (DESIGN.md §6)."""

from __future__ import annotations

import inspect

import numpy as np
import xarray as xr

from tipopac.anchor import anchor_pwv
from tipopac.atmgrid import PwvGrid
from tipopac.physics import predicted_tsys
from tipopac.spillover import SPILLOVER_TAU_DEFAULT, apply_spillover


def _tau_ds() -> xr.Dataset:
    """Dataset with a `tau_zenith(scan, antenna, spw)` holding a NaN pad cell."""
    tau = np.array([[[0.05, 0.02], [0.04, np.nan]]], dtype=np.float32)  # (1, 2, 2)
    return xr.Dataset({"tau_zenith": (("scan", "antenna", "spw"), tau)})


def test_subtracts_constant_and_records_attr() -> None:
    ds = _tau_ds()
    apply_spillover(ds, SPILLOVER_TAU_DEFAULT)
    assert ds.attrs["spillover_tau"] == SPILLOVER_TAU_DEFAULT
    finite = np.array([[[0.05, 0.02], [0.04, np.nan]]]) - SPILLOVER_TAU_DEFAULT
    np.testing.assert_allclose(ds["tau_zenith"].values[0, 0], finite[0, 0], rtol=1e-6)


def test_nan_cells_stay_nan() -> None:
    ds = _tau_ds()
    apply_spillover(ds, SPILLOVER_TAU_DEFAULT)
    assert np.isnan(ds["tau_zenith"].values[0, 1, 1])


def test_allows_negative_no_clamp() -> None:
    ds = xr.Dataset(
        {"tau_zenith": (("scan", "antenna", "spw"), np.array([[[0.001]]], np.float32))}
    )
    apply_spillover(ds, SPILLOVER_TAU_DEFAULT)
    assert ds["tau_zenith"].values[0, 0, 0] < 0.0


def test_none_and_zero_are_noops() -> None:
    for disabled in (None, 0, 0.0):
        ds = _tau_ds()
        before = ds["tau_zenith"].values.copy()
        apply_spillover(ds, disabled)
        assert ds.attrs["spillover_tau"] == 0.0
        np.testing.assert_array_equal(
            ds["tau_zenith"].values, before, "disabled must not touch tau_zenith"
        )


def test_default_is_disabled() -> None:
    """Both entry points default to no de-bias — δτ is stale (design.md §6)."""
    from tipopac.api import TippingAnalysis, tipopac

    for fn in (tipopac, TippingAnalysis.fit):
        assert inspect.signature(fn).parameters["spillover"].default is None


def test_predicted_tsys_round_trips_to_raw_fit() -> None:
    """De-bias + add-back reproduces the raw-fit Tsys; attr survives `.isel`."""
    rng = np.random.default_rng(0)
    dims = ("scan", "antenna", "spw", "polarization")
    shape = (1, 2, 2, 1)
    ds = xr.Dataset(
        {
            "tau_zenith": (
                ("scan", "antenna", "spw"),
                rng.uniform(0.02, 0.08, (1, 2, 2)).astype(np.float32),
            ),
            "T0": (dims, rng.uniform(5.0, 15.0, shape).astype(np.float32)),
            "Twmt": (
                ("scan", "spw"),
                rng.uniform(260.0, 280.0, (1, 2)).astype(np.float32),
            ),
            "tcal_fit": (
                ("scan", "antenna", "spw", "polarization"),
                np.ones(shape, np.float32),
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.ones((2, 2, 1), np.float32),
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.full((1, 2, 3), 45.0, np.float32),
            ),
        },
        coords={"frequency": ("spw", np.array([22e9, 33e9]))},
    )
    raw = predicted_tsys(ds)  # spillover_tau absent → treated as 0.0

    de_biased = ds.copy()
    apply_spillover(de_biased, SPILLOVER_TAU_DEFAULT)
    reconstructed = predicted_tsys(de_biased)

    xr.testing.assert_allclose(raw, reconstructed)
    # attr must survive subsetting so per-cell overlays add the offset back
    assert de_biased.isel(scan=0).attrs["spillover_tau"] == SPILLOVER_TAU_DEFAULT


def _toy_grid() -> PwvGrid:
    pwv = np.linspace(1.0, 10.0, 19)
    freq = np.linspace(10e9, 30e9, 41)
    tau = pwv[:, None] * (1.0 + 0.01 * freq[None, :] / 1e9) * 0.01
    tb = 270.0 * (1.0 - np.exp(-tau))
    return PwvGrid(pwv_mm=pwv, freq_Hz=freq, tau_z=tau, tb_z=tb, pwv_unscaled_mm=5.0)


def test_debias_lowers_anchored_pwv() -> None:
    """spillover=None ≡ baseline PWV; the opt-in de-bias lowers PWV."""
    grid = _toy_grid()
    freqs = np.linspace(12e9, 28e9, 16)
    tau_true, _ = grid.lookup(4.3, freqs)
    tau_z = np.tile(tau_true, (1, 2, 1)).astype(np.float32)
    tau_err = np.full_like(tau_z, 1e-3)
    grids = {0: grid}

    ds_base = xr.Dataset({"tau_zenith": (("scan", "antenna", "spw"), tau_z.copy())})
    apply_spillover(ds_base, None)
    np.testing.assert_array_equal(ds_base["tau_zenith"].values, tau_z)
    pwv_base, _ = anchor_pwv(ds_base["tau_zenith"].values, tau_err, grids, freqs)

    ds_deb = xr.Dataset({"tau_zenith": (("scan", "antenna", "spw"), tau_z.copy())})
    apply_spillover(ds_deb, SPILLOVER_TAU_DEFAULT)
    pwv_deb, _ = anchor_pwv(ds_deb["tau_zenith"].values, tau_err, grids, freqs)

    assert np.all(pwv_deb < pwv_base)
