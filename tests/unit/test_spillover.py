"""Unit tests for tipopac.spillover — Tsys forward-model η(ν) term (design.md §5.1, §6)."""

from __future__ import annotations

import inspect

import numpy as np
import xarray as xr

from tipopac.physics import k2nt, predicted_tsys
from tipopac.spillover import (
    ETA_MODEL_NAME,
    ETA_POLY_COEF,
    ETA_VALID_GHZ,
    eta_of_nu,
    spillover_tsys,
)


def test_eta_of_nu_matches_quadratic_in_range() -> None:
    """η(ν) equals the stored quadratic inside the trusted band."""
    c2, c1, c0 = ETA_POLY_COEF
    for nu_GHz in (4.0, 12.0, 22.0, 33.0, 45.0, 50.0):
        expected = c2 * nu_GHz**2 + c1 * nu_GHz + c0
        got = float(eta_of_nu(nu_GHz * 1e9))
        assert got == expected


def test_eta_of_nu_zero_outside_range() -> None:
    """η→0 below 4 GHz and above 50 GHz (edges of the JSON validity range)."""
    lo, hi = ETA_VALID_GHZ
    for nu_GHz in (3.5, lo - 0.01, hi + 0.01, 60.0):
        assert float(eta_of_nu(nu_GHz * 1e9)) == 0.0


def test_eta_of_nu_vectorized() -> None:
    """Array input evaluates element-wise with the same clamp."""
    freqs = np.array([3e9, 12e9, 33e9, 45e9, 55e9])
    eta = eta_of_nu(freqs)
    assert eta[0] == 0.0 and eta[4] == 0.0
    assert np.all(eta[1:4] > 0.0)


def test_spillover_tsys_formula() -> None:
    """spillover_tsys == η(ν)·k2nt(Tsurf,ν)·(1/cos z) at known inputs."""
    nu, T_surf, z = 33e9, 282.0, 40.0
    expected = eta_of_nu(nu) * float(k2nt(T_surf, nu)) / np.cos(np.deg2rad(z))
    got = float(spillover_tsys(nu, T_surf, z))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_spillover_tsys_scales_with_airmass() -> None:
    """The term is ∝ airmass = 1/cos z (grows toward the horizon)."""
    nu, T_surf = 22e9, 280.0
    s30 = float(spillover_tsys(nu, T_surf, 30.0))
    s60 = float(spillover_tsys(nu, T_surf, 60.0))
    np.testing.assert_allclose(
        s60 / s30, np.cos(np.deg2rad(30.0)) / np.cos(np.deg2rad(60.0)), rtol=1e-12
    )


def test_default_toggle_is_on() -> None:
    """Both entry points default ``spillover_model=True`` (the correct model)."""
    from tipopac.api import TippingAnalysis, tipopac

    for fn in (tipopac, TippingAnalysis.fit):
        assert inspect.signature(fn).parameters["spillover_model"].default is True


def _recon_ds() -> xr.Dataset:
    """Minimal dataset for a predicted_tsys reconstruction with weather_T."""
    rng = np.random.default_rng(0)
    dims = ("scan", "antenna", "spw", "polarization")
    shape = (1, 2, 2, 1)
    return xr.Dataset(
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
            "tcal_fit": (dims, np.ones(shape, np.float32)),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.ones((2, 2, 1), np.float32),
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.full((1, 2, 3), 45.0, np.float32),
            ),
            "weather_T": (("scan", "time"), np.full((1, 3), 282.0, np.float32)),
        },
        coords={"frequency": ("spw", np.array([22e9, 33e9]))},
    )


def test_predicted_tsys_adds_term_when_enabled() -> None:
    """With ``spillover_model`` set, predicted_tsys adds η·Bg·airmass inside /c."""
    ds = _recon_ds()
    baseline = predicted_tsys(ds)  # attr absent → no spillover

    enabled = ds.copy()
    enabled.attrs["spillover_model"] = ETA_MODEL_NAME
    reconstructed = predicted_tsys(enabled)

    T_surf = ds["weather_T"].mean(dim="time")
    spill = spillover_tsys(ds["frequency"], T_surf, ds["zenith_angle"])  # c == 1
    xr.testing.assert_allclose(reconstructed - baseline, spill.broadcast_like(baseline))


def test_predicted_tsys_no_term_when_disabled() -> None:
    """No ``spillover_model`` attr → reconstruction carries no spillover term."""
    ds = _recon_ds()
    # 22 GHz is inside the trusted band, so a term *would* be non-zero if applied.
    pred = predicted_tsys(ds)
    manual = predicted_tsys(ds.assign_attrs(spillover_model=ETA_MODEL_NAME))
    assert not np.allclose(pred.values, manual.values)
