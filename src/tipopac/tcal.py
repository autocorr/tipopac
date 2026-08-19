"""Stage C — anchor-pinned Tcal scale ``c``, in closed form.

With τ pinned to the Stage-B anchor ``τ_am`` the Stage-A model is linear in
``(T0/c, 1/c)``::

    Tsys(z) = (T0 + Tcmb·e^{−τ_am·a} + Twmt·(1 − e^{−τ_am·a}) + spill(z)) / c
            = A + B·pred(z),        A = T0/c,   B = 1/c

so a σ-weighted regression of the measured Tsys on the model brightness gives
``c = 1/B`` exactly in τ — no optimizer, no am call — with ``σ_c = σ_B/B²`` from
the same 2×2 normal equations. ``spill`` stays inside the numerator, matching
:func:`tipopac.fit._residuals_tcal` (ground pickup precedes the Tcal gain).

Runs only under ``independent_tau``: the anchor is fit to Stage-A ``tau_zenith``,
so under ``independent_tau_solve`` it would carry that mode's own opacity
inflation straight into ``c``.

One scalar per group is absorbed by construction — the anchor is am at the
group's fitted PWV, itself derived from these opacities — so the array-common
level of ``c`` is not a measurement. Only the per-antenna contrast and the
per-spw shape are.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from tipopac.defaults import DEFAULT_MIN_AIRMASS_SPAN
from tipopac.fit import (
    _MIN_SAMPLES,
    _RES_REJECT_CHI2,
    _RES_REJECT_MAX_PASS,
    valid_samples,
)
from tipopac.physics import T_CMB, k2nt
from tipopac.schema import SchemaError, select_group, surface_T_mean
from tipopac.spillover import spillover_tsys
from tipopac.tables import _model_at

__all__ = ["solve_tcal"]


def _wls(x: np.ndarray, y: np.ndarray, ivar: np.ndarray) -> tuple[float, float, float]:
    """Weighted straight-line fit y = A + B·x; returns (A, B, var_B)."""
    sw = float(ivar.sum())
    sx = float(ivar @ x)
    sy = float(ivar @ y)
    sxx = float(ivar @ (x * x))
    sxy = float(ivar @ (x * y))
    det = sw * sxx - sx * sx
    if det <= 0.0 or not np.isfinite(det):
        return np.nan, np.nan, np.nan
    A = (sxx * sy - sx * sxy) / det
    B = (sw * sxy - sx * sy) / det
    return A, B, sw / det


def _anchor_fit(
    pred: np.ndarray,
    tsys_R: np.ndarray,
    tsys_L: np.ndarray,
    ivar_R: np.ndarray,
    ivar_L: np.ndarray,
) -> dict:
    """Per-pol anchor-pinned regression with joint 4σ rejection.

    A time sample is dropped when either polarization exceeds the cutoff, as
    :func:`tipopac.fit._screen_antenna` does, so both polarizations share one
    sample set.
    """
    keep = np.ones(pred.size, dtype=bool)
    out: dict = {"n": 0}
    for _ in range(_RES_REJECT_MAX_PASS):
        if int(keep.sum()) < _MIN_SAMPLES:
            return {"n": 0}
        p = pred[keep]
        A_R, B_R, varB_R = _wls(p, tsys_R[keep], ivar_R[keep])
        A_L, B_L, varB_L = _wls(p, tsys_L[keep], ivar_L[keep])
        if not np.isfinite(B_R) or not np.isfinite(B_L):
            return {"n": 0}
        chi2_R = (tsys_R[keep] - (A_R + B_R * p)) ** 2 * ivar_R[keep]
        chi2_L = (tsys_L[keep] - (A_L + B_L * p)) ** 2 * ivar_L[keep]
        out = {
            "n": int(keep.sum()),
            "B": (B_R, B_L),
            "varB": (varB_R, varB_L),
            "keep": keep,
        }
        drop = (chi2_R > _RES_REJECT_CHI2) | (chi2_L > _RES_REJECT_CHI2)
        if not drop.any():
            break
        idx = np.flatnonzero(keep)
        # Copy before mutating: on the pass-exhaustion path `out` must keep
        # referencing the mask its B/varB were computed on.
        keep = keep.copy()
        keep[idx[drop]] = False
    return out


def _anchor_tau(ds: xr.Dataset, freq_Hz: np.ndarray, n_group: int) -> np.ndarray:
    """τ_am at the spw centres, per group. Shape (n_group, n_spw)."""
    tau_am = np.full((n_group, freq_Hz.size), np.nan)
    for k in range(n_group):
        tau_am[k] = _model_at(select_group(ds, k), freq_Hz)
    return tau_am


def solve_tcal(
    ds: xr.Dataset, *, min_airmass_span: float = DEFAULT_MIN_AIRMASS_SPAN
) -> None:
    """Estimate the Tcal scale against the Stage-B anchor; mutates *ds*.

    Overwrites ``tcal_fit`` with ``c·tcal_ref`` and adds ``sigma_tcal``. Cells
    with no estimate get NaN in both, the schema's "not measured" convention.

    Parameters
    ----------
    min_airmass_span
        Floor on ``max(airmass) − min(airmass)`` over the kept samples. Below
        it ``c`` is a poorly levered extrapolation and is not reported.
    """
    if "scan_group" not in ds.coords or "am_tau" not in ds.data_vars:
        raise SchemaError(
            "solve_tcal requires the Stage-B anchor ('scan_group', 'am_tau'); "
            "run fit() first"
        )

    freq_Hz = np.asarray(ds.coords["frequency"].values, dtype=np.float64)
    groups = np.asarray(ds.coords["scan_group"].values, dtype=int)
    n_scan = ds.sizes["scan"]
    n_ant = ds.sizes["antenna"]
    n_spw = ds.sizes["spw"]

    tau_am = _anchor_tau(ds, freq_Hz, int(ds.sizes["group"]))
    tcmb = np.asarray(k2nt(T_CMB, freq_Hz), dtype=np.float64)

    tsys = ds["Tsys"].values
    sigma = ds["sigma_Tsys"].values
    flag = ds["flag"].values
    zenith = ds["zenith_angle"].values
    twmt = np.asarray(ds["Twmt"].values, dtype=np.float64)
    tcal_ref = ds["tcal_ref"].values
    t_surf = surface_T_mean(ds).values
    apply_spillover = bool(ds.attrs.get("spillover_model"))

    c_out = np.full((n_scan, n_ant, n_spw, 2), np.nan)
    sigma_c_out = np.full((n_scan, n_ant, n_spw, 2), np.nan)

    for i_scan in range(n_scan):
        tau_scan = tau_am[groups[i_scan]]
        spill_on = apply_spillover and np.isfinite(t_surf[i_scan])
        for i_ant in range(n_ant):
            z_all = zenith[i_scan, i_ant, :]
            for i_spw in range(n_spw):
                tau0 = tau_scan[i_spw]
                if not np.isfinite(tau0):
                    continue
                cell = (i_scan, i_ant, i_spw)
                tsys_R, tsys_L = tsys[cell][0], tsys[cell][1]
                sig_R, sig_L = sigma[cell][0], sigma[cell][1]
                valid = valid_samples(
                    tsys_R, tsys_L, sig_R, sig_L, flag[cell][0], flag[cell][1]
                ) & np.isfinite(z_all)
                if int(valid.sum()) < _MIN_SAMPLES:
                    continue

                z_v = z_all[valid].astype(np.float64)
                airmass = 1.0 / np.cos(np.deg2rad(z_v))
                transp = np.exp(-tau0 * airmass)
                pred = tcmb[i_spw] * transp + twmt[i_scan, i_spw] * (1.0 - transp)
                if spill_on:
                    pred = pred + np.asarray(
                        spillover_tsys(freq_Hz[i_spw], t_surf[i_scan], z_v)
                    )

                res = _anchor_fit(
                    pred,
                    tsys_R[valid].astype(np.float64),
                    tsys_L[valid].astype(np.float64),
                    1.0 / sig_R[valid].astype(np.float64) ** 2,
                    1.0 / sig_L[valid].astype(np.float64) ** 2,
                )
                if res["n"] < _MIN_SAMPLES:
                    continue
                a_keep = airmass[res["keep"]]
                if a_keep.max() - a_keep.min() < min_airmass_span:
                    continue

                for i_pol in range(2):
                    B = res["B"][i_pol]
                    if not B > 0.0:
                        continue
                    c = 1.0 / B
                    c_out[i_scan, i_ant, i_spw, i_pol] = c
                    varB = res["varB"][i_pol]
                    if np.isfinite(varB):
                        sigma_c_out[i_scan, i_ant, i_spw, i_pol] = np.sqrt(varB) * c * c

    dims = ("scan", "antenna", "spw", "polarization")
    ds["tcal_fit"] = (dims, (c_out * tcal_ref[None, ...]).astype(np.float32))
    ds["sigma_tcal"] = (dims, (sigma_c_out * tcal_ref[None, ...]).astype(np.float32))
