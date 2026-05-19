"""Tipping-curve fitter for tipopac (DESIGN.md §6.3).

Public entry point: `fit_dataset(ds, mode)` — mutates the dataset in-place.
Only `tau_per_antenna` is implemented here (milestone 3); the global modes
(`global_tau`, `tcal_solve`) are milestone 5.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import least_squares

from tipopac.physics import k2nt, weighted_mean_atm_T

__all__ = ["fit_dataset"]

# QA thresholds from v2.6 (task_tipopac.py:1396-1412)
_STD_RESI: float = 3.0  # K — post-refit residual σ ceiling
_TR_UPPER: float = 300.0  # K — Tsys upper limit (per-sample validity + QA gate)
_MIN_SAMPLES: int = 3  # minimum unflagged time samples
_DZ_MIN: float = 10.0  # deg — minimum Δ(zenith angle) across scan
_MZ_MIN: float = 30.0  # deg — minimum ZA in valid samples (uses min, not mean)

_ALLOWED_MODES = ("tau_per_antenna", "global_tau", "tcal_solve")


def fit_dataset(ds: xr.Dataset, mode: str) -> None:
    """Fit tipping curves and write result variables into *ds* in-place.

    Adds: Tsys, tau_zenith, tau_err, T0, tcal_fit, fit_success, fit_reason.
    Raises ValueError for unrecognised mode; NotImplementedError for modes not
    yet implemented (global_tau, tcal_solve — milestone 5).
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES!r}, got {mode!r}")
    if mode != "tau_per_antenna":
        raise NotImplementedError(f"mode {mode!r} is not implemented yet (milestone 5)")

    Tsys_arr = _compute_tsys(ds)
    ds["Tsys"] = (("scan", "antenna", "spw", "polarization", "time"), Tsys_arr)

    n_scan = ds.sizes["scan"]
    n_ant = ds.sizes["antenna"]
    n_spw = ds.sizes["spw"]
    n_pol = ds.sizes["polarization"]

    tau_zenith = np.full((n_scan, n_ant, n_spw), np.nan, dtype=np.float32)
    tau_err = np.full((n_scan, n_ant, n_spw), np.nan, dtype=np.float32)
    T0_out = np.full((n_scan, n_ant, n_spw, n_pol), np.nan, dtype=np.float32)
    tcal_fit = np.full((n_scan, n_ant, n_spw, n_pol), np.nan, dtype=np.float32)
    fit_success = np.zeros((n_scan, n_ant, n_spw), dtype=bool)
    fit_reason = np.full((n_scan, n_ant, n_spw), "", dtype=object)

    flag_vals = ds["flag"].values  # (scan, ant, spw, pol, time)
    zenith_vals = ds["zenith_angle"].values  # (scan, ant, time)
    weather_T_vals = ds["weather_T"].values  # (scan, time)
    tcal_ref_vals = ds["tcal_ref"].values  # (ant, spw, pol)
    freq_vals = ds.coords["frequency"].values  # (spw,) Hz

    for i_scan in range(n_scan):
        for i_ant in range(n_ant):
            for i_spw in range(n_spw):
                freq_Hz = float(freq_vals[i_spw])
                tau_upper = 0.4 if freq_Hz > 45e9 else 0.3

                result = _fit_tau_per_antenna(
                    z_all=zenith_vals[i_scan, i_ant, :],
                    tsys_R_all=Tsys_arr[i_scan, i_ant, i_spw, 0, :],
                    tsys_L_all=Tsys_arr[i_scan, i_ant, i_spw, 1, :],
                    flag_R=flag_vals[i_scan, i_ant, i_spw, 0, :],
                    flag_L=flag_vals[i_scan, i_ant, i_spw, 1, :],
                    weather_T=weather_T_vals[i_scan, :],
                    freq_Hz=freq_Hz,
                    tau_upper=tau_upper,
                )

                reason = result["reason"]
                fit_reason[i_scan, i_ant, i_spw] = reason
                fit_success[i_scan, i_ant, i_spw] = reason == "ok"

                if reason == "ok":
                    tau_zenith[i_scan, i_ant, i_spw] = result["tau0"]
                    tau_err[i_scan, i_ant, i_spw] = result["tau_err"]
                    T0_out[i_scan, i_ant, i_spw, 0] = result["T0_R"]
                    T0_out[i_scan, i_ant, i_spw, 1] = result["T0_L"]
                    # tau_per_antenna: no Tcal correction — tcal_fit == tcal_ref
                    tcal_fit[i_scan, i_ant, i_spw, 0] = tcal_ref_vals[i_ant, i_spw, 0]
                    tcal_fit[i_scan, i_ant, i_spw, 1] = tcal_ref_vals[i_ant, i_spw, 1]

    ds["tau_zenith"] = (("scan", "antenna", "spw"), tau_zenith)
    ds["tau_err"] = (("scan", "antenna", "spw"), tau_err)
    ds["T0"] = (("scan", "antenna", "spw", "polarization"), T0_out)
    ds["tcal_fit"] = (("scan", "antenna", "spw", "polarization"), tcal_fit)
    ds["fit_success"] = (("scan", "antenna", "spw"), fit_success)
    ds["fit_reason"] = (("scan", "antenna", "spw"), fit_reason)
    ds.attrs["mode"] = mode


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_tsys(ds: xr.Dataset) -> np.ndarray:
    """Compute Tsys = (switched_sum/2) / switched_diff * tcal_ref (float32).

    Cells where switched_diff ≤ 0 or switched_sum ≤ 0 are NaN (v2.6:1230-1234).
    """
    diff = ds["switched_diff"].values  # (scan, ant, spw, pol, time)
    ssum = ds["switched_sum"].values
    tcal = ds["tcal_ref"].values  # (ant, spw, pol)
    tcal_b = tcal[None, :, :, :, None]  # → (1, ant, spw, pol, 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        tsys = (ssum / 2.0) / diff * tcal_b

    tsys = np.where((diff > 0) & (ssum > 0), tsys, np.nan)
    return tsys.astype(np.float32)


def _residuals(
    p: np.ndarray,
    z: np.ndarray,
    tsys_R: np.ndarray,
    tsys_L: np.ndarray,
    Twmt: float,
) -> np.ndarray:
    """Concatenated residuals for tau_per_antenna: [R_resid..., L_resid...]."""
    T0_R, T0_L, tau0 = p
    pred = Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z))))
    return np.concatenate([tsys_R - (T0_R + pred), tsys_L - (T0_L + pred)])


def _tau_err_from_jac(
    jac: np.ndarray,
    residuals: np.ndarray,
    n_params: int,
) -> float:
    """Return τ error (last parameter) from SVD of the least_squares Jacobian."""
    n_obs = len(residuals)
    if n_obs <= n_params or jac.shape[0] == 0:
        return float("nan")
    sigma2 = float(np.sum(residuals**2)) / (n_obs - n_params)
    U, s, Vt = np.linalg.svd(jac, full_matrices=False)
    if s[0] == 0.0:
        return float("nan")
    thresh = np.finfo(float).eps * max(jac.shape) * s[0]
    s_safe = np.where(s > thresh, s, thresh)
    cov = sigma2 * (Vt.T / s_safe**2) @ Vt
    return float(np.sqrt(max(cov[-1, -1], 0.0)))


def _stdtsys_threshold(freq_Hz: float) -> float:
    """Per-sample Tsys σ ceiling (v2.6:1398-1410), frequency-dependent."""
    if freq_Hz > 40e9:
        return 20.0
    if freq_Hz > 18e9:
        return 15.0
    return 5.0


def _fit_tau_per_antenna(
    z_all: np.ndarray,
    tsys_R_all: np.ndarray,
    tsys_L_all: np.ndarray,
    flag_R: np.ndarray,
    flag_L: np.ndarray,
    weather_T: np.ndarray,
    freq_Hz: float,
    tau_upper: float,
) -> dict:
    """Two-pass clip-and-refit for a single (scan, antenna, spw) cell.

    Returns a dict with at minimum key "reason" (str). On success, also
    "tau0", "tau_err", "T0_R", "T0_L" (all float).
    """
    # Validity: not flagged AND Tsys > 0 AND Tsys < 300 AND not NaN
    valid = (
        ~flag_R
        & ~flag_L
        & (tsys_R_all > 0)
        & (tsys_R_all < _TR_UPPER)
        & ~np.isnan(tsys_R_all)
        & (tsys_L_all > 0)
        & ~np.isnan(tsys_L_all)
    )

    if int(valid.sum()) < _MIN_SAMPLES:
        return {"reason": "too_few_samples"}

    z_v = z_all[valid]
    tsys_R_v = tsys_R_all[valid]
    tsys_L_v = tsys_L_all[valid]

    T_surf_mean = float(np.mean(weather_T[valid]))
    Twmt = float(k2nt(weighted_mean_atm_T(T_surf_mean), freq_Hz))

    p0 = [50.0, 50.0, 0.2]
    bounds = ([0.0, 0.0, 0.0], [_TR_UPPER, _TR_UPPER, tau_upper])

    # --- pass 1: initial fit for outlier detection ---
    try:
        res0 = least_squares(
            _residuals, p0, args=(z_v, tsys_R_v, tsys_L_v, Twmt), bounds=bounds
        )
    except Exception:
        return {"reason": "fit_failed"}

    T0_R0, T0_L0, tau0_0 = res0.x
    pred0 = Twmt * (1.0 - np.exp(-tau0_0 / np.cos(np.deg2rad(z_v))))
    resid_R0 = tsys_R_v - (T0_R0 + pred0)
    resid_L0 = tsys_L_v - (T0_L0 + pred0)
    std_R0 = float(np.std(resid_R0))
    std_L0 = float(np.std(resid_L0))

    # 2σ clip within the valid set
    clip = np.ones(int(valid.sum()), dtype=bool)
    if std_R0 > 0.0:
        clip &= np.abs(resid_R0) < 2.0 * std_R0
    if std_L0 > 0.0:
        clip &= np.abs(resid_L0) < 2.0 * std_L0

    z_c = z_v[clip]
    tsys_R_c = tsys_R_v[clip]
    tsys_L_c = tsys_L_v[clip]

    if len(z_c) < _MIN_SAMPLES:
        return {"reason": "too_few_samples"}

    # --- pass 2: refit on clipped data ---
    try:
        res = least_squares(
            _residuals, p0, args=(z_c, tsys_R_c, tsys_L_c, Twmt), bounds=bounds
        )
    except Exception:
        return {"reason": "fit_failed"}

    T0_R, T0_L, tau0 = res.x

    # --- post-clip QA (on clipped data + refit residuals) ---
    dz = float(np.max(z_c) - np.min(z_c))
    if dz <= _DZ_MIN:
        return {"reason": "dz_too_small"}

    mz = float(np.min(z_c))  # minimum ZA — legacy v2.6 semantics
    if mz <= _MZ_MIN:
        return {"reason": "mz_too_small"}

    std_tsys = _stdtsys_threshold(freq_Hz)
    if float(np.std(tsys_R_c)) >= std_tsys or float(np.std(tsys_L_c)) >= std_tsys:
        return {"reason": "tsys_std_too_high"}

    if float(np.mean(tsys_R_c)) >= _TR_UPPER or float(np.mean(tsys_L_c)) >= _TR_UPPER:
        return {"reason": "tsys_upper_limit"}

    pred = Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z_c))))
    std_R = float(np.std(tsys_R_c - (T0_R + pred)))
    std_L = float(np.std(tsys_L_c - (T0_L + pred)))
    if std_R >= _STD_RESI or std_L >= _STD_RESI:
        return {"reason": "resid_clip"}

    tau_err_val = _tau_err_from_jac(res.jac, res.fun, 3)

    return {
        "reason": "ok",
        "tau0": float(tau0),
        "tau_err": tau_err_val,
        "T0_R": float(T0_R),
        "T0_L": float(T0_L),
    }
