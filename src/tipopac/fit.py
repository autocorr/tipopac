"""Tipping-curve fitter for tipopac.

Public entry point: ``fit_dataset(ds, mode, grids=...)`` — mutates the dataset
in place.

Modes
-----
Stage 2 (forward-model atmosphere; PWV is the single atmospheric DOF):
    - ``per_antenna_pwv``: PWV per antenna, T0 per (ant, spw, pol)
    - ``shared_pwv``: one PWV per scan, T0 per (ant, spw, pol)
    - ``tcal_solve``: PWV per antenna, T0 + c per (ant, spw, pol)

Stage 1 (legacy per-spw τ; for back-compat, will be removed in a future
release):
    - ``tau_per_antenna`` (deprecated alias warns at call time)
    - ``global_tau`` (deprecated alias warns at call time)
    - ``tcal_solve_legacy`` (explicit opt-in to the Stage-1 ridge-prone fit)

Stage 2 modes require a per-scan ``PwvGrid`` (see
:mod:`tipopac.atmgrid`); pass them as ``grids={scan_idx: PwvGrid, ...}``.
``TippingAnalysis.build_atm_grids()`` populates the dict. Stage 1 modes do not
use the grid and accept ``grids=None``.
"""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
import scipy.sparse as _sp
import xarray as xr
from scipy.optimize import least_squares

from tipopac.atmgrid import PwvGrid
from tipopac.physics import k2nt, weighted_mean_atm_T

__all__ = ["fit_dataset"]

# Per-sample validity / physical bounds. The legacy 6-gate QA cascade
# (`_STD_RESI`, freq-dep `stdTsys` bins, `_DZ_MIN`, `_MZ_MIN`, mean-Tsys)
# has been replaced by σ-weighted robust loss + reduced-χ² + identifiability
# checks (see design/model_refactor.md §1.2–1.3).
_TR_UPPER: float = 300.0  # K — per-sample Tsys validity ceiling
_MIN_SAMPLES: int = 3  # minimum unflagged time samples
_C_LO: float = 0.5  # Tcal correction multiplier lower bound (physical prior)
_C_HI: float = 2.0  # Tcal correction multiplier upper bound (physical prior)
_TAU_HI: float = 1.0  # zenith τ upper bound (physical prior across VLA bands)

_LEGACY_MODES = ("tau_per_antenna", "global_tau", "tcal_solve_legacy")
_NEW_MODES = ("per_antenna_pwv", "shared_pwv", "tcal_solve")
_ALLOWED_MODES = _LEGACY_MODES + _NEW_MODES
_OUTLIER_PWV_FLOOR_MM: float = 1.0  # VLA A-config inter-antenna PWV variability
_OUTLIER_MAD_K: float = 3.0


def fit_dataset(
    ds: xr.Dataset,
    mode: str,
    grids: dict[int, PwvGrid] | None = None,
) -> None:
    """Fit tipping curves and write result variables into *ds* in-place.

    Adds (Stage 2 modes): Tsys, sigma_Tsys, tau_zenith (derived), tau_err
    (derived), T0, tcal_fit, fit_success, fit_reason, pwv, pwv_err,
    pwv_outlier, pwv_scan_median.

    Adds (legacy modes): Tsys, sigma_Tsys, tau_zenith, tau_err, T0, tcal_fit,
    fit_success, fit_reason.

    Parameters
    ----------
    ds:
        Canonical xarray.Dataset (schema §5).
    mode:
        One of the strings listed in the module docstring.
    grids:
        ``dict[scan_idx: int → PwvGrid]``. Required for Stage 2 modes; ignored
        for legacy modes.

    Raises
    ------
    ValueError
        On unrecognised mode or when a Stage 2 mode is invoked without grids.
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES!r}, got {mode!r}")

    if mode in ("tau_per_antenna", "global_tau"):
        new_eq = "per_antenna_pwv" if mode == "tau_per_antenna" else "shared_pwv"
        warnings.warn(
            f"mode={mode!r} is deprecated and routes to the Stage-1 legacy fit. "
            f"For the Stage-2 forward-model fit, call mode={new_eq!r} after "
            "TippingAnalysis.build_atm_grids() (semantics differ — "
            "see design/model_refactor.md §2.4).",
            DeprecationWarning,
            stacklevel=2,
        )

    if mode in _NEW_MODES and grids is None:
        raise ValueError(
            f"mode={mode!r} requires precomputed PWV grids; "
            "call TippingAnalysis.build_atm_grids() first."
        )

    Tsys_arr = _compute_tsys(ds)
    ds["Tsys"] = (("scan", "antenna", "spw", "polarization", "time"), Tsys_arr)

    sigma_Tsys_arr = _compute_sigma_tsys(ds, Tsys_arr)
    ds["sigma_Tsys"] = (
        ("scan", "antenna", "spw", "polarization", "time"),
        sigma_Tsys_arr,
    )

    if mode in _NEW_MODES:
        assert grids is not None  # narrowed by the earlier check
        _fit_dataset_stage2(ds, mode, grids, Tsys_arr, sigma_Tsys_arr)
        return

    # ---- Legacy (Stage 1) fit path ----
    # tcal_solve_legacy uses the same code as the v2.6-style tcal_solve.
    legacy_mode = "tcal_solve" if mode == "tcal_solve_legacy" else mode

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
    sigma_vals = sigma_Tsys_arr  # (scan, ant, spw, pol, time)
    freq_vals = ds.coords["frequency"].values  # (spw,) Hz

    for i_scan in range(n_scan):
        for i_spw in range(n_spw):
            freq_Hz = float(freq_vals[i_spw])
            tau_upper = _TAU_HI  # uniform physical bound — no freq-dep cliff

            if legacy_mode == "tau_per_antenna":
                for i_ant in range(n_ant):
                    result = _fit_tau_per_antenna(
                        z_all=zenith_vals[i_scan, i_ant, :],
                        tsys_R_all=Tsys_arr[i_scan, i_ant, i_spw, 0, :],
                        tsys_L_all=Tsys_arr[i_scan, i_ant, i_spw, 1, :],
                        sigma_R_all=sigma_vals[i_scan, i_ant, i_spw, 0, :],
                        sigma_L_all=sigma_vals[i_scan, i_ant, i_spw, 1, :],
                        flag_R=flag_vals[i_scan, i_ant, i_spw, 0, :],
                        flag_L=flag_vals[i_scan, i_ant, i_spw, 1, :],
                        weather_T=weather_T_vals[i_scan, :],
                        freq_Hz=freq_Hz,
                        tau_upper=tau_upper,
                    )
                    reason = result["reason"]
                    fit_reason[i_scan, i_ant, i_spw] = reason
                    # "ok" → success; "poorly_identified" → values present but
                    # fit_success=False so downstream code can opt-in.
                    fit_success[i_scan, i_ant, i_spw] = reason == "ok"
                    if reason in ("ok", "poorly_identified"):
                        tau_zenith[i_scan, i_ant, i_spw] = result["tau0"]
                        tau_err[i_scan, i_ant, i_spw] = result["tau_err"]
                        T0_out[i_scan, i_ant, i_spw, 0] = result["T0_R"]
                        T0_out[i_scan, i_ant, i_spw, 1] = result["T0_L"]
                        tcal_fit[i_scan, i_ant, i_spw, 0] = tcal_ref_vals[
                            i_ant, i_spw, 0
                        ]
                        tcal_fit[i_scan, i_ant, i_spw, 1] = tcal_ref_vals[
                            i_ant, i_spw, 1
                        ]

            else:
                # global_tau or tcal_solve: per-antenna screening then one global fit
                screens: list[dict | None] = []
                screen_reasons: list[str] = []
                for i_ant in range(n_ant):
                    sc = _screen_antenna(
                        z_all=zenith_vals[i_scan, i_ant, :],
                        tsys_R_all=Tsys_arr[i_scan, i_ant, i_spw, 0, :],
                        tsys_L_all=Tsys_arr[i_scan, i_ant, i_spw, 1, :],
                        sigma_R_all=sigma_vals[i_scan, i_ant, i_spw, 0, :],
                        sigma_L_all=sigma_vals[i_scan, i_ant, i_spw, 1, :],
                        flag_R=flag_vals[i_scan, i_ant, i_spw, 0, :],
                        flag_L=flag_vals[i_scan, i_ant, i_spw, 1, :],
                        weather_T=weather_T_vals[i_scan, :],
                        freq_Hz=freq_Hz,
                        tau_upper=tau_upper,
                    )
                    screen_reasons.append(sc["reason"])
                    # Use both "ok" and "poorly_identified" — the global fit
                    # constrains τ better than a single antenna.
                    screens.append(
                        sc if sc["reason"] in ("ok", "poorly_identified") else None
                    )

                passing: list[tuple[int, dict]] = []
                for i_ant in range(n_ant):
                    sc_i = screens[i_ant]
                    if sc_i is not None:
                        passing.append((i_ant, sc_i))

                if not passing:
                    for i_ant in range(n_ant):
                        fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]
                    continue  # fit_success stays False, tau_zenith stays NaN

                passing_screens = [s for _, s in passing]
                global_result = _fit_global(
                    passing_screens,
                    tcal_mode=(legacy_mode == "tcal_solve"),
                )

                global_reason = global_result["reason"]
                if global_reason not in ("ok", "poorly_identified"):
                    for i_ant in range(n_ant):
                        if screens[i_ant] is None:
                            fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]
                        else:
                            fit_reason[i_scan, i_ant, i_spw] = "fit_failed"
                    continue  # tau_zenith stays NaN

                tau0 = global_result["tau0"]
                tau_err_val = global_result["tau_err"]

                # tau_zenith broadcasts equal across ALL antennas (schema §5)
                tau_zenith[i_scan, :, i_spw] = tau0
                tau_err[i_scan, :, i_spw] = tau_err_val

                for i_ant in range(n_ant):
                    if screens[i_ant] is None:
                        fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]
                    else:
                        fit_reason[i_scan, i_ant, i_spw] = global_reason
                        fit_success[i_scan, i_ant, i_spw] = global_reason == "ok"

                for j, (i_ant, _) in enumerate(passing):
                    T0_out[i_scan, i_ant, i_spw, 0] = global_result["T0_R"][j]
                    T0_out[i_scan, i_ant, i_spw, 1] = global_result["T0_L"][j]
                    if legacy_mode == "global_tau":
                        tcal_fit[i_scan, i_ant, i_spw, 0] = tcal_ref_vals[
                            i_ant, i_spw, 0
                        ]
                        tcal_fit[i_scan, i_ant, i_spw, 1] = tcal_ref_vals[
                            i_ant, i_spw, 1
                        ]
                    else:  # tcal_solve
                        tcal_fit[i_scan, i_ant, i_spw, 0] = (
                            global_result["c_R"][j] * tcal_ref_vals[i_ant, i_spw, 0]
                        )
                        tcal_fit[i_scan, i_ant, i_spw, 1] = (
                            global_result["c_L"][j] * tcal_ref_vals[i_ant, i_spw, 1]
                        )

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


def _compute_sigma_tsys(
    ds: xr.Dataset, tsys: np.ndarray
) -> np.ndarray:
    """Per-sample σ_Tsys from radiometer-equation error propagation.

    Derivation. Tsys is formed from switched-power: with S = switched_sum,
    D = switched_diff, T_c = tcal_ref:
        Tsys = (S / 2) · T_c / D
    In steady state D ≈ T_c, so ∂Tsys/∂D ≈ -Tsys / D ≈ -Tsys / T_c. The
    switched-difference noise from a Dicke-style accumulation over total
    integration time τ_int is σ_D ≈ √2 · Tsys / √(Δν · τ_int) (per-state
    contributions add in quadrature). Dropping the S-side correlation term
    (it cancels in steady state) and the σ_S contribution (sub-dominant
    when Tsys ≫ T_c), the dominant propagation gives:

        σ_Tsys ≈ √2 · Tsys² / (T_c · √(Δν · τ_int))

    The Tsys/T_c amplification factor (~10–60× for VLA bands) is the
    physically essential part — dropping it would mis-scale σ by that
    factor and cause 4σ residual rejection to trip on most samples.
    Δν: per-spw bandwidth (coord). τ_int: per-sample exposure_time. T_c:
    tcal_ref per (antenna, spw, pol). See plan: design/model_refactor.md §1.1.
    """
    n_scan, n_ant, n_spw, n_pol, n_time = tsys.shape
    tcal = ds["tcal_ref"].values  # (ant, spw, pol)
    bw = ds.coords["bandwidth"].values  # (spw,) Hz
    expo = ds["exposure_time"].values  # (scan, time) seconds

    tcal_b = tcal[None, :, :, :, None]  # (1, ant, spw, pol, 1)
    bw_b = bw[None, None, :, None, None]  # (1, 1, spw, 1, 1)
    expo_b = expo[:, None, None, None, :]  # (scan, 1, 1, 1, time)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = tcal_b * np.sqrt(bw_b * expo_b)
        sigma = np.sqrt(2.0) * (tsys * tsys) / denom

    finite = np.isfinite(tsys) & np.isfinite(denom) & (denom > 0)
    return np.where(finite, sigma, np.nan).astype(np.float32)


def _residuals(
    p: np.ndarray,
    z: np.ndarray,
    tsys_R: np.ndarray,
    tsys_L: np.ndarray,
    sigma_R: np.ndarray,
    sigma_L: np.ndarray,
    Twmt: float,
) -> np.ndarray:
    """σ-weighted concatenated residuals for tau_per_antenna: [R..., L...]."""
    T0_R, T0_L, tau0 = p
    pred = Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z))))
    return np.concatenate(
        [(tsys_R - (T0_R + pred)) / sigma_R, (tsys_L - (T0_L + pred)) / sigma_L]
    )


def _residuals_global(
    p: np.ndarray,
    z_list: list[np.ndarray],
    tsys_R_list: list[np.ndarray],
    tsys_L_list: list[np.ndarray],
    sigma_R_list: list[np.ndarray],
    sigma_L_list: list[np.ndarray],
    Twmt: float,
) -> np.ndarray:
    """σ-weighted global_tau residuals.

    p = [T0_R_0, T0_L_0, ..., T0_R_{N-1}, T0_L_{N-1}, tau0].
    """
    tau0 = p[-1]
    parts = []
    for k in range(len(z_list)):
        T0_R = p[2 * k]
        T0_L = p[2 * k + 1]
        pred = Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z_list[k]))))
        parts.append((tsys_R_list[k] - (T0_R + pred)) / sigma_R_list[k])
        parts.append((tsys_L_list[k] - (T0_L + pred)) / sigma_L_list[k])
    return np.concatenate(parts)


def _jac_global(
    p: np.ndarray,
    z_list: list[np.ndarray],
    tsys_R_list: list[np.ndarray],
    tsys_L_list: list[np.ndarray],
    sigma_R_list: list[np.ndarray],
    sigma_L_list: list[np.ndarray],
    Twmt: float,
) -> _sp.csr_matrix:
    """σ-weighted analytical Jacobian for _residuals_global.

    Each row is divided by the corresponding σ so that JᵀJ → inverse covariance.
    """
    tau0 = p[-1]
    N = len(z_list)
    n_total = sum(2 * len(z) for z in z_list)
    J = np.zeros((n_total, 2 * N + 1))
    row = 0
    for k in range(N):
        n_k = len(z_list[k])
        cos_z = np.cos(np.deg2rad(z_list[k]))
        dpred_dtau = Twmt * np.exp(-tau0 / cos_z) / cos_z
        inv_sR = 1.0 / sigma_R_list[k]
        inv_sL = 1.0 / sigma_L_list[k]
        # R rows
        J[row : row + n_k, 2 * k] = -inv_sR
        J[row : row + n_k, -1] = -dpred_dtau * inv_sR
        row += n_k
        # L rows
        J[row : row + n_k, 2 * k + 1] = -inv_sL
        J[row : row + n_k, -1] = -dpred_dtau * inv_sL
        row += n_k
    return _sp.csr_matrix(J)


def _residuals_tcal(
    p: np.ndarray,
    z_list: list[np.ndarray],
    tsys_R_list: list[np.ndarray],
    tsys_L_list: list[np.ndarray],
    sigma_R_list: list[np.ndarray],
    sigma_L_list: list[np.ndarray],
    Twmt: float,
) -> np.ndarray:
    """σ-weighted tcal_solve residuals.

    p = [T0_R_0, c_R_0, T0_L_0, c_L_0, ..., tau0].
    Model: Tsys_meas = (T0 + Twmt·(1−exp(−τ/cos z))) / c.
    """
    tau0 = p[-1]
    parts = []
    for k in range(len(z_list)):
        T0_R = p[4 * k]
        c_R = p[4 * k + 1]
        T0_L = p[4 * k + 2]
        c_L = p[4 * k + 3]
        pred = Twmt * (1.0 - np.exp(-tau0 / np.cos(np.deg2rad(z_list[k]))))
        parts.append((tsys_R_list[k] - (T0_R + pred) / c_R) / sigma_R_list[k])
        parts.append((tsys_L_list[k] - (T0_L + pred) / c_L) / sigma_L_list[k])
    return np.concatenate(parts)


def _jac_tcal(
    p: np.ndarray,
    z_list: list[np.ndarray],
    tsys_R_list: list[np.ndarray],
    tsys_L_list: list[np.ndarray],
    sigma_R_list: list[np.ndarray],
    sigma_L_list: list[np.ndarray],
    Twmt: float,
) -> _sp.csr_matrix:
    """σ-weighted analytical Jacobian for _residuals_tcal."""
    tau0 = p[-1]
    N = len(z_list)
    n_total = sum(2 * len(z) for z in z_list)
    J = np.zeros((n_total, 4 * N + 1))
    row = 0
    for k in range(N):
        n_k = len(z_list[k])
        T0_R = p[4 * k]
        c_R = p[4 * k + 1]
        T0_L = p[4 * k + 2]
        c_L = p[4 * k + 3]
        cos_z = np.cos(np.deg2rad(z_list[k]))
        pred = Twmt * (1.0 - np.exp(-tau0 / cos_z))
        dpred_dtau = Twmt * np.exp(-tau0 / cos_z) / cos_z
        inv_sR = 1.0 / sigma_R_list[k]
        inv_sL = 1.0 / sigma_L_list[k]
        # r_R = (tsys_R − (T0_R + pred)/c_R) / σ_R
        J[row : row + n_k, 4 * k] = -inv_sR / c_R
        J[row : row + n_k, 4 * k + 1] = inv_sR * (T0_R + pred) / c_R**2
        J[row : row + n_k, -1] = -inv_sR * dpred_dtau / c_R
        row += n_k
        # r_L = (tsys_L − (T0_L + pred)/c_L) / σ_L
        J[row : row + n_k, 4 * k + 2] = -inv_sL / c_L
        J[row : row + n_k, 4 * k + 3] = inv_sL * (T0_L + pred) / c_L**2
        J[row : row + n_k, -1] = -inv_sL * dpred_dtau / c_L
        row += n_k
    return _sp.csr_matrix(J)


def _tau_err_from_jac(
    jac: "np.ndarray | _sp.csr_matrix",
    residuals: np.ndarray,
    n_params: int,
) -> float:
    """Return τ error (last parameter) from σ-weighted Jacobian and residuals.

    Residuals and Jacobian are already divided by per-sample σ_Tsys, so JᵀJ
    is the inverse covariance up to a goodness-of-fit factor. The diagonal
    element for τ (last param) gives σ_τ. A `sigma2 = Σr²/(n−p)` factor is
    retained as a goodness-of-fit inflator — equals reduced χ² when σ is
    well-calibrated, ≥1 otherwise.

    Note: scipy's `loss="soft_l1"` returns the unscaled Jacobian; we compute
    covariance under the L2 norm here. For samples with |r̃| ≪ f_scale this
    matches the actual fit; near/beyond f_scale this slightly inflates σ_τ
    relative to a true IRLS-equivalent computation. Conservative, not a bug.
    A proper IRLS reweighting (multiply rows by √ρ′(r̃²)) is deferred.
    """
    # scipy.optimize.least_squares may return res.jac as csr_matrix, csr_array,
    # or dense ndarray depending on `jac=` callable type — use the duck-typed
    # sparse check rather than isinstance against one concrete class.
    if _sp.issparse(jac):
        jac_dense: np.ndarray = np.asarray(cast(_sp.csr_matrix, jac).toarray())
    else:
        jac_dense = np.asarray(jac)
    n_obs = len(residuals)
    if n_obs <= n_params or jac_dense.shape[0] == 0:
        return float("nan")
    sigma2 = float(np.sum(residuals**2)) / (n_obs - n_params)
    U, s, Vt = np.linalg.svd(jac_dense, full_matrices=False)
    if s[0] == 0.0:
        return float("nan")
    thresh = np.finfo(float).eps * max(jac.shape) * s[0]
    s_safe = np.where(s > thresh, s, thresh)
    cov = sigma2 * (Vt.T / s_safe**2) @ Vt
    return float(np.sqrt(max(cov[-1, -1], 0.0)))


# Robust-loss control. 4σ residual cutoff for iterative rejection.
_SOFT_L1_FSCALE: float = 3.0
_RES_REJECT_CHI2: float = 16.0  # 4σ on σ-weighted residuals
_RES_REJECT_MAX_PASS: int = 3
_REDUCED_CHI2_MAX: float = 5.0
_TAU_REL_ERR_MAX: float = 0.5  # σ_τ/τ above this → poorly_identified


def _screen_antenna(
    z_all: np.ndarray,
    tsys_R_all: np.ndarray,
    tsys_L_all: np.ndarray,
    sigma_R_all: np.ndarray,
    sigma_L_all: np.ndarray,
    flag_R: np.ndarray,
    flag_L: np.ndarray,
    weather_T: np.ndarray,
    freq_Hz: float,
    tau_upper: float,
) -> dict:
    """σ-weighted robust per-antenna tipping fit (one scan, one spw).

    Single-pass soft_l1 fit with iterative 4σ residual rejection. Acceptance:
    reduced χ² < `_REDUCED_CHI2_MAX`. Identifiability: if `σ_τ/τ` exceeds
    `_TAU_REL_ERR_MAX`, returns reason="poorly_identified" with the fit
    values intact. The old QA cascade (dz, min(z), Tsys std bins, residual σ
    ceiling) is replaced by these two signals — see
    design/model_refactor.md §1.2–1.3.

    Returns {"reason": "ok" | "poorly_identified", "z_c", "tsys_R_c",
             "tsys_L_c", "sigma_R_c", "sigma_L_c", "Twmt", "T0_R", "T0_L",
             "tau0", "tau_err", "jac", "fun", "reduced_chi2"} on numerical
    success. Returns {"reason": <code>} on early failure.
    """
    valid = (
        ~flag_R
        & ~flag_L
        & (tsys_R_all > 0)
        & (tsys_R_all < _TR_UPPER)
        & np.isfinite(tsys_R_all)
        & (tsys_L_all > 0)
        & np.isfinite(tsys_L_all)
        & np.isfinite(sigma_R_all)
        & np.isfinite(sigma_L_all)
        & (sigma_R_all > 0)
        & (sigma_L_all > 0)
    )

    if int(valid.sum()) < _MIN_SAMPLES:
        return {"reason": "too_few_samples"}

    z_v = z_all[valid].astype(np.float64)
    tsys_R_v = tsys_R_all[valid].astype(np.float64)
    tsys_L_v = tsys_L_all[valid].astype(np.float64)
    sigma_R_v = sigma_R_all[valid].astype(np.float64)
    sigma_L_v = sigma_L_all[valid].astype(np.float64)

    T_surf_mean = float(np.mean(weather_T[valid]))
    Twmt = float(k2nt(weighted_mean_atm_T(T_surf_mean), freq_Hz))

    # T0 init from Tsys-vs-airmass linear intercept (replaces v2.6 hard-coded
    # T0=50). Per-mode tau init upgrade lands in Task #4.
    airmass = 1.0 / np.cos(np.deg2rad(z_v))
    if float(np.ptp(airmass)) < 1e-6:
        # No airmass leverage (flat tipping); fall back to sample mean. The
        # identifiability check below will flag this as poorly_identified.
        T0_R_init = float(np.clip(np.mean(tsys_R_v), 0.0, _TR_UPPER))
        T0_L_init = float(np.clip(np.mean(tsys_L_v), 0.0, _TR_UPPER))
    else:
        try:
            pR = np.polyfit(airmass, tsys_R_v, 1)
            pL = np.polyfit(airmass, tsys_L_v, 1)
            T0_R_init = float(np.clip(pR[1], 0.0, _TR_UPPER))
            T0_L_init = float(np.clip(pL[1], 0.0, _TR_UPPER))
        except (np.linalg.LinAlgError, ValueError):
            T0_R_init, T0_L_init = 50.0, 50.0
    tau_init = 0.05  # provisional; replaced by am-derived value in Task #4

    p0 = [T0_R_init, T0_L_init, min(tau_init, max(tau_upper * 0.5, 1e-3))]
    bounds = ([0.0, 0.0, 0.0], [_TR_UPPER, _TR_UPPER, tau_upper])

    # iterative-rejection loop: refit, drop samples with χ² > 16 (=4σ), repeat
    mask = np.ones(len(z_v), dtype=bool)
    res = None
    for _ in range(_RES_REJECT_MAX_PASS):
        n_keep = int(mask.sum())
        if n_keep < _MIN_SAMPLES:
            return {"reason": "too_few_samples"}
        try:
            res = least_squares(
                _residuals,
                p0,
                args=(
                    z_v[mask],
                    tsys_R_v[mask],
                    tsys_L_v[mask],
                    sigma_R_v[mask],
                    sigma_L_v[mask],
                    Twmt,
                ),
                bounds=bounds,
                loss="soft_l1",
                f_scale=_SOFT_L1_FSCALE,
            )
        except Exception:
            return {"reason": "fit_failed"}
        # χ² per (kept) sample, separately for R and L halves of res.fun
        chi2 = res.fun**2
        n_kept = int(mask.sum())
        chi2_R = chi2[:n_kept]
        chi2_L = chi2[n_kept:]
        keep_R = chi2_R < _RES_REJECT_CHI2
        keep_L = chi2_L < _RES_REJECT_CHI2
        # Drop the time sample if EITHER polarization exceeds 4σ. Per-pol
        # rejection would let half-bad samples through and bias T0.
        sample_keep = keep_R & keep_L
        if sample_keep.all():
            break
        # Map back into the original-mask indexing
        kept_idx = np.flatnonzero(mask)
        drop = kept_idx[~sample_keep]
        mask = mask.copy()
        mask[drop] = False
        p0 = list(map(float, res.x))

    assert res is not None
    T0_R, T0_L, tau0 = (float(v) for v in res.x)

    z_c = z_v[mask]
    tsys_R_c = tsys_R_v[mask]
    tsys_L_c = tsys_L_v[mask]
    sigma_R_c = sigma_R_v[mask]
    sigma_L_c = sigma_L_v[mask]
    n_data = len(res.fun)
    dof = max(1, n_data - 3)
    reduced_chi2 = float(np.sum(res.fun**2)) / dof
    if reduced_chi2 > _REDUCED_CHI2_MAX:
        return {"reason": "high_chi2"}

    tau_err_val = _tau_err_from_jac(res.jac, res.fun, 3)
    poorly_identified = (
        not np.isfinite(tau_err_val)
        or tau0 <= 0.0
        or tau_err_val / max(tau0, 1e-9) > _TAU_REL_ERR_MAX
    )
    reason = "poorly_identified" if poorly_identified else "ok"

    return {
        "reason": reason,
        "z_c": z_c,
        "tsys_R_c": tsys_R_c,
        "tsys_L_c": tsys_L_c,
        "sigma_R_c": sigma_R_c,
        "sigma_L_c": sigma_L_c,
        "Twmt": Twmt,
        "T0_R": T0_R,
        "T0_L": T0_L,
        "tau0": tau0,
        "tau_err": tau_err_val,
        "jac": res.jac,
        "fun": res.fun,
        "reduced_chi2": reduced_chi2,
    }


def _fit_tau_per_antenna(
    z_all: np.ndarray,
    tsys_R_all: np.ndarray,
    tsys_L_all: np.ndarray,
    sigma_R_all: np.ndarray,
    sigma_L_all: np.ndarray,
    flag_R: np.ndarray,
    flag_L: np.ndarray,
    weather_T: np.ndarray,
    freq_Hz: float,
    tau_upper: float,
) -> dict:
    """Per-(scan, antenna, spw) σ-weighted robust fit. Thin wrapper around
    `_screen_antenna` — the fit and the screen are now the same call.

    "reason" can be "ok" (clean fit), "poorly_identified" (fit converged but
    τ is data-limited; values still returned), or a failure code.
    """
    sc = _screen_antenna(
        z_all,
        tsys_R_all,
        tsys_L_all,
        sigma_R_all,
        sigma_L_all,
        flag_R,
        flag_L,
        weather_T,
        freq_Hz,
        tau_upper,
    )
    if sc["reason"] not in ("ok", "poorly_identified"):
        return sc
    return {
        "reason": sc["reason"],
        "tau0": sc["tau0"],
        "tau_err": sc["tau_err"],
        "T0_R": sc["T0_R"],
        "T0_L": sc["T0_L"],
    }


def _fit_global(
    screens: list[dict],
    *,
    tcal_mode: bool = False,
) -> dict:
    """σ-weighted robust global fit over all passing antennas for one (scan, spw).

    Single-pass `soft_l1` loss; one physical bound set (no escalation ladder).
    Identifiability mirrors per-antenna: returns reason="poorly_identified"
    when σ_τ/τ exceeds `_TAU_REL_ERR_MAX`.

    Per-antenna T0/c lists are returned in the order of `screens`. τ is
    shared. Tcal bounds c∈[_C_LO, _C_HI] are physical (~30% diode prior);
    τ∈[0, _TAU_HI] covers all VLA bands.
    """
    N = len(screens)
    Twmt = screens[0]["Twmt"]
    z_list = [s["z_c"] for s in screens]
    tsys_R_list = [s["tsys_R_c"] for s in screens]
    tsys_L_list = [s["tsys_L_c"] for s in screens]
    sigma_R_list = [s["sigma_R_c"] for s in screens]
    sigma_L_list = [s["sigma_L_c"] for s in screens]

    tau_init = float(np.median([s["tau0"] for s in screens]))
    T0_R_init = [float(s["T0_R"]) for s in screens]
    T0_L_init = [float(s["T0_L"]) for s in screens]

    if not tcal_mode:
        n_params = 2 * N + 1
        p0 = []
        for k in range(N):
            p0.extend([T0_R_init[k], T0_L_init[k]])
        p0.append(min(tau_init, _TAU_HI * 0.9))
        lb = [0.0, 0.0] * N + [0.0]
        ub = [_TR_UPPER, _TR_UPPER] * N + [_TAU_HI]
        try:
            res = least_squares(
                _residuals_global,
                p0,
                args=(
                    z_list,
                    tsys_R_list,
                    tsys_L_list,
                    sigma_R_list,
                    sigma_L_list,
                    Twmt,
                ),
                bounds=(lb, ub),
                jac=_jac_global,
                loss="soft_l1",
                f_scale=_SOFT_L1_FSCALE,
            )
        except Exception:
            return {"reason": "fit_failed"}
        tau0 = float(res.x[-1])
        tau_err_val = _tau_err_from_jac(res.jac, res.fun, n_params)
        dof = max(1, len(res.fun) - n_params)
        reduced_chi2 = float(np.sum(res.fun**2)) / dof
        poorly_identified = (
            not np.isfinite(tau_err_val)
            or tau0 <= 0.0
            or tau_err_val / max(tau0, 1e-9) > _TAU_REL_ERR_MAX
        )
        return {
            "reason": "poorly_identified" if poorly_identified else "ok",
            "tau0": tau0,
            "tau_err": tau_err_val,
            "reduced_chi2": reduced_chi2,
            "T0_R": [float(res.x[2 * k]) for k in range(N)],
            "T0_L": [float(res.x[2 * k + 1]) for k in range(N)],
        }

    # tcal_mode: single physical bound set, no escalation ladder.
    n_params = 4 * N + 1
    p0 = []
    for k in range(N):
        p0.extend([T0_R_init[k], 1.0, T0_L_init[k], 1.0])
    p0.append(min(tau_init, _TAU_HI * 0.9))
    lb = [0.0, _C_LO, 0.0, _C_LO] * N + [0.0]
    ub = [_TR_UPPER, _C_HI, _TR_UPPER, _C_HI] * N + [_TAU_HI]
    try:
        res = least_squares(
            _residuals_tcal,
            p0,
            args=(
                z_list,
                tsys_R_list,
                tsys_L_list,
                sigma_R_list,
                sigma_L_list,
                Twmt,
            ),
            bounds=(lb, ub),
            jac=_jac_tcal,
            loss="soft_l1",
            f_scale=_SOFT_L1_FSCALE,
        )
    except Exception:
        return {"reason": "fit_failed"}

    tau0 = float(res.x[-1])
    tau_err_val = _tau_err_from_jac(res.jac, res.fun, n_params)
    dof = max(1, len(res.fun) - n_params)
    reduced_chi2 = float(np.sum(res.fun**2)) / dof
    poorly_identified = (
        not np.isfinite(tau_err_val)
        or tau0 <= 0.0
        or tau_err_val / max(tau0, 1e-9) > _TAU_REL_ERR_MAX
    )
    return {
        "reason": "poorly_identified" if poorly_identified else "ok",
        "tau0": tau0,
        "tau_err": tau_err_val,
        "reduced_chi2": reduced_chi2,
        "T0_R": [float(res.x[4 * k]) for k in range(N)],
        "c_R": [float(res.x[4 * k + 1]) for k in range(N)],
        "T0_L": [float(res.x[4 * k + 2]) for k in range(N)],
        "c_L": [float(res.x[4 * k + 3]) for k in range(N)],
    }


# ---------------------------------------------------------------------------
# Stage 2 — forward-model fit with PwvGrid
# ---------------------------------------------------------------------------


def _param_err_from_jac(
    jac: "np.ndarray | _sp.csr_matrix",
    residuals: np.ndarray,
    n_params: int,
    idx: int,
) -> float:
    """Return σ for parameter ``idx`` from σ-weighted Jacobian + residuals.

    Same covariance computation as :func:`_tau_err_from_jac`, generalised to
    any parameter index (Stage 1 always returned σ_τ at the last column).
    """
    if _sp.issparse(jac):
        jac_dense: np.ndarray = np.asarray(cast(_sp.csr_matrix, jac).toarray())
    else:
        jac_dense = np.asarray(jac)
    n_obs = len(residuals)
    if n_obs <= n_params or jac_dense.shape[0] == 0:
        return float("nan")
    sigma2 = float(np.sum(residuals**2)) / (n_obs - n_params)
    try:
        _, s, Vt = np.linalg.svd(jac_dense, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    if s[0] == 0.0:
        return float("nan")
    thresh = np.finfo(float).eps * max(jac.shape) * s[0]
    s_safe = np.where(s > thresh, s, thresh)
    cov_diag = sigma2 * float(np.sum((Vt[:, idx] / s_safe) ** 2))
    return float(np.sqrt(max(cov_diag, 0.0)))


def _forward_predict(
    pwv: float,
    T0: np.ndarray,  # (n_pol, n_spw)
    c: np.ndarray,   # (n_pol, n_spw) — ones for opacity-only mode
    z_v: np.ndarray,  # (n_time,) zenith angle degrees
    freq_per_spw: np.ndarray,
    grid: PwvGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Forward model + per-(spw) atmospheric ingredients.

    Returns ``(pred, tau_z, tmean, dtau_dpwv, dtmean_dpwv)`` where
    ``pred`` has shape ``(n_spw, n_pol, n_time)``.
    """
    tau_z, tmean, dtau_dpwv, dtmean_dpwv = grid.lookup_with_grad(pwv, freq_per_spw)
    airmass = 1.0 / np.cos(np.deg2rad(z_v))  # (n_time,)
    tau_slant = tau_z[:, None] * airmass[None, :]  # (n_spw, n_time)
    absorb = -np.expm1(-tau_slant)  # 1 − exp(−τA)
    t_sky = tmean[:, None] * absorb  # (n_spw, n_time)
    # Broadcast T0[(n_pol, n_spw)] and c[(n_pol, n_spw)] over time.
    T0_b = np.transpose(T0)[:, :, None]  # (n_spw, n_pol, 1)
    c_b = np.transpose(c)[:, :, None]    # (n_spw, n_pol, 1)
    pred = (T0_b + t_sky[:, None, :]) / c_b  # (n_spw, n_pol, n_time)
    return pred, tau_z, tmean, dtau_dpwv, dtmean_dpwv


def _unpack_params(
    p: np.ndarray,
    n_spw: int,
    n_pol: int,
    *,
    tcal_mode: bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Decompose flat parameter vector into ``(pwv, T0, c)``.

    Layout:
      opacity: ``p = [pwv, T0[0,0], T0[0,1], …, T0[1,n_spw-1]]``
      tcal:    ``p = [pwv, T0[0,0], c[0,0], T0[0,1], c[0,1], …]``
    """
    pwv = float(p[0])
    T0 = np.empty((n_pol, n_spw))
    c = np.ones((n_pol, n_spw))
    if not tcal_mode:
        T0[:] = np.asarray(p[1 : 1 + n_pol * n_spw]).reshape(n_pol, n_spw)
    else:
        body = np.asarray(p[1 : 1 + 2 * n_pol * n_spw]).reshape(n_pol, n_spw, 2)
        T0[:] = body[..., 0]
        c[:] = body[..., 1]
    return pwv, T0, c


def _fit_per_antenna_pwv(
    z_all: np.ndarray,
    tsys_arr: np.ndarray,
    sigma_arr: np.ndarray,
    flag_arr: np.ndarray,
    freq_per_spw: np.ndarray,
    grid: PwvGrid,
    *,
    tcal_mode: bool,
    pwv_init: float | None = None,
    pwv_fixed: float | None = None,
) -> dict:
    """Forward-model fit for one (scan, antenna).

    Parameters
    ----------
    z_all, tsys_arr, sigma_arr, flag_arr:
        Per-(spw, pol, time) arrays already restricted to this antenna.
        ``z_all`` has shape ``(n_time,)``; the others ``(n_spw, n_pol, n_time)``.
    freq_per_spw:
        Per-spw centre frequencies (Hz).
    grid:
        Precomputed PwvGrid for this scan.
    tcal_mode:
        If True, also fits a per-(spw, pol) multiplicative ``c`` factor on
        ``Tcal`` (Stage-2 ``tcal_solve`` semantics).
    pwv_init:
        Initial PWV (mm). Defaults to ``grid.pwv_unscaled_mm`` if finite, else
        the grid midpoint.
    pwv_fixed:
        If given, pins PWV at this value by setting tight equal-bracketed
        bounds. Used by the ``shared_pwv`` two-pass procedure to refit T0
        (and c) only, after the per-antenna PWVs have been collapsed to
        the scan-level median.

    Returns
    -------
    dict
        Keys: ``reason``, ``pwv``, ``pwv_err``, ``tau_z`` (per-spw at fitted
        PWV), ``tau_err`` (per-spw, derived from σ_PWV via dτ/dPWV), ``T0_R``,
        ``T0_L``, ``c_R``, ``c_L`` (None if not ``tcal_mode``), ``reduced_chi2``.
        ``reason`` is one of ``ok``, ``poorly_identified``,
        ``too_few_samples``, ``high_chi2``, ``fit_failed``.
    """
    n_spw, n_pol, n_time = tsys_arr.shape

    valid = (
        ~flag_arr
        & np.isfinite(tsys_arr) & (tsys_arr > 0) & (tsys_arr < _TR_UPPER)
        & np.isfinite(sigma_arr) & (sigma_arr > 0)
    )
    # Require ≥ _MIN_SAMPLES across the entire (spw, pol) data; a per-spw-pol
    # check is too strict at noisy edges.
    if int(valid.sum()) < n_spw * n_pol * _MIN_SAMPLES:
        return {"reason": "too_few_samples"}

    airmass_all = 1.0 / np.cos(np.deg2rad(z_all))

    T0_init = np.full((n_pol, n_spw), 50.0, dtype=np.float64)
    for k in range(n_spw):
        for p in range(n_pol):
            v = valid[k, p, :]
            if int(v.sum()) >= 3 and float(np.ptp(airmass_all[v])) > 1e-6:
                try:
                    coefs = np.polyfit(airmass_all[v], tsys_arr[k, p, v], 1)
                    T0_init[p, k] = float(np.clip(coefs[1], 0.0, _TR_UPPER))
                except (np.linalg.LinAlgError, ValueError):
                    pass

    if pwv_init is None:
        if np.isfinite(grid.pwv_unscaled_mm):
            pwv_init = float(
                np.clip(grid.pwv_unscaled_mm, grid.pwv_mm[0], grid.pwv_mm[-1])
            )
        else:
            pwv_init = float(grid.pwv_mm[len(grid.pwv_mm) // 2])
    pwv_init = float(np.clip(pwv_init, grid.pwv_mm[0], grid.pwv_mm[-1]))

    if pwv_fixed is not None:
        # scipy.optimize.least_squares requires lb < ub strictly; pin via a
        # vanishingly small bracket. ε well below the grid step so the LM
        # cannot move PWV across an interpolation node.
        eps_pwv = 1e-6
        pwv_lb = float(
            np.clip(pwv_fixed - eps_pwv, grid.pwv_mm[0], grid.pwv_mm[-1])
        )
        pwv_ub = float(
            np.clip(pwv_fixed + eps_pwv, grid.pwv_mm[0], grid.pwv_mm[-1])
        )
        if pwv_ub <= pwv_lb:
            pwv_ub = pwv_lb + eps_pwv  # last-resort sentinel; grid clips anyway
        pwv_init = float(np.clip(pwv_fixed, pwv_lb, pwv_ub))
    else:
        pwv_lb = float(grid.pwv_mm[0])
        pwv_ub = float(grid.pwv_mm[-1])

    if not tcal_mode:
        n_params = 1 + n_pol * n_spw
        p0 = np.empty(n_params)
        p0[0] = pwv_init
        p0[1:] = T0_init.reshape(-1)
        lb = [pwv_lb] + [0.0] * (n_pol * n_spw)
        ub = [pwv_ub] + [_TR_UPPER] * (n_pol * n_spw)
    else:
        n_params = 1 + 2 * n_pol * n_spw
        body = np.stack([T0_init, np.ones_like(T0_init)], axis=-1)
        p0 = np.empty(n_params)
        p0[0] = pwv_init
        p0[1:] = body.reshape(-1)
        lb = [pwv_lb] + ([0.0, _C_LO] * (n_pol * n_spw))
        ub = [pwv_ub] + ([_TR_UPPER, _C_HI] * (n_pol * n_spw))

    def _resid(p: np.ndarray, keep: np.ndarray) -> np.ndarray:
        pwv, T0, c = _unpack_params(p, n_spw, n_pol, tcal_mode=tcal_mode)
        pred, *_ = _forward_predict(pwv, T0, c, z_all, freq_per_spw, grid)
        return ((tsys_arr - pred) / sigma_arr)[keep]

    def _jac(p: np.ndarray, keep: np.ndarray) -> np.ndarray:
        """Dense σ-weighted Jacobian, shape (n_keep, n_params).

        T0 / c partials are structurally sparse but the matrix is small
        enough at VLA scales (~480 × 33 worst case) that a dense build is
        simpler and faster than sparse bookkeeping.
        """
        pwv, T0, c = _unpack_params(p, n_spw, n_pol, tcal_mode=tcal_mode)
        _, tau_z, tmean, dtau_dpwv, dtmean_dpwv = _forward_predict(
            pwv, T0, c, z_all, freq_per_spw, grid
        )
        airmass = 1.0 / np.cos(np.deg2rad(z_all))  # (n_time,)
        tau_slant = tau_z[:, None] * airmass[None, :]  # (n_spw, n_time)
        exp_neg = np.exp(-tau_slant)
        absorb = 1.0 - exp_neg
        c_b = np.transpose(c)[:, :, None]  # (n_spw, n_pol, 1)
        T0_b = np.transpose(T0)[:, :, None]
        t_sky = tmean[:, None] * absorb  # (n_spw, n_time)
        inv_sigma = 1.0 / sigma_arr  # (n_spw, n_pol, n_time)

        # ∂pred/∂pwv = (1/c) · ∂t_sky/∂pwv
        # ∂t_sky/∂pwv = (∂tmean/∂pwv)·absorb + tmean · A · exp(-τA) · ∂τ/∂pwv
        dt_sky_dpwv = (
            dtmean_dpwv[:, None] * absorb
            + tmean[:, None] * airmass[None, :] * exp_neg * dtau_dpwv[:, None]
        )  # (n_spw, n_time)
        dpred_dpwv = dt_sky_dpwv[:, None, :] / c_b  # (n_spw, n_pol, n_time)

        jac_full = np.zeros(
            (n_spw, n_pol, n_time, n_params), dtype=np.float64
        )
        # Residual r = (Tsys - pred)/σ → ∂r/∂param = -(∂pred/∂param)/σ
        jac_full[..., 0] = -dpred_dpwv * inv_sigma
        if not tcal_mode:
            # ∂pred/∂T0_{p',k'} = δ/c → ∂r/∂T0 = -1/(c σ)
            for k in range(n_spw):
                for pp in range(n_pol):
                    col = 1 + pp * n_spw + k
                    jac_full[k, pp, :, col] = -1.0 / (c[pp, k] * sigma_arr[k, pp, :])
        else:
            # T0 columns: -1/(c σ); c columns: +(T0 + t_sky)/(c² σ)
            for k in range(n_spw):
                for pp in range(n_pol):
                    col_T0 = 1 + 2 * (pp * n_spw + k)
                    col_c = col_T0 + 1
                    jac_full[k, pp, :, col_T0] = -1.0 / (
                        c[pp, k] * sigma_arr[k, pp, :]
                    )
                    jac_full[k, pp, :, col_c] = (
                        (T0_b[k, pp, 0] + t_sky[k, :])
                        / (c[pp, k] ** 2 * sigma_arr[k, pp, :])
                    )
        return jac_full.reshape(-1, n_params)[keep.ravel()]

    keep_mask = valid.copy()
    res = None
    for _ in range(_RES_REJECT_MAX_PASS):
        if int(keep_mask.sum()) < n_spw * n_pol * _MIN_SAMPLES:
            return {"reason": "too_few_samples"}
        try:
            res = least_squares(
                _resid,
                p0,
                jac=_jac,
                args=(keep_mask,),
                bounds=(lb, ub),
                loss="soft_l1",
                f_scale=_SOFT_L1_FSCALE,
            )
        except Exception:
            return {"reason": "fit_failed"}
        chi2 = res.fun**2
        new_drop = chi2 > _RES_REJECT_CHI2
        if not new_drop.any():
            break
        kept_flat = np.flatnonzero(keep_mask.ravel())
        drop_flat = kept_flat[new_drop]
        keep_mask = keep_mask.copy()
        keep_mask.ravel()[drop_flat] = False
        p0 = res.x.copy()

    assert res is not None

    n_obs = len(res.fun)
    dof = max(1, n_obs - n_params)
    reduced_chi2 = float(np.sum(res.fun**2)) / dof
    if reduced_chi2 > _REDUCED_CHI2_MAX:
        return {"reason": "high_chi2"}

    pwv_fit, T0_fit, c_fit = _unpack_params(
        res.x, n_spw, n_pol, tcal_mode=tcal_mode
    )
    pwv_err = _param_err_from_jac(res.jac, res.fun, n_params, idx=0)

    poorly_identified = (
        not np.isfinite(pwv_err)
        or pwv_fit <= 0.0
        or pwv_err / max(pwv_fit, 1e-9) > _TAU_REL_ERR_MAX
    )

    # Derive per-spw τ_z and σ_τ from the fitted PWV and dτ/dPWV.
    tau_z_per_spw, _, dtau_dpwv, _ = grid.lookup_with_grad(pwv_fit, freq_per_spw)
    tau_err_per_spw = np.abs(dtau_dpwv) * pwv_err

    return {
        "reason": "poorly_identified" if poorly_identified else "ok",
        "pwv": float(pwv_fit),
        "pwv_err": float(pwv_err),
        "tau_z": tau_z_per_spw.astype(np.float64),
        "tau_err": tau_err_per_spw.astype(np.float64),
        "T0_R": T0_fit[0, :].astype(np.float64),
        "T0_L": T0_fit[1, :].astype(np.float64),
        "c_R": c_fit[0, :].astype(np.float64) if tcal_mode else None,
        "c_L": c_fit[1, :].astype(np.float64) if tcal_mode else None,
        "reduced_chi2": reduced_chi2,
    }


def _fit_dataset_stage2(
    ds: xr.Dataset,
    mode: str,
    grids: dict[int, PwvGrid],
    Tsys_arr: np.ndarray,
    sigma_Tsys_arr: np.ndarray,
) -> None:
    """Stage 2 fit loop: per-(scan, antenna) forward-model with PwvGrid.

    Always populates the Stage-2 PWV variables. Also fills the legacy
    ``tau_zenith``, ``tau_err``, ``T0``, ``tcal_fit``, ``fit_success``,
    ``fit_reason`` so downstream (caltables, plot) keeps working unchanged.
    """
    n_scan = ds.sizes["scan"]
    n_ant = ds.sizes["antenna"]
    n_spw = ds.sizes["spw"]
    n_pol = ds.sizes["polarization"]

    flag_vals = ds["flag"].values
    zenith_vals = ds["zenith_angle"].values
    tcal_ref_vals = ds["tcal_ref"].values
    freq_vals = ds.coords["frequency"].values.astype(np.float64)
    scan_ids = ds.coords["scan"].values

    tau_zenith = np.full((n_scan, n_ant, n_spw), np.nan, dtype=np.float32)
    tau_err = np.full((n_scan, n_ant, n_spw), np.nan, dtype=np.float32)
    T0_out = np.full((n_scan, n_ant, n_spw, n_pol), np.nan, dtype=np.float32)
    tcal_fit = np.full((n_scan, n_ant, n_spw, n_pol), np.nan, dtype=np.float32)
    fit_success = np.zeros((n_scan, n_ant, n_spw), dtype=bool)
    fit_reason = np.full((n_scan, n_ant, n_spw), "", dtype=object)

    pwv_out = np.full((n_scan, n_ant), np.nan, dtype=np.float32)
    pwv_err_out = np.full((n_scan, n_ant), np.nan, dtype=np.float32)
    pwv_outlier = np.zeros((n_scan, n_ant), dtype=bool)
    pwv_scan_median = np.full((n_scan,), np.nan, dtype=np.float32)

    is_tcal = mode == "tcal_solve"

    for i_scan in range(n_scan):
        scan_id = int(scan_ids[i_scan])
        grid = grids.get(scan_id)
        if grid is None:
            # No grid for this scan → mark all antennas as fit_failed.
            for i_ant in range(n_ant):
                for i_spw in range(n_spw):
                    fit_reason[i_scan, i_ant, i_spw] = "no_atm_grid"
            continue

        for i_ant in range(n_ant):
            result = _fit_per_antenna_pwv(
                z_all=zenith_vals[i_scan, i_ant, :].astype(np.float64),
                tsys_arr=Tsys_arr[i_scan, i_ant, :, :, :].astype(np.float64),
                sigma_arr=sigma_Tsys_arr[i_scan, i_ant, :, :, :].astype(np.float64),
                flag_arr=flag_vals[i_scan, i_ant, :, :, :],
                freq_per_spw=freq_vals,
                grid=grid,
                tcal_mode=is_tcal,
            )
            reason = result["reason"]
            for i_spw in range(n_spw):
                fit_reason[i_scan, i_ant, i_spw] = reason
                if reason in ("ok", "poorly_identified"):
                    tau_zenith[i_scan, i_ant, i_spw] = result["tau_z"][i_spw]
                    tau_err[i_scan, i_ant, i_spw] = result["tau_err"][i_spw]
                    T0_out[i_scan, i_ant, i_spw, 0] = result["T0_R"][i_spw]
                    T0_out[i_scan, i_ant, i_spw, 1] = result["T0_L"][i_spw]
                    if is_tcal:
                        tcal_fit[i_scan, i_ant, i_spw, 0] = (
                            result["c_R"][i_spw] * tcal_ref_vals[i_ant, i_spw, 0]
                        )
                        tcal_fit[i_scan, i_ant, i_spw, 1] = (
                            result["c_L"][i_spw] * tcal_ref_vals[i_ant, i_spw, 1]
                        )
                    else:
                        tcal_fit[i_scan, i_ant, i_spw, 0] = tcal_ref_vals[
                            i_ant, i_spw, 0
                        ]
                        tcal_fit[i_scan, i_ant, i_spw, 1] = tcal_ref_vals[
                            i_ant, i_spw, 1
                        ]
                    fit_success[i_scan, i_ant, i_spw] = reason == "ok"
            if reason in ("ok", "poorly_identified"):
                pwv_out[i_scan, i_ant] = result["pwv"]
                pwv_err_out[i_scan, i_ant] = result["pwv_err"]

        # Scan-level PWV consensus + outlier flag (advisor #shared_pwv-cheap-alt
        # is implemented in a wrapping pass below for that specific mode).
        scan_pwvs = pwv_out[i_scan, :][np.isfinite(pwv_out[i_scan, :])]
        if scan_pwvs.size > 0:
            median = float(np.median(scan_pwvs))
            mad = float(np.median(np.abs(scan_pwvs - median)))
            pwv_scan_median[i_scan] = median
            threshold = max(_OUTLIER_PWV_FLOOR_MM, _OUTLIER_MAD_K * mad)
            for i_ant in range(n_ant):
                pv = pwv_out[i_scan, i_ant]
                if np.isfinite(pv) and abs(pv - median) > threshold:
                    pwv_outlier[i_scan, i_ant] = True

    # shared_pwv mode (median-then-refit-T0): freeze PWV at scan median, re-fit
    # T0 (and c if applicable) per antenna. Avoids the 400+ parameter joint LM
    # while matching the user-visible semantics "one PWV per scan".
    if mode == "shared_pwv":
        for i_scan in range(n_scan):
            scan_id = int(scan_ids[i_scan])
            grid = grids.get(scan_id)
            if grid is None or not np.isfinite(pwv_scan_median[i_scan]):
                continue
            shared_pwv = float(pwv_scan_median[i_scan])
            for i_ant in range(n_ant):
                if fit_reason[i_scan, i_ant, 0] in ("too_few_samples", "no_atm_grid"):
                    continue
                result = _fit_per_antenna_pwv(
                    z_all=zenith_vals[i_scan, i_ant, :].astype(np.float64),
                    tsys_arr=Tsys_arr[i_scan, i_ant, :, :, :].astype(np.float64),
                    sigma_arr=sigma_Tsys_arr[i_scan, i_ant, :, :, :].astype(
                        np.float64
                    ),
                    flag_arr=flag_vals[i_scan, i_ant, :, :, :],
                    freq_per_spw=freq_vals,
                    grid=grid,
                    tcal_mode=False,
                    pwv_fixed=shared_pwv,
                )
                # Pin PWV: overwrite the per-antenna PWV with the consensus, and
                # recompute τ_z at the consensus value.
                if result["reason"] in ("ok", "poorly_identified"):
                    tau_z_shared, _ = grid.lookup(shared_pwv, freq_vals)
                    _, _, dtau_dpwv, _ = grid.lookup_with_grad(shared_pwv, freq_vals)
                    for i_spw in range(n_spw):
                        tau_zenith[i_scan, i_ant, i_spw] = tau_z_shared[i_spw]
                        # Common τ_err comes from the median absolute deviation
                        # of the per-antenna PWVs (a robust σ_PWV estimator).
                        pwv_consensus_err = float(
                            np.std(pwv_out[i_scan, :][np.isfinite(pwv_out[i_scan, :])])
                            / np.sqrt(
                                max(
                                    1,
                                    int(np.sum(np.isfinite(pwv_out[i_scan, :]))),
                                )
                            )
                        )
                        tau_err[i_scan, i_ant, i_spw] = (
                            abs(dtau_dpwv[i_spw]) * pwv_consensus_err
                        )
                        T0_out[i_scan, i_ant, i_spw, 0] = result["T0_R"][i_spw]
                        T0_out[i_scan, i_ant, i_spw, 1] = result["T0_L"][i_spw]
                    pwv_out[i_scan, i_ant] = shared_pwv
                    # Recompute outlier flag — once PWV is frozen, none are
                    # outliers in shared_pwv (semantics by design).
                    pwv_outlier[i_scan, i_ant] = False

    ds["tau_zenith"] = (("scan", "antenna", "spw"), tau_zenith)
    ds["tau_err"] = (("scan", "antenna", "spw"), tau_err)
    ds["T0"] = (("scan", "antenna", "spw", "polarization"), T0_out)
    ds["tcal_fit"] = (("scan", "antenna", "spw", "polarization"), tcal_fit)
    ds["fit_success"] = (("scan", "antenna", "spw"), fit_success)
    ds["fit_reason"] = (("scan", "antenna", "spw"), fit_reason)
    ds["pwv"] = (("scan", "antenna"), pwv_out)
    ds["pwv_err"] = (("scan", "antenna"), pwv_err_out)
    ds["pwv_outlier"] = (("scan", "antenna"), pwv_outlier)
    ds["pwv_scan_median"] = (("scan",), pwv_scan_median)
    ds.attrs["mode"] = mode
    ds.attrs["pwv_profile_source"] = {
        int(scan_ids[i]): grids[int(scan_ids[i])].profile_source
        for i in range(n_scan)
        if int(scan_ids[i]) in grids
    }
