"""Tipping-curve fitter for tipopac.

Public entry point: ``fit_dataset(ds, mode)`` — mutates the dataset in place.

Modes
-----
- ``tau_per_antenna``: per-(scan, ant, spw) opacity fit; 3-param LM
  (τ_z, T0_R, T0_L) per fit.
- ``tcal_solve``: per-(scan, spw) joint-across-antennas fit; shared τ_z,
  per-antenna (T0_R, c_R, T0_L, c_L). Sparse Jacobian.

PWV is not a fit parameter at this layer — the atmospheric anchor lives in
:mod:`tipopac.anchor` (post-hoc fit of PWV against τ_z(ν)). See
``design/independent_tau_fit.md``.
"""

from __future__ import annotations

import multiprocessing as _mp
import os as _os
from typing import cast

import numpy as np
import scipy.sparse as _sp
import xarray as xr
from scipy.optimize import least_squares

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

_ALLOWED_MODES = ("tau_per_antenna", "tcal_solve")


def fit_dataset(
    ds: xr.Dataset,
    mode: str,
    *,
    t_mean: np.ndarray | None = None,
    n_workers: int | None = None,
) -> None:
    """Fit tipping curves and write result variables into *ds* in-place.

    Adds: ``Tsys``, ``sigma_Tsys``, ``tau_zenith``, ``tau_err``, ``T0``,
    ``tcal_fit``, ``fit_success``, ``fit_reason``.

    Parameters
    ----------
    ds:
        Canonical xarray.Dataset (schema §5).
    mode:
        One of the strings listed in the module docstring.
    t_mean:
        Optional ``(n_scan, n_spw)`` array of effective radiating
        temperatures (noise K, already Rayleigh-Jeans-corrected via
        :func:`tipopac.physics.k2nt`). When provided, the per-cell value
        replaces the v2.6 ``k2nt(0.95·T_surface)`` Bevis heuristic for
        Stage A. ``NaN`` entries fall back to the Bevis form on that
        cell — see ``design/independent_tau_fit.md`` §1. ``T_mean`` from
        :func:`tipopac.anchor.compute_t_mean_grid` is the recommended
        feed.
    n_workers:
        If ``> 1``, dispatch Stage-A fit work via a
        :class:`multiprocessing.Pool` with the ``spawn`` start method;
        each worker exports single-threaded BLAS env vars. ``None`` or
        ``≤ 1`` runs serially in the calling process. Stage-A
        parallelism unit is ``(scan, ant, spw)`` for opacity mode and
        ``(scan, spw)`` for global/tcal modes.

    Raises
    ------
    ValueError
        On unrecognised mode or shape-mismatched ``t_mean``.
    """
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"mode must be one of {_ALLOWED_MODES!r}, got {mode!r}")

    Tsys_arr = _compute_tsys(ds)
    ds["Tsys"] = (("scan", "antenna", "spw", "polarization", "time"), Tsys_arr)

    sigma_Tsys_arr = _compute_sigma_tsys(ds, Tsys_arr)
    ds["sigma_Tsys"] = (
        ("scan", "antenna", "spw", "polarization", "time"),
        sigma_Tsys_arr,
    )

    n_scan = ds.sizes["scan"]
    n_ant = ds.sizes["antenna"]
    n_spw = ds.sizes["spw"]
    n_pol = ds.sizes["polarization"]

    if t_mean is not None and t_mean.shape != (n_scan, n_spw):
        raise ValueError(
            f"t_mean shape {t_mean.shape} != (n_scan={n_scan}, n_spw={n_spw})"
        )

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

    if mode == "tau_per_antenna":
        tasks = list(
            _build_opacity_tasks(
                n_scan,
                n_ant,
                n_spw,
                zenith_vals,
                Tsys_arr,
                sigma_vals,
                flag_vals,
                weather_T_vals,
                freq_vals,
                t_mean,
            )
        )
        for (i_scan, i_ant, i_spw), result in _dispatch(
            tasks, _opacity_worker, n_workers
        ):
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
                tcal_fit[i_scan, i_ant, i_spw, 0] = tcal_ref_vals[i_ant, i_spw, 0]
                tcal_fit[i_scan, i_ant, i_spw, 1] = tcal_ref_vals[i_ant, i_spw, 1]
    else:
        # tcal_solve: per-antenna screening then one global fit per
        # (scan, spw). Bundle one task per (scan, spw) so the inner
        # screening loop stays inside the worker.
        tasks = list(
            _build_global_tasks(
                n_scan,
                n_ant,
                n_spw,
                zenith_vals,
                Tsys_arr,
                sigma_vals,
                flag_vals,
                weather_T_vals,
                freq_vals,
                t_mean,
            )
        )
        for (i_scan, i_spw), packaged in _dispatch(tasks, _global_worker, n_workers):
            screen_reasons = packaged["screen_reasons"]
            passing_idx = packaged["passing_idx"]
            global_result = packaged["global_result"]

            if global_result is None:
                # No antenna passed screening; record per-ant screen reasons.
                for i_ant in range(n_ant):
                    fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]
                continue

            global_reason = global_result["reason"]
            if global_reason not in ("ok", "poorly_identified"):
                for i_ant in range(n_ant):
                    if i_ant in passing_idx:
                        fit_reason[i_scan, i_ant, i_spw] = "fit_failed"
                    else:
                        fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]
                continue

            tau0 = global_result["tau0"]
            tau_err_val = global_result["tau_err"]

            # tau_zenith broadcasts equal across ALL antennas (schema §5)
            tau_zenith[i_scan, :, i_spw] = tau0
            tau_err[i_scan, :, i_spw] = tau_err_val

            for i_ant in range(n_ant):
                if i_ant in passing_idx:
                    fit_reason[i_scan, i_ant, i_spw] = global_reason
                    fit_success[i_scan, i_ant, i_spw] = global_reason == "ok"
                else:
                    fit_reason[i_scan, i_ant, i_spw] = screen_reasons[i_ant]

            for j, i_ant in enumerate(passing_idx):
                T0_out[i_scan, i_ant, i_spw, 0] = global_result["T0_R"][j]
                T0_out[i_scan, i_ant, i_spw, 1] = global_result["T0_L"][j]
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


def _compute_sigma_tsys(ds: xr.Dataset, tsys: np.ndarray) -> np.ndarray:
    """Per-sample σ_Tsys from switched-power error propagation.

        σ_Tsys ≈ √2 · Tsys² / (T_c · √(Δν · τ_int))

    Why Tsys² and not the usual radiometer-equation Tsys. VLA switched
    power forms Tsys = (S/2) · T_c / D from two correlator accumulators
    (S = switched_sum, D = switched_diff). Unlike a total-power
    measurement where the receiver gain G is calibrated externally,
    here G is derived from D itself via G ≈ D/T_c. The noise diode is
    the on-line calibrator, and its SNR — D over σ_D ≈ T_c · √(Δν τ) /
    Tsys — sets the floor. The fractional error in D therefore
    propagates one-for-one into Tsys, with a Tsys/T_c amplification
    over the naive σ = Tsys/√(Δν τ). For VLA Tsys/T_c spans 10–50
    across bands, so σ_Tsys runs 10–70× the naive value; dropping the
    amplification trips the 4σ residual rejection on most samples.

    The full propagation drops the σ_S contribution (sub-dominant by
    ~(2 Tsys/T_c)²) and uses Cov(S, D) ≈ 0 in steady state, giving the
    expression above. Empirically (``run/sigma_tsys``), a multiple
    regression of detrended-Tsys MAD scatter over ~6000 cells gives
    (log Tsys, log T_c) exponents (+1.82, −0.89) vs the predicted
    (+2, −1) and the naive (+1, 0). The absolute normalization comes
    out ~1.8× low; about √2 of that is the per-state-vs-total-interval
    ambiguity in ``exposure_time`` for the Dicke accumulation, the
    rest residual gain drift and 3-bit quantization noise.

    T_c is ``tcal_ref`` rather than the solve-mode ``tcal_fit``:
    empirically that choice gives the cleaner σ, since ``tcal_fit``
    carries trajectory noise from the near-degenerate (T_0, c, τ)
    fit ridge. Computing σ here, pre-fit, also makes the noise model
    identical across fit modes.

    Inputs:
        Δν      — ``bandwidth`` coord (per spw), Hz
        τ_int   — ``exposure_time`` (per scan, time), s
        T_c     — ``tcal_ref`` (per antenna, spw, pol), K

    See ``design/design.md`` §5.3.
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
    Twmt_override: float | None = None,
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

    if Twmt_override is not None and np.isfinite(Twmt_override):
        Twmt = float(Twmt_override)
    else:
        # Bevis fallback: surface-T proxy when no grid T_mean is available
        # (e.g. legacy modes, or independent_tau with grid build failure).
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
    Twmt_override: float | None = None,
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
        Twmt_override=Twmt_override,
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


def _fit_global(screens: list[dict]) -> dict:
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
# Task builders and worker functions for serial / multiprocessing dispatch
# ---------------------------------------------------------------------------


def _build_opacity_tasks(
    n_scan: int,
    n_ant: int,
    n_spw: int,
    zenith_vals: np.ndarray,
    Tsys_arr: np.ndarray,
    sigma_vals: np.ndarray,
    flag_vals: np.ndarray,
    weather_T_vals: np.ndarray,
    freq_vals: np.ndarray,
    t_mean: np.ndarray | None,
):
    """Yield ``((i_scan, i_ant, i_spw), kwargs)`` for every opacity cell."""
    for i_scan in range(n_scan):
        for i_spw in range(n_spw):
            freq_Hz = float(freq_vals[i_spw])
            twmt = (
                float(t_mean[i_scan, i_spw])
                if t_mean is not None and np.isfinite(t_mean[i_scan, i_spw])
                else None
            )
            for i_ant in range(n_ant):
                kw = {
                    "z_all": zenith_vals[i_scan, i_ant, :],
                    "tsys_R_all": Tsys_arr[i_scan, i_ant, i_spw, 0, :],
                    "tsys_L_all": Tsys_arr[i_scan, i_ant, i_spw, 1, :],
                    "sigma_R_all": sigma_vals[i_scan, i_ant, i_spw, 0, :],
                    "sigma_L_all": sigma_vals[i_scan, i_ant, i_spw, 1, :],
                    "flag_R": flag_vals[i_scan, i_ant, i_spw, 0, :],
                    "flag_L": flag_vals[i_scan, i_ant, i_spw, 1, :],
                    "weather_T": weather_T_vals[i_scan, :],
                    "freq_Hz": freq_Hz,
                    "tau_upper": _TAU_HI,
                    "Twmt_override": twmt,
                }
                yield ((i_scan, i_ant, i_spw), kw)


def _build_global_tasks(
    n_scan: int,
    n_ant: int,
    n_spw: int,
    zenith_vals: np.ndarray,
    Tsys_arr: np.ndarray,
    sigma_vals: np.ndarray,
    flag_vals: np.ndarray,
    weather_T_vals: np.ndarray,
    freq_vals: np.ndarray,
    t_mean: np.ndarray | None,
):
    """Yield ``((i_scan, i_spw), kwargs)`` for every tcal_solve cell.

    Each task carries every antenna's screening inputs for this (scan, spw),
    so the worker performs screening + the global fit without further
    coordination.
    """
    for i_scan in range(n_scan):
        for i_spw in range(n_spw):
            freq_Hz = float(freq_vals[i_spw])
            twmt = (
                float(t_mean[i_scan, i_spw])
                if t_mean is not None and np.isfinite(t_mean[i_scan, i_spw])
                else None
            )
            per_ant: list[dict] = []
            for i_ant in range(n_ant):
                per_ant.append(
                    {
                        "z_all": zenith_vals[i_scan, i_ant, :],
                        "tsys_R_all": Tsys_arr[i_scan, i_ant, i_spw, 0, :],
                        "tsys_L_all": Tsys_arr[i_scan, i_ant, i_spw, 1, :],
                        "sigma_R_all": sigma_vals[i_scan, i_ant, i_spw, 0, :],
                        "sigma_L_all": sigma_vals[i_scan, i_ant, i_spw, 1, :],
                        "flag_R": flag_vals[i_scan, i_ant, i_spw, 0, :],
                        "flag_L": flag_vals[i_scan, i_ant, i_spw, 1, :],
                    }
                )
            kw = {
                "per_ant": per_ant,
                "weather_T": weather_T_vals[i_scan, :],
                "freq_Hz": freq_Hz,
                "tau_upper": _TAU_HI,
                "Twmt_override": twmt,
            }
            yield ((i_scan, i_spw), kw)


def _opacity_worker(args: tuple) -> tuple:
    """Module-level worker: invokes ``_fit_tau_per_antenna`` on one cell."""
    key, kwargs = args
    return key, _fit_tau_per_antenna(**kwargs)


def _global_worker(args: tuple) -> tuple:
    """Module-level worker: per-antenna screen + global fit for one cell."""
    key, kwargs = args
    per_ant = kwargs["per_ant"]
    weather_T = kwargs["weather_T"]
    freq_Hz = kwargs["freq_Hz"]
    tau_upper = kwargs["tau_upper"]
    Twmt_override = kwargs["Twmt_override"]

    screens: list[dict | None] = []
    screen_reasons: list[str] = []
    for ant_in in per_ant:
        sc = _screen_antenna(
            **ant_in,
            weather_T=weather_T,
            freq_Hz=freq_Hz,
            tau_upper=tau_upper,
            Twmt_override=Twmt_override,
        )
        screen_reasons.append(sc["reason"])
        screens.append(sc if sc["reason"] in ("ok", "poorly_identified") else None)

    passing: list[tuple[int, dict]] = [
        (i, s) for i, s in enumerate(screens) if s is not None
    ]
    if not passing:
        return key, {
            "screen_reasons": screen_reasons,
            "passing_idx": [],
            "global_result": None,
        }

    passing_idx = [i for i, _ in passing]
    passing_screens = [s for _, s in passing]
    global_result = _fit_global(passing_screens)
    return key, {
        "screen_reasons": screen_reasons,
        "passing_idx": passing_idx,
        "global_result": global_result,
    }


def _pool_init() -> None:
    """Pool initializer — single-threaded BLAS per worker.

    Parent already sets these via ``tipopac/__init__.py``, but ``spawn``
    workers inherit a clean environment by design — re-export here.
    """
    for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
        _os.environ[k] = "1"


def _dispatch(tasks: list, worker, n_workers: int | None):
    """Run *worker* across *tasks* either serially or via a process pool.

    Yields the worker's return values; ordering not guaranteed when a pool
    is used. Callers must address output cells by the key the worker
    returns, not by iteration order.
    """
    if n_workers is None or n_workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            yield worker(t)
        return

    ctx = _mp.get_context("spawn")
    with ctx.Pool(processes=int(n_workers), initializer=_pool_init) as pool:
        # imap_unordered streams results as workers finish; chunksize tuned
        # so each worker keeps busy on light per-cell work without
        # round-tripping every task through the queue.
        chunksize = max(1, len(tasks) // (int(n_workers) * 4))
        yield from pool.imap_unordered(worker, tasks, chunksize=chunksize)
