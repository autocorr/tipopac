"""Stage B — post-fit per-antenna PWV anchor against per-spw τ_z.

See ``design/independent_tau_fit.md`` §2. Given the Stage-A outputs
(`τ_z(scan, ant, spw)`, `σ_τ(scan, ant, spw)`) and a per-scan
:class:`tipopac.atmgrid.PwvGrid`, fit a single PWV per antenna by
weighted least-squares against ``τ_grid(PWV, ν_spw)``. σ_PWV comes from
the Cramér–Rao bound (Fisher information of the bilinear interpolant) —
no Hessian inversion, no SVD.

Also exposes :func:`compute_t_mean_grid` — the Stage-A Twmt input
(noise-K) per (scan, spw) sampled from the grid at each scan's unscaled
profile PWV. Stage A uses this when the new ``independent_tau`` /
``independent_tau_solve`` modes are active; the legacy modes keep the
v2.6 ``0.95·T_surface`` Bevis form.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar

from tipopac.atmgrid import PwvGrid
from tipopac.physics import k2nt

__all__ = ["anchor_pwv", "compute_t_mean_grid", "write_am_curve"]


# Default PWV search range. The grid axis caps the actual search; these
# defaults match the grid defaults in atmgrid.py.
_PWV_MIN_MM: float = 1.0
_PWV_MAX_MM: float = 50.0


def anchor_pwv(
    tau_z: np.ndarray,
    tau_err: np.ndarray,
    grids: dict[int, PwvGrid],
    freqs_Hz: np.ndarray,
    *,
    pwv_min_mm: float = _PWV_MIN_MM,
    pwv_max_mm: float = _PWV_MAX_MM,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one PWV per antenna against per-(scan, spw) zenith opacity.

    Parameters
    ----------
    tau_z, tau_err:
        Per-(scan, ant, spw) zenith opacity and its uncertainty. NaN cells
        (no fit / screened out) are dropped from the cost.
    grids:
        Per-scan :class:`PwvGrid`, keyed by *positional* scan index
        ``0..n_scan-1`` matching the first axis of *tau_z*. Scans missing
        from this dict are silently dropped from the cost for every
        antenna; they contribute nothing to PWV or σ_PWV.
    freqs_Hz:
        Spectral-window centre frequencies (Hz), shape ``(n_spw,)``,
        matching the third axis of *tau_z*.

    Returns
    -------
    pwv_ant, pwv_err_ant:
        Shape ``(n_ant,)`` arrays of fitted PWV (mm) and σ_PWV (mm). NaN
        sentinels for antennas with zero contributing cells or zero
        Fisher information.

    Notes
    -----
    Under ``independent_tau_solve`` mode the Stage-A τ_z(scan, spw) is
    broadcast equal across all antennas (per the schema §5 convention),
    so the per-antenna anchor returns identical PWV per antenna — the
    ``shared_pwv`` semantics noted in the design. Under
    ``independent_tau`` the τ_z varies per (scan, ant, spw) and the
    fit produces a distinct PWV per antenna.
    """
    if tau_z.shape != tau_err.shape:
        raise ValueError(f"tau_z shape {tau_z.shape} != tau_err shape {tau_err.shape}")
    if tau_z.ndim != 3:
        raise ValueError(f"tau_z must be 3-D (scan, ant, spw); got {tau_z.shape}")

    n_scan, n_ant, n_spw = tau_z.shape
    if freqs_Hz.shape != (n_spw,):
        raise ValueError(f"freqs_Hz shape {freqs_Hz.shape} != (n_spw={n_spw},)")

    # Restrict to scans we actually have a grid for, in positional order.
    scans_with_grid = [i for i in range(n_scan) if i in grids]

    pwv_ant = np.full(n_ant, np.nan, dtype=np.float64)
    pwv_err_ant = np.full(n_ant, np.nan, dtype=np.float64)

    for i_ant in range(n_ant):
        # Pull per-scan slices; mask non-finite cells.
        per_scan: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
        for i_scan in scans_with_grid:
            tz = tau_z[i_scan, i_ant, :].astype(np.float64)
            te = tau_err[i_scan, i_ant, :].astype(np.float64)
            valid = np.isfinite(tz) & np.isfinite(te) & (te > 0.0)
            if not valid.any():
                continue
            per_scan.append(
                (i_scan, tz[valid], te[valid], freqs_Hz[valid].astype(np.float64))
            )

        if not per_scan:
            continue

        def _cost(pwv: float, _ps=per_scan) -> float:
            chi2 = 0.0
            for i_sc, tz, te, fv in _ps:
                tau_pred, _ = grids[i_sc].lookup(pwv, fv)
                r = (tz - tau_pred) / te
                chi2 += float(np.dot(r, r))
            return chi2

        # Anchor the search to the grid's actual range if narrower than ours.
        lo = max(pwv_min_mm, max(grids[i_sc].pwv_mm[0] for i_sc, *_ in per_scan))
        hi = min(pwv_max_mm, min(grids[i_sc].pwv_mm[-1] for i_sc, *_ in per_scan))
        if hi <= lo:
            continue

        res = minimize_scalar(_cost, bounds=(lo, hi), method="bounded")
        pwv_star = float(res.x)
        pwv_ant[i_ant] = pwv_star

        # Cramér–Rao σ_PWV at the fitted PWV.
        inv_var = 0.0
        for i_sc, _tz, te, fv in per_scan:
            _, _, dtau_dpwv, _ = grids[i_sc].lookup_with_grad(pwv_star, fv)
            w = (dtau_dpwv / te) ** 2
            inv_var += float(np.sum(w))
        if inv_var > 0.0:
            pwv_err_ant[i_ant] = 1.0 / np.sqrt(inv_var)

    return pwv_ant, pwv_err_ant


def write_am_curve(
    ds: xr.Dataset,
    grids: dict[int, PwvGrid],
    pwv: np.ndarray,
) -> None:
    """Populate ``ds['am_freq_grid']`` and ``ds['am_tau']`` from Stage-B artifacts.

    Stage A+B reuses the per-scan :class:`PwvGrid` and the fitted per-antenna
    PWV — no second am run, no second open-meteo fetch. The dense curve is
    sampled at the median fitted PWV (falling back to a reference grid's
    ``pwv_unscaled_mm`` if every antenna is NaN), on the shared grid frequency
    axis. The result is a single 1-D ``frequency_dense`` slice suitable for the
    plotting overlay at ``plot.py:286–293``.

    Parameters
    ----------
    ds:
        Mutated in place. ``am_freq_grid`` / ``am_tau`` are added or
        overwritten with shape ``(n_freq,)`` and dtype ``float64`` matching
        the schema.
    grids:
        Per-scan :class:`PwvGrid` — same dict passed to :func:`anchor_pwv`.
        Must contain at least one grid, and all grids must share an identical
        ``freq_Hz`` axis (true by construction in
        :meth:`tipopac.api.TippingAnalysis.build_atm_grids`, which uses the
        same ``freq_min_Hz``/``freq_max_Hz``/``freq_step_Hz`` for every scan).
    pwv:
        Per-antenna PWV (mm) from :func:`anchor_pwv`. NaN values are ignored
        when picking the representative PWV.
    """
    if not grids:
        raise ValueError("write_am_curve requires at least one PwvGrid")

    grid_iter = iter(grids.values())
    ref_grid = next(grid_iter)
    for other in grid_iter:
        if not np.array_equal(other.freq_Hz, ref_grid.freq_Hz):
            raise ValueError(
                "PwvGrid freq_Hz axes disagree across scans — "
                "write_am_curve assumes a single shared frequency grid"
            )

    pwv_repr = float(np.nanmedian(pwv)) if np.any(np.isfinite(pwv)) else float(
        ref_grid.pwv_unscaled_mm
    )
    tau, _ = ref_grid.lookup(pwv_repr, ref_grid.freq_Hz)

    ds["am_freq_grid"] = (("frequency_dense",), ref_grid.freq_Hz.astype(np.float64))
    ds["am_tau"] = (("frequency_dense",), tau.astype(np.float64))


def compute_t_mean_grid(
    grids: dict[int, PwvGrid],
    freqs_Hz: np.ndarray,
    *,
    n_scan: int | None = None,
) -> np.ndarray:
    """Sample Stage-A T_mean (noise K) from each scan's grid.

    For each scan present in *grids*, looks up the bilinear-interpolated
    kinetic ``T_mean(ν_spw)`` at the grid's ``pwv_unscaled_mm`` (the
    profile's native PWV) and converts to Rayleigh-Jeans-corrected noise
    temperature via :func:`tipopac.physics.k2nt`. Rows for scans missing
    from *grids* are filled with NaN, signalling Stage A to fall back to
    the Bevis surface-temperature heuristic for those cells.

    Parameters
    ----------
    grids:
        Per-scan PwvGrid, keyed by positional scan index ``0..n_scan-1``.
    freqs_Hz:
        Spectral-window centre frequencies, shape ``(n_spw,)``.
    n_scan:
        Number of scans in the parent dataset. When ``None`` it is
        inferred as ``max(keys) + 1`` — but this misses trailing-scan
        build failures; callers that know the dataset shape should pass
        it explicitly.

    Returns
    -------
    t_mean:
        Shape ``(n_scan, n_spw)`` noise-K array suitable to pass as the
        ``t_mean`` kwarg to :func:`tipopac.fit.fit_dataset`.
    """
    if not grids:
        raise ValueError("compute_t_mean_grid requires at least one PwvGrid")

    if n_scan is None:
        n_scan = max(grids) + 1
    n_spw = int(freqs_Hz.shape[0])
    t_mean = np.full((n_scan, n_spw), np.nan, dtype=np.float64)
    freqs_d = freqs_Hz.astype(np.float64)
    for i_scan, grid in grids.items():
        _tau, tmean_K = grid.lookup(float(grid.pwv_unscaled_mm), freqs_d)
        # k2nt: kinetic K → Rayleigh-Jeans noise K (per-channel ν dependence).
        t_mean[i_scan, :] = np.asarray(
            [float(k2nt(float(t_k), float(f))) for t_k, f in zip(tmean_K, freqs_d)]
        )
    return t_mean
