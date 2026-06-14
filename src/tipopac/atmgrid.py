"""PWV-parameterised opacity / sky-brightness grid for the forward-model fit.

Stage 2 of the model refactor (design/model_refactor.md §2.1).

For each scan, the atmospheric profile (pressure, temperature, H₂O VMR) is
fixed; the single free atmospheric DOF is PWV. ``PwvGrid`` precomputes
``τ_z(ν, PWV)`` and ``Tb_z(ν, PWV)`` over a regular PWV axis by running am
many times in a process pool — one am call per grid point, never
``parallel=True``, per-worker ``cache_dir`` to avoid am-cache contention.

The fitter consumes the grid via :meth:`PwvGrid.lookup` (τ_z, T_mean)
and :meth:`PwvGrid.lookup_with_grad` (adds ∂/∂PWV via the linear
interpolant's analytical slope).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import astropy.units as u
import numpy as np

__all__ = ["PwvGrid", "build_pwv_grid", "pwv_mm_from_profile"]

_log = logging.getLogger(__name__)

# Default grid parameters — advisor flagged 0.05 mm as overkill; 0.5 mm gives
# linear interpolation accuracy ≲ 0.01 mm on a smooth function.
DEFAULT_PWV_MIN_MM: float = 1.0
DEFAULT_PWV_MAX_MM: float = 50.0
DEFAULT_PWV_STEP_MM: float = 0.5
DEFAULT_FREQ_STEP_HZ: float = 100e6  # 100 MHz, matches warm-am ≈ 25 ms

# Physical constants for the PWV integral.
_M_WATER_OVER_M_DRY: float = 18.015 / 28.9647
_G_EARTH: float = 9.80665  # m s⁻²
_RHO_LIQ_WATER: float = 1000.0  # kg m⁻³

# am's brightness_temperature column includes the CMB attenuated through the
# atmosphere (Tb = T_atm·(1−e^−τ) + T_cmb·e^−τ). Stage A's T_mean needs the
# atmosphere-only mean, so the CMB term is subtracted before dividing by the
# absorbed fraction.
_T_CMB: float = 2.725  # K (Fixsen 2009)


@dataclass(frozen=True)
class PwvGrid:
    """Bilinear lookup table for ``τ_z(ν, PWV)`` and ``T_mean(ν, PWV)``.

    Attributes
    ----------
    pwv_mm:
        Sorted ascending PWV axis (mm).
    freq_Hz:
        Sorted ascending frequency axis (Hz) — the am output grid.
    tau_z:
        Zenith opacity, shape ``(n_pwv, n_freq)``.
    tb_z:
        Zenith brightness temperature (K), shape ``(n_pwv, n_freq)``.
    pwv_unscaled_mm:
        PWV (mm) of the unscaled atmospheric profile. The grid was built by
        running am with ``troposphere_h2o_scaling = pwv_mm / pwv_unscaled_mm``;
        downstream code can use this to invert if needed.
    profile_source:
        Free-form label for which atmospheric profile underlies the grid
        (``"open_meteo"``, ``"afgl_midlatitude_summer"`` …). Stored on the
        Dataset as the ``pwv_profile_source(scan,)`` data var.
    """

    pwv_mm: np.ndarray
    freq_Hz: np.ndarray
    tau_z: np.ndarray
    tb_z: np.ndarray
    pwv_unscaled_mm: float = field(default=float("nan"))
    profile_source: str = field(default="unknown")

    def __post_init__(self) -> None:
        if self.pwv_mm.ndim != 1 or self.freq_Hz.ndim != 1:
            raise ValueError("pwv_mm and freq_Hz must be 1-D")
        if self.tau_z.shape != (self.pwv_mm.size, self.freq_Hz.size):
            raise ValueError(
                f"tau_z shape {self.tau_z.shape} mismatches "
                f"(n_pwv={self.pwv_mm.size}, n_freq={self.freq_Hz.size})"
            )
        if self.tb_z.shape != self.tau_z.shape:
            raise ValueError("tb_z and tau_z must have matching shapes")
        if not np.all(np.diff(self.pwv_mm) > 0):
            raise ValueError("pwv_mm must be strictly ascending")
        if not np.all(np.diff(self.freq_Hz) > 0):
            raise ValueError("freq_Hz must be strictly ascending")

    @cached_property
    def tmean(self) -> np.ndarray:
        """Atmosphere-only effective radiating temperature.

        ``T_mean = (Tb_z − T_cmb·exp(−τ_z)) / (1 − exp(−τ_z))``

        am's ``brightness_temperature`` includes the CMB attenuated through
        the atmosphere; the CMB term is subtracted so the returned value is
        the kinetic mean of the atmospheric emission alone. Without the
        subtraction, the second term ``T_cmb·exp(−τ)/(1−exp(−τ))`` diverges
        at low τ and inflates T_mean by hundreds of K at low-opacity bands.
        """
        absorb = -np.expm1(-self.tau_z)  # = 1 − exp(−τ), accurate for small τ
        eps = 1e-8
        tb_atm = self.tb_z - _T_CMB * np.exp(-self.tau_z)
        return tb_atm / np.maximum(absorb, eps)

    def lookup(
        self,
        pwv_mm: float,
        freq_Hz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate ``(τ_z, T_mean)`` at scalar ``pwv_mm`` and array ``freq_Hz``.

        Bilinear: linear in PWV (grid step 0.5 mm), linear in freq. PWV is
        clipped to the grid range.
        """
        pwv_c = float(np.clip(pwv_mm, self.pwv_mm[0], self.pwv_mm[-1]))
        i_hi = int(np.searchsorted(self.pwv_mm, pwv_c, side="right"))
        i_hi = min(i_hi, len(self.pwv_mm) - 1)
        i_lo = max(i_hi - 1, 0)
        if i_hi == i_lo:
            w = 0.0
        else:
            w = (pwv_c - self.pwv_mm[i_lo]) / (self.pwv_mm[i_hi] - self.pwv_mm[i_lo])

        tau_lo = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_lo, :])
        tau_hi = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_hi, :])
        tau_z = (1.0 - w) * tau_lo + w * tau_hi

        tmean_lo = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_lo, :])
        tmean_hi = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_hi, :])
        tmean = (1.0 - w) * tmean_lo + w * tmean_hi

        return tau_z, tmean

    def lookup_with_grad(
        self,
        pwv_mm: float,
        freq_Hz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """As :meth:`lookup`, plus ``∂τ_z/∂PWV`` and ``∂T_mean/∂PWV``.

        Derivatives are the analytical slope of the bilinear interpolant —
        exact for the linear approximation, no finite-difference noise.
        """
        pwv_c = float(np.clip(pwv_mm, self.pwv_mm[0], self.pwv_mm[-1]))
        i_hi = int(np.searchsorted(self.pwv_mm, pwv_c, side="right"))
        i_hi = min(i_hi, len(self.pwv_mm) - 1)
        i_lo = max(i_hi - 1, 0)
        if i_hi == i_lo:
            w = 0.0
            dpwv = 1.0  # value irrelevant; gradient is zero at the boundary
        else:
            dpwv = self.pwv_mm[i_hi] - self.pwv_mm[i_lo]
            w = (pwv_c - self.pwv_mm[i_lo]) / dpwv

        tau_lo = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_lo, :])
        tau_hi = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_hi, :])
        tau_z = (1.0 - w) * tau_lo + w * tau_hi
        dtau_dpwv = (tau_hi - tau_lo) / dpwv

        tmean_lo = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_lo, :])
        tmean_hi = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_hi, :])
        tmean = (1.0 - w) * tmean_lo + w * tmean_hi
        dtmean_dpwv = (tmean_hi - tmean_lo) / dpwv

        # Zero out gradient at clipped edges so the optimizer doesn't push there.
        if pwv_mm <= self.pwv_mm[0] or pwv_mm >= self.pwv_mm[-1]:
            dtau_dpwv = np.zeros_like(dtau_dpwv)
            dtmean_dpwv = np.zeros_like(dtmean_dpwv)

        return tau_z, tmean, dtau_dpwv, dtmean_dpwv


# ---------------------------------------------------------------------------
# PWV integration helper
# ---------------------------------------------------------------------------


def pwv_mm_from_profile(
    pressure: u.Quantity,
    h2o_vmr: u.Quantity | np.ndarray,
) -> float:
    """Compute PWV (mm) of an atmospheric profile from VMR by hydrostatic integral.

    ``PWV[m liquid] = (M_w / M_dry) / (g · ρ_w) · ∫ VMR dP``

    Pressure is taken as ``astropy.units.Quantity`` (any pressure unit accepted);
    VMR is dimensionless. The integral handles either pressure ordering — am's
    convention is surface (highest P) first.
    """
    p_Pa = pressure.to(u.Pa).value
    vmr = np.asarray(getattr(h2o_vmr, "value", h2o_vmr), dtype=np.float64)
    order = np.argsort(p_Pa)  # ascending P
    p_a = p_Pa[order]
    v_a = vmr[order]
    integral_pa = float(np.trapezoid(v_a, p_a))  # ∫ VMR dP, ≥ 0
    pwv_m = _M_WATER_OVER_M_DRY / (_G_EARTH * _RHO_LIQ_WATER) * integral_pa
    return float(pwv_m * 1000.0)


# ---------------------------------------------------------------------------
# Pool worker — module-level so it pickles cleanly.
# ---------------------------------------------------------------------------


_WORKER_STATE: dict = {}


def _worker_init(
    pressure_Pa: np.ndarray,
    temperature_K: np.ndarray,
    h2o_vmr: np.ndarray,
    freq_min_Hz: float,
    freq_max_Hz: float,
    freq_step_Hz: float,
    base_cache_dir: str,
) -> None:
    """Pool initializer: stash the model kwargs (as raw arrays) + per-worker
    cache_dir. The astropy unit attachment happens inside the worker to keep
    pickle payloads small."""
    _WORKER_STATE.clear()
    cache_dir = Path(base_cache_dir) / f"w{os.getpid()}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _WORKER_STATE.update(
        pressure_Pa=pressure_Pa,
        temperature_K=temperature_K,
        h2o_vmr=h2o_vmr,
        freq_min_Hz=freq_min_Hz,
        freq_max_Hz=freq_max_Hz,
        freq_step_Hz=freq_step_Hz,
        cache_dir=str(cache_dir),
    )


def _worker_run(scaling: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool task: build a fresh amwrap.Model with the given scaling and return
    ``(freq_Hz, tau, tb)`` as plain ndarrays."""
    import amwrap as _amwrap  # local import — workers don't need it at top level

    s = _WORKER_STATE
    m = _amwrap.Model(
        pressure=s["pressure_Pa"] * u.Pa,
        temperature=s["temperature_K"] * u.K,
        mixing_ratio={"h2o": s["h2o_vmr"] * u.dimensionless_unscaled},
        freq_min=s["freq_min_Hz"] * u.Hz,
        freq_max=s["freq_max_Hz"] * u.Hz,
        freq_step=s["freq_step_Hz"] * u.Hz,
        troposphere_h2o_scaling=float(scaling),
    )
    df = m.run(parallel=False, cache_dir=s["cache_dir"])
    freqs_Hz = df["frequency"].values * 1e9  # GHz → Hz
    tau = df["opacity"].values.astype(np.float64)
    tb = df["brightness_temperature"].values.astype(np.float64)
    return freqs_Hz, tau, tb


def _run_serial(scalings: np.ndarray, init_kwargs: dict) -> list[tuple]:
    """Sequential equivalent — used when n_workers ≤ 1 or for small grids."""
    _worker_init(**init_kwargs)
    out = [_worker_run(s) for s in scalings]
    _WORKER_STATE.clear()
    return out


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------


def build_pwv_grid(
    pressure: u.Quantity,
    temperature: u.Quantity,
    h2o_vmr: u.Quantity | np.ndarray,
    *,
    freq_min_Hz: float,
    freq_max_Hz: float,
    profile_source: str = "unknown",
    pwv_min_mm: float = DEFAULT_PWV_MIN_MM,
    pwv_max_mm: float = DEFAULT_PWV_MAX_MM,
    pwv_step_mm: float = DEFAULT_PWV_STEP_MM,
    freq_step_Hz: float = DEFAULT_FREQ_STEP_HZ,
    n_workers: int | None = None,
) -> PwvGrid:
    """Run am over a PWV grid and return a populated :class:`PwvGrid`.

    Parameters
    ----------
    pressure, temperature, h2o_vmr:
        Atmospheric profile (astropy Quantities; VMR may be a bare ndarray).
    freq_min_Hz, freq_max_Hz:
        Frequency span of the lookup table; should bracket all spw centres
        with a small margin (~5 %).
    profile_source:
        Free-form label stored on the grid for provenance.
    pwv_min_mm, pwv_max_mm, pwv_step_mm:
        PWV grid range and step. Defaults give 99 points over [1, 50] mm.
    freq_step_Hz:
        am output frequency step. 100 MHz keeps each run ~25 ms warm.
    n_workers:
        Process-pool size. Defaults to ``min(40, n_grid, os.cpu_count())``.

    Notes
    -----
    Per-worker ``cache_dir`` is mandatory — multiple workers sharing the
    default am cache race on its lockfile. See ``feedback_amwrap_parallel``.
    """
    if pwv_min_mm <= 0 or pwv_max_mm <= pwv_min_mm:
        raise ValueError(f"invalid pwv range [{pwv_min_mm}, {pwv_max_mm}]")

    h2o_q: u.Quantity = (
        h2o_vmr
        if isinstance(h2o_vmr, u.Quantity)
        else np.asarray(h2o_vmr) * u.dimensionless_unscaled
    )
    pwv_unscaled = pwv_mm_from_profile(pressure, h2o_q)
    if pwv_unscaled <= 0:
        raise ValueError(
            f"profile PWV is {pwv_unscaled:.3e} mm — cannot anchor scaling"
        )
    # NB. ``troposphere_h2o_scaling`` scales only the *tropospheric* H₂O column,
    # while ``pwv_mm_from_profile`` integrates the whole vertical profile. So
    # the grid axis ``pwv_mm`` literally means
    #   pwv_target_mm = scaling × pwv_unscaled_mm
    # where ``pwv_unscaled_mm`` includes a small (< 1 mm) stratospheric
    # contribution. The recovered ``pwv`` field is therefore the tropospheric
    # PWV expressed in scaled-total units, not the column-integrated PWV. The
    # difference is well within v1 precision for VLA conditions.

    pwv_axis = np.arange(
        pwv_min_mm, pwv_max_mm + 0.5 * pwv_step_mm, pwv_step_mm
    ).astype(np.float64)
    scalings = pwv_axis / pwv_unscaled

    init_kwargs = dict(
        pressure_Pa=pressure.to(u.Pa).value.astype(np.float64),
        temperature_K=temperature.to(u.K).value.astype(np.float64),
        h2o_vmr=np.asarray(h2o_q.value, dtype=np.float64),
        freq_min_Hz=float(freq_min_Hz),
        freq_max_Hz=float(freq_max_Hz),
        freq_step_Hz=float(freq_step_Hz),
    )

    n_grid = pwv_axis.size
    cpu = os.cpu_count() or 1
    n_eff = n_workers if n_workers is not None else min(40, n_grid, cpu)
    n_eff = max(1, min(n_eff, n_grid))

    if n_eff == 1:
        with tempfile.TemporaryDirectory(prefix="tipopac_amcache_") as tmp:
            init_kwargs_t = {**init_kwargs, "base_cache_dir": tmp}
            results = _run_serial(scalings, init_kwargs_t)
    else:
        with tempfile.TemporaryDirectory(prefix="tipopac_amcache_") as tmp:
            init_kwargs_t = {**init_kwargs, "base_cache_dir": tmp}
            ctx = mp.get_context("spawn")
            with ctx.Pool(
                processes=n_eff,
                initializer=_worker_init,
                initargs=tuple(init_kwargs_t.values()),
            ) as pool:
                results = pool.map(_worker_run, scalings.tolist(), chunksize=1)

    freq_ref = results[0][0]
    if not all(np.array_equal(r[0], freq_ref) for r in results):
        raise RuntimeError(
            "am returned inconsistent frequency grids across workers — check "
            "freq_step_Hz or amwrap version mismatch"
        )
    tau_z = np.stack([r[1] for r in results], axis=0).astype(np.float64)
    tb_z = np.stack([r[2] for r in results], axis=0).astype(np.float64)

    _log.info(
        "Built PwvGrid: source=%s, n_pwv=%d (%.2f→%.2f mm step %.3f mm), "
        "n_freq=%d (%.2f→%.2f GHz step %.1f MHz), profile PWV=%.3f mm",
        profile_source,
        n_grid,
        pwv_min_mm,
        pwv_max_mm,
        pwv_step_mm,
        freq_ref.size,
        freq_ref[0] / 1e9,
        freq_ref[-1] / 1e9,
        freq_step_Hz / 1e6,
        pwv_unscaled,
    )

    return PwvGrid(
        pwv_mm=pwv_axis,
        freq_Hz=freq_ref,
        tau_z=tau_z,
        tb_z=tb_z,
        pwv_unscaled_mm=pwv_unscaled,
        profile_source=profile_source,
    )
