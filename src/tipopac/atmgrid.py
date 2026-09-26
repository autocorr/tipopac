"""PWV-parameterised opacity / sky-brightness grid for the forward-model fit.

For each scan, the atmospheric profile (pressure, temperature, H₂O VMR) is
fixed; the single free atmospheric DOF is PWV. ``PwvGrid`` precomputes
``τ_z(ν, PWV)`` and ``Tb_z(ν, PWV)`` over a regular PWV axis by running am
many times in a process pool — one am call per grid point, never
``parallel=True``, per-worker ``cache_dir`` to avoid am-cache contention.

The fitter consumes the grid via :meth:`PwvGrid.lookup` (τ_z, T_mean)
and :meth:`PwvGrid.dtau_dpwv` (∂τ_z/∂PWV via the linear interpolant's
analytical slope). ``T_mean`` is returned in Rayleigh-Jeans
noise K — am's Planck ``Tb`` is converted via :attr:`PwvGrid.trj_z` before
any sky-term arithmetic, never after.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import astropy.units as u
import numpy as np
from amwrap import Model as _AmModel

from tipopac.physics import k2nt

_T_CMB: float = float(_AmModel.background_temperature.to_value(u.K))

__all__ = ["PwvGrid", "build_pwv_grid", "grid_freq_span"]

_log = logging.getLogger(__name__)

# Default grid parameters — advisor flagged 0.05 mm as overkill; 0.5 mm gives
# linear interpolation accuracy ≲ 0.01 mm on a smooth function.
DEFAULT_PWV_MIN_MM: float = 1.0
DEFAULT_PWV_MAX_MM: float = 50.0
DEFAULT_PWV_STEP_MM: float = 0.5
DEFAULT_FREQ_STEP_HZ: float = 100e6  # 100 MHz, matches warm-am ≈ 25 ms
DEFAULT_N_WORKERS: int = 8
GRID_FREQ_MIN_HZ: float = 1e9
GRID_FREQ_MAX_HZ: float = 51e9


def grid_freq_span(
    obs_min_Hz: float, obs_max_Hz: float, step_Hz: float
) -> tuple[float, float]:
    """am grid span: 1–51 GHz on exact nodes, extended to keep the ±5 % margin."""
    below = math.ceil(max(0.0, GRID_FREQ_MIN_HZ - obs_min_Hz * 0.95) / step_Hz)
    above = math.ceil(max(0.0, obs_max_Hz * 1.05 - GRID_FREQ_MAX_HZ) / step_Hz)
    return GRID_FREQ_MIN_HZ - below * step_Hz, GRID_FREQ_MAX_HZ + above * step_Hz


# Per-worker cache dir parent. tmpfs avoids HDD-backed /tmp on hosts where
# TMPDIR isn't pointed at RAM; falls back to tempfile's default if /dev/shm
# isn't present. Mirrors amwrap's own default (amwrap/driver.py CACHE_DIR).
_DEFAULT_CACHE_BASE: str | None = "/dev/shm" if Path("/dev/shm").is_dir() else None


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
        Zenith *Planck* brightness temperature (K), shape ``(n_pwv, n_freq)``
        — am's ``brightness_temperature`` column. Not radiance-linear; use
        :attr:`trj_z` for any arithmetic that combines sky terms.
    pwv_unscaled_mm:
        PWV (mm) that am holds for the unscaled atmospheric profile
        (``amwrap.Model.pwv``). Each grid row is an am run with
        ``target_pwv = pwv_mm``, so both are in am-column units.
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
    def trj_z(self) -> np.ndarray:
        """Zenith brightness in Rayleigh-Jeans noise K — the linear coordinate.

        am emits Planck ``Tb``; radiative transfer and noise power are linear
        in radiance, so any arithmetic combining sky terms must use this, not
        ``tb_z``. Equal to am's ``Trj`` column to ~1e-6 relative.
        """
        return np.asarray(k2nt(self.tb_z, self.freq_Hz[None, :]))

    @cached_property
    def tmean(self) -> np.ndarray:
        """Atmosphere-only effective radiating temperature, in RJ noise K.

        ``T_mean = (Trj_z − k2nt(T_cmb)·exp(−τ_z)) / (1 − exp(−τ_z))``

        The CMB is subtracted so the returned value describes the atmospheric
        emission alone; without it the term ``T_cmb·exp(−τ)/(1−exp(−τ))``
        diverges at low τ and inflates T_mean by hundreds of K at low-opacity
        bands. The decomposition is linear in radiance, so it runs on
        ``trj_z`` and uses the CMB's *radiation* temperature ``k2nt(T_cmb, ν)``
        — not Planck ``tb_z`` and a flat 2.725 K. The result is already noise
        K; callers must not apply ``k2nt`` again.
        """
        absorb = -np.expm1(-self.tau_z)  # = 1 − exp(−τ), accurate for small τ
        eps = 1e-8
        trj_atm = self.trj_z - k2nt(_T_CMB, self.freq_Hz[None, :]) * np.exp(-self.tau_z)
        return trj_atm / np.maximum(absorb, eps)

    def _pwv_bracket(self, pwv_mm: float) -> tuple[int, int, float, float]:
        """Bracketing indices, weight and PWV step (1.0 on a one-point grid)."""
        pwv_c = float(np.clip(pwv_mm, self.pwv_mm[0], self.pwv_mm[-1]))
        i_hi = int(np.searchsorted(self.pwv_mm, pwv_c, side="right"))
        i_hi = min(i_hi, len(self.pwv_mm) - 1)
        i_lo = max(i_hi - 1, 0)
        if i_hi == i_lo:
            return i_lo, i_hi, 0.0, 1.0
        dpwv = self.pwv_mm[i_hi] - self.pwv_mm[i_lo]
        return i_lo, i_hi, (pwv_c - self.pwv_mm[i_lo]) / dpwv, dpwv

    def lookup(
        self,
        pwv_mm: float,
        freq_Hz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate ``(τ_z, T_mean)`` at scalar ``pwv_mm`` and array ``freq_Hz``.

        Bilinear: linear in PWV (grid step 0.5 mm), linear in freq. PWV is
        clipped to the grid range.
        """
        i_lo, i_hi, w, _ = self._pwv_bracket(pwv_mm)

        tau_lo = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_lo, :])
        tau_hi = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_hi, :])
        tau_z = (1.0 - w) * tau_lo + w * tau_hi

        tmean_lo = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_lo, :])
        tmean_hi = np.interp(freq_Hz, self.freq_Hz, self.tmean[i_hi, :])
        tmean = (1.0 - w) * tmean_lo + w * tmean_hi

        return tau_z, tmean

    def dtau_dpwv(self, pwv_mm: float, freq_Hz: np.ndarray) -> np.ndarray:
        """``∂τ_z/∂PWV`` at scalar ``pwv_mm`` and array ``freq_Hz``.

        The analytical slope of the bilinear interpolant — exact for the
        linear approximation, no finite-difference noise. Zero outside the
        grid range so the optimizer is not pushed past the edges.
        """
        if pwv_mm <= self.pwv_mm[0] or pwv_mm >= self.pwv_mm[-1]:
            return np.zeros_like(np.asarray(freq_Hz, dtype=float))

        i_lo, i_hi, _w, dpwv = self._pwv_bracket(pwv_mm)
        tau_lo = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_lo, :])
        tau_hi = np.interp(freq_Hz, self.freq_Hz, self.tau_z[i_hi, :])
        return np.asarray((tau_hi - tau_lo) / dpwv)


# ---------------------------------------------------------------------------
# am model construction
# ---------------------------------------------------------------------------


def _am_model(
    pressure_Pa: np.ndarray,
    temperature_K: np.ndarray,
    h2o_vmr: np.ndarray,
    freq_min_Hz: float = 18e9,
    freq_max_Hz: float = 26.5e9,
    freq_step_Hz: float = 10e6,
    target_pwv_mm: float | None = None,
):
    """Build an ``amwrap.Model`` for the profile, optionally at a target PWV."""
    import amwrap as _amwrap  # local import — workers don't need it at top level

    return _amwrap.Model(
        pressure=pressure_Pa * u.Pa,
        temperature=temperature_K * u.K,
        mixing_ratio={"h2o": h2o_vmr * u.dimensionless_unscaled},
        freq_min=freq_min_Hz * u.Hz,
        freq_max=freq_max_Hz * u.Hz,
        freq_step=freq_step_Hz * u.Hz,
        target_pwv=None if target_pwv_mm is None else target_pwv_mm * u.mm,
    )


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


def _worker_run(pwv_mm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool task: run am on the profile at the given PWV and return
    ``(freq_Hz, tau, tb)`` as plain ndarrays."""
    s = _WORKER_STATE
    m = _am_model(
        s["pressure_Pa"],
        s["temperature_K"],
        s["h2o_vmr"],
        s["freq_min_Hz"],
        s["freq_max_Hz"],
        s["freq_step_Hz"],
        target_pwv_mm=float(pwv_mm),
    )
    df = m.run(parallel=False, cache_dir=s["cache_dir"])
    freqs_Hz = df["frequency"].values * 1e9  # GHz → Hz
    tau = df["opacity"].values.astype(np.float64)
    tb = df["brightness_temperature"].values.astype(np.float64)
    return freqs_Hz, tau, tb


def _run_serial(pwv_axis: np.ndarray, init_kwargs: dict) -> list[tuple]:
    """Sequential equivalent — used when n_workers ≤ 1 or for small grids."""
    _worker_init(**init_kwargs)
    out = [_worker_run(p) for p in pwv_axis]
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
    n_workers: int | None = DEFAULT_N_WORKERS,
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
        Process-pool size, capped at ``cpu_count``. ``None`` or ``≤ 1``
        runs serially in the calling process.

    Notes
    -----
    Per-worker ``cache_dir`` is mandatory — multiple workers sharing the
    default am cache race on its lockfile. See ``feedback_amwrap_parallel``.
    """
    if pwv_min_mm <= 0 or pwv_max_mm <= pwv_min_mm:
        raise ValueError(f"invalid pwv range [{pwv_min_mm}, {pwv_max_mm}]")

    p_Pa = pressure.to(u.Pa).value.astype(np.float64)
    t_K = temperature.to(u.K).value.astype(np.float64)
    vmr = np.asarray(getattr(h2o_vmr, "value", h2o_vmr), dtype=np.float64)
    init_kwargs = dict(
        pressure_Pa=p_Pa,
        temperature_K=t_K,
        h2o_vmr=vmr,
        freq_min_Hz=float(freq_min_Hz),
        freq_max_Hz=float(freq_max_Hz),
        freq_step_Hz=float(freq_step_Hz),
    )
    pwv_unscaled = float(_am_model(p_Pa, t_K, vmr).pwv.to_value(u.mm))
    if pwv_unscaled <= 0:
        raise ValueError(f"profile PWV is {pwv_unscaled:.3e} mm — cannot scale")

    pwv_axis = np.arange(
        pwv_min_mm, pwv_max_mm + 0.5 * pwv_step_mm, pwv_step_mm
    ).astype(np.float64)

    n_grid = pwv_axis.size
    cpu = os.cpu_count() or 1
    n_eff = 1 if n_workers is None else min(n_workers, cpu)
    n_eff = max(1, min(n_eff, n_grid))

    if n_eff == 1:
        with tempfile.TemporaryDirectory(
            prefix="tipopac_amcache_", dir=_DEFAULT_CACHE_BASE
        ) as tmp:
            init_kwargs_t = {**init_kwargs, "base_cache_dir": tmp}
            results = _run_serial(pwv_axis, init_kwargs_t)
    else:
        with tempfile.TemporaryDirectory(
            prefix="tipopac_amcache_", dir=_DEFAULT_CACHE_BASE
        ) as tmp:
            init_kwargs_t = {**init_kwargs, "base_cache_dir": tmp}
            ctx = mp.get_context("fork")
            with ctx.Pool(
                processes=n_eff,
                initializer=_worker_init,
                initargs=tuple(init_kwargs_t.values()),
            ) as pool:
                results = pool.map(_worker_run, pwv_axis.tolist(), chunksize=1)

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
