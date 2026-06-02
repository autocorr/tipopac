"""Identifiability regression tests for the post-refactor fit.

The legacy geometric QA gates (`dz > 10°`, `min(z) > 30°`) were removed in
favour of an identifiability check (`σ_τ/τ < _TAU_REL_ERR_MAX`). These tests
exercise (a) cases where the legacy gate would have rejected but the data
is actually well-identified — verifying that we no longer over-reject, and
(b) cases where τ is truly data-limited — verifying that the new check
fires and flags the antenna as `poorly_identified` rather than silently
accepting or hard-failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tipopac import physics, schema
from tipopac.fit import fit_dataset

import xarray as xr


def _make_synth_ds(
    *,
    za_range: tuple[float, float],
    n_time: int = 30,
    tau0: float = 0.04,
    T0_R: float = 50.0,
    T0_L: float = 48.0,
    freq_Hz: float = 10e9,
    noise_K: float = 0.05,
    n_ant: int = 1,
    rng_seed: int = 42,
) -> xr.Dataset:
    """Synthesize a tipping scan with the requested ZA span and noise level.

    Noise is injected, then exposure_time is set so the radiometer-equation
    σ_Tsys in the dataset matches the injected noise.
    """
    rng = np.random.default_rng(rng_seed)
    T_surf = 280.0
    Twmt = float(physics.k2nt(physics.weighted_mean_atm_T(T_surf), freq_Hz))
    z = np.linspace(*za_range, n_time)

    tsys_R_clean = physics.tsys_model(z, T0_R, tau0, Twmt)
    tsys_L_clean = physics.tsys_model(z, T0_L, tau0, Twmt)
    tsys_R = tsys_R_clean + rng.normal(0.0, noise_K, n_time)
    tsys_L = tsys_L_clean + rng.normal(0.0, noise_K, n_time)

    tcal = 5.0
    bandwidth_Hz = 2e9
    Tsys_typ = float(np.mean((tsys_R_clean + tsys_L_clean) / 2.0))
    sigma_eff = max(float(noise_K), 0.01)
    expo_s = 2.0 * Tsys_typ**4 / (tcal**2 * sigma_eff**2 * bandwidth_Hz)

    n_scan = 1
    n_spw = 1
    switched_diff = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    switched_sum = np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    for i_a in range(n_ant):
        switched_sum[0, i_a, 0, 0, :] = (2.0 * tsys_R / tcal).astype(np.float32)
        switched_sum[0, i_a, 0, 1, :] = (2.0 * tsys_L / tcal).astype(np.float32)

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
                np.full((n_ant, n_spw, 2), tcal, dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((n_scan, n_time), T_surf, dtype=np.float32),
            ),
            "weather_P": (
                ("scan", "time"),
                np.full((n_scan, n_time), 85000.0, dtype=np.float32),
            ),
            "weather_RH": (
                ("scan", "time"),
                np.full((n_scan, n_time), 0.3, dtype=np.float32),
            ),
            "exposure_time": (
                ("scan", "time"),
                np.full((n_scan, n_time), expo_s, dtype=np.float32),
            ),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=bool),
            ),
        },
        coords={
            "scan": np.array([1], dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.array([0], dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), np.array([freq_Hz], dtype=np.float64)),
            "bandwidth": (("spw",), np.array([bandwidth_Hz], dtype=np.float64)),
            "antenna_position": (
                ("antenna", "xyz"),
                np.zeros((n_ant, 3), dtype=np.float64),
            ),
            "scan_time_start": (("scan",), np.array([0.0])),
            "scan_time_end": (("scan",), np.array([90.0])),
            "time_utc": (
                ("scan", "time"),
                np.arange(n_time, dtype=np.float64)[np.newaxis, :],
            ),
        },
    )


@pytest.mark.parametrize(
    "za_range,label",
    [
        ((70.0, 75.0), "dz_5deg_near_horizon"),
        ((30.0, 35.0), "dz_5deg_mid"),
        ((40.0, 60.0), "dz_20deg_mid"),
    ],
)
def test_legacy_low_dz_still_recovers_at_vla_snr(
    za_range: tuple[float, float], label: str
) -> None:
    """Low-dz scans that v2.6 would have rejected (dz<10°) are NOT rejected.

    At realistic VLA SNR (per-sample σ ≈ 0.05 K from radiometer equation),
    a 5° ZA span still constrains τ well — σ_τ/τ ≪ 0.5. The legacy 10°
    geometric gate was conservative; the post-refactor identifiability
    check does not over-reject.
    """
    ds = _make_synth_ds(za_range=za_range, noise_K=0.05)
    fit_dataset(ds, mode="tau_per_antenna")
    reason = str(ds["fit_reason"].values[0, 0, 0])
    assert bool(ds["fit_success"].values[0, 0, 0]), (
        f"{label}: fit_success should be True, reason={reason}"
    )


def test_flat_za_triggers_poorly_identified() -> None:
    """Truly zero airmass leverage → poorly_identified, not silent success.

    With dz=0 the model can pick any (T0, τ) consistent with a single
    Tsys value; σ_τ/τ blows up. Replaces the legacy `dz > 10°` hard gate.
    """
    ds = _make_synth_ds(za_range=(45.0, 45.0), noise_K=0.05)
    fit_dataset(ds, mode="tau_per_antenna")
    reason = str(ds["fit_reason"].values[0, 0, 0])
    assert not bool(ds["fit_success"].values[0, 0, 0]), (
        f"flat-ZA: fit_success should be False, got reason={reason}"
    )
    assert reason == "poorly_identified", (
        f"flat-ZA: expected poorly_identified, got {reason}"
    )


def test_very_high_noise_triggers_poorly_identified() -> None:
    """When σ_Tsys is large enough that σ_τ/τ > 0.5, the check fires.

    Reduces the signal/noise on the airmass-curvature lever until the
    covariance-based identifiability check exceeds threshold.
    """
    # Per-sample σ ~ 30 K with τ=0.02, dz=10° → σ_τ/τ ≳ 0.5
    ds = _make_synth_ds(
        za_range=(40.0, 50.0),
        tau0=0.02,
        noise_K=30.0,
        rng_seed=11,
    )
    fit_dataset(ds, mode="tau_per_antenna")
    reason = str(ds["fit_reason"].values[0, 0, 0])
    assert reason in ("poorly_identified", "high_chi2"), (
        f"high-noise: expected poorly_identified or high_chi2, got {reason}"
    )
    assert not bool(ds["fit_success"].values[0, 0, 0])


def test_healthy_leverage_succeeds() -> None:
    """Sanity: a well-conditioned tipping scan succeeds with ok reason."""
    ds = _make_synth_ds(za_range=(30.0, 70.0), noise_K=0.05)
    fit_dataset(ds, mode="tau_per_antenna")
    assert bool(ds["fit_success"].values[0, 0, 0]), str(
        ds["fit_reason"].values[0, 0, 0]
    )
    assert ds["fit_reason"].values[0, 0, 0] == "ok"
