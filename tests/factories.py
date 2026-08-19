"""Synthetic dataset factories shared across the fast tests."""

from __future__ import annotations

import numpy as np
import xarray as xr

from tipopac import physics, schema
from tipopac.atmgrid import PwvGrid
from tipopac.timeutils import assign_groups


def make_pwv_grid(pwv_unscaled_mm: float = 5.0) -> PwvGrid:
    """Analytic grid: τ ∝ PWV · (1 + 0.01·ν), Tb = 270·(1 − exp(−τ)).

    Linear-in-PWV so the bilinear interpolant is exact, which lets the
    Cramér–Rao σ_PWV self-consistency check pass tightly.
    """
    pwv = np.linspace(1.0, 10.0, 19)  # 0.5 mm step matches default
    freq = np.linspace(10e9, 30e9, 41)  # 0.5 GHz step
    tau = pwv[:, None] * (1.0 + 0.01 * freq[None, :] / 1e9) * 0.01
    tb = 270.0 * (1.0 - np.exp(-tau))
    return PwvGrid(
        pwv_mm=pwv,
        freq_Hz=freq,
        tau_z=tau,
        tb_z=tb,
        pwv_unscaled_mm=pwv_unscaled_mm,
    )


def make_tipping_dataset(
    tau0: float = 0.04,
    freq_Hz: float = 10e9,
    n_time: int = 30,
    noise_K: float = 0.3,
    *,
    T0_R: float = 50.0,
    T0_L: float = 48.0,
    rng: np.random.Generator | None = None,
    seed: int = 42,
    flat_za: bool = False,
    za_range: tuple[float, float] = (35.0, 65.0),
    n_scan: int = 1,
    n_ant: int = 1,
    n_spw: int = 1,
    with_cmb: bool = True,
    per_cell_noise: bool = False,
    attrs: dict[str, str] | None = None,
) -> xr.Dataset:
    """Pre-fit dataset carrying a synthetic tipping curve.

    ``switched_diff = 1.0`` and ``tcal_ref = 5.0`` K, so
    Tsys = switched_sum / 2 * tcal_ref. *with_cmb* generates against the
    attenuated CMB the fitter models, so a round trip tests parameter
    recovery rather than a model mismatch; omitting it makes the recovered
    T0 low by ~Tcmb. *per_cell_noise* draws an independent realisation for
    every (scan, antenna, spw) instead of sharing one across all cells.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    T_surf = 280.0  # K
    Twmt = float(physics.k2nt(physics.mean_radiating_T(T_surf), freq_Hz))
    Tcmb = float(physics.k2nt(physics.T_CMB, freq_Hz)) if with_cmb else 0.0

    z = np.linspace(*za_range, n_time) if not flat_za else np.full(n_time, za_range[0])

    tsys_R_clean = physics.tsys_model(z, T0_R, tau0, Twmt, Tcmb)
    tsys_L_clean = physics.tsys_model(z, T0_L, tau0, Twmt, Tcmb)

    tcal = 5.0
    # Pick exposure_time so radiometer σ_Tsys ≈ max(noise_K, 0.01 K) at the
    # scan-mean Tsys. This keeps synthetic test data consistent with the
    # reader-derived σ that the fit consumes:
    #   σ = 2 · Tsys² / (Tcal · √(Δν·τ_int))  →  τ_int = 4·Tsys⁴ / (Tcal²·σ²·Δν)
    bandwidth_Hz = 2e9
    Tsys_typ = float(np.mean((tsys_R_clean + tsys_L_clean) / 2.0))
    sigma_eff = max(float(noise_K), 0.01)
    expo_s = float(4.0 * Tsys_typ**4 / (tcal**2 * sigma_eff**2 * bandwidth_Hz))

    switched_diff = np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    switched_sum = np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32)
    if not per_cell_noise:
        tsys_R = tsys_R_clean + rng.normal(0.0, noise_K, n_time)
        tsys_L = tsys_L_clean + rng.normal(0.0, noise_K, n_time)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            for i_w in range(n_spw):
                if per_cell_noise:
                    tsys_R = tsys_R_clean + rng.normal(0.0, noise_K, n_time)
                    tsys_L = tsys_L_clean + rng.normal(0.0, noise_K, n_time)
                switched_sum[i_sc, i_a, i_w, 0, :] = (2.0 * tsys_R / tcal).astype(
                    np.float32
                )
                switched_sum[i_sc, i_a, i_w, 1, :] = (2.0 * tsys_L / tcal).astype(
                    np.float32
                )

    zenith_arr = np.zeros((n_scan, n_ant, n_time), dtype=np.float32)
    for i_sc in range(n_scan):
        for i_a in range(n_ant):
            zenith_arr[i_sc, i_a, :] = z.astype(np.float32)

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
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
            "spw": np.arange(n_spw, dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), np.full(n_spw, freq_Hz, dtype=np.float64)),
            "bandwidth": (("spw",), np.full(n_spw, bandwidth_Hz, dtype=np.float64)),
            "band": (("spw",), np.array(["K"] * n_spw, dtype="U4")),
            "antenna_position": (
                ("antenna", "xyz"),
                np.zeros((n_ant, 3), dtype=np.float64),
            ),
            "scan_time_start": (
                ("scan",),
                np.arange(n_scan, dtype=np.float64) * 120.0,
            ),
            "scan_time_end": (
                ("scan",),
                np.arange(n_scan, dtype=np.float64) * 120.0 + 90.0,
            ),
            "time_utc": (
                ("scan", "time"),
                np.tile(np.arange(n_time, dtype=np.float64), (n_scan, 1))
                + np.arange(n_scan, dtype=np.float64)[:, None] * 120.0,
            ),
        },
        attrs=dict(attrs or {}),
    )


def make_fitted_dataset(
    *,
    n_scan: int = 1,
    n_ant: int = 1,
    n_spw: int = 1,
    success: bool = True,
    with_am: bool = False,
    with_atm: bool = False,
    with_stage_c: bool = False,
    with_pwv: bool = False,
    freq_Hz: float = 22.2e9,
    mode: str = "independent_tau_solve",
    tau_shared: bool = False,
) -> xr.Dataset:
    """Minimal post-fit dataset ready for PlotData.

    ZA values span 35-80 deg; Tsys is synthetic but positive. When
    *with_am* is True, ``am_freq_grid`` and ``am_tau`` are populated so
    the am-overlay path runs. ``mode`` sets ``ds.attrs["mode"]`` —
    save_all dispatches Tcal-fit / c plots from it. *with_stage_c* adds
    ``sigma_tcal``, the Stage-C output those plots also key on;
    *with_pwv* adds the Stage-B anchor vars and the object-dtype
    ``pwv_profile_source``.
    """
    n_time = 5

    za = np.linspace(35.0, 80.0, n_time, dtype=np.float32)
    za_arr = np.broadcast_to(za, (n_scan, n_ant, n_time)).copy()

    tsys_val = 80.0
    tsys = np.full((n_scan, n_ant, n_spw, 2, n_time), tsys_val, dtype=np.float32)

    tau0 = 0.05
    tau_zenith = np.full((n_scan, n_ant, n_spw), tau0, dtype=np.float32)
    if not tau_shared:
        # Antenna offsets summing to zero, so the weighted mean stays tau0.
        offset = 1e-3 * (np.arange(n_ant, dtype=np.float32) - (n_ant - 1) / 2)
        tau_zenith += offset[None, :, None]
    tau_err = np.full((n_scan, n_ant, n_spw), 0.002, dtype=np.float32)
    T0 = np.full((n_scan, n_ant, n_spw, 2), 50.0, dtype=np.float32)
    tcal_ref_val = 5.0
    tcal_ref = np.full((n_ant, n_spw, 2), tcal_ref_val, dtype=np.float32)
    tcal_fit = np.full((n_scan, n_ant, n_spw, 2), tcal_ref_val, dtype=np.float32)
    fit_success_arr = np.full((n_scan, n_ant, n_spw), success, dtype=bool)
    fit_reason = np.full(
        (n_scan, n_ant, n_spw), "ok" if success else "dz_too_small", dtype=object
    )

    freqs = np.linspace(freq_Hz, freq_Hz * 1.05, n_spw, dtype=np.float64)

    data_vars: dict = {
        "switched_diff": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.ones((n_scan, n_ant, n_spw, 2, n_time), dtype=np.float32),
        ),
        "switched_sum": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.full((n_scan, n_ant, n_spw, 2, n_time), 2.0, dtype=np.float32),
        ),
        "zenith_angle": (("scan", "antenna", "time"), za_arr),
        "tcal_ref": (("antenna", "spw", "polarization"), tcal_ref),
        "weather_T": (
            ("scan", "time"),
            np.full((n_scan, n_time), 280.0, dtype=np.float32),
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
            np.full((n_scan, n_time), 1.0, dtype=np.float32),
        ),
        "flag": (
            ("scan", "antenna", "spw", "polarization", "time"),
            np.zeros((n_scan, n_ant, n_spw, 2, n_time), dtype=bool),
        ),
        "Tsys": (("scan", "antenna", "spw", "polarization", "time"), tsys),
        "tau_zenith": (("scan", "antenna", "spw"), tau_zenith),
        "tau_err": (("scan", "antenna", "spw"), tau_err),
        "T0": (("scan", "antenna", "spw", "polarization"), T0),
        "tcal_fit": (("scan", "antenna", "spw", "polarization"), tcal_fit),
        "Twmt": (
            ("scan", "spw"),
            np.full((n_scan, n_spw), 270.0, dtype=np.float32),
        ),
        "fit_success": (("scan", "antenna", "spw"), fit_success_arr),
        "fit_reason": (("scan", "antenna", "spw"), fit_reason),
    }

    if with_stage_c:
        data_vars["sigma_tcal"] = (
            ("scan", "antenna", "spw", "polarization"),
            np.full((n_scan, n_ant, n_spw, 2), 0.05, dtype=np.float32),
        )

    if with_am:
        am_freq_grid = np.linspace(
            freqs.min() * 0.95, freqs.max() * 1.05, 50, dtype=np.float64
        )
        data_vars["am_freq_grid"] = (("frequency_dense",), am_freq_grid)
        data_vars["am_tau"] = (
            ("group", "frequency_dense"),
            np.full((1, am_freq_grid.size), tau0, dtype=np.float64),
        )

    if with_pwv:
        data_vars["pwv"] = (
            ("group", "antenna"),
            np.full((1, n_ant), 6.0, dtype=np.float32),
        )
        data_vars["pwv_err"] = (
            ("group", "antenna"),
            np.full((1, n_ant), 0.2, dtype=np.float32),
        )
        data_vars["pwv_profile_source"] = (
            ("scan",),
            np.full(n_scan, "open_meteo", dtype=object),
        )
        data_vars["pwv_model"] = (
            ("scan",),
            np.full(n_scan, 5.8, dtype=np.float32),
        )

    if with_atm:
        # 10-level synthetic profile, 850 → 10 hPa. Stored in Pa (schema §4).
        atm_p_hPa = np.array(
            [850, 700, 500, 300, 200, 100, 50, 30, 20, 10], dtype=np.float64
        )
        atm_p_Pa = atm_p_hPa * 100.0
        atm_T = np.linspace(280.0, 210.0, atm_p_hPa.size, dtype=np.float32)
        atm_vmr = np.logspace(-3, -6, atm_p_hPa.size).astype(np.float32)
        data_vars["atm_pressure"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_p_Pa, (n_scan, atm_p_hPa.size)).copy(),
        )
        data_vars["atm_temperature"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_T, (n_scan, atm_p_hPa.size)).copy(),
        )
        data_vars["atm_h2o_vmr"] = (
            ("scan", "atm_level"),
            np.broadcast_to(atm_vmr, (n_scan, atm_p_hPa.size)).copy(),
        )

    coords = {
        "scan": np.arange(1, n_scan + 1, dtype=np.intp),
        "antenna": [f"ea{i + 1:02d}" for i in range(n_ant)],
        "spw": np.arange(n_spw, dtype=np.intp),
        "polarization": list(schema.POL_VALUES),
        "xyz": ["X", "Y", "Z"],
        "frequency": (("spw",), freqs),
        "bandwidth": (("spw",), np.full(n_spw, 2e9, dtype=np.float64)),
        "band": (("spw",), np.full(n_spw, "K", dtype="U4")),
        "antenna_position": (
            ("antenna", "xyz"),
            np.zeros((n_ant, 3), dtype=np.float64),
        ),
        "scan_time_start": (
            ("scan",),
            np.linspace(
                5131296000.0, 5131296000.0 + 120.0 * n_scan, n_scan, dtype=np.float64
            ),
        ),
        "scan_time_end": (
            ("scan",),
            np.linspace(
                5131296090.0, 5131296090.0 + 120.0 * n_scan, n_scan, dtype=np.float64
            ),
        ),
        "time_utc": (
            ("scan", "time"),
            np.tile(np.linspace(5131296000.0, 5131296090.0, n_time), (n_scan, 1)),
        ),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.attrs["mode"] = mode
    # Stage-B group artifacts. Scans are 120 s apart, so the 1 h default
    # window puts them all in group 0.
    ds.coords["group"] = np.array([0], dtype=np.int32)
    ds.coords["scan_group"] = (
        ("scan",),
        assign_groups(ds.coords["scan_time_start"].values, 3600.0),
    )
    ds.coords["group_time_start"] = (
        ("group",),
        np.array([ds.coords["scan_time_start"].values.min()]),
    )
    ds.coords["group_time_end"] = (
        ("group",),
        np.array([ds.coords["scan_time_end"].values.max()]),
    )
    return ds
