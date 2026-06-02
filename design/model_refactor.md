# Tipopac robustness redesign — staged plan

## Context

The current rewrite (`src/tipopac/`) reproduces the v2.6 CASA `tipopac`
numerically but inherits its optimization fragility: ad-hoc parameter clamps,
a 6-gate per-antenna QA cascade, a 2-pass 2σ clip-and-refit, and a 3-pass
bound-escalation ladder in `tcal_solve`. The known `(T0, c, τ)` near-degenerate
ridge is the architectural root cause of `tcal_solve`'s sensitivity to
optimizer trajectory.

This plan replaces those heuristics with a principled approach in two stages:

1. **Stage 1 (solver)** — proper per-sample noise model, robust loss, single
   physical bound set, drop QA cascade and bound escalation.
2. **Stage 2 (physics)** — eliminate per-spw scalar `τ` as a free parameter.
   Use am to forward-model `τ(ν)` from a single per-antenna `PWV`, evaluated
   from a pre-computed PWV-grid. The atmospheric DOF drops from `N_spw`
   independent scalars to one scalar per antenna, exploiting the spectral
   shape of `τ(ν)` (especially across the 22 GHz H₂O line wings at K/Ka/Q
   band) to break the `(T0, τ)` degeneracy. The `(T0, c)` degeneracy that
   remains is broken by airmass curvature.

Deviation from v2.6 is acceptable; the new ground truth is am-generated
synthetic tipping curves with injected radiometer noise.

Out of scope this round: refraction / curved-Earth airmass, T0 physical
decomposition (CMB + Tspill + Trx), per-channel opacity within a spw, fitting
species beyond H₂O. MCMC / uncertainty quantification is deferred to a later
PR; default solver is MAP via Levenberg–Marquardt.

## Stage 1 — Solver-side cleanup

**Goal:** clean the heuristic surface area while keeping the current
parameterization (per-spw scalar `τ_zenith`). Validate against existing v2.6
comparison + a new synthetic fixture.

### 1.1 Per-sample noise model

- Add `sigma_Tsys[scan, antenna, time, spw, pol]` to the canonical
  xarray.Dataset (`design/initial_design.md` §5).
- Tipopac forms `Tsys = (switched_sum / 2) · Tcal_ref / switched_diff` (see
  `_compute_tsys` at fit.py:185–199). Error propagation through this ratio,
  with `σ_diff ≈ √2 · Tsys / √(Δν·τ_int)` (Dicke for the switched difference)
  and `diff ≈ Tcal_ref` in steady state, gives:

  ```
  σ_Tsys ≈ √2 · Tsys² / (Tcal_ref · √(Δν · τ_int))
  ```

  This is what the reader writes. The `Tsys / Tcal_ref` amplification factor
  is ~10–60× for VLA bands where Tsys ≫ Tcal — dropping it would underpredict
  σ by that factor and break the 4σ residual rejection. `Δν` is the spw
  bandwidth (sum of `CHAN_WIDTH` over unflagged channels); `τ_int` is
  `INTERVAL` (MS) or the SDM equivalent.
- Write the derivation as a comment block at the σ_Tsys site in each reader
  so future reviewers can verify.
- Both readers must produce identical schemas (SDM↔MS mapping in
  `design/initial_design.md` §4 gets a new row).

### 1.2 Robust loss in `least_squares`

- Pass `loss="soft_l1"`, `f_scale = 3.0` (in units of σ_Tsys) to all three
  `least_squares` calls in `fit.py`.
- **Remove** the 2-pass clip-and-refit logic at `fit.py:402–436`. The robust
  loss subsumes it in one pass.
- Update `_tau_err_from_jac()` (`fit.py:331–349`) to use the IRLS-equivalent
  weighted Jacobian when computing covariance.

### 1.3 Drop bound escalation and QA cascade

- `tcal_solve`: replace the 3-pass escalation ladder (`fit.py:552–595`,
  `_TCAL_LO`/`_TCAL_HI` constants) with a single physical bound set:
  `c ∈ [0.5, 2.0]`, `τ ∈ [0, 1.0]`. The bound widths come from receiver-system
  priors (Tcal is a 20–30% diode reference; ±50% is well outside any plausible
  drift), not from trial-and-error.
- Drop the 6-gate QA cascade in `_screen_antenna()` (`fit.py:~398–460`).
  Replace with two complementary checks:
  - **Noise-side**: drop samples whose `χ² = ((Tsys_meas −
    Tsys_model)/σ_Tsys)² > 16` (4σ), iteratively until stable. Per-antenna
    acceptance: ≥ N samples remain AND reduced χ² < 5.
  - **Identifiability-side**: post-fit, check `σ_τ / τ > 0.5`. If true, the
    scan has insufficient airmass leverage on τ regardless of how clean its
    residuals are. Flag as `poorly_identified` (a new `fit_reason` value)
    rather than `fit_failed`; the user / downstream caller decides whether
    to trust the value. This replaces the geometric `dz > 10°` and
    `min(z) > 30°` gates with a derived signal the fit itself provides.
- Remove `_STD_RESI`, the freq-dep `stdTsys` bins (5/15/20 K), `dz > 10°`,
  `min(z) > 30°`, `mean(Tsys) < 300 K` as hard gates. They may resurface
  as warnings (not gates) in QA logging.

### 1.4 Improved initialization

- For all three modes, compute `τ_init(ν_spw)` by evaluating am once with the
  open-meteo forecast PWV (or AFGL climatology if forecast fails). This
  replaces the hard-coded `τ=0.2` (`fit.py:399`) and the median-of-pre-screens
  introduced in commit `a5e8adb`.
- `T0_init` derived from the y-intercept of a linear fit of `Tsys` vs.
  `airmass` per (antenna, spw, pol) — already a one-liner with `np.polyfit`.

### 1.5 Acceptance for Stage 1

- Existing v2.6 integration test (`test_v26_parity` or equivalent) must still
  pass with the relaxed §11.3 tolerances. Some scans may shift; that's
  expected and the user has accepted it.
- New synthetic fixture in `tests/synth/` (see §3.1) recovers injected
  `(τ, T0, c)` within reported 1σ for ≥95% of scans.

## Stage 2 — Forward-model atmosphere

**Goal:** make `τ(ν)` a function of a single per-antenna `PWV` parameter
evaluated via a pre-computed am grid.

### 2.1 Pre-computed PWV grid (per scan)

- New module `src/tipopac/atmgrid.py` (or extend `atmosphere.py`).
- For each scan (defined by mid-time), construct the atmospheric profile
  once: `(P(z), T(z), H₂O_vmr(z))` from open-meteo / HRRR / ERA5 + measured
  surface pressure (already in MS WEATHER subtable; SDM equivalent in
  `Weather.xml`).
- Build a 981-point PWV grid `[1.0, 50.0]` mm step `0.05` mm.
- Run am in a `multiprocessing.Pool` (40 workers) over the grid, evaluating
  `τ(ν)` and `T_wmt(ν)` on a dense frequency grid spanning the observed spws
  (~10⁴ points). Pre-compute time: ~1.3 s wall on 40 cores.
- Output: `PwvGrid` object exposing `tau(pwv_mm, freq_Hz)` and
  `twmt(pwv_mm, freq_Hz)` via cubic spline interpolation in PWV and linear
  in freq (or vectorized for batch eval over spws).
- Cache pre-computed grids per scan in the dataset (`pwv_grid` opaque carrier
  variable or attached as an attribute keyed by scan).

### 2.2 New model + parameter vector

For each antenna (default mode `per_antenna`):

```
p = [pwv_mm, T0_R_k=1..K, T0_L_k=1..K]                          # opacity-only
p = [pwv_mm, T0_R_k, T0_L_k, c_R_k, c_L_k]                       # tcal_solve
```

Forward model per measurement `(time t, spw k, pol p)`:

```
airmass(t)        = 1 / cos(z(t))
τ_z(k)            = PwvGrid.tau(pwv_mm, ν_k)
T_wmt(k)          = PwvGrid.twmt(pwv_mm, ν_k)
T_sky(t, k)       = T_wmt(k) · (1 − exp(−τ_z(k) · airmass(t)))
Tsys_model        = T0_{p,k} + T_sky(t, k)                       # opacity-only
                  = (T0_{p,k} + T_sky(t, k)) / c_{p,k}            # tcal_solve
residual          = (Tsys_meas − Tsys_model) / σ_Tsys
```

Jacobian: analytical w.r.t. T0 and c; finite-difference w.r.t. PWV (one extra
spline eval, ~µs). Sparse blocks per antenna are diagonal except for the PWV
column, which couples all spws — well-conditioned by construction.

### 2.3 Antenna-level fit + scan-level aggregation

- Fit each antenna independently (embarrassingly parallel across antennas via
  `multiprocessing.Pool`). Parameter count per antenna is small (~17–33 for
  typical VLA scans with 8 spws); LM converges in ~10–50 iters.
- After all antennas: compute `pwv_scan_median = median(pwv_ant)` and
  `pwv_scan_mad`.
- Flag antennas as PWV outliers if `|pwv_ant − pwv_scan_median| > max(1.0 mm,
  k · pwv_scan_mad)` with `k=3` and the 1 mm floor reflecting VLA A-config
  inter-antenna PWV variability. Outliers retain their fit but get a
  `pwv_outlier` boolean flag — caller decides whether to use per-antenna or
  consensus value.

### 2.4 Mode reorganization (internal, partial API rename)

The three modes become configurations of the same joint model:

| Old name           | New name           | DOFs                                      |
|--------------------|--------------------|-------------------------------------------|
| `tau_per_antenna`  | `per_antenna_pwv`  | PWV per antenna, T0 per (ant, spw, pol)   |
| `global_tau`       | `shared_pwv`       | one PWV, T0 per (ant, spw, pol)           |
| `tcal_solve`       | `tcal_solve`       | PWV per antenna, T0 + c per (ant, spw, pol) |

`per_antenna_pwv` is the default. Old names remain accepted as deprecated
aliases for one release (a simple dict mapping in `api.py`); deprecation
warning emitted.

### 2.5 Schema changes (`design/initial_design.md` §5)

Add data variables:

- `pwv[scan, antenna]` — float, mm. Primary atmospheric output.
- `pwv_err[scan, antenna]` — 1σ from covariance.
- `pwv_outlier[scan, antenna]` — bool. True if antenna deviates from
  `pwv_scan_median` by more than the threshold.
- `pwv_scan_median[scan]` — float, mm. Robust consensus.
- `tau_zenith[scan, antenna, spw]` — retained (derived from PWV via grid
  lookup) so downstream callers and the `TOpac` caltable writer continue to
  work unchanged.

Update `design/initial_design.md` §5 (schema) and §7 (atmosphere — now
describes forward-model and grid, not post-hoc anchor) in the same commit
that lands the code.

### 2.6 Open-meteo / fallback hardening

- Keep AFGL climatology fallback. Promote the silent fallback warning to a
  `pwv_profile_source` per-scan attribute (`"open_meteo"`, `"hrrr"`, `"era5"`,
  `"afgl_midlatitude_summer"`, etc.) so users see in the dataset which scans
  used a degraded profile.
- Add open-meteo retry with backoff (3 attempts, 5/15/45 s); only fall back
  after exhaustion.

## Stage 3 (deferred) — MCMC uncertainty

Out of scope for this plan. Hook left: the LM fit produces a covariance
estimate; an opt-in `uncertainty="mcmc"` flag on `tipopac()` would run emcee
around the MAP for ~5 s/scan on 40 cores. Design note added to
`design/initial_design.md` under "Future work".

## Critical files

- `src/tipopac/fit.py` — gutted and rebuilt around the joint model; both
  Stages.
- `src/tipopac/atmosphere.py` — gains the `PwvGrid` builder. Existing
  `pwv_scaling` anchor logic deleted (it becomes redundant once `τ` is no
  longer free).
- `src/tipopac/schema.py` — new variables per §2.5.
- `src/tipopac/readers/ms.py`, `src/tipopac/readers/sdm.py` — compute
  `sigma_Tsys` from radiometer equation; emit consistent schema.
- `src/tipopac/api.py` — mode aliases for backwards compat; orchestration
  unchanged except calling new fit signature.
- `tests/synth/` (new) — am-driven synthetic-data generator + recovery tests.
- `tests/unit/test_fit.py` — update or replace with synth-based tests.
- `tests/integration/` — relax tolerances on existing v2.6 parity test;
  current tolerances assume close numerical match that no longer applies.
- `design/initial_design.md` — §5, §6.1, §7, §11.3, §13 updated alongside code.

## Verification

1. `tests/synth/test_recovery.py` — generate 100 synthetic scans spanning
   PWV ∈ [2, 30] mm, with realistic per-sample radiometer noise, optional
   injected Tcal errors. Assert recovered `(PWV, T0, c)` within 1σ for ≥95%.
   Coverage must include a low-leverage subset: `dz ≈ 5°`, `min(z) ≈ 70°`.
   Assert these scans are flagged `poorly_identified`, not `fit_failed` or
   silently accepted — this is the regression test for removing the
   geometric QA gates.
2. `tests/synth/test_outlier_antenna.py` — inject one antenna with PWV
   biased by 3 mm; assert `pwv_outlier` flag fires for that antenna only.
3. `tests/synth/test_ridge_diagnostics.py` — run `tcal_solve` on a synthetic
   scan; assert reported `pwv_err` and `c_err` are consistent with the local
   Hessian (sanity check on covariance pipeline).
4. `tests/integration/test_v26_parity.py` — relax tolerances per §11.3
   update; smoke-test only (real MS at `data/tip_test.ms`).
5. Manual run: `uv run python -m tipopac data/tip_test.ms --mode
   per_antenna_pwv` should complete within ~30 s wall on 40 cores for the
   provided test MS, with `pwv_scan_median` values in the 3–8 mm range
   consistent with VLA-Q seasonal PWV statistics.
6. `uv run pytest tests/unit` and `uv run ty check src/tipopac` clean.

## Sequencing

PR 1 (Stage 1.1–1.5): noise model, robust loss, drop heuristics. Lands behind
existing API and modes.

PR 2 (Stage 2): pre-computed PWV grid, joint forward model, antenna-parallel
LM, schema additions, mode renaming with deprecation aliases. Lands after
PR 1 is merged.

`design/initial_design.md` updates in each PR alongside code.
