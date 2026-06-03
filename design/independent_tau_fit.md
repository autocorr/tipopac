# Independent per-spw τ fit + post-hoc PWV anchor

A redirection of the Stage-2 forward-model architecture. Instead of
one joint LM per antenna over `(PWV, T0[spw,pol], …)`, fit each spw
independently for `(τ_z, T0, …)` and then fit a single PWV per
antenna (or globally) against the resulting `τ_z(ν)` samples via the
precomputed am grid.

This is essentially v2.6's architecture with a modernised anchor: per-
spw observational opacity estimation first, atmospheric model second.

## Why revisit this

`design/model_refactor.md` §2 chose to make PWV a free parameter
inside the per-antenna LM to "exploit the spectral shape of τ(ν) to
break the (T0, τ) degeneracy." The Stage-2 fit has since shipped, and
the profiling work in `design/performance_refactor_considerations.md`
exposes three issues with that joint architecture:

1. At full `n_spw=112` the Stage-2 Jacobian is structurally rank-
   deficient (192 of 225 columns are zero on any given scan because
   each scan covers only one 16-spw band), and scipy's TRF still does
   a dense SVD on the full matrix every accepted step. This is the
   dominant cost.
2. The "spectral shape constraint" the joint fit advertises is
   recoverable post-hoc with proper `σ_τ` propagation, modulo a small
   correlation between `τ_z` and `T0` per spw. Airmass curvature
   decouples that correlation: with VLA tipping coverage the residual
   covariance is small (≲10 % efficiency loss vs joint MLE).
3. The `(T0, c, τ)` ridge in `tcal_solve` is broken by cross-antenna
   coupling, not cross-spw coupling. The existing Stage-1 path
   `_fit_global(tcal_mode=True)` (fit.py:719, with sparse Jacobian
   from commit 28b5544) already provides the right level of joint
   inference for this — per-(scan, spw), joint across antennas — at
   45 s wall on the full MS even with the harmful default BLAS
   threading still in play.

The architecture described below splits the problem along the lines
the data and the degeneracies actually justify.

## Architecture

```
Stage A — per-spw observational fit (independent per (scan, ant, spw))
  Model:  Tsys(z, pol) = T0(pol) + T_mean(spw) · (1 − exp(−τ_z · airmass(z)))   [opacity]
                       = (T0(pol) + T_mean(spw) · (...)) / c(pol)               [tcal-solve, see §2.2]
  Params: (T0_R, T0_L, τ_z)            — 3 params, per_antenna_pwv equivalent
  Atmospheric input: T_mean(spw) only  — see §2.1.
  Output: τ_z(scan, ant, spw),  σ_τ(scan, ant, spw),  T0(scan, ant, spw, pol)

Stage B — atmospheric anchor (per antenna, or shared across antennas)
  Model:  τ_model(ν, spw) = τ_grid(PWV, ν_spw)        [from PwvGrid.lookup]
  Cost:   χ²(PWV) = Σ_{scan,spw}  [(τ_z(spw) − τ_model(PWV, ν_spw))² / σ_τ²(spw)]
  Params: PWV per antenna  (shared across that antenna's 7 scans by default)
  Output: PWV(ant), σ_PWV(ant), pwv_outlier per (scan, ant) from residual scan
```

The two stages are separable: Stage A's output `(τ_z, σ_τ)` is the
sufficient statistic for Stage B given the chosen model. Stage A
never sees the am grid; Stage B never re-touches the time-domain
data.

## Detail

### 1. Stage A — per-spw fit

The opacity-mode case maps directly onto the existing legacy
`_fit_tau_per_antenna` (fit.py:678): 3-param LM per (scan, ant, spw).
Stage-1 robustness work (soft_l1, σ_Tsys, residual rejection) carries
over unchanged.

**T_mean atmospheric input.** Two cheap options:

- v2.6-style: `T_mean ≈ 0.95 · T_surface` (surface weather temperature
  from MS WEATHER subtable). Simplest; biases τ by ~1–2 % per K of
  T_mean error.
- Grid-lookup at climatology PWV: `T_mean(ν_spw) =
  PwvGrid.lookup_tmean(pwv_climatology, ν_spw)` for each scan,
  evaluated once before any fit. The PwvGrid is already built (Stage 2
  precompute), so this is a one-line lookup, no extra am call.

Recommend the grid-lookup form because the grid is built anyway. If
the grid build itself is later removed in favour of a leaner Stage A
path, fall back to `0.95 · T_surface`.

**tcal_solve variant.** The (T0, c) per-(ant, spw) ridge is not broken
within one antenna's elevation sweep — global-across-antennas coupling
is required. Use `_fit_global(tcal_mode=True)` at fit.py:719: one fit
per (scan, spw) jointly determining τ_z and per-antenna (T0_R, c_R,
T0_L, c_L). Stage-1 already has sparse-CSR Jacobian here from commit
28b5544 — keep it. 7 × 112 = 784 such fits per MS, Jacobian ~`4590 ×
109`, sparse LSMR wins (the regime where the 28b5544 trick pays).

**Parallelism unit for Stage A.** Either:

- Opacity mode: `(scan, ant, spw)` — **3024 independent fits** on this
  validation MS (27 ants × 7 scans × 16 spws-per-scan-band).
- tcal_solve mode: `(scan, spw)` — **784 fits**.

Both saturate any plausible core count. Workers must export single-
threaded BLAS — see §3 below.

### 2. Stage B — atmospheric anchor

`atmosphere.anchor(...)` (atmosphere.py:305) already implements the
weighted least-squares anchor against `τ_obs`, `σ_τ` over a 1-D
`scipy.optimize.minimize_scalar(bounded)` on a single scaling
parameter. Adapt to:

- Variable parameter: PWV in mm directly (not a multiplicative
  scaling) via `PwvGrid.lookup` rather than a per-call am run. This is
  already what `_fit_dataset_stage2` does, just exposed as an
  independent stage.
- Aggregation level: **per antenna across that antenna's 7 scans** by
  default. Each antenna's PWV is constrained by `Σ_scan Σ_spw 1 ≈ 112`
  data points (one τ_z per spw across all bands). Single 1-D
  minimisation per antenna, ~milliseconds. Sub-cases:
  - `per_antenna_pwv` semantics: PWV per antenna.
  - `shared_pwv` semantics: one PWV across all antennas — single 1-D
    fit using all 3024 τ_z points.
  - Per-(scan, ant) PWV for diagnostics (cheap, no production use).
- Outlier handling: residual-scan after the PWV fit. Any (scan, ant,
  spw) τ_z that lies > kσ from the model is downweighted or flagged.
  Replaces the current `pwv_outlier` flag with a more direct
  observable.

**σ_PWV propagation.** From the per-spw `σ_τ` and the local sensitivity
`∂τ_grid/∂PWV` at the fitted PWV (already exposed by
`PwvGrid.lookup_with_grad`):

```
σ_PWV² = 1 / Σ_{scan,spw}  (∂τ/∂PWV)² / σ_τ²
```

Standard Cramér–Rao for a 1-D nonlinear fit. No SVD, no Hessian
inversion.

### 3. Schema changes

Stage A outputs map onto existing variables; Stage B's PWV outputs
already exist in the schema from the Stage-2 work:

| Variable | Source | Notes |
|---|---|---|
| `tau_zenith[scan, ant, spw]` | Stage A | direct observational, not derived from PWV |
| `tau_err[scan, ant, spw]` | Stage A | σ_τ from per-spw Jacobian |
| `T0[scan, ant, spw, pol]` | Stage A | per-spw |
| `tcal_fit[scan, ant, spw, pol]` | Stage A (tcal_solve) | from per-spw `_fit_global` |
| `pwv[ant]` *or* `pwv[scan, ant]` | Stage B | semantic shift — see below |
| `pwv_err[ant]` *or* `pwv_err[scan, ant]` | Stage B | σ from Cramér–Rao |
| `fit_success[scan, ant, spw]` | Stage A | per-spw, drops the (scan,ant)-level all-or-nothing failure modes that Stage 2 conflates |
| `fit_reason[scan, ant, spw]` | Stage A | per-spw |

**Semantic shift in `pwv`.** Today `pwv[scan, ant]` records the joint-
fit PWV per (scan, ant). Under this architecture the natural object
is `pwv[ant]` — one PWV per antenna shared across all scans, since
that is what was actually fit. Per-scan PWV becomes a derived
diagnostic (rerun the anchor with per-scan grouping if needed). This
is consistent with the user-stated physical justification (4-min
inter-scan spacing → atmospheric stability) and matches the
parallelism unit of Stage B.

If `pwv[scan, ant]` must be retained for downstream callers,
broadcast the per-antenna value across the scan axis, the same way
`tau_zenith` is broadcast across the antenna axis in `global_tau` and
`tcal_solve` modes (per §5 of `design/initial_design.md`,
"Representation choices").

### 4. Public API

Add a new mode `independent_tau`. Routing in `fit_dataset(...)`:

```
mode="independent_tau":
    Stage A: per-(scan, ant, spw) opacity fit using _fit_tau_per_antenna
    Stage B: per-antenna PWV anchor via atmosphere.anchor (adapted)

mode="independent_tau_solve":
    Stage A: per-(scan, spw) global fit using _fit_global(tcal_mode=True)
    Stage B: per-antenna PWV anchor (same as above)
```

`per_antenna_pwv`, `shared_pwv`, `tcal_solve` stay available but
become aliases routing to the joint Stage-2 path, marked experimental
pending validation against this new mode.

## Performance considerations carried over

(Subset of `design/performance_refactor_considerations.md` that
applies to this architecture. Items irrelevant here — the §2
structural `n_params` bug, the (d) band-slicing fix, the (f) joint-LM
discussion — are omitted because this architecture doesn't have those
shapes.)

### BLAS multithreading is harmful at this matrix size

scipy-bundled OpenBLAS with no env limit picks ~10 threads per call
on the per-antenna fit matrices. Measured: BLAS=1 is 20× faster wall
than the default for `_fit_per_antenna_pwv` at n_spw=16. Stage A's
per-spw fits use *even smaller* matrices (~3-param, ~170 rows) — they
fall further into the regime where BLAS multithreading is pure
overhead.

**Action.** Set `OPENBLAS_NUM_THREADS=1` (and `MKL_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`) before numpy import — both in the application
entry point and exported to any worker subprocesses. This is also a
hard prerequisite for the process-level parallelism in §5 below: 40
workers × 10 BLAS threads each on 40 cores would be a 10× over-
subscription.

### Sparse Jacobian is the right call for the Stage A tcal_solve global fit, the wrong call for Stage A opacity-mode and Stage B

scipy's TRF picks `tr_solver='exact'` (direct SVD) for dense Jacobian
input and `tr_solver='lsmr'` (iterative) for sparse. The break-even
matrix size is somewhere between `(1360 × 33)` (small — dense SVD
wins) and `(4590 × 109)` (large — sparse LSMR wins). Concretely:

- **Stage A opacity per-(scan, ant, spw) fit**: matrix `(170 × 3)`.
  Dense. Do not wrap in CSR — measured 2.6–4.9× regression on the
  larger per-antenna case.
- **Stage A tcal global per-(scan, spw) fit**: matrix `(4590 × 109)`.
  Sparse-CSR Jacobian (already in place from commit 28b5544). Keep.
- **Stage B PWV anchor**: 1-D scalar minimisation. No Jacobian.

### Process-level parallelism for Stage A

Stage A is the entire fit workload after the matrix-size fix. Each
unit is a `_fit_tau_per_antenna` (opacity) or `_fit_global` (tcal)
call — both are already closure-free at module scope and pickle-able
as-is. Use `multiprocessing.Pool` with `OPENBLAS_NUM_THREADS=1`
exported per worker. The closure-lift problem that complicated (b)
for the joint Stage-2 fit doesn't apply here.

Expected wall on 40 cores: dominated by serial parts (MS read 2–3 s,
grid build 20–30 s, dataset assembly < 1 s). The fit stage itself
should be ≪ 5 s for either mode given the per-fit sizes.

### What's left on the serial-overhead floor

- **MS read** (`MSReader.from_path(...).read()`): ~2.3 s on the
  validation MS. Single-threaded, casatools-bound.
- **Grid build** (`build_pwv_grid` per scan): ~20–30 s for 7 scans
  with the default am pool internally parallelising. Could be
  parallelised across scans on top of internal workers, but each
  scan's am build already touches multiple cores so the win is
  modest. Stage B only needs grid lookups, not new am runs.
- **Dataset assembly**: < 1 s.

So `fit_dataset(mode='independent_tau')` on the full validation MS
should land near `read + grids + ~1 s fit + assembly ≈ 25–35 s`,
practically all serial overhead. Compare with the joint Stage-2
extrapolation of 5–30 min single-threaded for the same MS.

## Validation strategy

1. **Numerical parity with v2.6 on the legacy fixtures.** Stage A
   opacity is mechanically the v2.6 architecture; the existing v2.6
   regression tests at `design/initial_design.md` §11.3 (opacity
   within `max(0.005, 0.05·τ_v26)`, Tcal within 1 %) should still
   pass.
2. **σ_PWV self-consistency check.** Synthesise tipping curves with
   known PWV + injected noise via `am`. Confirm that the Stage B σ_PWV
   matches the standard deviation across realisations within
   √(2/(n−1)) — the standard test for proper error propagation.
3. **Comparison against Stage-2 joint fit on the same synthetic.** On
   data without `(T0, c)` ridge pathologies, the two modes should
   agree on PWV within their respective σ. The expected 5–10 %
   efficiency gap from §2 above should show up as a slightly larger
   Stage B σ_PWV than Stage-2's marginal σ_PWV, but both should be
   unbiased.
4. **Real-data sanity on tip_test.ms.** Run both modes; compare
   `tau_zenith(spw)` and `pwv` distributions across antennas. Stage A
   per-spw τ should look smooth in frequency (the user-facing
   diagnostic that's hardest to inspect in the Stage-2 joint output).

## Open empirical questions

- Magnitude of the `(τ, T0)` correlation per spw on real VLA tipping
  geometry. The 5–10 % efficiency claim is a rule-of-thumb; the actual
  number depends on elevation range, σ_Tsys, and band. Measure on
  synthetic before claiming parity.
- T_mean sensitivity in production: does the grid-lookup-at-
  climatology-PWV form yield τ within tolerance of an am-anchored
  T_mean (which would require Stage B to iterate)? Likely yes by a
  wide margin, but worth one synthetic test.
- Whether `pwv[scan, ant]` retention as a broadcast of `pwv[ant]` is
  acceptable to downstream callers, or whether per-scan PWV must be
  re-derived per-scan from Stage B. The cost is trivial — runs Stage B
  7× per ant — but the API semantics need a decision.
