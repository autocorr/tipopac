# Performance refactor considerations

Findings from profiling the Stage-2 forward-model fit on `data/tip_test.ms`
(27 ant × 112 spw × 7 scan × 2 pol × 85 time). All measurements taken on
the working tree of `feat/model-refactor` at the point described, on a
40-core Xeon Gold 6242R. The full-scale baseline was not run to
completion — extrapolation caveats below.

## 1. BLAS multithreading is actively harmful on this workload

numpy is built against scipy's bundled OpenBLAS (`MAX_THREADS=64`,
`NO_AFFINITY=1`, `DYNAMIC_ARCH`). With no env limit, OpenBLAS chooses
~10 threads per call on the per-antenna fit's Jacobian / SVD shapes
(~`2720 × 33` at n_spw=16). Sweep on the same `_fit_per_antenna_pwv`
problem:

| `OPENBLAS_NUM_THREADS` | wall | CPU | CPU/wall |
|---|---|---|---|
| 1 | **0.633 s** | 0.61 s | 0.97 |
| 2 | 0.671 s | 2.00 s | 2.99 |
| 4 | 1.070 s | 4.26 s | 3.98 |
| 8 | 0.891 s | 7.10 s | 7.97 |
| (unset → ~10) | 0.969 s | 9.80 s | 10.11 |

Single-threaded BLAS is **fastest wall AND uses ~16× less CPU**. The
LM matrices are too small for parallel SVD/GEMM to amortise OpenBLAS's
synchronisation overhead. At the 27 ant × 7 scan × 16 spw scale, BLAS=1
is **20.5×** faster than the default (1.98 s vs 40.7 s for per-antenna
mode).

**Implication.** Earlier "baseline" numbers quoted in profiling reports
(commit 28b5544 message, the bench_fit.py output, and the first
benchmark logs in this branch) were measured with default BLAS and so
were paying this contention cost. They are not representative of the
true single-thread cost.

**Action.** Set `OPENBLAS_NUM_THREADS=1` (also `MKL_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`) before invoking `fit_dataset`, or pin it at module
import via `os.environ` before numpy is imported. This is also a
prerequisite for any process-level parallelism (each worker process
needs single-thread BLAS to avoid `n_workers × n_blas_threads`
oversubscription on the same cores).

## 2. Structural bug: `n_params` sized for all spws, but each scan has data in only one band

The MS layout for VLA tipping: 7 scans, ~4 min apart, each scan covers
a disjoint 16-spw band block (K → Ka → Ka → X → Ku → Q → Q for this
MS). Inside `_fit_per_antenna_pwv`, however:

```python
n_spw, n_pol, n_time = tsys_arr.shape  # (112, 2, 85) at full scale
n_params = 1 + n_pol * n_spw           # = 225  (opacity)
                                       # = 449  (tcal_solve)
```

For any given scan only ~16 of those 112 spws have real data; the
other 96 are entirely NaN-flagged. The Jacobian therefore has ≥192
columns that are *structurally zero* — no row anywhere will populate
them. scipy's TRF does not detect this; it does a dense SVD on the
full `(m × 225)` Jacobian every accepted step.

This was masked at small `n_spw` in the benchmarks because slicing
`spw=slice(0, k)` happened to align with the first scan's band. At
`n_spw ≥ 16` it started to bite: BLAS=1 scaling for `per_antenna_pwv`:

| n_spw | wall (s) | fits succ. | per-fit (ms) |
|---|---|---|---|
| 8 | 0.77 | 216 | 3.6 |
| 16 | 1.98 | 432 | 4.6 |
| 32 | 41.54 | 1312 | 31.7 |

The 16→32 per-fit jump of 6.9× (against ~4× from O(m·n²) on real-rank
work) is the SVD walking through more zero columns as `n_spw` grows.
At full `n_spw=112`, extrapolation suggests a single-thread baseline
in the 5–30 min range for `per_antenna_pwv` and 5–10× longer for
`tcal_solve`. Not confirmed empirically — the run was interrupted.

## 3. Sparse-Jacobian wrap is a regression for per-antenna fits

Commit 28b5544 sparsified `_jac_global` / `_jac_tcal` (legacy global
fits, Jacobian ~`4590 × 109`) and saw `fit_dataset(global_tau)` go
from 231 s → 45 s. The win comes from scipy switching `tr_solver` from
`'exact'` (SVD) to `'lsmr'` (iterative) when handed a sparse Jacobian.

Porting the same trick to `_fit_per_antenna_pwv._jac` (Jacobian
~`1360 × 33`) was attempted in this session and **regressed
2.6–4.9×** across all three Stage-2 modes. LSMR's per-step
convergence does not amortise on small matrices; direct dense SVD
wins. The in-code comment at `fit.py:1066` ("the matrix is small
enough at VLA scales … that a dense build is simpler and faster than
sparse bookkeeping") was right. Do not repeat this port.

If §2 is fixed first (`n_params` falls from 225 to ~33 per fit),
the per-antenna Jacobian remains in the small-matrix dense-SVD regime
and a sparse port stays a regression. If a future change unifies
scans into one big per-antenna Jacobian (§4 option (f)), the matrix
grows back into the regime where sparse pays off.

## 4. Refactor options and their tradeoffs on a 40-core box

Five distinct levers, increasing scope:

**(a) Cache atmospheric ingredients between `_resid` and `_jac` for
the same PWV.** The closure currently calls `_forward_predict` in
`_resid` and again in `_jac` (`fit.py:1073`), discarding `pred` the
second time — same `grid.lookup_with_grad` work twice per accepted LM
step. ~10–15% saving on fit work. Low risk, ~30 lines.

**(b) Parallelise per-antenna fits.** Each `_fit_per_antenna_pwv` call
is independent across `(scan, ant)` — 189 fits per mode are
embarrassingly parallel. Implementation requires lifting `_resid` /
`_jac` out of the closure into module-level functions with explicit
args (today they capture `tsys_arr`, `sigma_arr`, `z_all`, `grid`,
etc. from the enclosing scope) so they are picklable for
`multiprocessing.Pool`. Each worker process must export
`OPENBLAS_NUM_THREADS=1` to avoid `40 × 10 = 400` BLAS threads on 40
cores. Expected speedup ≈ 30–35× on 40 cores (Amdahl-bounded by load
imbalance + serial parts).

**(d) Per-scan band slicing.** Inside `_fit_dataset_stage2`, before
calling `_fit_per_antenna_pwv`, slice the `n_spw` axis to the spws
that have data in that scan. The fit then sees `n_spw≈16`,
`n_params≈33` — the regime the code was tuned for. No semantic
change, only stops allocating and SVD-ing zero columns. This is the
single highest-impact change: it converts the super-linear scaling
in §2 back to roughly linear in `n_spw`. Independent of (a) and (b)
and compositional with both.

**(e) Consensus PWV across scans — generalise existing `shared_pwv`.**
Two-pass mirror of the existing scan-internal consensus, but
combining across scans within an antenna instead of across antennas
within a scan:

```
pass 1: fit (scan, ant) → PWV(scan, ant)               # 189 fits
combine: PWV(ant) = median or IVW over k of PWV(scan_k, ant)
pass 2: refit (scan, ant) with PWV pinned → T0(scan, ant)  # 189 fits
```

PWV constrained by all 7 elevation sweeps per antenna. Statistical
efficiency: median ≈ 64 %, IVW → 100 % under Gaussian noise. Code
cost: ~50 lines, reuses the `pwv_fixed=` hook already used by
`shared_pwv`. Parallelism unit stays `(scan, ant)` → full 40-core
saturation.

**(f) Joint LM per antenna.** Stack all 7 scans' time samples into
one residual vector per antenna; one `least_squares` call per
antenna fits joint PWV + per-(scan, band) T0 (+ c). Statistically
optimal under soft_l1 by construction; gives a covariance on PWV
that already includes inter-scan information. Code cost is the
highest of the options: closure lift, per-(scan, band) initial T0,
robust-loss / sample-rejection semantics re-derived for the joint
residual. **Parallelism unit becomes the antenna — only 27 fits, so
13/40 cores idle.** Jacobian re-enters the regime (~`19040 × 225`)
where a sparse build pays off, so this would naturally also entail
(c) the deferred sparse-Jacobian port.

## 5. Recommended path

1. `OPENBLAS_NUM_THREADS=1` immediately — one env line, 20× on the
   real-scale fit, no code risk.
2. (d) per-scan band slicing — biggest single algorithmic win; expected
   to push full-scale `per_antenna_pwv` from minutes into seconds even
   single-threaded.
3. (b) per-`(scan, ant)` `multiprocessing.Pool` with single-thread BLAS
   per worker — straightforward once `_resid` / `_jac` are lifted; the
   24-core saturation argument favours (b) over (f).
4. (a) cache atmospheric ingredients — only after (d), since with (d)
   the per-fit absolute cost is small enough that the 10–15 % is
   correspondingly small.
5. (e) consensus PWV across scans — physics improvement, stacks cleanly
   on top of the above.
6. (f) joint LM — only if (e) consensus is statistically inadequate
   and the PWV-covariance reporting matters more than parallel
   throughput.

(a)+(b)+(d) together, with (e) optional, should bring
`fit_dataset(per_antenna_pwv)` on the full validation MS to a few
seconds wall on 40 cores, dominated by serial grid build and MS read
rather than by the fit itself.

## 6. Open empirical questions

- True single-thread baseline at `n_spw=112` was not measured. The
  super-linear scaling in §2 means the 5–30 min range is a guess.
- (d) per-scan band-slice has not been implemented or measured; the
  predicted speedup factor (~SVD cost ratio `(225/33)² ≈ 47×` per
  step) needs empirical confirmation.
- Process-level parallel speedup on this workload depends on whether
  pickle overhead of per-fit inputs (`tsys_arr` slice, `grid`, etc.)
  is amortised — needs a prototype.
- Grid build is currently serial across scans (~30 s on this MS) and
  could parallelise across scans on top of the existing per-scan
  internal worker pool — separate concern but relevant to the
  serial-overhead floor in §5's "few seconds" estimate.
