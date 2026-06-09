# σ_Tsys derivation for VLA switched-power Tsys

Reference / pedagogical writeup of the formula computed in
`tipopac.fit._compute_sigma_tsys`. The canonical implementation contract
is `design/design.md` §5.3; this file shows the math step by step and
spells out the assumptions.

The headline result:

```
σ_Tsys  ≈  √2 · Tsys² / ( T_c · √(Δν · τ_int) )
```

The surprising piece is the **Tsys²**, not the textbook Tsys you get
from the radiometer equation. The reason is that VLA switched-power
uses the noise tube as its own gain calibrator, and the SNR of *that*
calibration is what limits σ.

---

## 1. Physical setup

Each VLA front end injects a stable noise tube of equivalent input
temperature `T_c` (Kelvin) into the signal path through a Walsh-switched
coupler. During every correlator integration `τ_int` (~1 s), the noise
tube is keyed ON and OFF at f_W ≈ 10 Hz, and the correlator accumulates
two sums per (antenna, spw, polarization):

```
P_off = G · Tsys                        ← mean digital power, tube OFF
P_on  = G · (Tsys + T_c)                ← mean digital power, tube ON

S  = P_on + P_off  =  G · (2 Tsys + T_c)        ← MS column SWITCHED_SUM
D  = P_on − P_off  =  G · T_c                   ← MS column SWITCHED_DIFF
```

Symbols:

| symbol  | meaning                                                              |
| ------- | -------------------------------------------------------------------- |
| `G`     | receiver chain gain (counts/K). Unknown, drifts slowly.              |
| `Tsys`  | system temperature (K). What we want.                                |
| `T_c`   | noise-tube equivalent input temperature (K). Known from CALDEVICE.   |
| `Δν`    | bandwidth (Hz). Per-spw `bandwidth` coord.                           |
| `τ_int` | per-state integration time (s). Per-sample `exposure_time`.          |

The whole point of the switching scheme is that `T_c` is stable and
known to ≲1% from on-table calibration, while `G` is none of those
things. The combination `S/D` cancels `G` and lets us back out `Tsys`
in absolute Kelvin without ever measuring `G` directly.

Solving for Tsys: from `D = G T_c` we have `G = D/T_c`. Plug into S:

```
Tsys = T_c · (S − D) / (2 D)                    ← exact

     ≈ (S/2) · T_c / D                          ← used by the code,
                                                  valid when Tsys ≫ T_c
```

The fractional error of the approximation is `D/(S−D) ≈ T_c/(2 Tsys)`,
roughly 2.5% at the VLA-typical Tsys/T_c ≈ 20 and 10% in the low-band
limit Tsys/T_c ≈ 5. We carry the approximation into the noise
propagation, since both contributions track each other and the dominant
noise term swamps a few-percent bias.

---

## 2. The standard radiometer equation (baseline)

For a power measurement integrating bandwidth `Δν` over time `τ`, the
fractional uncertainty in the measured power is

```
σ_P / ⟨P⟩  =  1 / √(Δν · τ)
```

So for our two single-state accumulators, each integrated over `τ_int`
of total time on that state:

```
σ_P_off ≈ G · Tsys / √(Δν τ_int)
σ_P_on  ≈ G · (Tsys + T_c) / √(Δν τ_int)  ≈  G · Tsys / √(Δν τ_int)
```

The last approximation uses `Tsys ≫ T_c` again. Cov(P_on, P_off) = 0
because the two accumulators draw from disjoint time samples at the
Walsh rate.

### A note on the τ_int convention

There are two conventions in the literature for what "integration time"
means in the radiometer equation, and they differ by √2:

- **Per-state convention** (used above): `τ_int` is the integration
  time accumulated *in each Walsh state*. `σ_D = √2 · G Tsys / √(Δν τ_int)`.
- **Total-interval convention**: `τ_int` is the full ON+OFF interval.
  Each state then gets `τ_int/2`, giving `σ_D = 2 · G Tsys / √(Δν τ_int)`.

The code uses the per-state convention, which is what produces the
`√2` prefactor in `_compute_sigma_tsys`. Whether the MS `EXPOSURE`
column reports the per-state or total integration time is *the* known
ambiguity in the absolute normalization of σ_Tsys; empirically
(`run/sigma_tsys`) the predicted σ comes out ≈1.8× low, and a √2 of
that is plausibly this convention mismatch. **This affects σ_Tsys by
an O(1) constant**, identical across all cells, so it does not bias
the χ² weighting in the tipping fit; it would only matter for absolute
interpretation of reduced χ² or the σ_τ that drops out of the
covariance.

---

## 3. Step-by-step propagation of σ_Tsys

Given `Tsys(S, D) = (S/2) · T_c / D`, the general first-order
propagation is:

```
σ²_Tsys = (∂Tsys/∂S)² · σ²_S
        + (∂Tsys/∂D)² · σ²_D
        + 2 · (∂Tsys/∂S) · (∂Tsys/∂D) · Cov(S, D)
```

### 3.1 Partials

```
∂Tsys/∂S  =  T_c / (2 D)                =  Tsys / S
∂Tsys/∂D  =  −(S/2) · T_c / D²          =  −Tsys / D
```

The minus sign in `∂Tsys/∂D` is the noise-tube calibration mechanism:
a positive noise excursion in `D` makes the implied gain too large,
which makes the inferred Tsys too small. The magnitude is the
fractional gain uncertainty — exactly what we expect from any
self-calibrated measurement.

### 3.2 Variances of S and D

S and D are linear combinations of P_on and P_off; with Cov(P_on,
P_off) = 0:

```
Var(S) = Var(P_on + P_off) = Var(P_on) + Var(P_off)
       ≈ 2 · G² Tsys² / (Δν τ_int)

Var(D) = Var(P_on − P_off) = Var(P_on) + Var(P_off)
       ≈ 2 · G² Tsys² / (Δν τ_int)

  →  σ_S  ≈  σ_D  ≈  √2 · G Tsys / √(Δν τ_int)
```

That `√2` over the single-state σ is the well-known Dicke penalty:
switching costs √2 in noise compared to a hypothetical total-power
measurement with the same total integration time.

### 3.3 Covariance of S and D

```
Cov(S, D)  =  Cov(P_on + P_off, P_on − P_off)
           =  Var(P_on) − Var(P_off)
           ≈  G² · ((Tsys+T_c)² − Tsys²) / (Δν τ_int)
           ≈  G² · 2 Tsys T_c / (Δν τ_int)
```

The relative magnitude of the cross-term in σ²_Tsys vs the D-only
term is `T_c/Tsys`, which is at most ~10% at VLA, and the cross-term
*reduces* the variance slightly (the sign cancels with the negative
∂Tsys/∂D). We drop it; the formula is conservative by ≤10%.

### 3.4 Plugging in

S-side contribution, using `G = D/T_c` ⇒ `G²/D² = 1/T_c²`:

```
(∂Tsys/∂S)² σ²_S  =  ( T_c / (2D) )² · 2 G² Tsys² / (Δν τ_int)
                  =  ( T_c² / (4 D²) ) · 2 (D²/T_c²) Tsys² / (Δν τ_int)
                  =  Tsys² / ( 2 Δν τ_int )
```

The factor inside the parentheses simplifies because the explicit `D²`
in the variance cancels the implicit one hidden in `G`. What's left is
proportional to Tsys²/(Δν τ_int) — i.e., the **naive radiometer-equation
variance**, modulo a factor of 2.

D-side contribution, same substitution:

```
(∂Tsys/∂D)² σ²_D  =  ( Tsys / D )² · 2 G² Tsys² / (Δν τ_int)
                  =  ( Tsys² / D² ) · 2 (D²/T_c²) Tsys² / (Δν τ_int)
                  =  2 · Tsys⁴ / ( T_c² · Δν τ_int )
```

### 3.5 Ratio of D-side to S-side

```
D-side / S-side  =  [ 2 Tsys⁴ / (T_c² Δν τ) ]  /  [ Tsys² / (2 Δν τ) ]
                 =  4 · (Tsys / T_c)²
```

For VLA Tsys/T_c ≈ 10–50, the D-side dominates by 400–10000×.
**Three orders of magnitude or more.** So the D-side alone gives the
right answer to within a fraction of a percent, and we can collapse:

```
σ²_Tsys  ≈  (∂Tsys/∂D)² σ²_D  =  2 Tsys⁴ / ( T_c² · Δν τ_int )

→  σ_Tsys  ≈  √2 · Tsys² / ( T_c · √(Δν τ_int) )
```

---

## 4. The physical intuition

It is worth pausing to understand *why* the answer has Tsys² rather
than Tsys.

Conceptually:

- A direct calibrated total-power measurement of Tsys is limited by
  the fundamental shot/thermal noise in the integrated power, giving
  σ ≈ Tsys/√(Δν τ). That's the textbook radiometer equation.
- The VLA switched-power scheme does **not** have an externally
  calibrated gain. It invents a calibrator on the fly: the noise tube,
  whose K-value is known.
- But the *signal* from that calibrator — the digital power difference
  `D = G T_c` — is buried in the bath of Tsys K of receiver noise. Its
  SNR is

  ```
  SNR(D)  ≈  T_c · √(Δν τ) / Tsys
  ```

  At VLA with Tsys = 100 K, T_c = 5 K, Δν = 128 MHz, τ_int = 1 s, that
  comes out to ≈ 565 — fine, but **not** the 10⁴–10⁵ that an external
  calibrator would give you.
- The fractional uncertainty in the *gain* you derive from the noise
  tube is just 1/SNR(D) ≈ Tsys / (T_c · √(Δν τ)).
- And the fractional uncertainty in Tsys equals the fractional
  uncertainty in 1/G, which equals the fractional uncertainty in D.
  So:

  ```
  σ_Tsys / Tsys  ≈  Tsys / ( T_c · √(Δν τ) )

  σ_Tsys  ≈  Tsys² / ( T_c · √(Δν τ) )    (+ the √2 for switching)
  ```

The Tsys² is the fingerprint of a self-calibrating measurement where
the calibration signal is small compared to the noise background.
Whenever you see σ ∝ T_signal² / T_calibrator in radio astronomy, that
is what is going on.

A useful reframe: σ_Tsys is what you'd get from the naive radiometer
equation **multiplied by the Tsys/T_c amplification factor**. If you
forget the amplification (e.g., by using a flat or v2.6-style unit
weighting), at VLA bands you underweight high-Tsys samples by a factor
of 10–70, which is exactly the wrong direction for tipping fits where
the high-airmass samples carry most of the τ leverage.

---

## 5. Assumptions, sanity checks, and where this breaks

Approximations made and their cost (all small at VLA):

| Approximation                                | Fractional error |
| -------------------------------------------- | ---------------- |
| `Tsys ≫ T_c` in `Tsys = (S/2) T_c / D`       | T_c/(2 Tsys) ~ 2–5% |
| Same approximation in `σ_P_on`               | T_c/Tsys ~ 5–10% |
| `Cov(P_on, P_off) = 0`                       | true to ≲1% from gain drift on > τ_int timescales |
| `Cov(S, D)` dropped                          | conservative by ≤ T_c/Tsys ~ 5–10% |
| S-side contribution dropped                  | 1/(4·(Tsys/T_c)²) ≲ 0.25% |
| Linearization (first-order propagation)      | (σ_D/D)² · higher-order, ≲ 0.5% |

The dominant *uncontrolled* corrections are not in the math above:

- **3-bit quantization**: VLA samplers add Van Vleck noise; the effective
  σ is roughly 1.13× the ideal radiometer-eq prediction. Real.
- **Gain drift within τ_int**: G isn't perfectly constant across one
  switching cycle. If σ_G/G ~ ε, this adds ε² · Tsys² in quadrature
  to σ²_Tsys. For the VLA cryogenic chain over 1 s this is sub-percent.
- **Atmospheric fluctuation** on sub-second timescales: real sky noise
  that the radiometer equation does not see.
- **τ_int convention** (per-state vs total interval): an O(√2) absolute
  scale ambiguity, identical across cells.

Failure modes worth knowing:

- **High-band (K, Q) at low elevation, no noise tube**: if T_c → 0 or
  is mis-tabulated, the formula blows up. `_compute_sigma_tsys` masks
  cells with `tcal_ref ≤ 0` to NaN.
- **Sun in sidelobe, RFI**: those raise Tsys real-time and the formula
  faithfully tracks it, *but* the assumption of stationary Tsys
  underlying the radiometer equation is violated. Garbage in, garbage
  out.
- **T_c retabulation between observation and analysis**: if the
  CALDEVICE column was patched after the data was taken, the gain you
  back out is wrong by `T_c_recorded / T_c_true`. σ_Tsys scales as
  1/T_c so a 10% error in T_c is a 10% error in σ. Worth checking
  before running the fit on archival data.

---

## 6. Choice of T_c: `tcal_ref` vs `tcal_fit`

In `independent_tau_solve` (tcal_solve) mode, the fit produces a per-cell
`tcal_fit = c · tcal_ref`, where `c` is the solved Tcal-correction
factor in the (T_0, c, τ) model. One might naively expect that the
"true" Tcal for the noise budget is `tcal_fit` rather than `tcal_ref`.

Empirically (see `run/sigma_tsys/test_sigma_tsys.py`), substituting
`tcal_fit` for `tcal_ref` slightly **degrades** the agreement between
predicted and empirical σ:

| Tcal source                            | emp/pred median ratio | MR exponents (Tsys, T_c) |
| -------------------------------------- | --------------------- | ------------------------ |
| `tcal_ref` (`independent_tau` mode)    | 1.83                  | (+1.82, −0.89)           |
| `tcal_fit` (`independent_tau_solve`)   | 1.93                  | (+1.68, −0.79)           |

The reason: `tcal_fit` is a fitted parameter sitting near a (T_0, c, τ)
degeneracy ridge. It carries trajectory noise from the optimizer that
is decorrelated with the real switched-power noise budget. Better to
use the stable reference value.

A useful side effect of this choice is that `σ_Tsys` is identical
across `independent_tau` and `independent_tau_solve` modes (it's
computed pre-fit), so χ² and σ_τ are directly comparable across modes.

---

## 7. Empirical confirmation summary

From `run/sigma_tsys/test_sigma_tsys.py`, 6048 (scan, ant, spw, pol)
cells:

```
sig_emp = MAD * 1.4826 of Tsys(t) detrended against a quadratic in 1/cos(z)
sig_pred = ds["sigma_Tsys"]      (= the formula above)
sig_naive = Tsys / √(Δν τ_int)
```

Findings:

```
median sig_emp / sig_pred   = 1.83     ← absolute scale ~1.8× off,
                                         likely √2 from convention + Van Vleck
median sig_emp / sig_naive  = 54       ← matches Tsys/T_c × √2 = 21 × √2 = 30
                                         (within the absolute-scale offset)

multiple-regression slopes on log Tsys, log T_c, holding log(BT) at −0.5:
    log Tsys exponent:  +1.82          (rewrite expects +2, naive +1)
    log T_c  exponent:  −0.89          (rewrite expects −1, naive  0)
```

Strong support for the Tsys²/T_c shape over the naive Tsys/√BT,
modulo an O(1) absolute prefactor we have not chased into the ground.

---

## 8. Pointers

- Implementation: `src/tipopac/fit.py` → `_compute_sigma_tsys`
- Spec contract: `design/design.md` §5.3
- Empirical validation: `run/sigma_tsys/`
- Related: AIPS Memo 199 (Perley/Butler) on VLA switched power; Rohlfs
  & Wilson *Tools of Radio Astronomy* §4 on the Dicke radiometer.
