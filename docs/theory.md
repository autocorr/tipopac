---
icon: material/function-variant
---

# Theory &amp; method

This page summarizes the physics and the fit. The authoritative
specification is [`design/design.md`](https://github.com/autocorr/tipopac/blob/main/design/design.md)
in the repository; this is a narrative overview of it.

The pipeline runs in three stages: **A** fits the tipping curve for zenith
opacity per `(scan, antenna, spw)`, **B** anchors a precipitable-water-vapor
value per `(group, antenna)` against a precomputed `am` opacity grid, and
**C** solves the noise-diode (Tcal) scale in closed form with the opacity
pinned to that anchor.

## The tipping curve

As an antenna scans down in elevation, the line of sight passes through
more atmosphere, so the measured system temperature rises. With a
flat-earth, no-refraction geometry the **airmass** is $\sec z = 1/\cos z$,
where the zenith angle is

$$
z = 90^\circ - \mathrm{rad2deg}(\mathrm{el}_\mathrm{encoder}).
$$

The system temperature as a function of zenith angle is modeled as

$$
T_\mathrm{sys}(z) = T_0
 + T_\mathrm{cmb}\,e^{-\tau_0/\cos z}
 + T'_\mathrm{wmt}\,\bigl(1 - e^{-\tau_0/\cos z}\bigr)
 + T_\mathrm{spill}(z),
$$

where $T_0$ is the receiver plus every elevation-*constant* pickup term,
$\tau_0$ is the **zenith opacity** we want, and $T'_\mathrm{wmt}$ is the
weighted-mean atmospheric temperature. The atmosphere both emits and
attenuates: the CMB behind it is transmitted as
$T_\mathrm{cmb}e^{-\tau_0/\cos z}$, so the amplitude the tipping curve
actually measures is $(T'_\mathrm{wmt} - T_\mathrm{cmb})$. Dropping the CMB
term biases $\tau_0$ low by roughly 0.8 %. $T_\mathrm{spill}(z)$ is
instrumental ground pickup, below.

Because power is measured (not field), brightness temperatures carry a
Nyquist / Rayleigh–Jeans correction,

$$
\mathrm{k2nt}(T,\nu) = T\,\frac{h\nu/kT}{e^{h\nu/kT}-1},
$$

so $T_\mathrm{cmb} = \mathrm{k2nt}(2.725\,\mathrm{K}, \nu)$ — not 2.725 K
at the high VLA bands.

The weighted-mean atmospheric temperature $T'_\mathrm{wmt}$ comes from each
scan's atmospheric grid, sampled at the profile's native PWV. Grid
temperatures are already noise K, so no further `k2nt` is applied to them.
Only when a grid cell is unavailable does the code fall back to the
Ulvestad (1987) surface-temperature relation,

$$
T'_\mathrm{wmt} = \mathrm{k2nt}\bigl(256.9 + 0.445\,T_\mathrm{surf},\ \nu\bigr),
$$

with $T_\mathrm{surf}$ in degrees Celsius. Its input is a kinetic
temperature, so this path — and only this path — is `k2nt`-corrected.

### Spillover

Ground emission entering the antenna sidelobes is not attenuated by the sky
column, so it adds a $T_\mathrm{sys}$ contribution proportional to airmass
that a naive opacity fit partly absorbs into $\tau_0$:

$$
T_\mathrm{spill}(z) = \eta(\nu)\,\mathrm{k2nt}(T_\mathrm{surf},\nu)\,\frac{1}{\cos z}.
$$

$\eta(\nu)$ is the stored **spillover efficiency** — a quadratic in $\nu$
running from $\approx 0.47\,\%$ at 4 GHz to $\approx 0.14\,\%$ at 50 GHz,
and zero outside that range. It was derived from cross-band closure against
a forward-model atmosphere rather than from the tipping curves themselves.

Carrying the term inside the model, rather than correcting the opacity
afterwards, matters in two ways. It does not depend on $\tau$, so it shifts
the model without adding a free parameter and leaves
$\partial\,\mathrm{pred}/\partial\tau$ unchanged. And because the fit
supplies each scan's own airmass sampling, the opacity bias it removes
adapts to that scan instead of being a stored constant. `tau_zenith` is
therefore spillover-free at the Stage-A output, and Stage B anchors PWV on
it directly with no add-back. The `spillover_model=True` default enables
this; `False` reproduces the older fit that absorbed ground pickup into the
opacity.

## System temperature and its noise

The VLA records switched power: a sum $S$ and a difference $D$ from the
noise diode of temperature $T_c$ (`tcal_ref`). The system temperature is

$$
T_\mathrm{sys} = \frac{S}{2}\,\frac{T_c}{D},
$$

with $S = \texttt{switched\_sum}$, $D = \texttt{switched\_diff}$. Because
the noise diode is its own gain calibrator, propagating measurement noise
through this expression gives a **radiometer-equation** uncertainty with a
characteristic $T_\mathrm{sys}^2$ dependence:

$$
\sigma_{T_\mathrm{sys}} \approx \frac{2\,T_\mathrm{sys}^2}{T_c\,\sqrt{\Delta\nu\,\tau_\mathrm{int}}},
$$

where $\Delta\nu$ is the per-SPW bandwidth and $\tau_\mathrm{int}$ is the
total ON+OFF Walsh interval (each state accumulates $\tau_\mathrm{int}/2$).
The $T_\mathrm{sys}/T_c$ amplification (~10–60× across VLA bands) is the
physically essential part: it is the SNR with which the diode calibrates
its own gain, and dropping it would mis-scale $\sigma$ and trip outlier
rejection on most samples. A pedagogical derivation lives in
`design/sigma_tsys_derivation.md`.

## Stage A — zenith opacity fit

Stage A fits the tipping curve directly to the data with
`scipy.optimize.least_squares`, using a `soft_l1` robust loss
($f_\mathrm{scale}=3.0$) on $\sigma$-weighted residuals

$$
r_i = \frac{T_{\mathrm{sys},i} - \mathrm{model}_i}{\sigma_{T_\mathrm{sys},i}}.
$$

The fit unit is one `(scan, antenna, spw)` cell and the free parameters are
$T_{0,R}$, $T_{0,L}$, and $\tau_z$ — one opacity per antenna, with both
polarizations sharing it. The lab Tcal is held fixed here ($c \equiv 1$);
the **Tcal correction factor** $c$ is estimated later, in closed form at
pinned $\tau$ (Stage C, below).

The fit runs under a single set of physical bounds — no escalation ladder:

$$
T_0 \in [0, 300]\ \mathrm{K}, \qquad
\tau \in [0, 1.0].
$$

Samples whose per-point $\chi^2 = ((T_\mathrm{sys}-\mathrm{model})/\sigma)^2 > 16$
(4$\sigma$) in either polarization are dropped and the fit repeated, up to
three passes. Per-parameter uncertainties come from an SVD of the Jacobian,
$\tilde J = U S V^\top \Rightarrow \mathrm{cov} = \sigma^2\,V S^{-2} V^\top$
with $\sigma^2 = \sum \tilde r^2/(n-p)$; the $\tau_z$ diagonal becomes
`tau_err`.

### Quality flags

Each cell gets one `fit_reason`:

- **`ok`** — converged, reduced $\chi^2 \le 5$, and $\sigma_\tau/\tau < 0.5$.
- **`poorly_identified`** — converged, but $\sigma_\tau/\tau > 0.5$,
  $\sigma_\tau$ is non-finite, or $\tau_z \le 0$; values are written,
  `fit_success` is `False`. The $\tau_z \le 0$ route is where a
  non-physical opacity lands — it is reported as unidentified rather than
  repaired.
- **`too_few_samples`** — fewer than 3 unflagged samples after rejection.
- **`high_chi2`** — reduced $\chi^2 > 5$.
- **`fit_failed`** — the optimizer raised or refused to converge, or the
  cell had no usable $T'_\mathrm{wmt}$ (no grid value and no finite surface
  temperature).

## Stage B — per-antenna PWV anchor

Stage A gives $\tau_z$ at the SPW center frequencies. Stage B converts
those into a single **precipitable water vapor** value per antenna by
matching them against a precomputed opacity grid.

Because a TCAL block can span a day, the scans are first partitioned into
greedy sequential windows of at most `group_duration_s` seconds (default
7200) so that one PWV is not asked to cover the atmosphere's diurnal
variation. Everything below is per **group**: PWV is fit per
`(group, antenna)`, and each group also yields one dense $\tau(\nu)$ curve
sampled from `am` at that group's median fitted PWV — the curve Stage C
and the plot overlays consume.

For antenna $a$ within a group it minimizes

$$
\chi^2(\mathrm{PWV}; a) = \sum_{\mathrm{scan}\,\in\,g}\ \sum_{\mathrm{spw}}
\left(\frac{\tau_z(\mathrm{scan}, a, \mathrm{spw}) - \tau_\mathrm{grid}(\mathrm{PWV}, \nu_\mathrm{spw})}{\sigma_\tau(\mathrm{scan}, a, \mathrm{spw})}\right)^2
$$

with `scipy.optimize.minimize_scalar(method="bounded")`. The uncertainty is
the Cramér–Rao bound at the fitted PWV,

$$
\sigma_\mathrm{PWV}^2 = \left[\sum_{\mathrm{scan}\,\in\,g}\ \sum_{\mathrm{spw}}
\frac{(\partial\tau_\mathrm{grid}/\partial\mathrm{PWV})^2}{\sigma_\tau^2}\right]^{-1},
$$

where $\partial\tau_\mathrm{grid}/\partial\mathrm{PWV}$ is the analytical
slope of the bilinear interpolant — no Hessian inversion.

Since Stage A solves $\tau_z$ per antenna, `pwv[group, antenna]` is
genuinely per-antenna, and the antenna dimension is retained on every
downstream variable even where it happens to be degenerate.

## Stage C — the Tcal scale at pinned opacity

Evaluating Stage B's per-group `am` opacity curve at the SPW centers gives
a $\tau_\mathrm{am}$ that came from outside the individual tipping curve.
With $\tau$ held there, the Stage-A model stops being nonlinear — it is a
straight line in the model brightness:

$$
T_\mathrm{sys}(z) = \frac{T_0 + T_\mathrm{cmb}e^{-\tau_\mathrm{am}a}
+ T'_\mathrm{wmt}(1 - e^{-\tau_\mathrm{am}a}) + T_\mathrm{spill}(z)}{c}
= A + B\,\mathrm{pred}(z), \qquad A = \frac{T_0}{c},\ B = \frac{1}{c}
$$

so $c = 1/B$ is an inverse-variance weighted regression slope from the
$2\times2$ normal equations — exact in $\tau$, with
$\sigma_c = \sigma_B / B^2$ falling out of the same covariance. No
optimizer, no second `am` run. Sample screening and the joint-polarization
4σ rejection are Stage A's, so the two stages see the same data.

**What is and is not measurable.** The anchor is `am` at the group's
*fitted* PWV, which was itself derived from these opacities. One scalar per
group is therefore absorbed by construction: the array-common level of $c$
is not a measurement, and any array-common model error — $T'_\mathrm{wmt}$,
the 22 GHz water line, the spillover $\eta(\nu)$ — lands there. What
survives is the per-antenna contrast and the per-SPW shape.

**Per-scan $c$ is not calibration-grade.** A single tip's $c$ carries a
scatter an order of magnitude above $\sigma_c$, and the reproducible
per-antenna structure itself drifts over months. Average over a window
before treating $c$ as a Tcal correction. Cells with too little airmass
leverage (`min_airmass_span`) are not reported at all: with $\tau$ pinned,
$c$ is a *level* measurement, and a short tip does not constrain it.

`tcal_fit` $= c \cdot$ `tcal_ref` and `sigma_tcal` $= \sigma_c \cdot$
`tcal_ref`, both NaN where no estimate was made — a finite `sigma_tcal` is
what marks a cell as measured.

## The atmosphere model

The opacity grid $\tau_\mathrm{grid}(\mathrm{PWV}, \nu)$ and the
weighted-mean temperature come from Scott Paine's `am` radiative-transfer
code, run through `amwrap`:

1. **`fetch_atm_profile`** — the only network stage. It pulls vertical
   temperature, humidity, and geopotential-height profiles for the
   observation from Open-Meteo's GFS/HRRR pressure-level grid (closest
   hourly slice per scan), with retry/backoff. If the date predates the
   archive (~2021-03-23) or the request fails, it falls back deterministically
   to **AFGL climatologies** (`midlatitude_summer` / `midlatitude_winter`,
   chosen from the observation month under `"auto"`).
2. **`build_atm_grids`** — runs `am` **once per scan** to build a
   `PwvGrid`: a bilinear lookup over `(pwv_mm, freq_Hz)` for zenith opacity
   and brightness temperature. PWV is varied by scaling the tropospheric
   water column in `am`.

Crucially, `am` runs only during grid construction — **never inside the
per-sample fit loop**. Every stage reads from the precomputed grid: Stage A
takes its $T_\mathrm{mean}$ input from it, Stage B fits PWV against its
$\tau_z(\nu)$, and Stage C reuses the curve Stage B already wrote.

## The canonical dataset

Every reader (MS or SDM) produces one `xarray.Dataset` with dimensions
`scan, antenna, spw, polarization (R, L), time`, plus profile/frequency
axes. The `time` axis is **per-scan-local and NaN-padded** (no MultiIndex);
a `flag` array masks the padding and bad data, so flag-respecting
reductions must go through `tipopac.schema.apply_flags(ds, var)`. Variable
groups cover reader inputs (switched power, zenith angle, Tcal, weather),
fit results ($T_\mathrm{sys}$, $\sigma_{T_\mathrm{sys}}$, `tau_zenith`,
`tau_err`, `tcal_fit`, `sigma_tcal`, `fit_reason`), the atmospheric
profile, and the PWV anchor. The schema is defined in `src/tipopac/schema.py` and §4 of the
design document.
