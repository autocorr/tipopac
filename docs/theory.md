---
icon: material/function-variant
---

# Theory &amp; method

This page summarizes the physics and the fit. The authoritative
specification is [`design/design.md`](https://github.com/autocorr/tipopac/blob/main/design/design.md)
in the repository; this is a narrative overview of it.

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
T_\mathrm{sys}(z) = T_0 + T'_\mathrm{wmt}\,\bigl(1 - e^{-\tau_0/\cos z}\bigr),
$$

where $T_0$ is the receiver/ground contribution, $\tau_0$ is the
**zenith opacity** we want, and $T'_\mathrm{wmt}$ is the
weighted-mean atmospheric temperature. Fitting this curve to
$T_\mathrm{sys}$ versus $\sec z$ yields $\tau_0$.

Because power is measured (not field), brightness temperatures carry a
Nyquist / Rayleigh–Jeans correction,

$$
\mathrm{k2nt}(T,\nu) = T\,\frac{h\nu/kT}{e^{h\nu/kT}-1}.
$$

The weighted-mean atmospheric temperature $T'_\mathrm{wmt}$ comes from each
scan's atmospheric grid (sampled at the profile's native PWV and corrected
with `k2nt`). Only when a grid cell is unavailable does the code fall back
to the Bevis (1992) surface-temperature heuristic,

$$
T_\mathrm{wmt} = 70.2 + 0.72\,T_\mathrm{surf}.
$$

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

There are two modes:

| Mode | Fit unit | Free parameters |
| --- | --- | --- |
| `independent_tau` | `(scan, antenna, spw)` | $T_{0,R},\ T_{0,L},\ \tau_z$ |
| `independent_tau_solve` (default) | `(scan, spw)` | per-antenna $(T_{0,R}, c_R, T_{0,L}, c_L)$ and one shared $\tau_z$ |

`independent_tau` trusts the lab Tcal and fits opacity per antenna.
`independent_tau_solve` additionally solves a **Tcal correction factor**
$c$ per (antenna, polarization), sharing a single $\tau_z$ across all
antennas in a (scan, SPW); its Jacobian is sparse (block-diagonal in the
per-antenna $(T_0, c)$ columns, dense in the shared $\tau_z$ column).

The fit runs under a single set of physical bounds — no escalation ladder:

$$
T_0 \in [0, 300]\ \mathrm{K}, \qquad
\tau \in [0, 1.0], \qquad
c \in [0.5, 2.0].
$$

Samples whose per-point $\chi^2 = ((T_\mathrm{sys}-\mathrm{model})/\sigma)^2 > 16$
(4$\sigma$) in either polarization are dropped and the fit repeated, up to
three passes. Per-parameter uncertainties come from an SVD of the Jacobian,
$\tilde J = U S V^\top \Rightarrow \mathrm{cov} = \sigma^2\,V S^{-2} V^\top$
with $\sigma^2 = \sum \tilde r^2/(n-p)$; the $\tau_z$ diagonal becomes
`tau_err`.

### Quality flags

Each cell gets one `fit_reason`:

- **`ok`** — converged, reduced $\chi^2 < 5$, and $\sigma_\tau/\tau < 0.5$.
- **`poorly_identified`** — converged but $\sigma_\tau/\tau > 0.5$; values
  are written, `fit_success` is `False`.
- **`too_few_samples`** — fewer than 3 unflagged samples after rejection.
- **`high_chi2`** — reduced $\chi^2 \ge 5$.
- **`fit_failed`** — the optimizer raised or refused to converge.

## Stage B — per-antenna PWV anchor

Stage A gives $\tau_z$ at the SPW center frequencies. Stage B converts
those into a single **precipitable water vapor** value per antenna by
matching them against a precomputed opacity grid. For antenna $a$ it
minimizes

$$
\chi^2(\mathrm{PWV}; a) = \sum_{\mathrm{scan},\,\mathrm{spw}}
\left(\frac{\tau_z(\mathrm{scan}, a, \mathrm{spw}) - \tau_\mathrm{grid}(\mathrm{PWV}, \nu_\mathrm{spw})}{\sigma_\tau(\mathrm{scan}, a, \mathrm{spw})}\right)^2
$$

with `scipy.optimize.minimize_scalar(method="bounded")`. The uncertainty is
the Cramér–Rao bound at the fitted PWV,

$$
\sigma_\mathrm{PWV}^2 = \left[\sum_{\mathrm{scan},\,\mathrm{spw}}
\frac{(\partial\tau_\mathrm{grid}/\partial\mathrm{PWV})^2}{\sigma_\tau^2}\right]^{-1},
$$

where $\partial\tau_\mathrm{grid}/\partial\mathrm{PWV}$ is the analytical
slope of the bilinear interpolant — no Hessian inversion.

Under `independent_tau` the per-antenna $\tau_z$ varies, so `pwv[antenna]`
is genuinely per-antenna. Under `independent_tau_solve` the shared $\tau_z$
is broadcast equal across antennas, so the anchor returns identical
`pwv[antenna]` — shared-PWV semantics that fall out of the per-antenna fit.
The antenna dimension is retained either way.

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
per-sample fit loop**. Both stages read from the precomputed grid: Stage A
takes its $T_\mathrm{mean}$ input from it, and Stage B fits PWV against its
$\tau_z(\nu)$.

## The canonical dataset

Every reader (MS or SDM) produces one `xarray.Dataset` with dimensions
`scan, antenna, spw, polarization (R, L), time`, plus profile/frequency
axes. The `time` axis is **per-scan-local and NaN-padded** (no MultiIndex);
a `flag` array masks the padding and bad data, so flag-respecting
reductions must go through `tipopac.schema.apply_flags(ds, var)`. Variable
groups cover reader inputs (switched power, zenith angle, Tcal, weather),
fit results ($T_\mathrm{sys}$, $\sigma_{T_\mathrm{sys}}$, `tau_zenith`,
`tau_err`, `tcal_fit`, `fit_reason`), the atmospheric profile, and the PWV
anchor. The schema is defined in `src/tipopac/schema.py` and §4 of the
design document.
