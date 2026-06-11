# `tipopac` — Design Specification

A clean, importable Python rewrite of the CASA `tipopac` task. Estimates
VLA zenith opacity and noise-diode temperatures from `DO_SKYDIP`
tipping-scan data, without requiring a CASA runtime.

> **Scope of this document.** Requirements / contract for the current
> implementation. This file supersedes `old_context/initial_design.md`
> and `old_context/independent_tau_fit.md`. When implementation drifts
> from this spec, this file is updated in the same commit (see CLAUDE.md).

---

## 1. Overview & goals

### Preserved from `tipopac_v2.6`

- The physical model: `Tsys = T0 + Twmt'·(1 − exp(−τ₀/cos z))`, with
  `Twmt'` the Nyquist-corrected weighted-mean atmospheric temperature.
- VLA-specific assumptions: dual circular (R/L) polarization,
  two-row CALDEVICE (noise-tube + solar-filter), AZELGEO encoder
  elevation, MS/SDM scan-intent `*DO_SKYDIP*`.
- Application of online and user-file flagging before fit acceptance.

### Changed vs `tipopac_v2.6`

- Plain Python package; no `buildmytasks` / CASA-task wrapper.
  `casatools` is an ordinary library import for table I/O and the
  optional caltable writers.
- In-memory representation is one canonical `xarray.Dataset` produced
  by either an MS or SDM reader.
- Atmospheric modelling moves from `casatools.atmosphere` to Scott
  Paine's `am`, accessed via the local `amwrap` Python wrapper.
- Vertical profiles come from open-meteo's `historical-forecast-api`
  pressure-level grid (gfs_hrrr model); offline fallback is amwrap's
  AFGL climatologies.
- Fit architecture is **Stage A + Stage B**: per-spw zenith-opacity
  fit from the observed data, then a post-hoc per-antenna PWV anchor
  against the resulting `τ_z(ν)` samples via a precomputed
  `PwvGrid`. am is run **once per analysis** (during grid build),
  never inside the per-sample fit loop.
- Modern Python: `pyproject.toml` with `uv`, type hints, `ty` for
  type-checking, `ruff` for lint/format, `pytest` for tests.

---

## 2. Public API

Two surfaces over the same internal stages.

```python
# --- functional one-shot ---
from tipopac import tipopac, Result

result: Result = tipopac(
    path,                                            # MS or SDM (auto-detected)
    *,
    scans=None,                                      # iterable[int] of DO_SKYDIP scan numbers; None = all
    bands=None,                                      # iterable[str] of VLA bands; None = ("Ku","K","Ka","Q")
    mode="independent_tau_solve",                    # | "independent_tau"
    flags_online=True,
    flags_file=None,
    atm_profile_source="open-meteo",                 # | "afgl"
    afgl_climatology="auto",                         # | "midlatitude_summer" | ...
    n_workers=None,                                  # int → multiprocessing.Pool
    plot_dir=None,                                   # if set, write PNGs here
    caltable_opacity=None,                           # if set, write CASA TOpac table
    caltable_tcal=None,                              # if set, write CALDEVICE-style table
)

# --- class-based for staged / notebook use ---
from tipopac import TippingAnalysis

ta = TippingAnalysis.from_path("data/tip_test.ms", scans=None, bands=None)
ta.apply_flags(online=True, file=None)
ta.fetch_atm_profile(source="open-meteo")
ta.build_atm_grids()
ta.fit(mode="independent_tau_solve")
ta.plot(out_dir="plots/")
ta.write_caltables(opacity="z.cal", tcal="t.cal")
result = ta.result
```

By default the readers keep only the high-frequency VLA bands (`Ku, K,
Ka, Q`) where tipping-curve fits are well-conditioned. Pass
`bands=["L", "S", ...]` to opt into low bands explicitly. Scan
filtering further narrows the DO_SKYDIP set — e.g. on a 24-hour
observation with a tipping scan every six hours, `scans=[12]` fits
only that block. Filtering is applied at read time; excluded scans
and SPWs are never loaded.

`Result` is a frozen dataclass:

```python
@dataclass(frozen=True)
class Result:
    dataset: xr.Dataset                   # the canonical schema (§4)
    mode: str                             # the public mode used
    input_path: Path
    input_format: Literal["ms", "sdm"]
    software_versions: dict[str, str]     # tipopac, casatools, sdmpy, amwrap, am
```

The freeze applies to the dataclass field bindings, not the underlying
`xr.Dataset`. Callers that need an unchanging snapshot take
`result.dataset.copy(deep=True)`.

### 2.1 Modes

Two public modes; both run **Stage A → Stage B**. They differ only in
the Stage-A fit routing.

| Mode                      | Stage-A unit           | Stage-A free parameters                                 |
| ------------------------- | ---------------------- | ------------------------------------------------------- |
| `independent_tau`         | `(scan, antenna, spw)` | `T0_R, T0_L, τ_z`                                       |
| `independent_tau_solve`   | `(scan, spw)`          | per-antenna `(T0_R, c_R, T0_L, c_L)` + one shared `τ_z` |

`independent_tau_solve` is the default. `independent_tau` is the
no-Tcal-correction variant for callers that trust the laboratory Tcal.

### 2.2 Staging contract

Each `TippingAnalysis` method mutates `self._ds` in place:

- `from_path(path, scans=None, bands=None)` — auto-detect reader,
  apply scan and band selection at read time, build dataset, validate
  against §4 schema. Defaults: every DO_SKYDIP scan, only high-
  frequency receivers (`Ku, K, Ka, Q`).
- `apply_flags(online, file)` — populate `ds["flag"]`.
- `fetch_atm_profile(source, afgl_climatology)` — attach atmospheric
  profile to `ds` (§7.1). Idempotent.
- `build_atm_grids(pwv_step_mm, freq_step_Hz, n_workers)` — precompute
  per-scan `PwvGrid` and stash on `self._grids` (§7.2). Auto-calls
  `fetch_atm_profile` if needed.
- `fit(mode, n_workers)` — Stage A + Stage B. Auto-calls
  `build_atm_grids` if needed. After this `result` is available.
- `plot(out_dir)` — per-(scan, antenna, spw) PNGs via
  `plot.PlotData(ds).save_all`.
- `write_caltables(opacity, tcal)` — optional CASA-format outputs.

---

## 3. Reader abstraction

A single Protocol; two concrete implementations; one dispatcher.

```python
class TippingReader(Protocol):
    @classmethod
    def supports(cls, path: Path) -> bool: ...
    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        scans: Sequence[int] | None = None,
        bands: Sequence[str] | None = None,
    ) -> "TippingReader": ...
    def read(self) -> xr.Dataset: ...
```

`tipopac.api` walks the registered readers; the first whose
`supports(path)` returns True is constructed and `read()`. Heuristics:

- **MSReader.supports**: directory containing `table.dat` and a
  `SYSPOWER/` subtable.
- **SDMReader.supports**: directory containing `ASDM.xml`.

Both readers apply scan and band selection at read time so excluded
scans and SPWs are never loaded. Scan filtering narrows the
DO_SKYDIP set returned by `scansforintent` (MS) or `Scan.scanIntent`
(SDM); band filtering reads the receiver-set band label from the
SPW NAME (`SPECTRAL_WINDOW.NAME` in the MS, `SpectralWindow.xml`
`<name>` in the SDM) — both carry the same `EVLA_<BAND>#…` string —
and drops SPWs whose band is not in the user's allowlist. The SDM's
`Receiver.xml` `<frequencyBand>` carries the same information and is
the equivalent SDM-only source. A scan whose SPWs are all dropped by
the band filter is removed from the dataset. The selection helpers
(`band_for_spw_name`, `normalize_bands`, `validate_scan_selection`,
`select_spws_by_band`) live in `tipopac.bands` and are shared between
both readers — the MS↔SDM parity contract extends to selection
behaviour.

### SDM ↔ MS column mapping

Both readers must produce datasets that pass `schema.validate(ds)` —
identical dims, coords, dtypes. The mapping below is the parity
contract.

| MS subtable / column                                      | SDM table                  | sdmpy access pattern                                                |
| --------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- |
| `ANTENNA.NAME`                                            | `Antenna.xml`              | `sdm['Antenna'][i].name`                                            |
| `SPECTRAL_WINDOW.REF_FREQUENCY/NUM_CHAN/TOTAL_BANDWIDTH`  | `SpectralWindow.xml`       | `sdm['SpectralWindow'][spw_id]`                                     |
| `POINTING.TIME/ENCODER`                                   | `Pointing.xml`             | `sdm['Pointing'][ant_id, time_id]`                                  |
| `SYSPOWER.TIME/SWITCHED_DIFF/SWITCHED_SUM`                | `SysPower.xml`             | `sdm['SysPower'][ant_id, feed_id, spw_id]`                          |
| `CALDEVICE.NOISE_CAL`                                     | `CalDevice.xml`            | iterate rows; row key `(antennaId, feedId, spectralWindowId)`; load 0 = noise tube; R = col 3, L = col 3+ncols |
| `WEATHER.TIME/TEMPERATURE/REL_HUMIDITY/PRESSURE`          | `Weather.xml`              | `sdm['Weather'][station, time]`                                     |
| scan intent `*DO_SKYDIP*`                                 | `Scan.xml` + `Subscan.xml` | `sdm['Scan'][i].scanIntent` / `sdm['Subscan'][i,j].subscanIntent`   |
| `FLAG_CMD` (online flags)                                 | — (no SDM equivalent)      | `SDMReader` returns an empty flag command set                       |

---

## 4. Canonical `xarray.Dataset` schema

The single in-memory representation. Optional vars are tolerated when
absent but must conform when present.

```text
Dimensions
  scan           (n_scans,)         int        DO_SKYDIP scan numbers
  antenna        (n_antennas,)      str        e.g. "ea05"
  spw            (n_spw,)           int        spectral-window id
  polarization   (2,)               str        "R", "L"
  time           (n_time,)          int        per-scan local sample index
                                               (0..max_n_samples−1); ragged
                                               across scans, padding masked by
                                               the flag array
  xyz            (3,)               -          ITRF axis label for antenna_position
  atm_level      (n_levels,)        -          pressure-level axis from §7.1
  frequency_dense (n_freq,)         -          dense am output grid axis

Coords
  frequency(spw)                   Hz             spw reference frequency
  bandwidth(spw)                   Hz             spw total bandwidth
  band(spw)                        str (U4)       VLA receiver band label ("Ku","K","Ka","Q","X","C","S","L","P","4")
  antenna_position(antenna, xyz)   m              ITRF X, Y, Z
  scan_time_start(scan)            s              UTC MJD-seconds
  scan_time_end(scan)              s              UTC MJD-seconds
  time_utc(scan, time)             float64        non-dim 2D coord; per-sample
                                                  UTC MJD-seconds, NaN at the pad

Data variables — inputs (filled by readers)
  switched_diff (scan, antenna, spw, polarization, time)   float32
  switched_sum  (scan, antenna, spw, polarization, time)   float32
  zenith_angle  (scan, antenna, time)                      float32   deg
  tcal_ref      (antenna, spw, polarization)               float32   K   (CALDEVICE row 0)
  weather_T     (scan, time)                               float32   K   surface kinetic T
  weather_P     (scan, time)                               float32   Pa
  weather_RH    (scan, time)                               float32   fractional (0–1)
  exposure_time (scan, time)                               float32   s   per-sample integration
  flag          (scan, antenna, spw, polarization, time)   bool

Data variables — fit results (filled by fit.py)
  Tsys          (scan, antenna, spw, polarization, time)   float32   K
  sigma_Tsys    (scan, antenna, spw, polarization, time)   float32   K   radiometer-eq per-sample σ
  tau_zenith    (scan, antenna, spw)                       float32   nepers
  tau_err       (scan, antenna, spw)                       float32
  T0            (scan, antenna, spw, polarization)         float32   K
  tcal_fit      (scan, antenna, spw, polarization)         float32   K
  fit_success   (scan, antenna, spw)                       bool
  fit_reason    (scan, antenna, spw)                       str

Data variables — atmospheric profile (filled by atmosphere.attach_profile)
  atm_pressure     (atm_level,)                            float64   Pa
  atm_temperature  (scan, atm_level)                       float32   K
  atm_h2o_vmr      (scan, atm_level)                       float32   volumetric mixing ratio

Data variables — atmospheric anchor (filled by anchor.anchor_pwv / write_am_curve)
  pwv              (antenna,)                              float32   mm   per-antenna fitted PWV
  pwv_err          (antenna,)                              float32   mm   1σ from Cramér–Rao
  am_freq_grid     (frequency_dense,)                      float64   Hz   dense am output axis
  am_tau           (frequency_dense,)                      float64   nepers, at representative PWV

Attrs
  source_path         : str
  source_format       : "ms" | "sdm"
  observatory         : "VLA"
  mode                : str  (the public mode used)
  software_versions   : dict[str, str]
  scans_requested     : "all" | list[int]   (user-supplied scans argument, or "all")
  bands_requested     : "default_high_freq" | list[str]  (user-supplied bands argument)
  selected_scans      : list[int]           (resolved DO_SKYDIP scans kept after filtering)
  selected_bands      : list[str]           (sorted unique band labels present after filtering)
  atm_profile_source  : "open_meteo" | "afgl_<climatology>"
  open_meteo_query    : dict | None      (provenance: lat, lon, time, endpoint, model)
  surface_pressure_hPa: dict[int, float] (per-scan median surface pressure)
  pwv_profile_source  : dict[int, str]   (per-scan grid provenance)
```

### 4.1 Representation choices

- **`band(spw)` is a label, not a unique key.** Multiple SPWs may
  share the same band label — e.g. a single tipping scan often covers
  both low-Ka and high-Ka, and both SPWs label as `"Ka"`. Likewise for
  low-Q / high-Q. Code that groups by band must not assume one SPW
  per band.
- **Antenna dim is retained even when degenerate.** Under
  `independent_tau_solve` the Stage-A `tau_zenith(scan, spw)` is
  broadcast to every antenna (so caltable writers see the same shape
  regardless of mode); the per-antenna PWV anchor consequently emits
  identical `pwv[ant]` across antennas. This is deliberate.
- **Time axis is per-scan-local and NaN-padded.** No MultiIndex.
  `ds["flag"]` masks the pad and any flagged sample.
- **Flag-respecting projections.** `schema.apply_flags(ds, var)`
  returns `ds[var].where(~ds.flag)` (with `flag` reduced over any
  dims missing from `var`). All reductions over `time` must go
  through this helper; touching `ds[var]` directly silently
  contaminates the reduction with NaN-padding and flagged samples.
- **Pure-Python `xr.Dataset` is the science output.** Caltable writers
  (§9.2) are optional, gated, and the only path that links
  `casatools`. `result.dataset.to_netcdf(...)` / `.to_zarr(...)` is
  the recommended archive format.

---

## 5. Physics and fit

### 5.1 Physics primitives (`physics.py`)

- `tsys_model(z_deg, T0, tau0, Twmt) = T0 + Twmt·(1 − exp(−τ₀/cos z))`.
- `k2nt(T_K, ν_Hz) = T·(hν/kT) / (exp(hν/kT) − 1)` — Nyquist
  (Rayleigh-Jeans) correction.
- `airmass(z_deg) = 1/cos(z)` — flat-earth, no refraction (matches v2.6).
- `weighted_mean_atm_T(T_surf_K) = 70.2 + 0.72·T_surf` — Bevis (1992).
  Used inside Stage A only as the fallback when the grid-derived
  `T_mean` is unavailable for a (scan, spw) cell.

### 5.2 Geometry (`geometry.py`)

`zenith_angle(el_encoder_rad) = 90° − rad2deg(el_encoder_rad)`,
vectorized. AZELGEO encoder elevation is geodetic; with refraction
disabled, no frame transform is needed.

### 5.3 Per-sample noise model

`sigma_Tsys` is added to the dataset by Stage A. Derivation: `Tsys =
(S/2)·T_c/D` with `S = switched_sum`, `D = switched_diff`,
`T_c = tcal_ref`. In steady state `D ≈ T_c`, so the dominant error
propagation gives

```
σ_Tsys ≈ √2 · Tsys² / (T_c · √(Δν · τ_int))
```

with `Δν` the per-spw bandwidth and `τ_int` the per-sample
`exposure_time`. The `Tsys / T_c` amplification (~10–60× for VLA
bands) is the physically essential part — dropping it would mis-scale
σ and trip 4σ residual rejection on most samples.

### 5.4 Stage A fit

Computes `Tsys = (S/2)/D · T_c` and `sigma_Tsys` per §5.3, then fits
the tipping curve. Two routings:

| Mode                    | Unit                   | Free parameters per fit                                   |
| ----------------------- | ---------------------- | --------------------------------------------------------- |
| `independent_tau`       | `(scan, antenna, spw)` | `T0_R, T0_L, τ_z`                                         |
| `independent_tau_solve` | `(scan, spw)`          | `T0_R_a, c_R_a, T0_L_a, c_L_a` ∀ passing antenna; `τ_z`  |

Both use `scipy.optimize.least_squares` with `soft_l1` robust loss
(`f_scale = 3.0`) and σ-weighted residuals
`r_i = (Tsys_i − model_i) / σ_Tsys,i`.

**`independent_tau_solve` Jacobian.** Sparse CSR — block-diagonal in
the per-antenna `(T0, c)` columns, dense in the shared `τ_z` column.
SciPy's TRF then dispatches `tr_solver='lsmr'`. The `independent_tau`
opacity fit uses the dense default; the matrix is too small for the
sparse path to pay.

**T_mean atmospheric input.** Per (scan, spw), noise-K
`T_mean(spw)` is sampled from each scan's `PwvGrid` at the profile's
native PWV via `anchor.compute_t_mean_grid` and Rayleigh-Jeans-
corrected through `k2nt`. NaN cells fall back to `k2nt(0.95 ·
T_surface)` per the v2.6 Bevis heuristic.

**Single physical bound set** (no escalation ladder):

- `T0 ∈ [0, 300 K]`
- `τ ∈ [0, 1.0]` (covers all VLA bands)
- `c ∈ [0.5, 2.0]` (Tcal correction; physical receiver prior)

**Initialisation.** `T0_init` comes from a per-(antenna, spw, pol)
linear `Tsys` vs. `airmass` y-intercept; `τ_init = 0.05`.

**Per-fit loop.**

1. σ-weighted `soft_l1` `least_squares` with the bounds above.
2. Drop time samples whose `χ² = ((Tsys − model)/σ)² > 16` (4σ) in
   *either* polarization; refit. Repeat up to 3 iterations or until
   no sample is rejected.
3. For `independent_tau_solve` the per-antenna screening above
   produces a passing set; one global LM follows over all passing
   antennas.

**Per-parameter error.** From `OptimizeResult.jac` via SVD:
`J̃ = U S Vᵀ → cov = σ² · V S⁻² Vᵀ`, with `σ² = Σr̃² / (n−p)`. The
diagonal entry on `τ_z` is stored as `tau_err`.

**Identifiability and QA gates.** Single tier, five reason strings:

- `ok` — fit converged, reduced χ² < 5, σ_τ / τ < 0.5.
- `poorly_identified` — fit converged but σ_τ / τ > 0.5. Fit values
  are still written; `fit_success` is False so downstream callers
  decide whether to consume them.
- `too_few_samples` — fewer than 3 unflagged time samples after
  rejection.
- `high_chi2` — reduced χ² ≥ 5 after iterative rejection.
- `fit_failed` — `least_squares` raised or refused to converge.

The legacy v2.6 cascade (`_STD_RESI`, freq-dependent `_stdtsys` bins,
`_DZ_MIN`, `_MZ_MIN`, mean-Tsys upper-limit, 3-pass bound escalation)
is gone. The behaviours it guarded are captured by the χ² gate and
the identifiability ratio.

**Parallelism.** When `n_workers > 1`, Stage A units are dispatched
through `multiprocessing.Pool` with the `spawn` start method. Each
worker exports `OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=OMP_NUM_THREADS=1`.
Package import sets these env vars by default (BLAS multithreading on
the per-fit matrix sizes here is pure overhead and 20× slower at
scale).

---

## 6. Stage B atmospheric anchor

A 1-D bounded scalar fit per antenna against the precomputed
`PwvGrid`. The grid is built once per scan during `build_atm_grids`
and is never recomputed inside Stage B.

**Inputs.** `tau_zenith(scan, ant, spw)`, `tau_err(scan, ant, spw)`,
the per-scan `PwvGrid` dict, and the spectral-window centre
frequencies.

**Cost** for antenna `a`:

```
χ²(PWV; a) = Σ_{scan,spw}  [(τ_z(scan, a, spw) − τ_grid(PWV, ν_spw)) / σ_τ(scan, a, spw)]²
```

Cells with non-finite `τ_z`, non-finite `σ_τ`, or `σ_τ ≤ 0` are
dropped. Minimisation is `scipy.optimize.minimize_scalar(method=
"bounded")` against the intersection of the configured PWV search
range and each contributing grid's `pwv_mm` axis.

**σ_PWV.** Cramér–Rao at the fitted PWV:

```
σ_PWV² = 1 / Σ_{scan,spw}  (∂τ_grid/∂PWV)² / σ_τ²
```

`∂τ_grid/∂PWV` is the analytical slope of the bilinear interpolant
(`PwvGrid.lookup_with_grad`). No Hessian inversion, no SVD.

**Per-mode semantics.** Under `independent_tau` the per-antenna
τ_z varies, so `pwv[ant]` differs across antennas — a true
per-antenna PWV. Under `independent_tau_solve` the Stage-A τ_z is
broadcast equal across antennas, so the per-antenna anchor returns
identical `pwv[ant]` (shared-PWV semantics fall out of the per-antenna
fit).

**Outputs on the dataset.**

- `pwv(antenna)`, `pwv_err(antenna)`.
- `am_freq_grid(frequency_dense)`, `am_tau(frequency_dense)` — a
  representative am τ(ν) slice sampled from the reference grid at
  the median fitted PWV (falling back to `pwv_unscaled_mm` if every
  antenna is NaN). Used for the plot overlay.

---

## 7. Atmospheric profile pipeline (contract)

Two stages, both attached to `TippingAnalysis`. Neither is invoked
inside the per-sample fit loop.

### 7.1 `attach_profile(ds, source, afgl_climatology)`

The **single** network-touching stage. Idempotent: re-running on a
dataset that already has `atm_pressure` is a no-op.

- **`source="open-meteo"` (default).** One HTTP call against
  `historical-forecast-api.open-meteo.com/v1/forecast` with
  `models=gfs_hrrr` and pressure-level variables (temperature,
  relative humidity, geopotential height) on the 1000 → 10 hPa
  coarse grid. The fetch covers the full observation UTC date span;
  the closest hourly slice is selected per scan. Transient failures
  retry with backoff (default 4 attempts at offsets 0, 5, 15, 45 s).
  Deterministic failure (no pressure-level data in the response —
  date predates the gfs_hrrr archive, ≈ 2021-03-23) bails to AFGL
  immediately without retrying.
- **`source="afgl"`.** Skip the network call entirely.
  `afgl_climatology="auto"` (default) resolves to
  `midlatitude_summer` / `midlatitude_winter` from the observation's
  median UTC month. Explicit names are passed through to
  `amwrap.Climatology`.

The fetched profile is clipped at a single surface pressure (median
of per-scan `weather_P` medians; per-scan variation is <2 hPa at the
VLA, well below am modelling precision) so `atm_level` is constant
across scans.

**Writes to `ds`:**

- Data vars `atm_pressure(atm_level)` (Pa, 1-D),
  `atm_temperature(scan, atm_level)` (K),
  `atm_h2o_vmr(scan, atm_level)` (dimensionless VMR).
- Attrs `atm_profile_source` (e.g. `"open_meteo"` or
  `"afgl_midlatitude_summer"`), `open_meteo_query` (lat, lon, time
  range, endpoint, model; absent on AFGL path), `surface_pressure_hPa`
  (per-scan provenance).

### 7.2 `build_atm_grids(pwv_step_mm, freq_step_Hz, n_workers)`

Precomputes one `PwvGrid` per scan from the attached profile and
stores them on `TippingAnalysis._grids[scan_id]`. Auto-calls
`fetch_atm_profile` if `atm_pressure` is not yet on the dataset.
Writes `ds.attrs["pwv_profile_source"][scan_id]` for provenance.

### 7.3 `PwvGrid` contract

A frozen dataclass — bilinear lookup table for
`τ_z(PWV, ν)` and zenith brightness temperature `Tb_z(PWV, ν)`:

```python
@dataclass(frozen=True)
class PwvGrid:
    pwv_mm: np.ndarray              # (n_pwv,) ascending PWV axis, mm
    freq_Hz: np.ndarray             # (n_freq,) ascending freq axis, Hz
    tau_z: np.ndarray               # (n_pwv, n_freq) zenith opacity, nepers
    tb_z: np.ndarray                # (n_pwv, n_freq) zenith Tb, K
    pwv_unscaled_mm: float          # PWV of the un-scaled input profile
    profile_source: str             # provenance label
```

Two read methods:

- `lookup(pwv_mm, freqs_Hz) -> (τ, T_mean_K)` — `T_mean` is the
  atmosphere-only mean brightness, derived from `tb_z` with the CMB
  contribution subtracted and divided by the absorbed fraction
  `(1 − e^{−τ})`.
- `lookup_with_grad(pwv_mm, freqs_Hz) -> (τ, T_mean, ∂τ/∂pwv,
  ∂T_mean/∂pwv)` — same plus the analytical slope of the linear
  interpolant in the PWV direction; used by Stage B's Cramér–Rao
  σ_PWV.

The grid is parameterised by `troposphere_h2o_scaling = pwv_mm /
pwv_unscaled_mm` in am; this means the same underlying profile drives
both Stage A's `T_mean` input (sampled at `pwv_unscaled_mm`) and
Stage B's PWV fit — there is no second am run downstream.

---

## 8. Flagging

A single entry point: `flags.apply(ds, online: bool, file: Path |
None) -> None` updates `ds["flag"]` in place.

- **Online flags (MS only).** Read `FLAG_CMD` rows whose `REASON`
  is **not** in `{ANTENNA_NOT_ON_SOURCE, SHADOW, CLIP_ZERO_ALL}` —
  the v2.6 inclusion contract. Each `COMMAND` field is parsed by a
  single regex into `(antenna_name, t_start, t_end)`. SDM has no
  `FLAG_CMD`; the online path is a no-op there.
- **User file.** Lines of the form
  `antenna='ea05' spw='7' timerange='YYYY/MM/DD/HH:MM:SS~YYYY/MM/DD/HH:MM:SS'`.
  Single regex. `*`, empty, or missing means "all" for the
  corresponding field.
- **Application.** One interval-overlap mask broadcast across
  `(scan, antenna, spw, polarization, time)` via the `time_utc`
  coord:
  `(ds.time_utc >= t_start) & (ds.time_utc <= t_end)`. The v2.6
  four-case interval expansion (start-inside, end-inside,
  spanning, etc.) collapses to this single expression.

---

## 9. Outputs

### 9.1 Primary

`Result.dataset` — the canonical `xarray.Dataset`. Persistable via
`.to_netcdf(...)` or `.to_zarr(...)`.

### 9.2 Optional CASA caltables

Gated by the corresponding `caltable_*` argument being non-None.
Both keep `casatools.calibrater` / `casatools.table` as runtime
imports. "No CASA at runtime" in this project means we don't depend
on `buildmytasks` or a `casa` process; it does not mean zero
`casatools` imports.

- **Opacity caltable (`TOpac`).** `cb.createcaltable(name, "Real",
  "TOpac", True)`, then `tb.putcell` row-by-row with TIME, FIELD_ID,
  SPECTRAL_WINDOW_ID, ANTENNA1, ANTENNA2=−1, SCAN_NUMBER, FPARAM=τ₀,
  PARAMERR, FLAG, SNR. Schema matches v2.6 so downstream `applycal`
  works unchanged. Requires `tau_zenith`, `tau_err`, `fit_success`.
- **Tcal caltable (CALDEVICE clone).** Copy the source CALDEVICE
  subtable and write `[[tcal_fit_R, tcal_fit_L], [0., 0.]]` per
  `(antenna, spw)` cell — row 0 is the fitted noise-tube values, row
  1 (solar-filter slot) is zeroed for v2.6 output-format parity.
  Requires `tcal_fit` (i.e. `mode="independent_tau_solve"`).

### 9.3 Plots

`plot.PlotData(ds).save_all(out_dir=...)` writes one interactive
vega-altair `.html` per plot plus a top-level `index.html`. Hover
tooltips carry `(scan, antenna, spw, polarization)` identity;
colour encodes status (good / flagged / weighted mean), not
identity. Per successfully-fit `(scan, antenna, spw)`: an elevation
curve (Tsys vs. zenith angle, both pols, fitted curve overlaid).
Per scan with any successful fit: a τ vs frequency log-scatter with
optional am τ(ν) overlay from `am_freq_grid` / `am_tau`, and —
when `tcal_fit` actually differs from `tcal_ref` — a `T_cal` vs
frequency and a `c = T_cal,fit / T_cal,ref` plot.

`out_dir` is created with `Path.mkdir(parents=True, exist_ok=True)`.

---

## 10. Acceptance criteria

- All unit tests pass (`uv run pytest`).
- `uv run ruff check .`, `uv run ruff format --check .`,
  `uv run ty check src/tipopac` all clean.
- **Synthetic Stage-A fixtures.** Well-leveraged synthetic tipping
  curves recover injected τ and T0 to within the σ-propagated
  bounds; flat-ZA / high-noise scans return `poorly_identified`
  rather than spurious values. This is the regression test for the
  legacy geometric-QA-gate removal (no `_DZ_MIN`, no `_MZ_MIN`).
- **Stage-B σ_PWV self-consistency.** On synthetic tipping curves
  with known injected PWV and noise, Cramér–Rao σ_PWV from
  `anchor.anchor_pwv` matches the empirical Monte-Carlo σ across
  realisations within √(2/(n−1)) — standard test for correct error
  propagation.
- **v2.6 numerical-parity check.** Retained as a **smoke test only**
  with loose tolerances (≈ 10× the original `max(0.005, 0.05·τ_v26)`
  and 10× the 1 % Tcal tolerance). Systematic drift is expected:
  v2.6 uses unit weights + L2 + 2σ clip + 3-pass bound escalation +
  geometric `dz`/`min(z)` gates; this rewrite uses radiometer-eq σ +
  `soft_l1` + single-tier bounds + an identifiability ratio. Parity
  is a sanity floor, not a contract.
- Integration test on `data/tip_test.ms` runs end-to-end in all
  three pipeline modes (open-meteo, AFGL fallback, AFGL-forced).
  Open-meteo network calls are monkeypatched in CI via the fixture at
  `tests/integration/reference/open_meteo_response.json`; a separate
  `pytest -m network` test exercises the live endpoint.

---

## 11. Out of scope / explicitly retired

- **`tau_extrapolated(scan, spw_all)`.** The old per-spw am
  extrapolation written by the scalar `pwv_scaling` anchor. Removed
  with the single-anchor flow (commit `3e53b46`). Consumers read
  `tau_zenith` directly and use `PwvGrid.lookup` for off-grid spws.
- **`pwv_scaling` attr.** Replaced by the per-antenna `pwv(antenna)`
  data variable.
- **Legacy public modes** `global_tau`, `per_antenna_pwv`,
  `shared_pwv`. Gone. `tcal_solve` is retained as the private
  Stage-A backend for `independent_tau_solve`.
- **am-based forward fit (Stage 2 of `model_refactor.md`).** Reverted.
  PWV is fitted post-hoc against the per-spw τ samples (§6), never as
  an inner LM parameter.
- **Replacement of Bevis `weighted_mean_atm_T` as the default Twmt.**
  Stage A uses grid-derived `T_mean` by default; Bevis is the
  NaN-cell fallback only.
- **Pure-Python CASA-table writer.** Caltable output keeps
  `casatools.table`.
- **Standalone CLI entry point** (`python -m tipopac ...`). Easy to
  add but not in scope.
- **Multi-observatory support.** VLA-only assumptions (dual circular
  pol, CALDEVICE shape, AZELGEO encoder) are baked in.
- **Negative-τ mean fallback** (v2.6 `task_tipopac.py:1509–1511`).
  The rewrite returns the fitted τ as-is and lets `fit_reason`
  inform downstream decisions.
- **`besta` lowest-σ antenna rescue** (v2.6 lines 1480–1492).
  The rewrite emits no τ for a `(scan, spw)` where every antenna
  fails QA.
- **Upper-atmosphere splicing above 10 hPa.** Needed for accuracy at
  the top of K/Q bands; handled inside `amwrap` (external work).
  `tipopac` will accept the spliced profile transparently when that
  support lands.
