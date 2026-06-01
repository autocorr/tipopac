# `tipopac` — Design Document

A clean, importable Python rewrite of the CASA `tipopac` task. Estimates VLA
zenith opacity and noise-diode temperatures from `DO_SKYDIP` tipping-scan data,
without requiring a CASA runtime.

> **Scope of this document.** This is the implementation contract for v1. It
> commits to specific API shapes, data schemas, algorithms, and acceptance
> criteria. Sections labelled "Deferred" are explicitly out of scope for v1.

---

## 1. Overview & goals

### What we preserve from `tipopac_v2.6`

- The physical model:
  `Tsys = T0 + Twmt' * (1 − exp(−τ₀/cos z))`, with Twmt' the Nyquist-corrected
  weighted-mean atmospheric temperature.
- The three solver configurations (per-antenna τ, global τ, global τ + Tcal
  correction).
- VLA-specific assumptions (dual circular R/L polarization, two-row CALDEVICE,
  AZELGEO pointing encoder, MS/SDM scan-intent `*DO_SKYDIP*`).
- The data-quality gates (delta-zenith-angle, Tsys upper limits, σ-clipping)
  applied before a fit is accepted.

### What we change

- No `buildmytasks`/CASA-task wrapper. The module is plain Python, imported as
  `tipopac`.
- `casatools`/`casatasks` are used as ordinary library imports for table I/O
  and the optional CASA-format caltable writers; they are **not** required for
  the science output (an `xarray.Dataset`).
- Atmospheric modelling moves from `casatools.atmosphere` to Scott Paine's `am`
  via the local `amwrap` Python wrapper, fed by vertical profiles from
  open-meteo (`openmeteo-requests`) with amwrap's bundled AFGL climatologies as
  the offline fallback.
- The in-memory representation is a single canonical `xarray.Dataset` produced
  by either an MS reader or an SDM reader (`sdmpy`).
- Modern Python practice: type hints throughout, `ty` for type-checking, `ruff`
  for lint/format, `pytest` for unit + integration coverage.

### What is deferred (not in v1)

- Replacing the simple 2-parameter Tsys fit with an am-based forward model.
- Using am to compute Twmt' (kept on the Bevis 1992 empirical relation for v1).
- A pure-Python CASA-caltable writer (v1 keeps `casatools.table` for that
  output path).

---

## 2. Public API

Two surfaces. Pick the one that matches the call site.

```python
# --- functional one-shot ---
from tipopac import tipopac, Result

result: Result = tipopac(
    path,                                            # MS or SDM (auto-detected)
    *,
    mode="tcal_solve",                               # "tau_per_antenna" | "global_tau" | "tcal_solve"
    flags_online=True,
    flags_file=None,
    atm_model=True,                                  # run am + open-meteo extrapolation
    atm_profile_source="open-meteo",                 # "open-meteo" | "afgl"
    afgl_climatology="midlatitude_summer",           # used as fallback or when forced
    plot_dir=None,                                   # if set, write PNGs here
    caltable_opacity=None,                           # if set, write CASA TOpac table
    caltable_tcal=None,                              # if set, write CALDEVICE-style table
)

# --- class-based for staged / notebook use ---
from tipopac import TippingAnalysis

ta = TippingAnalysis.from_path("data/tip_test.ms")
ta.apply_flags(online=True, file=None)
ta.fit(mode="tcal_solve")
ta.extrapolate(atm_profile_source="open-meteo")
ta.plot(out_dir="plots/")
ta.write_caltables(opacity="z.cal", tcal="t.cal")
result = ta.result
```

`Result` is a small dataclass:

```python
@dataclass(frozen=True)
class Result:
    dataset: xr.Dataset          # the canonical schema (§5) populated with fit outputs
    mode: str                    # the fit mode used
    input_path: Path
    input_format: Literal["ms", "sdm"]
    software_versions: dict[str, str]   # tipopac, casatools, sdmpy, amwrap, am
```

All other state (per-scan fit success, Tcal corrections, the am extrapolation,
PWV scaling) lives inside `Result.dataset` per §5.

`frozen=True` only freezes the dataclass field bindings, not the underlying
`xr.Dataset`. The staged API (`apply_flags`, `fit`, `extrapolate`) mutates
`Result.dataset` in place. Callers that need an unchanging snapshot should
take `result.dataset.copy(deep=True)`.

---

## 3. Module layout

```
tip_rewrite/
├── DESIGN.md                            # this document
├── pyproject.toml                       # ruff, ty, pytest, deps
├── data/                                # symlink → ../data/, holds test MS
├── amwrap/                              # local checkout of github.com/autocorr/amwrap
├── tipopac_v2.6/                        # legacy reference, kept read-only
├── src/tipopac/
│   ├── __init__.py                      # re-exports `tipopac`, `TippingAnalysis`, `Result`
│   ├── api.py                           # one-shot function + TippingAnalysis class
│   ├── readers/
│   │   ├── base.py                      # TippingReader Protocol (§4)
│   │   ├── ms.py                        # MSReader (casatools.table)
│   │   └── sdm.py                       # SDMReader (sdmpy)
│   ├── schema.py                        # build + validate the canonical xr.Dataset
│   ├── flags.py                         # online + user-file flag parsing (§8)
│   ├── geometry.py                      # astropy-based zenith-angle helpers
│   ├── physics.py                       # k2nt, airmass, Tsys model
│   ├── fit.py                           # three fit modes (§6); scipy.optimize.least_squares
│   ├── atmosphere.py                    # am + open-meteo + AFGL fallback (§7)
│   ├── caltables.py                     # optional CASA caltable writers
│   └── plot.py                          # per-(scan,antenna,spw) panels with am overlay
└── tests/
    ├── unit/                            # synthetic-data fit, schema, flag, atm tests
    └── integration/                     # full-pipeline test on data/tip_test.ms
```

`src/`-layout matches the existing repo skeleton (`src/tipopac/` already
exists). Tests sit at top-level `tests/` so they are not shipped.

---

## 4. Reader abstraction

A single Protocol; two concrete implementations; one dispatcher.

```python
# src/tipopac/readers/base.py
from typing import Protocol, ClassVar

class TippingReader(Protocol):
    """Parse a tipping-data source into the canonical xarray.Dataset (§5)."""

    @classmethod
    def supports(cls, path: Path) -> bool: ...

    @classmethod
    def from_path(cls, path: Path) -> "TippingReader": ...

    def read(self) -> xr.Dataset: ...
```

`tipopac.api` walks the registered reader classes; for the first whose
`supports(path)` returns True it calls `R.from_path(path).read()`. Both
construction and dispatch are part of the Protocol so the typechecker can
verify the chain end-to-end. Heuristics for `supports`:

- **MSReader.supports**: `path` is a directory containing `table.dat` and a
  `SYSPOWER/` subtable.
- **SDMReader.supports**: `path` is a directory containing `ASDM.xml`.

### SDM ↔ MS column mapping

The two readers must converge on the §5 schema. The mapping below is the
implementation contract.

| MS subtable / column                                  | SDM table                          | sdmpy access pattern                                                |
| ----------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------- |
| `ANTENNA.NAME`                                        | `Antenna.xml`                      | `sdm['Antenna'][i].name`                                            |
| `SPECTRAL_WINDOW.REF_FREQUENCY/NUM_CHAN/TOTAL_BANDWIDTH` | `SpectralWindow.xml`            | `sdm['SpectralWindow'][spw_id]`                                     |
| `POINTING.TIME/ENCODER`                               | `Pointing.xml`                     | `sdm['Pointing'][ant_id, time_id]`                                  |
| `SYSPOWER.TIME/SWITCHED_DIFF/SWITCHED_SUM`            | `SysPower.xml`                     | `sdm['SysPower'][ant_id, feed_id, spw_id]`                          |
| `CALDEVICE.NOISE_CAL`                                 | `CalDevice.xml`                    | iterate rows of `sdm['CalDevice']`; row key is (antennaId, feedId, spectralWindowId); load 0 = noise tube; receptor R = column 3, receptor L = column 3+ncols (parity verified by `tests/unit/test_sdm_reader.py::test_sdm_ms_parity_tcal_ref`) |
| `WEATHER.TIME/TEMPERATURE/REL_HUMIDITY/PRESSURE`      | `Weather.xml`                      | `sdm['Weather'][station, time]`                                     |
| scan intent `*DO_SKYDIP*` (via STATE/SOURCE)          | `Scan.xml` + `Subscan.xml`         | `sdm['Scan'][i].scanIntent` / `sdm['Subscan'][i,j].subscanIntent`   |
| `FLAG_CMD` (online flags)                             | — (no SDM equivalent)              | `SDMReader` returns an empty flag command set                       |

If `MSReader.read()` and `SDMReader.read()` ever produce datasets that diverge
in dims, coords, dtypes, or units, the abstraction has failed and `schema.validate()`
will catch it in CI.

---

## 5. Canonical `xarray.Dataset` schema

Single in-memory representation produced by either reader, mutated in place by
flagging, fitting, and extrapolation.

```text
Dimensions
  scan          (n_scans,)        int       DO_SKYDIP scan numbers
  antenna       (n_antennas,)     str       e.g. "ea05"
  spw           (n_spw,)          int       spectral-window id
  polarization  (2,)              str       "R", "L"
  time          (n_time,)         int       per-scan local sample index
                                            (0..max_n_samples−1); ragged
                                            across scans, padding masked by
                                            the flag array; absolute time
                                            lives in the time_utc coord below
  pressure_level (n_levels,)      float     hPa, present only after .extrapolate()

Coords
  frequency(spw)               Hz             spw reference frequency
  bandwidth(spw)               Hz             spw total bandwidth
  antenna_position(antenna,3)  m              ITRF X, Y, Z
  scan_time_start(scan)        s              UTC seconds (MJD-sec)
  scan_time_end(scan)          s              UTC seconds (MJD-sec)
  time_utc(scan, time)         float64        non-dim 2D coord; UTC MJD-seconds
                                              per sample, NaN where the time
                                              axis is padded. Used for
                                              absolute-time queries (§8
                                              user-file flag matching).

Data variables — inputs (filled by readers)
  switched_diff(scan, antenna, spw, polarization, time)   float32
  switched_sum (scan, antenna, spw, polarization, time)   float32
  zenith_angle(scan, antenna, time)                       float32  deg
  tcal_ref    (antenna, spw, polarization)                float32  K   (CALDEVICE row 0)
  weather_T   (scan, time)                                float32  K   surface kinetic T (interp)
  weather_P   (scan, time)                                float32  Pa
  weather_RH  (scan, time)                                float32  (0–1, fractional RH)
  flag        (scan, antenna, spw, polarization, time)    bool

Data variables — fit results (filled by fit.py)
  Tsys        (scan, antenna, spw, polarization, time)    float32  K
  tau_zenith  (scan, antenna, spw)                        float32  nepers
  tau_err     (scan, antenna, spw)                        float32
  T0          (scan, antenna, spw, polarization)          float32  K
  tcal_fit    (scan, antenna, spw, polarization)          float32  K
  fit_success (scan, antenna, spw)                        bool
  fit_reason  (scan, antenna, spw)                        str      ""/"ok" or QA failure code

Data variables — am extrapolation (filled by atmosphere.py)
  tau_extrapolated(scan, spw_all)                         float32  nepers, every spw in source
  am_freq_grid                                            (frequency_dense,) Hz
  am_tau                                                  (frequency_dense,) nepers

Attrs
  source_path        : str
  source_format      : "ms" | "sdm"
  observatory        : "VLA"
  mode               : str (the fit mode used)
  software_versions  : dict[str, str]
  atm_profile_source : "open-meteo" | "afgl"
  afgl_climatology   : str
  pwv_scaling        : float | None     (the anchor-fit result; §7)
  open_meteo_query   : dict | None      (provenance: lat, lon, time, endpoint)
```

**Representation choices.**

- `tau_zenith` keeps an `antenna` dim even in the non-per-antenna modes; values
  broadcast equal across antennas. The dim cost is trivial and downstream code
  simplifies. In `global_tau` and `tcal_solve`, `tau_zenith` is written to **all**
  antennas when the global fit succeeds — including antennas that failed per-antenna
  screening. An antenna excluded by screening has `fit_success=False` and `T0`/
  `tcal_fit` set to NaN, but `tau_zenith` is still populated with the global τ₀ so
  downstream caltable writers can populate every antenna row without special-casing.
- `tau_extrapolated` is populated for every spw in the source (§7) — including
  those with a successful per-(scan, antenna) fit — so the am curve can serve
  as a QA cross-check overlay. Downstream consumers should prefer `tau_zenith`
  over `tau_extrapolated` for `(scan, spw)` where `fit_success=True`;
  `tau_extrapolated` is the authoritative value only for fit-failure / no-data
  spws.
- Times are kept per-scan-local on a single padded `time` axis; the `flag`
  array masks the pad and any flagged sample. No MultiIndex. Reductions over
  `time` go through `schema.apply_flags(ds, var)` (defined below) so the pad
  and the flag array are always respected together.

`schema.py` provides two helpers used throughout the package:

- `validate(ds)` asserts dims/coords/dtypes; called by both readers before
  returning, and by tests.
- `apply_flags(ds, var: str) -> xr.DataArray` returns `ds[var].where(~ds.flag)`
  — the flag-respecting view used by every reduction over the `time` axis
  (Tsys statistics, residual σ, σ-clip masking, etc.). Skipping the helper
  and touching `ds[var]` directly silently contaminates the reduction with
  NaN-padding and flagged samples.

---

## 6. Physics and fit

### 6.1 Physics primitives (`physics.py`)

- `tsys_model(z_deg, T0, tau0, Twmt) -> ndarray` — exact v2.6 formula.
- `k2nt(T_K, nu_Hz) -> ndarray` — Nyquist correction:
  `T · (hν/kT) / (exp(hν/kT) − 1)`.
- `weighted_mean_atm_T(T_surf_K) -> ndarray` — Bevis 1992:
  `70.2 + 0.72 · T_surf`. **Default for v1.** An alternative
  `weighted_mean_atm_T_from_am(profile, freq)` is reserved as a v2 upgrade path
  and is not the default.
- `airmass(zenith_angle_deg) -> ndarray` — `1 / cos(z)` (flat-earth, matches
  v2.6; no refraction correction).

### 6.2 Geometry (`geometry.py`)

`zenith_angle(el_encoder_rad) -> deg = 90.0 - np.rad2deg(el_encoder_rad)`,
vectorized over `(scan, antenna, time)`. AZELGEO encoder elevation is the
geodetic elevation; with refraction disabled (§6.1) no frame transform is
required, so the legacy CASA `me.measure(..., 'AZEL')` step collapses to a
single subtraction. No `astropy` dependency for this module.

### 6.3 Fit modes (`fit.py`)

| `mode`              | Solved per       | Free parameters                                          |
| ------------------- | ---------------- | -------------------------------------------------------- |
| `"tau_per_antenna"` | (scan, antenna, spw) | T0_R, T0_L, τ₀                                       |
| `"global_tau"`      | (scan, spw)      | T0 for each (antenna, pol) plus a single τ₀              |
| `"tcal_solve"`      | (scan, spw)      | T0 for each (antenna, pol), per-antenna Tcal correction, τ₀ |

`"tcal_solve"` corresponds to v2.6's `calcTcals=True` and (matching v2.6) forces
per-antenna τ off — a single shared τ₀ is solved alongside the Tcal corrections.

All three modes use `scipy.optimize.least_squares` with explicit bounds.
Covariance comes from `OptimizeResult.jac` via the SVD-based formula:
`J = U S Vᵀ → cov = σ² · V S⁻² Vᵀ` with `σ² = Σr² / (n − p)` and singular
values below a small relative threshold clipped to avoid the rank-deficient
case. Per-parameter error is `PARAMERR = sqrt(diag(cov))`; the τ entry is
stored as `tau_err` in the dataset, and the TOpac caltable's `SNR` column
is `|τ| / tau_err`. For `tcal_solve`, a **3-pass escalation** is used, matching v2.6's
`fitting_Tcal` (`task_tipopac.py:161–236`):

| Pass | c bounds | Escalate when |
|------|----------|---------------|
| 1 | `[0.8, 1.2]` | tau at bound, or > 8 params within 2% of bound |
| 2 | `[0.7, 1.2]` | tau at bound, or > 10 params within 2% of bound |
| 3 | `[0.7, 1.3]` | — (final pass, always accepted) |

Pass 1's tight c window prevents convergence to a spurious high-c local
minimum (c ≈ 1.14, τ ≈ 0.031) that the optimizer reaches when starting from
τ = 0.2 with wide bounds. The τ initial guess is the median of per-antenna
τ₀ values from the prescreening step (rather than a flat 0.2).

`global_tau` uses a single `least_squares` call (no retry needed; no c
parameters whose bounds can trap the optimizer). Common bounds:

- `T0 ∈ [0, trUpperLimit]` where `trUpperLimit = 300 K`
  (`task_tipopac.py:1397`).
- `τ ∈ [0.0, tauUpperLimit]`: `tauUpperLimit = 0.4` for spw center > 45 GHz,
  else `0.3` (`task_tipopac.py:1398–1412`).

**`tau_per_antenna` fit flow.** The fit is run **twice** per `(scan, antenna, spw)`
as a clip-and-refit sequence: initial `least_squares` → flag samples with
`|residual| > 2σ` → re-run `least_squares` with the clipped samples masked.
The refit result is the published fit. This mirrors v2.6 (`task_tipopac.py:1421–1431`)
and is essential for matching v2.6's τ numerics on noisy scans.

**`global_tau` / `tcal_solve` fit flow.** These modes have a two-phase structure:
1. **Per-antenna screening** (using the no-Tcal `tau_per_antenna` residuals). For
   each antenna a two-pass clip-and-refit is run to identify and remove 2σ outliers
   and to apply all QA gates (dz, mz, tsys_std, tsys_upper, resid_clip). Antennas
   that fail any gate are excluded from the global fit and their failure reason is
   stored in `fit_reason`; `tau_zenith` is still broadcast to them on global-fit
   success (see §5 note above). Twmt is taken from the first passing antenna's valid
   samples (matching v2.6's `getTruw` flag).
2. **Global least_squares** over all passing antennas. Parameter vector:
   - `global_tau`: `[T0_R_0, T0_L_0, ..., T0_R_{N-1}, T0_L_{N-1}, τ₀]` (2N+1 params)
   - `tcal_solve`: `[T0_R_0, c_R_0, T0_L_0, c_L_0, ..., τ₀]` (4N+1 params),
     where `c` is the Tcal correction multiplier (model: `Tsys_meas = (T0+pred)/c`).
   `tcal_solve` uses 3-pass escalation (see bounds table above); `global_tau`
   uses a single call.

QA gates (from v2.6) move into a single `quality_check(...)` function
returning a typed reason string from the enum below. Thresholds match v2.6
(`task_tipopac.py:1396–1461`):

- `ok`
- `too_few_samples` — fewer than `minTipInts = 3` unflagged time samples in
  either polarization (`task_tipopac.py:1252`).
- `dz_too_small` — Δ(zenith angle) ≤ 10° across the scan.
- `mz_too_small` — mean zenith angle ≤ 30°.
- `tsys_std_too_high` — per-sample Tsys σ exceeds a frequency-dependent
  threshold (`task_tipopac.py:1398–1410`):
    - spw center ≤ 18 GHz: 5 K
    - 18 < spw center ≤ 40 GHz: 15 K
    - spw center > 40 GHz: 20 K
- `tsys_upper_limit` — mean Tsys ≥ `trUpperLimit = 300 K`.
- `resid_clip` — refit residual σ still exceeds `stdResi = 3 K` after the
  two-pass clip-and-refit (`task_tipopac.py:1396`).
- `fit_failed` — `least_squares` raised or refused to converge.

This replaces v2.6's per-antenna QA composite at
`task_tipopac.py:1452–1461` plus the bound-hit / min-samples checks
scattered nearby. The reason string is stored in `ds.fit_reason`.

---

## 7. Atmospheric model: am + open-meteo + AFGL fallback

The legacy `doModel` feature is replaced by an anchored am extrapolation. The
am model is run **once per analysis**, never inside the per-scan fit loop.

### 7.1 Pipeline

1. **Fit per-spw τ from the data** (§6). These fitted τ values are the
   ground truth that the am model is anchored to.
2. **Build the vertical atmospheric profile.**
   - **`atm_profile_source="open-meteo"` (default).** Query open-meteo via
     `openmeteo_requests` for the observation's lat/lon and time, pulling
     `pressure_level` variables (temperature, relative humidity,
     geopotential_height) on the GFS / HRRR pressure-level grid (1000 → 10
     hPa). Use the `/v1/forecast` endpoint for recent dates and the archive
     endpoint for historical timestamps (chosen at call time from the scan
     UTC). Convert RH → H₂O volumetric mixing ratio via Magnus-Tetens or
     equivalent.

     **Upper-atmosphere splicing** (above 10 hPa) is required for correct
     opacity at Q-band and higher, but is handled inside `amwrap` itself
     (extension to be added externally to this project). `tipopac` exposes a
     pass-through argument once that support lands; v1 calls amwrap with the
     open-meteo profile as-is and accepts reduced accuracy at the top of K/Q
     bands until splicing is enabled.
   - **`atm_profile_source="afgl"` (forced) or open-meteo failure.** Use
     `amwrap.Climatology(afgl_climatology)` (default `"midlatitude_summer"`).
3. **Run am.** Construct `amwrap.Model(pressure=..., temperature=...,
   mixing_ratio={"h2o": ...})` over the full frequency range covered by all
   spws (extend a few %  past the band edges).
4. **Anchor algorithm — fit a single `pwv_scaling` scalar.**
   - For each (scan, spw) with a successful τ fit, evaluate the am-predicted
     zenith opacity τ_am(ν_spw, pwv_scaling) using
     `model.troposphere_h2o_scaling = pwv_scaling`.
   - Minimise
     `Σ (τ_fit − τ_am(scaling))² / τ_err²`
     over `pwv_scaling` (1-D scalar fit, `scipy.optimize.minimize_scalar` with
     a sensible positive bound, e.g. `[0.1, 5.0]`).
5. **Extrapolate.** Re-run am with the fitted scaling and report
   `tau_extrapolated[scan, spw_all]` at every spw in the source — including
   spws with a successful per-(scan, antenna) fit, so the am curve can serve
   as a cross-check overlay (see §9.3). Downstream consumers prefer
   `tau_zenith` where `fit_success=True` and fall back to `tau_extrapolated`
   only for fit-failure / no-data spws (§5). Also store the dense am
   frequency grid (`am_freq_grid`, `am_tau`) for plotting.
6. **Fallback policy.** If the open-meteo client raises or times out (5 s),
   log a warning, set `atm_profile_source` in attrs to `"afgl"`, and re-run
   step 2 with the AFGL climatology. The anchor algorithm is identical.
7. **Provenance.** `pwv_scaling`, `atm_profile_source`, `afgl_climatology`,
   and `open_meteo_query` (a dict of lat/lon/time/endpoint) are written to
   `Result.dataset.attrs`.

### 7.2 Why this anchor design

The fit (§6) is the only place τ is determined from observation; am is used
solely to interpolate / extrapolate across frequency. A scalar PWV-scaling
anchor is enough degrees of freedom to align an AFGL or open-meteo profile
to the actual sky, without dragging am into the per-scan least-squares
problem (which would be slow, harder to debug, and would couple v2.6's
validated numerics to an external model we're introducing for the first time).

---

## 8. Flagging

A single `flags.apply(ds, online: bool, file: Path | None) -> xr.Dataset`.

- **Online flags.** Read `FLAG_CMD` rows whose `REASON` is in the v2.6
  inclusion list — i.e. **exclude** `ANTENNA_NOT_ON_SOURCE`, `SHADOW`,
  `CLIP_ZERO_ALL` (the inclusion-list query at `task_tipopac.py:886`; parse
  of the `COMMAND` string into `(antenna, time_start, time_end)` at lines
  887–896). The rewrite uses one regex over `COMMAND`.
- **User file.** Each line is
  `antenna='ea05' spw='7' timerange='YYYY/MM/DD/HH:MM:SS~YYYY/MM/DD/HH:MM:SS'`.
  Single regex; `*`, empty, or missing means "all" for that field.
- **Application.** One interval-overlap function broadcasting against the
  `(scan, antenna, spw, polarization, time)` `flag` array. Matching uses the
  `time_utc(scan, time)` coord (§5): a single
  `(ds.time_utc >= t_start) & (ds.time_utc <= t_end)` mask broadcast across
  the remaining data dims. v2.6's four-case interval expansion at
  `task_tipopac.py:1116–1199` (start/end-inside, start-before-end-inside,
  start-inside-end-after, spanning) collapses to that single call.
- SDM input: only the user-file path is exercised (SDM has no `FLAG_CMD`).

---

## 9. Outputs

### 9.1 Primary: the `xarray.Dataset` inside `Result`

Callers can `result.dataset.to_netcdf(...)` or `.to_zarr(...)` for archive.

### 9.2 Optional CASA caltables

Both gated by the corresponding `caltable_*` argument being non-None.

- **Opacity caltable (`TOpac`).** Created via `casatools.calibrater`
  (`cb.createcaltable(name, "Real", "TOpac", True)`) and populated row-by-row
  with `casatools.table` (TIME, FIELD_ID, SPECTRAL_WINDOW_ID, ANTENNA1,
  ANTENNA2=-1, SCAN_NUMBER, FPARAM=τ₀, PARAMERR, FLAG, SNR). Schema matches
  the legacy task so downstream `applycal` works unchanged.
- **Tcal caltable (CALDEVICE clone).** Copy the source CALDEVICE subtable and
  write `np.array([[tcal_fit_R, tcal_fit_L], [0., 0.]])` per `(antenna, spw)`
  cell — row 0 holds the fitted noise-tube values, row 1 (solar-filter slot)
  is zeroed to match v2.6 output format (`task_tipopac.py:1633`).

> **Explicit caveat.** v1 keeps the `casatools.table` / `casatools.calibrater`
> import path for these two writers. "No CASA at runtime" in this project
> means we don't depend on `buildmytasks` or a `casa` process; it does not
> mean zero CASA modules in `sys.modules`. Building a pure-Python CASA-table
> writer is deferred (§12).

### 9.3 Plots

Matplotlib is a full dependency (not optional). One PNG per `(scan, antenna,
spw)` written under `plot_dir/`:

- Top panel: Tsys vs zenith angle for both polarizations, with the fitted
  curve overlaid.
- Bottom panel: am-predicted Tsys curve (using the anchored `pwv_scaling`)
  overlaid for cross-check.

The output directory is created with `Path.mkdir(parents=True, exist_ok=True)`
— never `os.system("mkdir ...")`.

---

## 10. Dev tooling and dependencies

### 10.1 `pyproject.toml`

Already in place; extend it during implementation to include the missing
scientific deps (`scipy`, `xarray`, `astropy`, `matplotlib`, `casatasks`). The
existing pieces of `pyproject.toml` that match this design:

- `requires-python = ">=3.13"`
- `dependencies`: `casatools`, `sdmpy`, `openmeteo-requests`, `amwrap`,
  `numpy`, `pandas`, `requests-cache`, `retry-requests`
- `[tool.uv.sources] amwrap = { git = "https://github.com/autocorr/amwrap" }`
  (the local `amwrap/` checkout is the editable source; the git URL is the
  reproducible pin)
- `dev`: `pytest`, `ruff`, `ty`, `ipython`, `ipdb`

To add during implementation:
`scipy>=1.13`, `xarray>=2024.10`, `matplotlib>=3.9`, `casatasks>=6.7.5`.
(`astropy` is not required; geometry collapses to a one-line subtraction per
§6.2.)

### 10.2 Type-checking

**`ty`** (not mypy). Already in the dev group. Run via `uv run ty check
src/tipopac` in CI.

### 10.3 Lint / format

`ruff` (already in dev). One config block in `pyproject.toml` covering both
lint and format.

### 10.4 Test runner

`pytest`. Two trees:

- `tests/unit/` — fast, no large fixtures.
- `tests/integration/` — needs `data/tip_test.ms`, gated by a `slow` marker.

---

## 11. Testing and validation

### 11.1 Unit tests

- **`tests/unit/test_physics.py`.** `k2nt` against the analytic limits
  (hν ≪ kT → T; hν → ∞ → 0). `tsys_model` boundary cases.
- **`tests/unit/test_fit.py`.** Synthetic Tsys curves with known T0/τ₀ at
  realistic noise; recover all three modes within tolerance. Includes a
  failure case (constant Tsys → fit refuses, returns `dz_too_small`).
- **`tests/unit/test_schema.py`.** Build a minimal `xarray.Dataset` by hand,
  pass `schema.validate(ds)`; mutate one dtype and confirm it raises.
- **`tests/unit/test_flags.py`.** Five overlap cases (no overlap, point,
  partial-left, partial-right, full containment). Confirms the one-overlap
  function replaces the v2.6 four-case block.
- **`tests/unit/test_atmosphere.py`.** Monkeypatch the open-meteo client to
  return a fixed profile; drive `atmosphere.anchor(τ_fit, τ_err, freqs)`
  with synthetic τ_fit generated from am at known `pwv_scaling`; confirm the
  anchor recovers the scaling to within 1%.

### 11.2 Integration test

`tests/integration/test_full_pipeline.py`:

1. **Atmosphere fixture.** Capture the open-meteo `pressure_level` response
   for `data/tip_test.ms`'s observation time once and commit it as
   `tests/integration/reference/open_meteo_response.json`. The integration
   test monkeypatches the `openmeteo_requests` client to return this payload
   so the pipeline result is deterministic across weather-model revisions.
2. Run the full pipeline on `data/tip_test.ms` for one DO_SKYDIP scan in each
   of the three modes (with the fixture engaged).
3. Compare against legacy v2.6 output **on first execution after
   implementation** (one-time side-by-side run inside CASA), then freeze the
   resulting dataset as a NetCDF reference in `tests/integration/reference/`.
4. Subsequent runs compare `Result.dataset` against the frozen reference.
5. A separate network-gated test (`pytest -m network`) exercises the live
   open-meteo call so endpoint-shape regressions are still caught, even
   though it does not participate in the reference comparison.

### 11.3 Acceptance criteria for v1 (numerical agreement vs v2.6)

- Zenith opacity per `(scan, spw)`: agreement within
  `max(0.005, 0.05 · τ_v26)` nepers — comparison scoped to `(scan, spw)`
  where v2.6 reports `ok` without invoking the negative-τ mean fallback,
  the `besta` rescue, or layer-2 / layer-3 bounds escalation (see §12).
  Marginal `(scan, spw)` that v2.6 rescued via those paths will not agree
  by construction and are excluded from the v1 acceptance comparison.
- Tcal corrections (mode `tcal_solve`): agreement within
  `max(0.01 K, 0.06 · |Tcal_v26|)` per `(antenna, spw, polarization)`.
  Rationale: the tipping-curve residual `Tsys − (T0 + Twmt·(1−exp(−τ·A)))/c` is
  near-degenerate in `(T0, c, τ)` at low airmass and low τ — sub-tolerance τ
  shifts (∼0.001 nepers, well within the 0.005-neper τ acceptance above) are
  absorbed by joint `(T0, c)` shifts at very small RMS penalty. Empirically and
  by back-of-envelope (Δc/c ≈ Twmt·A·Δτ / (T0 + Twmt·τ) ≈ 5–6% for typical VLA
  K/Ka/Q-band tipping geometry), 1% c agreement is not achievable when v1 and
  v2.6 take different optimizer trajectories (different `tau_init`, different
  pass-1 τ lower bounds, scipy version differences, analytical vs finite-diff
  Jacobian). The 6% threshold reflects the largest c spread induced by the
  τ-test tolerance and is matched by the worst observed v1↔v2.6 cell deviation
  on `tip_test.ms` (max 5.94%, 99.5th pct 5.5%).
- `pwv_scaling` is **not** compared to v2.6 (different ATM stack); it is
  sanity-checked by the unit test in §11.1.
- All unit tests pass; `ruff check`, `ruff format --check`, and
  `ty check` are clean.

If any of the above fail at integration time, the rewrite is not yet "v1
ready"; investigate before freezing the reference.

---

## 12. Deferred / explicitly out of scope for v1

- Replacing the 2-parameter Tsys fit with an am-based forward model
  (§7 chooses anchored extrapolation instead).
- Using am-derived `weighted_mean_atm_T_from_am` as the default (Bevis 1992
  retained in v1).
- A pure-Python CASA-table writer (v1 keeps `casatools.table`).
- A standalone CLI entry point (`python -m tipopac ...`); easy to add but not
  required.
- Multi-observatory support; v1 is VLA-only by assumption (dual circular pol,
  CALDEVICE shape, etc., are VLA-specific).
- **Negative-τ mean fallback.** v2.6 (`task_tipopac.py:1509–1511`) replaces
  a fitted `τ < 0` with `np.mean(tauTemp[tauTemp>0])` — the mean of the
  positive per-antenna τ values for the same `(scan, spw)` — and sets
  `efit[-1] = 3·np.std(tauTemp[tauTemp>0])`. The rewrite returns `τ < 0`
  as-is with the appropriate `fit_reason`, leaving the decision to
  downstream consumers.
- **`besta` lowest-σ antenna rescue.** v2.6 (`task_tipopac.py:1480–1492`)
  keeps the antenna with the smallest combined residual σ when every antenna
  fails QA for a `(scan, spw)`. The rewrite emits no τ for that
  `(scan, spw)` and marks `fit_success=False`; consumers can read
  `fit_reason` for the cause.
- ~~Multi-layer fit-retry escalation.~~ **Implemented in v1.** The `tcal_solve`
  global fit uses the same 3-pass escalation as v2.6 (`fitting_Tcal`,
  `task_tipopac.py:161–236`); see §6.3 for the bound table.
- **Upper-atmosphere splicing above 10 hPa.** Required for opacity accuracy
  at Q-band and above. Handled inside `amwrap` (work to be done externally
  to this project); `tipopac` exposes a pass-through argument once that
  support lands (§7.1 step 2).

---

## 13. Implementation milestone order

A suggested order so the dataset schema is exercised end-to-end early:

1. `schema.py` + `tests/unit/test_schema.py` — pin the contract first.
2. `readers/ms.py` for one scan on `data/tip_test.ms` — produces a real
   dataset that conforms to `schema.validate(ds)`.
3. `physics.py`, `geometry.py`, `fit.py` (mode `tau_per_antenna` first) +
   synthetic unit tests.
4. `flags.py` + unit tests.
5. Extend `fit.py` to `global_tau` and `tcal_solve` modes.
6. `caltables.py` (CASA-format writers).
7. `atmosphere.py` (open-meteo client + am anchor) + unit tests.
8. `plot.py`.
9. `readers/sdm.py` once a representative SDM is available.
10. `tests/integration/` reference dataset frozen.

Each milestone leaves the package green: `ruff`, `ty`, and `pytest -m "not slow"`
should pass at the end of every step.
