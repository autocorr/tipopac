# tipopac code tour

A module-by-module walkthrough of `src/tipopac/`, written for the project's author. Skip the physics primer; refer to `design/design.md` for the spec.

## Overview

The package is a clean-Python rewrite of CASA's `tipopac` task that takes a VLA Measurement Set (or SDM directory) of DO_SKYDIP scans and produces zenith opacity τ_z(ν, antenna, spw) plus, optionally, fitted Tcal. It exposes two public entry points in `api.py`:

- `tipopac(...)` — one-shot pipeline. Returns a frozen `Result` dataclass and (optionally) writes outputs to disk. Thin wrapper around the staged class — useful for scripts and the `Result.from_tipopac(...)` style call site.
- `TippingAnalysis(...)` — staged pipeline class for notebook use. Each stage is a named method (`from_path`, `apply_flags`, `fetch_atm_profile`, `build_atm_grids`, `fit`, `plot`, `weblog`, `write_caltables`); the underlying `xr.Dataset` lives at `self._ds` and is mutated in place. This is the right entry point when you want to peek at intermediate state in Jupyter.

A separate CLI entry, `python -m tipopac.summary`, bypasses the full pipeline entirely. It just opens the MS / SDM metadata, prints the DO_SKYDIP scan table (id / UTC start / band / SPW ids), and exits. Useful for picking which scans to feed `tipopac()` with.

The pipeline flow:

```
detect_reader → MSReader|SDMReader.read       →  schema-valid xr.Dataset
     ↓
flags.apply                                   (online FLAG_CMD + user text file)
     ↓
atmosphere.attach_profile                     (one open-meteo HTTP call per analysis)
     ↓
atmgrid.build_pwv_grid                        (one am-pool per scan; never per-sample)
     ↓
anchor.compute_t_mean_grid  →  fit.fit_dataset (Stage A: τ_z, T0, Tcal)
                                ↓
anchor.anchor_pwv (Stage B)  ← τ_z(scan, spw, ant)
     ↓
anchor.write_am_curve                         (dense τ(ν) sampled from same PwvGrid)
     ↓
plot.PlotData.save_all  +  weblog.build_weblog  +  caltables.write_{opacity,tcal}
```

### Fit modes

Three fit modes are exposed through `fit.fit_dataset(mode=...)` (resolved from `TippingAnalysis(..., mode=...)`):

- `tau_per_antenna` — each `(scan, ant, spw)` cell gets its own 3-parameter LM fit (`T0_R`, `T0_L`, `τ`). Stage B then anchors per-antenna PWV against the per-antenna τ_z(ν) curve.
- `tcal_solve` — joint fit per `(scan, spw)`: shared `τ` across antennas, per-antenna `(T0, c)` where `c = tcal_fit / tcal_ref`. This is the mode that yields the Tcal caltable.
- `independent_tau_solve` — degenerate per-antenna mode where each antenna's τ is treated independently but the antenna dim is retained for shape compatibility; `tau_zenith` and `pwv(antenna)` carry the antenna dim with broadcast-equal values. (See CLAUDE.md: "Antenna dim is retained even when degenerate.")

### Architectural invariants

From `CLAUDE.md` and `design/design.md`, these are the rules every contributor needs to respect; they show up implicitly throughout the modules below:

- The canonical `xr.Dataset` schema in `schema.py` is the contract. Both readers must produce datasets that pass `schema.validate`.
- The `time` axis is per-scan-local and NaN-padded. There is no MultiIndex; the boolean `flag` array masks the pad. Reductions over `time` MUST go through `schema.apply_flags(ds, var)`, never `ds[var]` directly.
- `am` (the atmospheric model) runs exactly once per analysis, inside `atmgrid.build_pwv_grid`. Stage B is a 1-D bounded scalar fit against the precomputed `PwvGrid` — no second am call.
- Online flag application is **one** broadcast interval-overlap expression, not the four-case interval expansion of v2.6 lines ~1117–1199.
- v2.6 numerical parity is a smoke test, not a contract. The rewrite uses radiometer-eq σ + `soft_l1` + single-tier bounds + an identifiability ratio in place of v2.6's unit-weight L2 + 2σ clip + 3-pass bound escalation. Drift is expected.

---

## `api.py` — pipeline glue

User-facing surface and the only module that wires every other module together. NetCDF/TSV writers are local helpers; everything else delegates.

The `TippingAnalysis` lifecycle is essentially a state machine: `__init__` constructs from kwargs; `from_path` reads the dataset; subsequent stage methods each set a flag (`_flags_applied`, `_atm_profile_attached`, `_atm_grids_built`, `_fit_done`) so that calling them out of order is either a no-op or raises. `write_outputs` is the convenience method that runs everything left to run and emits NetCDF + TSV + plots + weblog + (optionally) caltables into a single directory.

- `_software_versions` — `src/tipopac/api.py:26` — collect optional-dep version strings (numpy, xarray, scipy, astropy, amwrap, casatools, sdmpy) for `Result.provenance`. Best-effort; missing imports become `None`.
- `_coerce_attr_for_netcdf` — `src/tipopac/api.py:65` — map in-memory attr types (`dict`, `list`, `Path`, `None`) to NetCDF-encodable values. Recurses on dicts and lists; encodes `None` as the empty string with a sentinel attr.
- `_write_dataset_netcdf` — `src/tipopac/api.py:94` — sanitise object-dtype vars and attrs on a shallow copy, then `to_netcdf`. The copy avoids polluting the live dataset with sentinel attrs.
- `_write_model_opacity_tsv` — `src/tipopac/api.py:115` — write the Stage-B τ(ν) model curve (`am_freq_grid`, `am_tau`) as a two-column TSV for downstream consumers that don't want to crack NetCDF.
- `Result` — `src/tipopac/api.py:130` — frozen dataclass returned by both `tipopac()` and `TippingAnalysis.result`. Holds `dataset` plus `provenance` (input path, software versions, run timestamp, fit mode, PWV profile source).
- `tipopac` — `src/tipopac/api.py:141` — one-shot pipeline; instantiates `TippingAnalysis`, runs every stage, optionally writes outputs, returns `Result`.
- `TippingAnalysis` — `src/tipopac/api.py:232` — staged pipeline class; constructor captures all the fit/plot/output configuration but does no I/O.
- `TippingAnalysis.from_path` — `src/tipopac/api.py:246` — dispatch on the reader registry via `readers.detect_reader`, instantiate the reader with `(scans=, bands=)` selection, call its `read()`.
- `TippingAnalysis.apply_flags` — `src/tipopac/api.py:259` — delegate to `flags.apply(self._ds, online=..., file=...)`. Idempotent: subsequent calls no-op.
- `TippingAnalysis.fetch_atm_profile` — `src/tipopac/api.py:269` — idempotent open-meteo / AFGL profile attach via `atmosphere.attach_profile`. Records the profile source on `ds.attrs["pwv_profile_source"]`.
- `TippingAnalysis.build_atm_grids` — `src/tipopac/api.py:299` — loop scans, call `atmgrid.build_pwv_grid` once per scan with that scan's `(pressure, T, VMR)` slice. Result is a `dict[int, PwvGrid]` keyed by scan id and stashed on `self._grids_by_scan`.
- `TippingAnalysis.fit` — `src/tipopac/api.py:352` — orchestrates the Stage A ↔ Stage B handshake: `anchor.compute_t_mean_grid` → `fit.fit_dataset(t_mean=...)` (Stage A) → `anchor.anchor_pwv` (Stage B) → `anchor.write_am_curve`. All three downstream calls share the same `_grids_by_scan` dict — no second am invocation.
- `TippingAnalysis.plot` — `src/tipopac/api.py:403` — defer to `plot.PlotData(self._ds).save_all(plot_dir)`.
- `TippingAnalysis.weblog` — `src/tipopac/api.py:408` — defer to `weblog.build_weblog(plot_dir)`. Depends only on filenames the previous step wrote.
- `TippingAnalysis.write_caltables` — `src/tipopac/api.py:413` — opt-in TOpac / CALDEVICE writers via `caltables.write_opacity` / `caltables.write_tcal`. Tcal write requires `mode="tcal_solve"`.
- `TippingAnalysis.write_outputs` — `src/tipopac/api.py:426` — bundle NetCDF + TSV + plots + weblog + caltables into `output_dir`. Calls every remaining stage method, so it's safe even if you only ran up to `fit()`.
- `TippingAnalysis.result` (property) — `src/tipopac/api.py:453` — build the `Result` once `fit()` has run; raises if called too early.

---

## `schema.py` — the §5 contract

Defines the canonical `xarray.Dataset` everything else respects. Both readers must produce a dataset that passes `validate(ds)`; the fit, anchor, plot, and caltable modules assume that contract holds and access variables directly.

The dim layout is `(scan, antenna, spw, polarization, time)` plus `xyz`. `polarization` is fixed to `("R", "L")`. The `time` dim is **per-scan-local** — each scan's data sits at indices `[0, n_scan_samples_i)` in the time axis, and unused tail entries are NaN; the boolean `flag` array masks both online-flagged samples and the NaN pad. `apply_flags` is the mandatory accessor: it OR-reduces `flag` over dims a target var lacks (e.g. `weather_T(scan, time)` doesn't carry `antenna/spw/pol`, so flags get collapsed), then returns `ds[var].where(~flag)`. Any reduction over `time` that bypasses `apply_flags` will silently fold NaN pad and flagged samples into the answer.

Input vars (required, full-rank `(scan, antenna, spw, polarization, time)`): `switched_diff`, `switched_sum`, `zenith_angle`, `tcal_ref(antenna, spw, polarization)`, `weather_T/P/RH(scan, time)`, `exposure_time(scan, time)`, `flag`. Required coords: `frequency`, `bandwidth`, `band` on `spw`; `antenna_position` on `(antenna, xyz)`; `scan_time_start/end` on `scan`; `time_utc` on `(scan, time)`.

- `SchemaError` — `src/tipopac/schema.py:25` — typed exception for §5 contract violations; preserves the var/coord name and dim context in its message.
- `_dtype_matches` — `src/tipopac/schema.py:115` — accept-equivalent-dtype check (object dtype special-cased, otherwise same subdtype + itemsize). Tolerates float32 vs float64 to keep the reader implementations free to choose.
- `_check_var` — `src/tipopac/schema.py:121` — verify a single var's dims tuple and dtype, raising `SchemaError` with context.
- `validate` — `src/tipopac/schema.py:140` — full §5 conformance check: required dims present and nonzero, `polarization` coord exactly `("R","L")`, all required coords/inputs present with correct dims/dtype, `flag` shape matches `switched_diff`, optional vars (if present) match their declared signature.
- `apply_flags` — `src/tipopac/schema.py:182` — flag-respecting projection: reduce `flag` over dims absent from `var` via `.any`, then `.where(~flag)`. The contract is "if it touches `time`, route through here."

---

## `readers/` — MS and SDM ingest

Both readers produce the same schema-valid dataset; the SDM↔MS column-mapping in `design/design.md §3` is the parity contract, not something to re-derive. The two reader implementations are deliberately structured in parallel so that the column-mapping table maps onto private helpers one-for-one (`_read_antenna`, `_read_spectral_window`, `_read_scan_meta`, `_read_caldevice`, `_read_pointing`, `_read_weather`, `_build_dataset`).

The SYSPOWER (MS) / SysPower (SDM) sub-table is the time axis. For each scan, the reader queries SYSPOWER for sample timestamps and switched-power values, then interpolates weather and pointing onto those timestamps via nearest-neighbour. This is why `weather_T/P/RH` lives on `(scan, time)` not its own time axis — there's only one time axis in the dataset and it's the SYSPOWER one.

### `readers/__init__.py` — lazy registry

The `casatools` / `sdmpy` imports are deferred so that `import tipopac` doesn't drag CASA into the process for SDM-only sessions (and vice versa).

- `_get_readers` — `src/tipopac/readers/__init__.py:10` — first-call import + cache of `MSReader`, `SDMReader`.
- `detect_reader` — `src/tipopac/readers/__init__.py:25` — pick the class whose `supports()` matches the given path.

### `readers/base.py` — shared interface

- `TippingReader` (Protocol) — `src/tipopac/readers/base.py:12` — `supports` / `from_path` / `read` contract both readers implement. Also requires `list_skydip_scans` for the `summary` CLI fast path.
- `SkydipScanInfo` (dataclass) — `src/tipopac/readers/base.py:24` — `(scan_id, start_mjd_s, spw_ids, bands)` summary record used by `summary`.

### `readers/ms.py` — CASA MS reader

Returns a schema-valid `xr.Dataset` with `source_format="ms"`; online flag application is left to `flags.apply` downstream so the reader can be tested in isolation. Unit conversions (hPa→Pa, %→fraction, degree→radian where needed) are documented in the module docstring and centralised in the `_read_*` helpers.

- `MSReader` — `src/tipopac/readers/ms.py:33` — public reader class.
- `MSReader.supports` — `src/tipopac/readers/ms.py:52` — look for `table.dat` and `SYSPOWER/` as the MS marker.
- `MSReader.from_path` — `src/tipopac/readers/ms.py:57` — canonical factory taking `scans`/`bands` selection.
- `MSReader.list_skydip_scans` — `src/tipopac/readers/ms.py:67` — lightweight metadata-only path used by `summary` (no syspower/pointing/weather load).
- `MSReader.read` — `src/tipopac/readers/ms.py:93` — orchestrate sub-table reads, apply selection, build dataset, run `schema.validate`.
- `_apply_selection` — `src/tipopac/readers/ms.py:147` — filter scans + SPWs by user request; informative `ValueError`s on empties, missing scan ids, or band-filter contract violations.
- `_read_antenna` — `src/tipopac/readers/ms.py:213` — antenna names + ITRF positions from `ANTENNA`.
- `_read_spectral_window` — `src/tipopac/readers/ms.py:227` — `(refFreq, totalBW, band-label)` per SPW; labels via `bands.band_for_spw_name`.
- `_read_scan_meta` — `src/tipopac/readers/ms.py:248` — DO_SKYDIP scan ids and per-scan SPWs + time windows from `msmetadata`. Filters out non-tipping scans before the heavy reads run.
- `_read_caldevice` — `src/tipopac/readers/ms.py:271` — noise-tube `tcal_ref(ant, spw, pol)`. Implements the v2.6 previous-spw fallback: when a `(ant, spw, pol)` cell is missing, fall back to the same antenna's previous spw value in band order. Avoids spurious NaN in the canonical dataset.
- `_read_pointing` — `src/tipopac/readers/ms.py:313` — per-antenna `(time, zenith_angle_deg)` from `POINTING.ENCODER[1]`. ENCODER[1] is elevation; `geometry.zenith_angle` converts to ZA.
- `_read_weather` — `src/tipopac/readers/ms.py:348` — `(time, T_K, P_Pa, RH_frac)` with documented hPa→Pa / %→fraction conversions.
- `_nearest_idx` — `src/tipopac/readers/ms.py:379` — nearest-time index lookup used by pointing-on-syspower alignment.
- `_build_dataset` — `src/tipopac/readers/ms.py:391` — allocate `(scan, antenna, spw, pol, time)` arrays, query SYSPOWER per scan, interpolate weather and pointing onto its timestamps, assemble canonical `xr.Dataset`. This is where the per-scan-local NaN-padded time axis is constructed.

### `readers/sdm.py` — SDM reader

Mirror of `MSReader`. The canonical dataset is identical apart from `source_format="sdm"` and an all-false `flag` (the SDM has no FLAG_CMD equivalent, so online flags are skipped — see `flags._apply_online_flags`). The SDM-specific complications (joining `Antenna` × `Station` for ITRF positions, locating `Station_0` for the WX monitor, runtime field-name detection on SysPower) are all hidden inside `_read_*`.

- `SDMReader` — `src/tipopac/readers/sdm.py:37` — public reader class.
- `SDMReader.supports` — `src/tipopac/readers/sdm.py:56` — look for `ASDM.xml`.
- `SDMReader.from_path` — `src/tipopac/readers/sdm.py:60` — factory with selection args.
- `SDMReader.list_skydip_scans` — `src/tipopac/readers/sdm.py:70` — `summary` fast path; opens the SDM and reads Scan / SpectralWindow / SysPower only.
- `SDMReader.read` — `src/tipopac/readers/sdm.py:99` — open SDM, run sub-table reads + selection, build the canonical dataset, validate.
- `_apply_selection` — `src/tipopac/readers/sdm.py:162` — same scan/band filter logic as the MS reader.
- `_read_antenna` — `src/tipopac/readers/sdm.py:228` — join `Antenna` and `Station` tables to recover names + ITRF positions; builds `ant_id_to_idx`.
- `_read_spectral_window` — `src/tipopac/readers/sdm.py:255` — `(refFreq, totalBW, band, id→idx)` per SpectralWindow row.
- `_read_scan_meta` — `src/tipopac/readers/sdm.py:273` — filter Scan rows for `DO_SKYDIP`; derive per-scan SPWs from SysPower in the time window (no scan↔SPW map in SDM metadata).
- `_read_caldevice` — `src/tipopac/readers/sdm.py:311` — noise-tube `tcal_ref` from `CalDevice.coupledNoiseCal` with the same previous-spw NaN fallback as MS.
- `_read_pointing` — `src/tipopac/readers/sdm.py:355` — per-antenna `(time, zenith_angle)` from `Pointing.encoder[:, 0, 1]`.
- `_read_weather` — `src/tipopac/readers/sdm.py:393` — filter Weather to `Station_0` (the WX monitor, not the per-pad stations); parse SDM's `timeInterval`.
- `_nearest_idx` — `src/tipopac/readers/sdm.py:431` — local duplicate of the MS-reader nearest-time helper.
- `_build_dataset` — `src/tipopac/readers/sdm.py:443` — derive per-scan timestamps from SysPower, allocate canonical arrays, populate weather/pointing/syspower. Detects the syspower interval field at runtime — older SDMs spell it `interval`, newer ones `duration` or `integrationTime`.

---

## `flags.py` — §8 online + user-file flags

Replaces the v2.6 four-case interval expansion (lines ~1117–1199) with one broadcast `(time_utc >= t_start) & (time_utc <= t_end)` over `(scan, antenna, spw, polarization, time)`. The trick: because `time_utc` lives on `(scan, time)` and the interval bounds are scalar MJD-seconds, the comparison broadcasts to `(scan, time)` and then OR-reduces into the full-rank `flag` array along the antenna/spw/pol slice the command targets. No case splitting, no per-interval loop allocations.

Online flags come from `FLAG_CMD` (MS only — SDM has no equivalent, so `_apply_online_flags` silently no-ops when `source_format != "ms"`); user flags come from a CASA-flagcmd-like text file with `antenna=`, `spw=`, `timerange=` clauses.

- `_ymd_to_mjd_sec` — `src/tipopac/flags.py:41` — parse `'YYYY/MM/DD/HH:MM:SS[.fff]'` → MJD-seconds (float64).
- `_parse_command` — `src/tipopac/flags.py:51` — regex-extract `(antenna, t_start, t_end)` from a `FLAG_CMD.COMMAND` string; returns `None` on mismatch (warned and skipped).
- `_parse_user_line` — `src/tipopac/flags.py:68` — parse a user-file line into `(antenna, spw, t_start, t_end)`; empty / `*` / legacy `-1` are wildcards.
- `_apply_interval` — `src/tipopac/flags.py:99` — OR the time-interval mask into the flag array for the selected `(antenna, spw)` slice, broadcasting `(n_scan, n_time)` to full rank. Unknown antenna/spw names silently no-op (the same FLAG_CMD entry may target antennas that aren't in the selection).
- `apply` — `src/tipopac/flags.py:142` — public entry: dispatch to online/user helpers; returns the same `ds`.
- `_apply_online_flags` — `src/tipopac/flags.py:169` — open `FLAG_CMD` via `casatools.table`, exclude `_REASON_EXCLUDE` (`ANTENNA_NOT_ON_SOURCE`, `SHADOW`, `CLIP_ZERO_ALL`) via TaQL — those reasons are routinely set by online systems and would flag everything if applied. Parse each surviving `COMMAND` and call `_apply_interval`.
- `_apply_user_flags` — `src/tipopac/flags.py:213` — line-by-line parse of the user flag file (skipping `#` / blanks), warn on unparseable lines, apply each interval.

---

## `atmosphere.py` — profile fetch

The only network-touching stage. Attaches a vertical atmospheric profile (`atm_pressure`, `atm_temperature`, `atm_h2o_vmr`, `surface_pressure_hPa`) on a single `pressure_level` coord. Two sources, in priority order:

1. **open-meteo `historical-forecast-api` with `models=gfs_hrrr`** (see project memory: pressure-level vars only available on this endpoint, archive starts ~2021-03-23; SDK `PressureLevel()` always returns 0, must pair by index). One HTTP call covers the entire scan window; `_pick_hourly_per_scan_and_clip` picks the closest hourly slice per scan.
2. **AFGL climatology fallback** via `amwrap.Climatology`. Triggered by `_NoPressureLevelData` (deterministic — no point retrying open-meteo) or by an explicit `afgl_climatology=` argument. `"auto"` resolves to month-of-year.

In both cases, the profile is clipped at the site surface pressure (`_compute_surface_pressure` reduces per-scan `weather_P` to a single median value) so that am isn't asked to integrate above the actual atmospheric column.

- `_NoPressureLevelData` — `src/tipopac/atmosphere.py:30` — sentinel exception for the open-meteo no-upper-air case; the caller catches it and falls back to AFGL without retrying.
- `attach_profile` — `src/tipopac/atmosphere.py:107` — public entry; resolves source (`open_meteo` | `afgl`), fetches a profile, applies per-scan hour pick + surface clip, writes the four vars + provenance attrs.
- `_compute_surface_pressure` — `src/tipopac/atmosphere.py:261` — reduce per-scan `weather_P` to a single median surface pressure for the clip and a per-scan provenance array.
- `_utc_date_range` — `src/tipopac/atmosphere.py:285` — span scan times into a `("YYYY-MM-DD", "YYYY-MM-DD")` pair for the open-meteo request.
- `_pick_hourly_per_scan_and_clip` — `src/tipopac/atmosphere.py:296` — choose closest hourly slice per scan and apply the median-surface clip via `amwrap.interp_by_pressure`.
- `_pick_climatology_for_date` — `src/tipopac/atmosphere.py:367` — resolve `afgl_climatology="auto"` from observation month.
- `_fetch_open_meteo` — `src/tipopac/atmosphere.py:382` — single HTTP call to the SDK; parse the response into `(pressure, T, H₂O VMR, hour timestamps, meta)` Quantities. Pairs pressure-level vars by index because the SDK's `PressureLevel()` is broken.
- `_afgl_profile` — `src/tipopac/atmosphere.py:511` — return an AFGL climatology profile via `amwrap.Climatology`, clipped to site surface pressure.

---

## `atmgrid.py` — am invocation + PWV grid

`build_pwv_grid` is the only place `am` runs in the entire pipeline. It builds a 2-D lookup table `(pwv_mm × freq_Hz) → (τ_z, T_b)` by scaling the troposphere H₂O column up and down (`troposphere_h2o_scaling`) and running `amwrap.Model` at each PWV grid point. Stage A then samples `(τ_z, T_mean)` at the per-scan grid via bilinear interp; Stage B fits PWV against Stage-A τ_z using `lookup_with_grad` for the analytical ∂τ/∂PWV slope.

The parallelisation strategy is mandatory (see project memory): **never** `amwrap.Model.run(parallel=True)` — am's threadpool fights with multiprocessing and lockfile-contends with itself. Instead, use `mp.Pool(spawn)` with `_worker_init` carving out a per-worker `cache_dir` so the am cache lockfile doesn't deadlock under concurrent reads. `_pool_init` re-pins single-threaded BLAS inside each worker because spawn doesn't inherit env vars set in the parent process.

- `PwvGrid` (dataclass) — `src/tipopac/atmgrid.py:52` — holds `(pwv_mm, freq_Hz, tau_z, tb_z)` lookup tables + profile provenance.
- `PwvGrid.tmean` — `src/tipopac/atmgrid.py:99` — cached atmosphere-only effective T (CMB-subtracted) used by Stage A. Computed lazily on first access from the stored `(τ, T_b)` and the CMB term.
- `PwvGrid.lookup` — `src/tipopac/atmgrid.py:115` — bilinear interp of `(τ_z, T_mean)` at scalar PWV, array ν. Fit hot path.
- `PwvGrid.lookup_with_grad` — `src/tipopac/atmgrid.py:144` — same plus analytical ∂/∂PWV slopes; the anchor's Cramér–Rao bound depends on this.
- `pwv_mm_from_profile` — `src/tipopac/atmgrid.py:188` — hydrostatic column-integral of VMR → PWV (mm); the grid's scaling anchor (PWV=1.0 in the grid = the profile's natural column).
- `_worker_init` — `src/tipopac/atmgrid.py:218` — pool initializer; stashes the frozen profile arrays + carves a per-worker am `cache_dir` to avoid lockfile contention.
- `_worker_run` — `src/tipopac/atmgrid.py:244` — pool task: build a fresh `amwrap.Model` with `troposphere_h2o_scaling=scaling` and return `(freq, τ, Tb)`.
- `_run_serial` — `src/tipopac/atmgrid.py:266` — sequential equivalent for `n_workers ≤ 1` or for unit tests that don't want spawn.
- `build_pwv_grid` — `src/tipopac/atmgrid.py:279` — public builder; sets up the PWV axis (log-spaced bracketing the profile PWV), dispatches am runs via `mp.Pool(spawn)`, stacks results into a `PwvGrid`.

`api.TippingAnalysis.build_atm_grids` loops scans and calls `build_pwv_grid` once per scan with that scan's `(pressure, T, VMR)` slice — i.e. one am-pool invocation per scan, not per analysis and not per sample.

---

## `fit.py` — Stage A kernel

The opacity-fitting kernel. Single public entry `fit_dataset`; everything else is internal. The flow inside `fit_dataset`:

1. Compute Tsys = (ssum/2)/diff · tcal_ref per `(scan, ant, spw, pol, time)` via `_compute_tsys`. The denominator is the switched-power difference; NaN-mask cells where `diff` is non-finite or non-positive.
2. Compute σ_Tsys per sample via `_compute_sigma_tsys`, propagating noise from switched power through the radiometer equation. σ_Tsys is the fit weighting; this is one of the rewrite's deliberate departures from v2.6's unit-weight L2.
3. Resolve the per-`(scan, spw)` Twmt grid via `_compute_twmt_grid`. If the externally-supplied `t_mean=` (from `anchor.compute_t_mean_grid`) has a finite value at this cell, use it; otherwise fall back to the Bevis (1992) form `70.2 + 0.72·T_surf` on weather_T. The same fallback rule appears in `anchor.compute_t_mean_grid` at the other end of the Stage A↔B loop.
4. Build per-cell fit tasks and dispatch through `_opacity_worker` (per-`(scan, ant, spw)` 3-param LM: `T0_R`, `T0_L`, `τ`) or `_global_worker` (per-`(scan, spw)` joint LM: shared `τ`, per-antenna `(T0, c)` for `tcal_solve`).
5. Inside each worker, `_screen_antenna` runs the robust soft_l1 fit with iterative 4σ rejection on the σ-weighted residuals. The kept-sample arrays are reused — they feed `_fit_global` directly in `tcal_solve` mode, so the screening cost isn't paid twice.
6. The fit reports a categorical `fit_reason` (one of `ok`, `poorly_identified`, `too_few_samples`, `solver_failed`) which is what `plot.FitQualityHeatmap` colour-codes by.

The "identifiability ratio" called out in `CLAUDE.md` is checked inside `_screen_antenna`: if the bounded LM solution sits at a bound or the τ-vs-T0 condition number exceeds a threshold, `fit_reason` becomes `poorly_identified` and the cell's τ_z is kept but its σ_τ is inflated.

- `fit_dataset` — `src/tipopac/fit.py:45` — public Stage-A driver; writes Tsys, sigma_Tsys, tau_zenith, T0, tcal_fit, fit_success / fit_reason into `ds` in place.
- `_compute_tsys` — `src/tipopac/fit.py:230` — Tsys = `(ssum/2)/diff · tcal_ref` with NaN-masking on bad denominators.
- `_compute_sigma_tsys` — `src/tipopac/fit.py:247` — per-sample σ_Tsys from error propagation on switched power; the fit weighting and the σ used in `_tau_err_from_jac`.
- `_compute_twmt_grid` — `src/tipopac/fit.py:294` — resolve per-`(scan, spw)` Twmt: grid value when finite, else Bevis on weather_T.
- `_residuals` — `src/tipopac/fit.py:321` — σ-weighted residuals for the 3-param per-antenna model (T0_R, T0_L, τ).
- `_residuals_tcal` — `src/tipopac/fit.py:338` — σ-weighted residuals for the joint `tcal_solve` model (per-antenna T0, c plus shared τ).
- `_jac_tcal` — `src/tipopac/fit.py:365` — analytical sparse Jacobian of `_residuals_tcal`. Sparse because each row only touches the antenna it belongs to plus the shared τ column.
- `_tau_err_from_jac` — `src/tipopac/fit.py:404` — Cramér–Rao σ_τ from the σ-weighted Jacobian, inflated by reduced χ² so the reported uncertainty tracks fit-of-model rather than just propagated noise.
- `_screen_antenna` — `src/tipopac/fit.py:451` — core per-antenna robust soft_l1 fit with iterative 4σ rejection; emits `ok`/`poorly_identified` plus the kept-sample arrays reused by the global fit.
- `_fit_tau_per_antenna` — `src/tipopac/fit.py:614` — thin wrapper that re-shapes `_screen_antenna` output for the per-antenna mode.
- `_fit_global` — `src/tipopac/fit.py:657` — single-pass joint LM solve over all passing antennas in one `(scan, spw)` for `tcal_solve` mode.
- `_build_opacity_tasks` — `src/tipopac/fit.py:733` — generator yielding one fit task per `(scan, ant, spw)`.
- `_build_global_tasks` — `src/tipopac/fit.py:771` — generator yielding one fit task per `(scan, spw)`, bundling every antenna's screening inputs.
- `_opacity_worker` — `src/tipopac/fit.py:820` — pickle-clean worker that runs `_fit_tau_per_antenna` on one task.
- `_global_worker` — `src/tipopac/fit.py:826` — pickle-clean worker that screens every antenna then runs `_fit_global`.
- `_pool_init` — `src/tipopac/fit.py:868` — spawn-pool initializer that re-pins single-threaded BLAS.
- `_dispatch` — `src/tipopac/fit.py:878` — serial / `mp.Pool(spawn)` switch driving the workers; serial is the default when `n_workers ≤ 1` or in CI.

---

## `anchor.py` — Stage B

Stage B is the 1-D bounded scalar fit `CLAUDE.md` mentions. Given Stage-A τ_z(scan, spw, ant) and the per-scan `PwvGrid`, `anchor_pwv` fits PWV per antenna by minimising `Σ ((τ_z_obs - τ_z_model(PWV))/σ_τ)²` over spw. The grid is precomputed, so this is cheap — no second am call. σ_PWV comes from the Cramér–Rao bound using the analytical ∂τ/∂PWV from `PwvGrid.lookup_with_grad`.

The Stage A ↔ Stage B loop is closed via `compute_t_mean_grid`: each scan's `PwvGrid` is sampled at its **unscaled** profile PWV (i.e., the open-meteo-derived column) to produce a per-`(scan, spw)` T_mean. That grid is what `fit_dataset` consumes as `t_mean=`. No iteration — one pass through.

`write_am_curve` samples a dense τ(ν) curve at the median fitted PWV onto `am_freq_grid` / `am_tau` so `plot.TauVsFrequency` can overlay an AM-model line. Same `PwvGrid`, so still no second am call.

- `anchor_pwv` — `src/tipopac/anchor.py:35` — per-antenna weighted χ² fit of PWV against Stage-A τ_z; σ_PWV from CR bound using `PwvGrid.lookup_with_grad`.
- `write_am_curve` — `src/tipopac/anchor.py:137` — sample a dense τ(ν) curve at the median fitted PWV onto `am_freq_grid` / `am_tau` for the plot overlay.
- `compute_t_mean_grid` — `src/tipopac/anchor.py:190` — sample per-`(scan, spw)` T_mean from each scan's `PwvGrid` at its unscaled-profile PWV; the Stage-A `t_mean` kwarg input.

---

## `physics.py`, `geometry.py`, `bands.py` — shared primitives

### `physics.py`

Stateless physics shared between `fit`, `anchor`, and the plots. Nothing here touches xarray state apart from `predicted_tsys`.

- `k2nt` — `src/tipopac/physics.py:28` — Nyquist correction of kinetic K → noise K at frequency ν. Important at high ν where `hν/kT` is no longer ≪ 1; used in both the Bevis fallback and `PwvGrid.tmean`.
- `tsys_model` — `src/tipopac/physics.py:37` — reference scalar tipping-curve formula `T0 + Twmt·(1 − exp(−τ/cos z))`. The "what we're fitting" formula in one place; both `_residuals` variants in `fit.py` and `predicted_tsys` re-derive from it.
- `predicted_tsys` — `src/tipopac/physics.py:50` — xarray-aware Tsys reconstruction from persisted fit fields. Includes the `c = tcal_fit / tcal_ref` divisor for `tcal_solve` mode so the model line in `plot.ElevationCurve` overlays correctly on observed Tsys.
- `airmass` — `src/tipopac/physics.py:73` — flat-earth `1/cos z`. No curvature correction — matches v2.6 and is fine at the ZAs DO_SKYDIP covers.
- `weighted_mean_atm_T` — `src/tipopac/physics.py:78` — Bevis (1992) `T_atm = 70.2 + 0.72·T_surf`. The surface-T proxy when no grid T_mean is available (e.g. AFGL fallback hasn't been wired to that path, or `weather_T` is the only thing finite).

### `geometry.py`

Single utility — elevation encoder → zenith angle.

- `zenith_angle` — `src/tipopac/geometry.py:10` — `90° − rad2deg(el_encoder)`; no refraction correction, matching v2.6.

### `bands.py`

VLA receiver-band labels and scan/SPW selection helpers; used at read time, not in the fit hot path.

- `VLA_BAND_LABELS` — `src/tipopac/bands.py:37` — canonical ordered band tuple.
- `HIGH_FREQ_DEFAULT` — `src/tipopac/bands.py:50` — default `(Ku, K, Ka, Q)` selection. The bands where tipping curves are useful.
- `band_for_spw_name` — `src/tipopac/bands.py:56` — parse canonical band label from an `EVLA_<BAND>#…` SPW NAME.
- `normalize_bands` — `src/tipopac/bands.py:87` — validate / canonicalize user band selection (None → high-freq default).
- `validate_scan_selection` — `src/tipopac/bands.py:117` — resolve user scan ids against the available DO_SKYDIP set with informative errors.
- `attach_selection_attrs` — `src/tipopac/bands.py:149` — record scan/band selection provenance on `ds.attrs`.
- `select_spws_by_band` — `src/tipopac/bands.py:176` — filter candidate SPW ids by allowed bands, preserving the original order.

---

## `timeutils.py` — MJD↔Unix

The smallest module. One conversion used wherever code needs to talk to pandas / `datetime`-aware libraries (open-meteo SDK timestamps, altair axes).

- `mjd_s_to_unix_s` — `src/tipopac/timeutils.py:17` (with `@overload`s at L13–16) — subtract `MJD_UNIX_EPOCH * 86400.0` (`MJD_UNIX_EPOCH = 40587.0`); broadcasts over numpy arrays.

Consumers: `summary.py:15`, `plot.py:30`, `atmosphere.py:25` (aliased `_mjd_s_to_unix_s`).

---

## `plot.py` — Altair diagnostic plots

vega-altair `LayerChart`s saved as standalone `.html`. The diagnostic-plot library is the runtime QA surface — every fit run produces a directory full of these and `weblog.build_weblog` stitches them into a navigable index.

The class hierarchy is `Plot` (base, shared style + `save` + `_finalize`) → concrete plots. The "vs frequency" family shares a `_QuantityVsFrequency` scaffold for scan subsetting and mean-layer construction. Heatmaps share `_Heatmap` (`mark_rect` faceted by scan). `Summary` is the odd one out — a non-altair textual HTML page that serves as the weblog's landing view (added in commit `6164f22`).

The implicit contract `weblog.py` depends on: `PlotData.save_all` writes specific filenames (`elevation__scan{N}__ant{NAME}__spw{ID}.html`, `tau_vs_frequency__scan{N}.html`, `summary.html`, etc.) and `weblog.py`'s glob patterns / regex constants mirror those names. Renaming a plot file is a cross-module change.

The data-shrinking helpers `_to_df` / `_round` exist because altair embeds JSON inline; without them an `ElevationCurve` page can exceed 1 MB for the full sample density (see commit `8a61bbe`). `_to_df` projects columns and drops NaN rows; `_round` trims float precision.

- module setup (`alt.data_transformers.disable_max_rows()`, `_Z_GRID`) — `src/tipopac/plot.py:50` — disables altair's 5000-row cap; defines model-curve ZA grid.
- `_scan_title` — `src/tipopac/plot.py:57` — title prefix for one vs many scans.
- `Plot` — `src/tipopac/plot.py:64` — base class with shared colour palette, sizes, and save helpers.
- `Plot.save` — `src/tipopac/plot.py:89` — force `.html` suffix and write via altair.
- `Plot._finalize` — `src/tipopac/plot.py:96` — apply width/height/title; toggle `interactive()`.
- `_validate_scans` — `src/tipopac/plot.py:112` — normalise `scans=None | int | list[int]` against `ds["scan"]`.
- `_to_df` — `src/tipopac/plot.py:118` — xarray → tidy DataFrame with NaN-dropping and column projection (cuts JSON bloat).
- `_round` — `src/tipopac/plot.py:142` — float32→float64 then per-column rounding to trim embedded-JSON precision.
- `_QuantityVsFrequency` — `src/tipopac/plot.py:155` — shared scaffold (scan subsetting, freq domain, mean-layer builder) for the three "vs frequency" plots.
- `_QuantityVsFrequency._mean_layer` — `src/tipopac/plot.py:170` — firebrick mean-per-spw scatter layer reused by every subclass.
- `ElevationCurve` — `src/tipopac/plot.py:194` — Tsys vs ZA scatter + model line per `(scan, antenna, spw)`; calls `physics.predicted_tsys` on a dense ZA grid (`_Z_GRID`).
- `TauVsFrequency` — `src/tipopac/plot.py:288` — zenith opacity vs ν with weighted mean and the optional AM-model line from `am_freq_grid`/`am_tau`.
- `TcalVsFrequency` — `src/tipopac/plot.py:396` — fitted/reference Tcal vs ν with per-pol/antenna scatter; branches on `kind` (`tcal_fit` vs `tcal_ref`) because `tcal_ref` has no scan dim.
- `CVsFrequency` — `src/tipopac/plot.py:491` — Tcal correction ratio `tcal_fit / tcal_ref` vs ν. The diagnostic that says "is the noise tube right?"
- `AtmosphericProfile` — `src/tipopac/plot.py:558` — vertical T and H₂O VMR profiles on a log-pressure y-axis with independent x-scales. Provenance for the open-meteo / AFGL source.
- `_Heatmap` — `src/tipopac/plot.py:687` — `mark_rect` heatmap scaffold faceted by scan; hooks defer metric/colour/tooltip to subclasses.
- `_Heatmap._flag_fraction` — `src/tipopac/plot.py:732` — per-`(scan, ant, spw)` flagged fraction over observed cells.
- `_Heatmap.build` — `src/tipopac/plot.py:741` — assemble DataFrame, size facets to per-scan spw counts, optionally facet.
- `FitQualityHeatmap` — `src/tipopac/plot.py:797` — categorical `fit_reason` heatmap (ordered palette best→worst). The first plot you look at when something's off.
- `ResidualRmsHeatmap` — `src/tipopac/plot.py:842` — Tsys residual RMS heatmap; rebuilds the model via `physics.predicted_tsys` and reduces.
- `Summary` — `src/tipopac/plot.py:880` — non-altair textual HTML run summary (input metadata + per-scan stats); the weblog landing view.
- `Summary._render` — `src/tipopac/plot.py:921` — emit standalone HTML page with inline CSS.
- `Summary._scan_stats` — `src/tipopac/plot.py:1000` — compute UTC start, observed bands, centre freq, mean τ, flag fraction, fit success per scan.
- `PlotData` — `src/tipopac/plot.py:1064` — public façade: holds the dataset, returns each `Plot` subclass on demand, bulk-writes via `save_all`.
- `PlotData.save_all` — `src/tipopac/plot.py:1107` — emit every applicable plot to disk under the filenames `weblog.py` consumes; gates `tcal_fit` / `c` on the non-`independent_tau` mode.

---

## `weblog.py` — HTML index

Post-plot generator. Globs the plot directory for files matching `PlotData.save_all`'s naming scheme and emits a self-contained `index.html` with a dropdown picker plus per-cell elevation-curve drilldowns. **Decoupled from the dataset** — only filenames drive the dropdown options, so the weblog can be regenerated against a static plot dump without re-running the fit.

- plot-name constants (`_ELEVATION_RE`, `_SUMMARY_PLOT`, `_AGGREGATE_PLOTS`, `_ELEVATION_LABEL`) — `src/tipopac/weblog.py:28` — mirror of `PlotData.save_all`'s names; the filename contract lives here.
- `build_weblog` — `src/tipopac/weblog.py:43` — glob `plot_dir`, build the scan→spws→ants map, write `index.html`.
- `_render_html` — `src/tipopac/weblog.py:83` — emit the inline-CSS / inline-JS page with select widgets and an iframe loader.

---

## `summary.py` — `python -m tipopac.summary` CLI

The fast pre-flight: prints the DO_SKYDIP scan table (id / UTC start / band / SPW ids) without doing the full reader load. Useful when you want to pick scan ids to pass to `tipopac(scans=...)` and don't want to wait through SYSPOWER/POINTING/WEATHER reads to find out.

- `summarize_skydip_scans` — `src/tipopac/summary.py:26` — detect the reader, call its `list_skydip_scans` (the metadata-only path), format, write to stdout or file.
- `_format_skydip_table` — `src/tipopac/summary.py:58` — column-align `SkydipScanInfo` rows into the printable table.
- `_format_mjd_utc` — `src/tipopac/summary.py:102` — MJD-seconds → `YYYY-MM-DD HH:MM:SS UTC` via `timeutils.mjd_s_to_unix_s`.
- argparse `__main__` block — `src/tipopac/summary.py:110` — CLI entry.

---

## `caltables.py` — opt-in CASA caltable writers

Only reachable via `TippingAnalysis.write_outputs(caltable_*=True)`. Writers call `casatools`; row-builders are pure-Python and unit-testable without CASA.

Two outputs are supported. `write_opacity` produces a TOpac caltable bootstrapped via `casatools.calibrater` — this is the standard product, valid in every fit mode. `write_tcal` clones the source MS's CALDEVICE schema (it needs the exact column layout to be ingested back into CASA) and writes Tcal rows; it is only valid in `tcal_solve` mode because that's the only mode that produces a `tcal_fit` to write.

- `write_opacity` — `src/tipopac/caltables.py:32` — bootstrap a TOpac caltable via `casatools.calibrater`, then bulk-fill rows from `_build_opacity_rows`.
- `write_tcal` — `src/tipopac/caltables.py:62` — clone CALDEVICE schema from `ds.attrs["source_path"]` and write Tcal rows from `_build_tcal_rows`; raises if `mode != "tcal_solve"`.
- `_build_opacity_rows` — `src/tipopac/caltables.py:102` — emit one TOpac row dict per `(scan, antenna, spw)`, zeroing failed-fit cells (`fit_success == False`).
- `_build_tcal_rows` — `src/tipopac/caltables.py:143` — emit one CALDEVICE row dict per `(scan, antenna, spw)` with R/L noise-tube values and a zeroed solar-filter slot (CALDEVICE schema requires the column even though we don't fit it).

---

## `__init__.py` — package bootstrap

Two things happen at import time. First, single-threaded BLAS is forced via `os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")` (and `MKL_NUM_THREADS`, `OMP_NUM_THREADS`) before any numerical import. This is mandatory for the `mp.Pool` dispatch in `fit.py` and `atmgrid.py`: under spawn, each worker re-imports tipopac, which re-runs this block, but the *parent* needs the cap set before numpy is loaded so the workers don't inherit oversubscribed BLAS pools via fork on Linux. The `setdefault` means a user who really wants threaded BLAS can override by exporting first.

Second, `tipopac.summary` is lazy-loaded via `__getattr__` so that `python -m tipopac.summary` doesn't double-load the module. Public re-exports are `tipopac`, `TippingAnalysis`, `Result` from `api`.

- BLAS thread-cap loop — `src/tipopac/__init__.py:12` — set `OPENBLAS/MKL/OMP_NUM_THREADS=1` via `setdefault`.
- `__getattr__` — `src/tipopac/__init__.py:24` — defer `tipopac.summary` import to first attribute access.

---

## Legacy `v2.6` → new layout

`tipopac_v2.6/lastversion/tipping/private/task_tipopac.py` is one ~1900-line function (single `def tipopac(...)` at L37) with inline helpers and a procedural body. The landmarks below are the table of contents; line numbers are v2.6-side.

- v2.6 L37 `def tipopac(...)` (single monolith) → `api.tipopac` (one-shot) + `api.TippingAnalysis` (staged) in `src/tipopac/api.py`. The big functional split: one function with eight implicit phases → eight named methods.
- v2.6 L120–236 inline `model` / `fitting_Tcal` / least-squares residual closures → `fit.py` (Stage A, per-antenna and joint variants) + `physics.tsys_model` / `physics.predicted_tsys` (the reference formula in one place).
- v2.6 L238–268 `makeplot` → `plot.py` (the entire altair plot family + `Summary`; uses `timeutils.mjd_s_to_unix_s` for the time axis).
- v2.6 L414–530 `getAtmDetails` / `estimateOpacity` / `model` (per-frequency `casatools.atmosphere` invocations) → `atmosphere.attach_profile` (one am profile fetch, open-meteo or AFGL) + `atmgrid.build_pwv_grid` (one am-pool per scan building a precomputed PWV grid; never per-sample).
- v2.6 L542–730 `fitATM` (frequency-vs-opacity ATM fit, residuals helper) → `anchor.py` (post-fit per-antenna PWV anchor against τ_z(ν), see `design/independent_tau_fit.md`).
- v2.6 L769–857 `ANTENNA` / `SPECTRAL_WINDOW` / `POINTING` table opens → `readers/ms.py` (and `readers/sdm.py` for SDM); both produce a schema-conformant `xr.Dataset` per `design/design.md §3` MS↔SDM column-mapping contract.
- v2.6 L859 `WEATHER` open → readers populate `weather_T/P/RH(scan, time)` per `schema.INPUT_DATA_VARS`.
- v2.6 L876–890 `FLAG_CMD` open + REASON exclude list → `flags._apply_online_flags` (same `_REASON_EXCLUDE` set, TaQL exclusion).
- v2.6 L989–1063 `CALDEVICE` open + Tcal caltable write → `caltables.write_tcal` (uses `ds.attrs["source_path"]` to copy CALDEVICE schema).
- v2.6 L1066–1199 SYSPOWER ingest + four-case interval expansion of online flags → replaced by the single broadcast `(time_utc >= t_start) & (time_utc <= t_end)` in `flags._apply_interval`; per the module docstring this directly replaces v2.6 L1116–1199.
- v2.6 L1225–1275 per-sample `Tsys = (Psum/2)/Pdif · Tcal` and per-sample flagging → vectorized into `switched_diff` / `switched_sum` ingest in the readers; downstream Tsys statistics consume `schema.apply_flags(ds, "Tsys")`.
- v2.6 L1278–1545 OPTION 1/2/3 per-scan-per-spw nested fitting loops (prior fit → outlier σ-clip → final `fitting_Tcal`) → `fit.fit_dataset` operating against `atmgrid.build_pwv_grid` outputs; the three OPTIONs collapse into the `tau_per_antenna` / `tcal_solve` / `independent_tau_solve` modes.

Numerical parity with v2.6 is a smoke test, not a contract. The rewrite uses radiometer-eq σ + `soft_l1` + single-tier bounds + an identifiability ratio in place of v2.6's unit-weight L2 + 2σ clip + 3-pass bound escalation + geometric `dz` / `min(z)` gates. Drift is expected; see project memory `tcal_solve convergence ridge` for one concrete example where ~3-5% Tcal mismatch is optimizer-trajectory sensitivity on a near-degenerate (T0, c, τ) ridge, not a logic bug.
