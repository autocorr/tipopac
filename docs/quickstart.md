---
icon: material/rocket-launch
---

# Quickstart

This page walks through a full run: pointing at an input, listing its
tipping scans, fitting, and opening the results. For the precise argument
lists see the [API reference](api.md).

## 1. Inputs: MS or SDM

`tipopac` accepts a path to **either** a CASA Measurement Set **or** an SDM
directory. The format is auto-detected:

- a path containing `table.dat` and a `SYSPOWER/` subdirectory is read as
  an **MS**;
- a path containing `ASDM.xml` is read as an **SDM** (via `sdmpy`, with no
  BDF required).

Both produce the same canonical `xarray.Dataset`, so everything downstream
is identical regardless of input format.

!!! note "Online flags are MS-only"
    `FLAG_CMD` online flags can be applied for an MS. SDM inputs have no
    equivalent, so `flags_online` has no effect there.

## 2. List the tipping scans

Before fitting, see which `DO_SKYDIP` scans an input contains:

```bash
uv run python -m tipopac.summary data/tip_test.ms
```

This prints a table of scan id, UTC start, band, and SPW ids. Add
`-o scans.txt` to write it to a file instead of stdout. Use the scan ids
and bands you see here to drive the `scans=` / `bands=` selection below.

## 3. Run the pipeline (one-shot)

The simplest path is the `tipopac()` function. It runs every stage and,
when `output_dir` is set, writes all artifacts:

```python
from tipopac import tipopac

result = tipopac(
    "data/tip_test.ms",
    mode="independent_tau_solve",   # default; per-(scan, spw) Tcal-solve + PWV anchor
    n_workers=8,                    # Stage-A process-pool parallelism (None = serial)
    output_dir="run",               # write outputs here; None = compute-only
)

ds = result.dataset                 # the canonical xarray.Dataset
print(ds["tau_zenith"], ds["pwv"], ds["tcal_fit"])
```

### Common arguments

| Argument | Meaning |
| --- | --- |
| `scans` | `DO_SKYDIP` scan numbers to keep. `None` keeps all skydip scans. |
| `bands` | VLA receiver bands (case-insensitive, e.g. `["Ku", "K"]`). `None` keeps the well-conditioned high bands `Ku, K, Ka, Q`. |
| `mode` | `"independent_tau_solve"` (default) or `"independent_tau"`. See [Theory](theory.md). |
| `flags_online` | Apply online flags (MS `FLAG_CMD` / SDM `Flag.xml`). Default `True`. |
| `flags_file` | Path to a user flag file (`antenna/spw/timerange` per line). |
| `atm_profile_source` | `"open-meteo"` (default, one HTTP call) or `"afgl"` (offline). |
| `n_workers` | Stage-A fit parallelism. `None` runs serially. |
| `output_dir` | Where artifacts are written; `None` for compute-only. |
| `caltable_opacity` / `caltable_tcal` | Opt-in CASA caltables. |

## 4. Run the pipeline (staged)

For notebooks or to inspect the dataset between stages, drive
[`TippingAnalysis`](api.md#tipopac.TippingAnalysis) directly. Each stage
mutates the dataset in place:

```python
from tipopac import TippingAnalysis

ta = TippingAnalysis.from_path("data/tip_test.ms", bands=["K", "Ka"])
ta.apply_flags(online=True)
ta.fetch_atm_profile(source="open-meteo")   # the only network stage
ta.build_atm_grids()                         # runs `am` once per scan
ta.fit(mode="independent_tau_solve", n_workers=8)

ds = ta.dataset
ta.write_outputs("run")                      # or ta.plot(...) / ta.weblog(...)
```

The order is fixed: `from_path` → `apply_flags` → `fetch_atm_profile` →
`build_atm_grids` → `fit`. `build_atm_grids` will call `fetch_atm_profile`
with defaults if you skipped it, and `fit` will build grids if needed —
but calling them explicitly lets you choose the profile source and inspect
intermediate state.

!!! tip "Offline / reproducible runs"
    Pass `source="afgl"` to `fetch_atm_profile` (or
    `atm_profile_source="afgl"` to `tipopac()`) to skip the network
    entirely and use AFGL climatologies. This is what the integration test
    uses for determinism.

## 5. Outputs and the weblog

When `output_dir` is set, every run writes into that directory:

| File | Contents |
| --- | --- |
| `tipopac.nc` | The full canonical `xarray.Dataset` as NetCDF. |
| `model_opacity.tsv` | Stage-B model opacity $\tau(\nu)$ on the uniform 1–51 GHz am grid — `frequency_Hz`, `tau_model`. |
| `measured_opacity.tsv` | Fitted and model $\tau$ at the SPW centre frequencies, one row per `(scan, spw)`. |
| `*.html` plots | Interactive Vega-Altair charts (opacity vs frequency, Tcal, fit-quality and residual heatmaps, atmospheric profile, …). |
| `index.html` | The **weblog** — a self-contained browser for all the plots. |
| `tipopac.opacity` / `tipopac.tcal` | Opt-in CASA caltables (`caltable_opacity` / `caltable_tcal`). |

Open the weblog in a browser to explore the results:

```bash
xdg-open run/index.html      # or just open run/index.html
```

It provides a dropdown to switch between plot types, with per-scan,
per-antenna, and per-SPW selectors for the elevation (tipping) curves.

## 6. Inspect results programmatically

The returned `Result` exposes the dataset, the resolved input format, and
the software versions used:

```python
result.input_format          # "ms" or "sdm"
result.mode                  # the fit mode label
result.software_versions     # {"am": ..., "casatools": ..., "tipopac": ...}

ds = result.dataset
ds["tau_zenith"]             # zenith opacity (scan, antenna, spw)
ds["tau_err"]                # its 1-sigma uncertainty
ds["pwv"], ds["pwv_err"]     # per-antenna precipitable water vapor [mm]
ds["tcal_fit"]               # fitted Tcal; NaN where not measured
ds["sigma_tcal"]             # its 1-sigma uncertainty (independent_tau, Stage C)
ds["fit_reason"]             # per-cell QA label (ok, poorly_identified, ...)
```

Reductions over the (NaN-padded, per-scan-local) `time` axis must respect
the `flag` array — go through `tipopac.schema.apply_flags(ds, var)` rather
than indexing `ds[var]` directly.
