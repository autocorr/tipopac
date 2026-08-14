---
icon: lucide/radio-tower
---

# tipopac

**Fit VLA tipping scans to measure zenith atmospheric opacity and
noise-diode (Tcal) temperatures** — a clean, importable Python rewrite of
the legacy CASA task `tipopac`, with no CASA runtime required.

```python
from tipopac import tipopac

result = tipopac("data/tip_test.ms", output_dir="run")
ds = result.dataset                 # canonical xarray.Dataset
print(ds["tau_zenith"], ds["pwv"], ds["tcal_fit"])
```

## What it does

During a VLA tipping scan (`DO_SKYDIP`), an antenna sweeps in elevation
while recording system temperature. Because the atmospheric path length
grows toward the horizon, the rise in $T_\mathrm{sys}$ with airmass
encodes the **zenith opacity** $\tau_0$, and the switched-power
measurement simultaneously constrains the **noise-diode temperature**
$T_\mathrm{cal}$. `tipopac` reads those scans, fits a physically grounded
atmospheric model, and reports per-spectral-window opacity, per-antenna
precipitable water vapor (PWV), and Tcal corrections.

- **Reads MS *or* SDM** — a CASA Measurement Set or an SDM file (no BDF
  required) is auto-detected and read into one canonical
  `xarray.Dataset`. Both readers satisfy the same schema.
- **Physically grounded fit** — atmospheric radiative transfer comes from
  Scott Paine's `am` (via the
  [`amwrap`](https://github.com/autocorr/amwrap) wrapper), driven by
  vertical profiles from NCEP's HRRR forecast analysis (Open-Meteo), with
  AFGL climatologies as an offline fallback.
- **Three-stage solver** — Stage A fits the tipping curve for zenith
  opacity; Stage B anchors a per-antenna PWV against a precomputed
  opacity grid; Stage C solves the Tcal scale in closed form at that
  pinned opacity. See [Theory &amp; method](theory.md).
- **Self-contained outputs** — a NetCDF dataset, interactive HTML plots, a
  browsable weblog, and opt-in CASA caltables. See [Quickstart](quickstart.md).
- **Modern tooling** — `uv` + `pyproject.toml`, type hints checked with
  `ty`, `ruff`, and `pytest`. `casatools` is an ordinary library import;
  "no CASA at runtime" means no `casa` process and no `buildmytasks`.

## Where to next

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — `uv sync`, the
  external `am` binary, and the optional test dataset.
- :material-rocket-launch: **[Quickstart](quickstart.md)** — point it at an
  MS/SDM, run the pipeline, and open the weblog.
- :material-function-variant: **[Theory &amp; method](theory.md)** — the
  tipping-curve model, the radiometer-equation noise, and the
  Stage-A/B/C fit.
- :material-api: **[API reference](api.md)** — `tipopac()`,
  `TippingAnalysis`, and `Result`.

</div>

## Lineage

This rewrite is based on Chris Hales' `tipopac_v1.0` with contributions
from Pedro Beaklini (`tipopac_v2.6`); both live under `vendor/`. The
numerical method has been modernized (robust loss, radiometer-equation
weighting, a single-tier QA gate), so exact parity with v2.6 is a smoke
test rather than a contract.
