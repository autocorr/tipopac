# tipopac

Fit VLA tipping scans to measure zenith opacity and calibration device
temperatures. This library reads either CASA Measurement Sets or SDM files (no
BDF required). It then fits a per-(scan, antenna, spw) atmospheric model using
predictions from the atmospheric radiative transfer code AM using vertical
atmospheric profiles from NCEP's High-Resolution Rapid Refresh forecast
analysis, falling back to AFGL climatologies when those are unavailable.
This rewrite is based off prior work written by Chris Hales (see
`vendor/tipopac_v1.0`) with contributions from Pedro Beaklini
(`vendor/tipopac_v2.6`).

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/). To install this
repository run:

```bash
git clone https://github.com/autocorr/tipopac
cd tipopac
uv sync --no-dev
source .venv/bin/activate
```

This creates a virtual environment in `.venv/` and installs the needed
dependencies, including the AM wrapper
[`amwrap`](https://github.com/autocorr/amwrap) from the pinned source on
GitHub.

To run the slow tests link the shared tipping dataset into `data/`: the
Measurement Set as `tip_test.ms` and the matching SDM as `tip_test.sdm`.
With only one of the two present, `pytest -m slow` aborts; narrow the
selection to the tests that read the form you have.

## Quickstart

### Library usage

The example below runs the pipeline and produces optional caltables and plots:

```python
from tipopac import tipopac

result = tipopac(
    "data/tip_test.ms",
    n_workers=8,                           # am-grid and Stage-A pool size (None = serial)
    output_dir="run",                      # optional
)

ds = result.dataset                        # xarray.Dataset
print(ds["tau_zenith"], ds["pwv"], ds["tcal_fit"])
```

For staged / notebook use, instantiate the class directly and inspect the
dataset between stages:

```python
from tipopac import TippingAnalysis

ta = TippingAnalysis.from_path("data/tip_test.ms")
ta.apply_flags(online=True)
ta.fetch_atm_profile(source="open-meteo")
ta.build_atm_grids()
ta.fit(n_workers=8)
ta.plot(out_dir="run/plots")
ta.write_caltables(opacity="run/topac.cal", tcal="run/tcal.cal")
ds = ta.dataset
```

The fit runs in three stages: a per-(scan, antenna, spw) opacity fit at
`c ≡ 1`, then a per-antenna PWV anchor against the precomputed `am` grid,
then a closed-form Tcal scale pinned to that anchor.

## Development

```bash
uv sync                                    # install + lock deps

uv run pytest                              # unit tests (fast)
uv run pytest tests/unit                   # explicit
uv run pytest -m slow                      # needs data/tip_test.ms + tip_test.sdm
uv run pytest -m network                   # hits live open-meteo
uv run pytest tests/unit/test_fit.py::test_fit_tau_per_antenna_recovers_params

uv run ruff check .                        # lint
uv run ruff format .                       # format

uv run ty check src/tipopac                # type-check
```

By default `pytest` skips both `slow` (needs the shared test data) and
`network` markers, see `[tool.pytest.ini_options]` in `pyproject.toml`.

The package layout is `src/tipopac/`; readers live under
`src/tipopac/readers/`. The legacy task at
`vendor/tipopac_v2.6/lastversion/tipping/private/task_tipopac.py`
is reference only.
