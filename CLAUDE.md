# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A clean-Python rewrite of the legacy CASA task `tipopac` (VLA tipping-curve
opacity / Tcal estimation). The new package is `tipopac` under `src/tipopac/`;
the legacy reference implementation lives **read-only** at
`tipopac_v2.6/lastversion/tipping/private/task_tipopac.py` (~1900 lines).

**`DESIGN.md` is the authoritative implementation contract** for v1. Anything
ambiguous about API shape, data schema, fit modes, atmospheric-model anchoring,
or acceptance criteria is answered there before you read the code. If the code
and DESIGN.md disagree, the bug is in the code unless a follow-up has been
explicitly agreed.

Implementation order is in DESIGN.md §13 (schema → MS reader → physics/fit →
flags → other modes → caltables → atmosphere → plot → SDM reader → integration
reference). As of writing, `src/tipopac/` is an empty skeleton — only the
package directory exists.

The `references/` directory contains other authoritative references. These
include:

- `references/ms_v2_memo_299.html`: Specification for the MeasurementSet v2.

## Toolchain & commands

The repo uses **`uv`** (Astral) with `pyproject.toml` + `uv.lock`. Python ≥
3.13 (`.python-version` pins `3.13`). Always use `uv run python` and never run
bare `python` commands.

```bash
# install / sync deps (creates .venv/, respects uv.lock)
uv sync

# tests
uv run pytest                          # all tests
uv run pytest tests/unit               # fast unit tests only
uv run pytest -m "not slow"            # skip integration (needs data/tip_test.ms)
uv run pytest tests/unit/test_fit.py::test_global_tau  # single test

# lint + format
uv run ruff check .
uv run ruff format .

# type-check — use `ty`, NEVER mypy
uv run ty check src/tipopac
```

The integration test (DESIGN.md §11.2) is gated by `pytest.mark.slow` and needs
the ~7 GB MS at `data/tip_test.ms` (symlink to `../data/`).

## Architecture — the parts that span multiple files

Everything funnels through one in-memory representation: the canonical
`xarray.Dataset` defined in **DESIGN.md §5**. Read that section before adding
a data variable, changing a dim order, or touching either reader — it is the
contract the rest of the codebase relies on.

The pipeline is:

```
readers/{ms,sdm}.py  →  schema.validate(ds)  →  flags.apply  →  fit.fit_scan  →  atmosphere.anchor + extrapolate  →  caltables / plot
```

- **`readers/`** — `MSReader` (uses `casatools.table`) and `SDMReader` (uses
  `sdmpy`) implement the `TippingReader` Protocol in `readers/base.py`. **Both
  must produce the same `xarray.Dataset` schema.** The SDM↔MS column mapping is
  the table in DESIGN.md §4 — that is the contract for parity between readers,
  not something to re-derive.
- **`fit.py`** — three modes (`tau_per_antenna`, `global_tau`, `tcal_solve`)
  matching v2.6's three solver configurations. `tcal_solve` forces global τ —
  matches v2.6's `calcTcals` → `tauPerAnt=False` constraint. Uses
  `scipy.optimize.least_squares` (not the legacy `leastsq`) so covariance comes
  for free; no manual `linalg.inv(J^T J)`.
- **`atmosphere.py`** — am model via the local `amwrap/` package, anchored by
  fitting a single scalar `pwv_scaling` against the per-spw fitted τ values
  (DESIGN.md §7). am is run **once per analysis**, never inside the fit loop.
  Vertical profiles come from open-meteo (`openmeteo-requests`); the fallback
  on HTTP/timeout is an amwrap AFGL climatology (default
  `midlatitude_summer`). The local `amwrap/` directory is a checkout; the
  pinned source is the git URL in `pyproject.toml` `[tool.uv.sources]`.
- **`caltables.py`** — optional CASA `TOpac` opacity table and CALDEVICE-style
  Tcal table. **These keep `casatools.calibrater` / `casatools.table` as
  imports.** "No CASA at runtime" in this project means we don't depend on
  `buildmytasks` or a `casa` process; it does not mean zero CASA modules.

The two public surfaces in `api.py` (`tipopac(...)` one-shot function and the
`TippingAnalysis` class) wrap the same internal stages — the class exists
purely to let notebook users inspect the dataset between stages.

## Conventions that aren't visible from any one file

- **`ty`, not `mypy`.** `ty` is in the dev group and is the typechecker for
  this project. Do not propose mypy commands or `[tool.mypy]` config.
- **`tipopac_v2.6/` is reference, not a dependency.** Read it to understand
  what the rewrite must match numerically (DESIGN.md §11.3 acceptance: opacity
  within `max(0.005, 0.05·τ_v26)`; Tcal corrections within 1%). Do not import
  from it; do not modify it.
- **Antenna dim is kept even when redundant.** `tau_zenith` carries an
  `antenna` dim in all three fit modes — in `global_tau` and `tcal_solve` the
  values broadcast equal across antennas. This is deliberate (downstream code
  simplification), spelled out in DESIGN.md §5 "Representation choices".
- **Time axis is per-scan-local and NaN-padded.** No MultiIndex; the `flag`
  array masks the pad.
- **Online-flag application is one interval-overlap call**, not the four-case
  expansion at v2.6 lines ~1117–1199. If a contributor reintroduces case
  splitting, push back.
- **The `data/` directory is a symlink** to `../data/` outside the repo; the
  MS is large and shared. Don't write to it.

## When DESIGN.md and reality drift

`DESIGN.md` is versioned in-repo and is meant to track v1. If implementation
forces a change to the schema, the anchor algorithm, the fit-mode semantics,
or the acceptance criteria, update DESIGN.md in the same commit. Do not let
code-vs-doc skew accumulate silently.
