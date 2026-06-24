---
icon: material/wrench
---

# Development

## Setup

The project uses [`uv`](https://docs.astral.sh/uv/) with `pyproject.toml`
and `uv.lock`. Sync the full environment (runtime + dev) once:

```bash
uv sync
```

Always run tools through `uv run` so the project environment is used —
never bare `python`.

## Tests

`pytest` markers gate the expensive tests; by default both `slow` and
`network` are skipped (`[tool.pytest.ini_options]` in `pyproject.toml`).

```bash
uv run pytest                              # fast unit + synth tests
uv run pytest -m "not slow"                # everything except integration
uv run pytest -m slow                      # integration; needs data/tip_test.ms
uv run pytest -m network                   # hits the live Open-Meteo endpoint
uv run pytest tests/unit/test_fit.py::test_global_tau   # a single test
```

- **`slow`** — the full-pipeline integration test against the ~7 GB
  `data/tip_test.ms` (see [Installation](installation.md#optional-the-test-dataset)).
  It uses the AFGL profile source for determinism. Worker count comes from
  `TIPOPAC_TEST_WORKERS` (unset → serial).
- **`network`** — tests that exercise the live Open-Meteo API.

Tests are organized under `tests/unit/` (one module per source module),
`tests/synth/` (synthetic-data checks), and `tests/integration/`.

## Lint, format, type-check

```bash
uv run ruff check .                        # lint
uv run ruff format .                       # format
uv run ty check src/tipopac                # type-check
```

!!! warning "Use `ty`, not mypy"
    Type-checking is done with [`ty`](https://github.com/astral-sh/ty).
    Do not introduce mypy.

## Repository layout

```text
src/tipopac/        # the package
  readers/          # MSReader, SDMReader → one canonical Dataset
  schema.py         # the canonical xarray schema + apply_flags()
  physics.py        # tipping-curve model, k2nt, Bevis T_wmt
  fit.py            # Stage A least-squares solver
  anchor.py         # Stage B PWV anchor + T_mean grid
  atmosphere.py     # profile fetch (Open-Meteo / AFGL)
  atmgrid.py        # PwvGrid + build_pwv_grid (runs am)
  plot.py, weblog.py, caltables.py, summary.py, api.py
tests/              # unit / synth / integration
design/             # design.md (the contract) + derivations
docs/               # this documentation
run/                # example driver scripts
vendor/             # read-only legacy tipopac_v1.0 / v2.6 references
```

## The design document is the contract

`design/design.md` specifies the API shape, the dataset schema, the
Stage-A/Stage-B fit architecture, and the acceptance criteria. **If an
implementation change forces a change to any of those, update
`design/design.md` in the same commit** — do not let code-vs-doc skew
accumulate. The schema in `src/tipopac/schema.py` and the SDM↔MS mapping in
§3 are parity contracts, not things to re-derive per reader.

The legacy task at
`vendor/tipopac_v2.6/lastversion/tipping/private/task_tipopac.py` is
reference only — read it to understand behavior, but do not import from it
or modify it. v2.6 numerical parity is a smoke test, not a contract.

## Building the documentation

These docs are built with [Zensical](https://zensical.org) (an
MkDocs-Material fork). The config is `zensical.toml` at the repository
root; pages live under `docs/`. The API Reference is generated from
NumPy-style docstrings via `mkdocstrings` (installed with the dev group).

```bash
uv run zensical serve                       # live preview at http://127.0.0.1:8000
uv run zensical build --clean               # render the static site into site/
```

LaTeX is rendered with MathJax (configured globally in
`docs/javascripts/mathjax.js`), so equations can be written with
`$...$` / `$$...$$` on any page. On a push to `main`, the
`.github/workflows/docs.yml` workflow builds and deploys to GitHub Pages.
