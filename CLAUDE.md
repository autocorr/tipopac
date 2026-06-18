# CLAUDE.md

This is a clean-Python rewrite of the legacy CASA task `tipopac` (VLA
tipping-curve opacity / Tcal estimation). The package is `tipopac` under
`src/tipopac/`.

- `design/design.md` — the spec / requirements contract. Check it
  first for ambiguities concerning API shape, schema, fit modes, the
  Stage-A/Stage-B fit architecture, or acceptance criteria. Flag
  disagreements with code and raise them to the user.
- `tipopac_v2.6/lastversion/tipping/private/task_tipopac.py` — **read-
  only** legacy reference (~1900 lines). Useful for understanding
  behaviour; do not import from it and do not modify it.
- `old_context/` — superseded design notes (`initial_design.md`,
  `independent_tau_fit.md`, `model_refactor.md`, …). Historical
  context only; `design/design.md` overrides any of these on conflict.
- `references/ms_v2_memo_299.html` — MeasurementSet v2 spec.

## Toolchain & commands

`uv` with `pyproject.toml` + `uv.lock`. Python ≥ 3.13. Always use
`uv run python`, never bare `python`.

```bash
uv sync                                # install / sync deps
uv run pytest                          # all tests
uv run pytest -m "not slow"            # skip integration (needs data/tip_test.ms)
uv run ruff check . && uv run ruff format .
uv run ty check src/tipopac            # type-check — `ty`, NEVER mypy
```

The integration test is gated by `pytest.mark.slow` and needs the
~7 GB MS at `data/tip_test.ms`.

## Conventions that aren't visible from any one file

- **`design/design.md` is the contract.** If implementation forces a
  change to the schema, the fit architecture, the anchor algorithm,
  or acceptance criteria, update `design/design.md` in the same
  commit. Do not let code-vs-doc skew accumulate silently.
- **The canonical `xarray.Dataset` schema is defined in
  `src/tipopac/schema.py` and spec'd in design.md §4.** Both readers
  (MS, SDM) must produce the same dataset; the SDM↔MS column-mapping
  table in design.md §3 is the parity contract, not something to
  re-derive.
- **Antenna dim is retained even when degenerate.** `tau_zenith`
  carries an `antenna` dim in both modes; under `independent_tau_solve`
  the per-antenna values broadcast equal. Same for `pwv(antenna)`.
  Downstream code simplification — deliberate.
- **Time axis is per-scan-local and NaN-padded.** No MultiIndex; the
  `flag` array masks the pad. Reductions over `time` must go through
  `schema.apply_flags(ds, var)`, never `ds[var]` directly.
- **Online-flag application is one interval-overlap expression**, not
  the four-case interval expansion at v2.6 lines ~1117–1199. If a
  contributor reintroduces case splitting, push back.
- **am runs once per analysis** (during `build_atm_grids`), never
  inside the per-sample fit loop. Stage B is a 1-D bounded scalar
  fit against the precomputed `PwvGrid` — no second am call.
- **Local `amwrap/`** is a checkout; the reproducible pin is the git
  URL in `pyproject.toml` `[tool.uv.sources]`.
- **"No CASA at runtime"** means no `buildmytasks` and no `casa`
  process. `casatools.table` / `casatools.calibrater` are ordinary
  imports used by readers and the optional caltable writers.
- **v2.6 numerical parity is a smoke test, not a contract.** The
  rewrite uses radiometer-eq σ + `soft_l1` + single-tier bounds + an
  identifiability ratio in place of v2.6's unit-weight L2 + 2σ clip +
  3-pass bound escalation + geometric `dz`/`min(z)` gates. Drift is
  expected.
- **`data/` is a symlink** to `../data/`. The MS is large and shared;
  don't write to it.
