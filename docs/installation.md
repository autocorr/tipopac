---
icon: material/download
---

# Installation

## Requirements

- **Python ≥ 3.13**.
- A **Unix-like OS** (Linux or macOS). The atmospheric model `am` is
  compiled from source during install, so you also need **GNU Make** and a
  **C compiler** (e.g. GCC). The parallel build of `am` wants a compiler
  with **OpenMP** support.
- [`uv`](https://docs.astral.sh/uv/) for dependency management.

No CASA installation is required. `casatools` is pulled in as an ordinary
PyPI dependency and used only as a library (table I/O and the optional
caltable writers) — there is no `casa` process and no `buildmytasks`.

## Install

```bash
git clone https://github.com/autocorr/tipopac
cd tipopac
uv sync --no-dev          # runtime deps only
source .venv/bin/activate
```

`uv sync` creates `.venv/` and installs the locked dependencies, including
the AM wrapper [`amwrap`](https://github.com/autocorr/amwrap) from its
pinned GitHub source (`[tool.uv.sources]` in `pyproject.toml`).

!!! note "AM is built for you"
    Installing `amwrap` **automatically compiles AM from the source files
    it ships**. If a copy of `am` is already found on your `PATH`, that one
    is used instead. You do not need to download or install AM separately.

For development work (tests, linting, type-checking, building these docs),
sync the dev group as well:

```bash
uv sync                   # runtime + dev dependencies
```

Always invoke Python through `uv` so the project environment is used:

```bash
uv run python -c "import tipopac; print(tipopac.__name__)"
```

## Key dependencies

| Dependency | Role |
| --- | --- |
| `amwrap` / `am` | Atmospheric radiative transfer (compiled from source on install). |
| `casatools` | MS table I/O and optional CASA caltable output. |
| `sdmpy` | SDM reading (no BDF required). |
| `xarray`, `numpy`, `pandas` | The canonical dataset and numerics. |
| `openmeteo-requests`, `requests-cache`, `retry-requests` | Fetching vertical atmospheric profiles. |
| `altair` | Interactive HTML plots. |

## Optional: the test dataset

The slow tests (`pytest -m slow`) read the shared tipping dataset in both
of its forms: the ~7 GB Measurement Set and the matching SDM. `data/` is a
symlink to a shared location; link both into it:

```text
data/tip_test.ms  -> .../THIG0007.sb39095133.eb39266164.59246.04231435186.ms/
data/tip_test.sdm -> .../THIG0007.sb39095133.eb39266164.59246.04231435186/
```

Each slow test declares which of the two it needs, and selecting one whose
data is absent aborts the run rather than skipping it — so a green run is
never a silent no-op. An MS-only checkout can still *select* the MS-only
tests by path, but a bare `pytest -m slow` selects the SDM tests too and
aborts. Without either form, the fast unit tests still run (`uv run
pytest`, which skips the `slow` and `network` markers by default). See
[Development](development.md) for the full test matrix.
