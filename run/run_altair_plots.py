#!/usr/bin/env python
"""End-to-end driver: build every altair plot for ``data/tip_test.ms``.

Runs the full pipeline (read MS → flags → atm profile → grids → fit) and
writes one interactive ``.html`` per plot plus an ``index.html`` under
``run/altair_plots/``. Caches the fitted dataset alongside as
``dataset.nc`` so re-runs go straight to plotting; delete the cache to
force a fresh pipeline run.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_MS = REPO / "data" / "tip_test.ms"
OUT_DIR = REPO / "run" / "altair_plots"
CACHE_PATH = OUT_DIR / "dataset.nc"

MODE = "independent_tau_solve"
PROFILE_SOURCE = "open-meteo"
AFGL_CLIMATOLOGY = "auto"
N_WORKERS = 40


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("run_altair_plots")

    import xarray as xr

    from tipopac.plot import PlotData

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        log.info("loading cached fitted dataset from %s", CACHE_PATH)
        ds = xr.open_dataset(CACHE_PATH).load()
    else:
        if not DATA_MS.exists():
            sys.exit(f"ERROR: {DATA_MS} not found")
        ds = _run_pipeline(log)
        log.info("caching fitted dataset to %s", CACHE_PATH)
        ds.to_netcdf(CACHE_PATH)

    log.info("plotting to %s", OUT_DIR)
    t0 = time.perf_counter()
    PlotData(ds).save_all(OUT_DIR)
    log.info("plots done in %.1f s", time.perf_counter() - t0)
    log.info("open %s in a browser", OUT_DIR / "index.html")


def _run_pipeline(log: logging.Logger):
    from tipopac import flags as _flags
    from tipopac.api import TippingAnalysis
    from tipopac.readers.ms import MSReader

    log.info("reading %s", DATA_MS)
    t0 = time.perf_counter()
    ds = MSReader.from_path(DATA_MS).read()
    log.info("read in %.1f s; sizes=%s", time.perf_counter() - t0, dict(ds.sizes))

    log.info("applying online flags")
    ds = _flags.apply(ds, online=True, file=None)

    ta = TippingAnalysis(ds, DATA_MS)
    log.info("fetching atm profile (source=%s)", PROFILE_SOURCE)
    ta.fetch_atm_profile(source=PROFILE_SOURCE, afgl_climatology=AFGL_CLIMATOLOGY)
    log.info("building per-scan PWV grids")
    ta.build_atm_grids()
    log.info("fitting (mode=%s, n_workers=%d)", MODE, N_WORKERS)
    t0 = time.perf_counter()
    ta.fit(mode=MODE, n_workers=N_WORKERS)
    log.info("fit done in %.1f s", time.perf_counter() - t0)
    return ta.result.dataset


if __name__ == "__main__":
    main()
