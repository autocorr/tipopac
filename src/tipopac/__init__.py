# Single-threaded BLAS is materially faster than the default on the per-fit
# matrix sizes here (~3-param LM, ~170 rows for opacity; up to ~109-param,
# ~4590 rows for the global tcal fit). 20× wall speedup measured at full
# scale; pure overhead at small. Also a hard prerequisite for the
# multiprocessing.Pool dispatch in fit.py — 40 workers × ~10 default BLAS
# threads would oversubscribe a 40-core box.  See
# design/performance_refactor_considerations.md §1.
#
# `setdefault` so an explicit upstream export still wins.
import os as _os

for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    _os.environ.setdefault(_k, "1")
del _os, _k

from tipopac.api import Result, TippingAnalysis, tipopac  # noqa: E402

__all__ = ["Result", "TippingAnalysis", "summarize_skydip_scans", "tipopac"]


# Lazy: defer `tipopac.summary` import until first attribute access so
# `python -m tipopac.summary` does not double-load the module via the
# package's eager import path.
def __getattr__(name: str):
    if name == "summarize_skydip_scans":
        from tipopac.summary import summarize_skydip_scans

        return summarize_skydip_scans
    raise AttributeError(f"module 'tipopac' has no attribute {name!r}")
