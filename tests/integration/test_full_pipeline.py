"""Full-pipeline integration test on the validation MS.

Runs the Stage A+B ``independent_tau_solve`` pipeline on data/tip_test.ms.

Uses AFGL climatology for the atmospheric model so the test is fully
deterministic without network access. A separate @pytest.mark.network
test exercises the live open-meteo call.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"


def _n_workers() -> int | None:
    """Read `TIPOPAC_TEST_WORKERS` env var to set Stage-A fit parallelism.

    CI / local runs can export this var to a sensible value
    (e.g. ``min(16, cpu_count())``); unset → serial (the historical
    default).
    """
    v = os.environ.get("TIPOPAC_TEST_WORKERS")
    return int(v) if v else None


# ---------------------------------------------------------------------------
# Tests — independent_tau_solve (Stage A + B, design/independent_tau_fit.md)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_independent_tau_solve():
    """Run the Stage-A + Stage-B path end-to-end on the validation MS.

    AFGL profile (no network) so the test is reproducible; the grid is
    built once and drives both Stage-A T_mean and Stage-B PWV anchor.
    """
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.build_atm_grids()
    ta.fit(mode="independent_tau_solve", n_workers=_n_workers())
    return ta.dataset


@pytest.mark.slow
def test_independent_tau_solve_schema(ds_independent_tau_solve):
    """Pipeline output (including pwv, pwv_err) must satisfy the schema."""
    from tipopac import schema

    schema.validate(ds_independent_tau_solve)


@pytest.mark.slow
def test_independent_tau_solve_outputs_populated(ds_independent_tau_solve):
    """Stage A τ + Tcal and Stage B PWV must be finite for some antenna."""
    ds = ds_independent_tau_solve

    # Mode label is the *public* mode, not the Stage-A backend name.
    assert ds.attrs["mode"] == "independent_tau_solve"

    # Stage A wrote tau_zenith / tcal_fit.
    for name in ("tau_zenith", "tau_err", "tcal_fit", "fit_success"):
        assert name in ds.data_vars, f"missing Stage-A output: {name}"

    # At least one (scan, antenna, spw) cell should have a successful fit.
    assert bool(ds["fit_success"].values.any()), "no Stage-A fits succeeded"
    assert np.isfinite(ds["tau_zenith"].values).any(), "all tau_zenith are NaN"

    # Stage B wrote pwv + pwv_err per antenna.
    assert "pwv" in ds.data_vars
    assert "pwv_err" in ds.data_vars
    assert ds["pwv"].dims == ("antenna",)
    assert ds["pwv_err"].dims == ("antenna",)

    # In tcal_solve backend, τ_z is broadcast equal across antennas, so
    # the per-antenna PWV anchor returns identical values per antenna
    # (the `shared_pwv` semantics in the design). At least one antenna
    # must have produced a finite anchor.
    finite_mask = np.isfinite(ds["pwv"].values) & np.isfinite(ds["pwv_err"].values)
    assert finite_mask.any(), "Stage-B PWV anchor produced no finite values"
    # σ_PWV must be positive where finite.
    assert (ds["pwv_err"].values[finite_mask] > 0).all()
