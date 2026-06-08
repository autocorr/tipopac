"""Full-pipeline integration test vs v2.6 reference output.

Runs the tipopac pipeline on data/tip_test.ms for all three fit modes and
compares numerical results against the frozen v2.6 reference in
tests/integration/reference/v26/.

Uses AFGL climatology for the atmospheric model so the test is fully
deterministic without network access.  A separate @pytest.mark.network
test exercises the live open-meteo call.

Acceptance thresholds (DESIGN.md §11.3):
  - Zenith opacity: max(0.005, 0.05 · |τ_v26|) nepers
  - Tcal corrections: within 1% of v2.6 per (antenna, spw, polarization)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"
REF_DIR = Path(__file__).parent / "reference" / "v26"


def _n_workers() -> int | None:
    """Read `TIPOPAC_TEST_WORKERS` env var to set Stage-A fit parallelism.

    The legacy ``global_tau`` / ``tcal_solve`` modes have a per-(scan, spw)
    Jacobian large enough that the serial path takes ~hours on the full
    MS. CI / local runs can export this var to a sensible value
    (e.g. ``min(16, cpu_count())``); unset → serial (the historical
    default).
    """
    v = os.environ.get("TIPOPAC_TEST_WORKERS")
    return int(v) if v else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_ref(mode: str) -> dict:
    path = REF_DIR / mode / "reference.json"
    if not path.exists():
        pytest.skip(f"reference not found: {path}")
    with path.open() as f:
        return json.load(f)


def _ref_array(ref: dict, name: str) -> np.ndarray:
    """Reconstruct an ndarray from the JSON reference, replacing null with NaN."""
    entry = ref["data_vars"][name]
    flat = _to_float(entry["data"])
    dims = entry["dims"]
    coord_lens = {
        "scan": len(ref["coords"]["scan"]),
        "antenna": len(ref["coords"]["antenna"]),
        "spw": len(ref["coords"]["spw"]),
        "polarization": len(ref["coords"]["polarization"]),
    }
    shape = tuple(coord_lens[d] for d in dims)
    return flat.reshape(shape)


def _to_float(obj) -> np.ndarray:
    """Recursively flatten a nested list of float|None → float64 array."""
    arr = np.array(obj, dtype=object)
    out = np.empty(arr.shape, dtype=np.float64)
    none_mask = arr == None  # noqa: E711
    out[none_mask] = np.nan
    out[~none_mask] = arr[~none_mask].astype(np.float64)
    return out.flatten()


def _excluded_set(ref: dict) -> set[tuple[int, int]]:
    """Parse acceptance_excluded_cells into a set of (scan_idx, spw_idx) pairs."""
    raw = ref["attrs"].get("acceptance_excluded_cells", "")
    if not raw:
        return set()
    return {tuple(int(x) for x in p.split(",")) for p in raw.split(";") if p}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_tau_per_antenna():
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.fit(mode="tau_per_antenna", n_workers=_n_workers())
    ta.extrapolate()
    return ta.dataset


@pytest.fixture(scope="module")
def ds_global_tau():
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.fit(mode="global_tau", n_workers=_n_workers())
    ta.extrapolate()
    return ta.dataset


@pytest.fixture(scope="module")
def ds_tcal_solve():
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.fit(mode="tcal_solve", n_workers=_n_workers())
    ta.extrapolate()
    return ta.dataset


# ---------------------------------------------------------------------------
# Tests — tau_per_antenna
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tau_per_antenna_schema(ds_tau_per_antenna):
    """Pipeline output must satisfy the canonical schema."""
    from tipopac import schema

    schema.validate(ds_tau_per_antenna)


@pytest.mark.slow
def test_tau_per_antenna_vs_v26(ds_tau_per_antenna):
    """Per-antenna τ must agree with v2.6 within max(0.005, 0.05·τ_v26)."""
    ref = _load_ref("tau_per_antenna")
    exc = _excluded_set(ref)

    n_scan = len(ref["coords"]["scan"])
    n_ant = len(ref["coords"]["antenna"])
    n_spw = len(ref["coords"]["spw"])

    pytest.skip(
        "v2.6 numerical-parity retired as a hard acceptance gate after the "
        "Stage-1 solver refactor (radiometer-eq σ, soft_l1 loss, single physical "
        "bounds, identifiability gate). See design/initial_design.md §11.3 and "
        "design/model_refactor.md. Synthetic fixtures in tests/synth/ are the "
        "primary acceptance for the post-refactor pipeline. This test remains "
        "as a manual smoke check — rerun with `--run-v26-comparison` (not "
        "implemented) when needed."
    )
    # Reference variables retained for future manual diff inspection:
    _ = (ref, exc, n_scan, n_ant, n_spw, ds_tau_per_antenna)


# ---------------------------------------------------------------------------
# Tests — global_tau
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_global_tau_schema(ds_global_tau):
    from tipopac import schema

    schema.validate(ds_global_tau)


@pytest.mark.slow
def test_global_tau_vs_v26(ds_global_tau):
    """Global τ comparison is skipped: DESIGN.md §12 scope exclusion.

    v2.6 global_tau uses a three-pass optimizer with progressively relaxed
    bounds (tauPerAnt=False, calcTcals=False). v1 uses a single pass with
    layer-3 bounds. The two solvers converge to different local minima
    systematically for this mode; the v1 acceptance criteria in DESIGN.md §11.3
    explicitly excludes cells that required layer-2/layer-3 escalation, but
    does not commit to matching v2.6 global_tau numerically.
    """
    pytest.skip(
        "global_tau τ comparison not valid vs v2.6: "
        "multi-pass optimizer escalation is out of scope for v1 (DESIGN.md §12)"
    )


# ---------------------------------------------------------------------------
# Tests — tcal_solve
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_tcal_solve_schema(ds_tcal_solve):
    from tipopac import schema

    schema.validate(ds_tcal_solve)


@pytest.mark.slow
def test_tcal_solve_tau_vs_v26(ds_tcal_solve):
    """tcal_solve global τ vs v2.6 — retired (see initial_design.md §11.3)."""
    pytest.skip(
        "v2.6 numerical-parity retired as a hard acceptance gate after the "
        "Stage-1 solver refactor (radiometer-eq σ, soft_l1 loss, single physical "
        "bounds, identifiability gate). See design/initial_design.md §11.3 and "
        "design/model_refactor.md. Synthetic fixtures in tests/synth/ are the "
        "primary acceptance for the post-refactor pipeline."
    )
    _ = ds_tcal_solve


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


@pytest.mark.slow
def test_tcal_solve_tcal_vs_v26(ds_tcal_solve):
    """Fitted Tcal vs v2.6 — retired (see initial_design.md §11.3)."""
    pytest.skip(
        "v2.6 numerical-parity retired as a hard acceptance gate after the "
        "Stage-1 solver refactor. See design/initial_design.md §11.3 and "
        "design/model_refactor.md."
    )
    ref = _load_ref("tcal_solve")
    if "tcal_fit" not in ref["data_vars"]:
        pytest.skip("no tcal_fit in reference")

    exc = _excluded_set(ref)
    n_scan = len(ref["coords"]["scan"])
    n_spw = len(ref["coords"]["spw"])
    n_ant = len(ref["coords"]["antenna"])

    tcal_v26 = _ref_array(ref, "tcal_fit")  # (n_scan, n_ant, n_spw, 2)
    tcal_v1 = ds_tcal_solve["tcal_fit"].values  # (n_scan, n_ant, n_spw, 2)

    failures: list[str] = []
    n_compared = 0

    for si in range(n_scan):
        for wi in range(n_spw):
            if (si, wi) in exc:
                continue
            for ai in range(n_ant):
                for pi, pol in enumerate(["R", "L"]):
                    t26 = tcal_v26[si, ai, wi, pi]
                    t1 = tcal_v1[si, ai, wi, pi]
                    if not np.isfinite(t26) or not np.isfinite(t1):
                        continue
                    # -999 is the sentinel written when caltableT row count
                    # doesn't match expected (n_scan × n_ant × n_spw).
                    if t26 < 0:
                        continue
                    # Tolerance: max(0.01 K, 6% of v2.6 tcal value).
                    # See DESIGN.md §11.3 — convergence-ridge sensitivity.
                    tol = max(0.01, 0.06 * abs(t26))
                    if abs(t1 - t26) > tol:
                        ant = ref["coords"]["antenna"][ai]
                        spw = ref["coords"]["spw"][wi]
                        failures.append(
                            f"scan_idx={si} ant={ant} spw={spw} pol={pol}: "
                            f"v1={t1:.4f} v26={t26:.4f} tol={tol:.4f}"
                        )
                    n_compared += 1

    if n_compared == 0:
        pytest.skip("no Tcal cells to compare")
    assert not failures, (
        f"{len(failures)}/{n_compared} Tcal cells outside max(0.01 K, 6%) tolerance:\n"
        + "\n".join(failures[:20])
    )
