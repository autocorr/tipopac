"""Full-pipeline integration test on the validation MS.

Runs both public modes on data/tip_test.ms: the default ``independent_tau``
(Stage A+B+C) and the legacy ``independent_tau_solve`` (Stage A+B).

Uses AFGL climatology for the atmospheric model so the test is fully
deterministic without network access. The profile-source routes and the
live open-meteo call are covered in tests/unit/test_atmosphere.py.
"""

from __future__ import annotations


import numpy as np
import pytest

from tests.conftest import MS_PATH

pytestmark = pytest.mark.needs_ms


# ---------------------------------------------------------------------------
# Tests — independent_tau_solve (Stage A + B, design/independent_tau_fit.md)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_independent_tau_solve(n_workers):
    """Run the Stage-A + Stage-B path end-to-end on the validation MS.

    AFGL profile (no network) so the test is reproducible; the grid is
    built once and drives both Stage-A T_mean and Stage-B PWV anchor.
    """
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.build_atm_grids()
    ta.fit(mode="independent_tau_solve", n_workers=n_workers)
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

    # Stage B wrote pwv + pwv_err per (time group, antenna). This MS spans
    # ~25 min, so the 1 h default groups every scan together.
    assert "pwv" in ds.data_vars
    assert "pwv_err" in ds.data_vars
    assert ds["pwv"].dims == ("group", "antenna")
    assert ds["pwv_err"].dims == ("group", "antenna")
    assert ds.sizes["group"] == 1
    assert (ds.coords["scan_group"].values == 0).all()
    assert ds["am_tau"].dims == ("group", "frequency_dense")

    # In tcal_solve backend, τ_z is broadcast equal across antennas, so
    # the per-antenna PWV anchor returns identical values per antenna
    # (the `shared_pwv` semantics in the design). At least one antenna
    # must have produced a finite anchor.
    finite_mask = np.isfinite(ds["pwv"].values) & np.isfinite(ds["pwv_err"].values)
    assert finite_mask.any(), "Stage-B PWV anchor produced no finite values"
    # σ_PWV must be positive where finite.
    assert (ds["pwv_err"].values[finite_mask] > 0).all()


@pytest.mark.slow
def test_independent_tau_solve_band_selection(ds_independent_tau_solve):
    """Default `bands=None` keeps only high-frequency bands and records provenance."""
    ds = ds_independent_tau_solve

    bands_present = set(ds.coords["band"].values.tolist())
    assert bands_present <= {"Ku", "K", "Ka", "Q"}, (
        f"low-band SPWs survived default filter: {bands_present}"
    )

    assert ds.attrs["scans_requested"] == "all"
    assert ds.attrs["bands_requested"] == "default_high_freq"
    assert set(ds.attrs["selected_bands"]) == bands_present
    assert ds.attrs["selected_scans"] == list(ds.coords["scan"].values.tolist())


@pytest.mark.slow
def test_independent_tau_solve_skips_stage_c(ds_independent_tau_solve):
    """Stage C must not run against an anchor fit to the solve mode's τ."""
    assert "sigma_tcal" not in ds_independent_tau_solve.data_vars


# ---------------------------------------------------------------------------
# Tests — independent_tau (Stage A + B + C)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_independent_tau(n_workers):
    """Run the Stage-A + B + C path end-to-end on the validation MS.

    Deliberately passes no `mode`, so this exercises the library default.
    """
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.build_atm_grids()
    ta.fit(n_workers=n_workers)
    return ta.dataset


@pytest.mark.slow
def test_default_mode_runs_stage_c(ds_independent_tau):
    """The default mode is independent_tau, and it populates Stage C."""
    assert ds_independent_tau.attrs["mode"] == "independent_tau"
    assert "sigma_tcal" in ds_independent_tau.data_vars


@pytest.mark.slow
def test_independent_tau_schema(ds_independent_tau):
    """Stage-C outputs must satisfy the schema."""
    from tipopac import schema

    schema.validate(ds_independent_tau)


@pytest.mark.slow
def test_stage_c_outputs_populated(ds_independent_tau):
    """Stage C writes a finite c and a positive σ_c on some cells."""
    ds = ds_independent_tau

    for name in ("tcal_fit", "sigma_tcal"):
        assert name in ds.data_vars, f"missing Stage-C output: {name}"

    c = ds["tcal_fit"].values / ds["tcal_ref"].values[None, ...]
    sigma_c = ds["sigma_tcal"].values / ds["tcal_ref"].values[None, ...]
    finite = np.isfinite(c)
    assert finite.any(), "Stage C produced no finite c"

    # c and σ_c are measured together; σ_c is the measured-ness flag.
    np.testing.assert_array_equal(finite, np.isfinite(sigma_c))
    assert (sigma_c[finite] > 0).all()

    # The array-common level is absorbed by the anchor, so c sits near 1.
    assert 0.5 < float(np.median(c[finite])) < 2.0


@pytest.mark.slow
def test_stage_c_respects_min_airmass_span(n_workers):
    """An unreachable leverage floor gates every cell to NaN."""
    from tipopac import TippingAnalysis

    ta = TippingAnalysis.from_path(MS_PATH)
    ta.apply_flags(online=True)
    ta.fetch_atm_profile(source="afgl")
    ta.build_atm_grids()
    ta.fit(mode="independent_tau", n_workers=n_workers, min_airmass_span=100.0)

    assert not np.isfinite(ta.dataset["tcal_fit"].values).any()
    assert not np.isfinite(ta.dataset["sigma_tcal"].values).any()
