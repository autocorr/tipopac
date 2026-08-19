"""Unit tests for tipopac.api (PwvGrid cache in build_atm_grids)."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tipopac.api import TippingAnalysis, _module_version
from tipopac.atmgrid import PwvGrid


def _stub_grid() -> PwvGrid:
    pwv = np.array([1.0, 2.0], dtype=np.float64)
    freq = np.array([20e9, 25e9], dtype=np.float64)
    tau = np.array([[0.01, 0.012], [0.02, 0.024]], dtype=np.float64)
    tb = np.array([[10.0, 12.0], [20.0, 22.0]], dtype=np.float64)
    return PwvGrid(pwv_mm=pwv, freq_Hz=freq, tau_z=tau, tb_z=tb, pwv_unscaled_mm=1.5)


def _make_ds(
    n_scan: int,
    *,
    atm_p_Pa: np.ndarray,
    surface_P_hPa: np.ndarray | None,
) -> xr.Dataset:
    """Minimal dataset for build_atm_grids: per-scan profile + frequency coord."""
    n_level = atm_p_Pa.shape[1]
    atm_T = np.broadcast_to(
        np.linspace(280.0, 210.0, n_level, dtype=np.float32), (n_scan, n_level)
    ).copy()
    atm_h = np.broadcast_to(
        np.logspace(-3, -6, n_level, dtype=np.float32), (n_scan, n_level)
    ).copy()
    data_vars: dict = {
        "atm_pressure": (("scan", "atm_level"), atm_p_Pa.astype(np.float64)),
        "atm_temperature": (("scan", "atm_level"), atm_T),
        "atm_h2o_vmr": (("scan", "atm_level"), atm_h),
    }
    if surface_P_hPa is not None:
        data_vars["surface_pressure_hPa"] = (
            ("scan",),
            surface_P_hPa.astype(np.float64),
        )

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "frequency": (("spw",), np.array([22.2e9], dtype=np.float64)),
        },
        attrs={"atm_profile_source": "afgl_midlatitude_winter"},
    )


def test_module_version_falls_back_to_distribution_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Modules with no ``__version__`` (casatools, sdmpy) report the installed one."""
    import importlib
    import importlib.metadata
    import types

    monkeypatch.setattr(importlib, "import_module", lambda name: types.ModuleType(name))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.71.2")

    assert _module_version("sdmpy") == "1.71.2"


def test_module_version_unavailable_when_import_fails() -> None:
    """A module that cannot be imported reports the sentinel, not a crash."""
    assert _module_version("no_such_module_qqq") == "unavailable"


def test_software_versions_attr_written_at_construction() -> None:
    """The §4 attr lands on the dataset, so any archive of it keeps provenance."""
    ds = _make_ds(1, atm_p_Pa=np.array([[85000.0, 50000.0]]), surface_P_hPa=None)

    TippingAnalysis(ds, Path("fake.ms"))

    versions = json.loads(ds.attrs["software_versions"])
    assert set(versions) == {"tipopac", "casatools", "sdmpy", "amwrap", "am"}
    assert all(isinstance(v, str) and v for v in versions.values())


def test_build_atm_grids_reuses_grid_for_identical_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three scans with identical profiles and tight surface P → one build."""
    n_scan = 3
    atm_p_Pa = np.broadcast_to(
        np.array([85000, 70000, 50000, 30000], dtype=np.float64), (n_scan, 4)
    ).copy()
    ds = _make_ds(
        n_scan,
        atm_p_Pa=atm_p_Pa,
        surface_P_hPa=np.array([850.0, 850.1, 849.9]),
    )

    call_count = {"n": 0}

    def _stub(*args: object, **kwargs: object) -> PwvGrid:
        call_count["n"] += 1
        return _stub_grid()

    monkeypatch.setattr("tipopac.atmgrid.build_pwv_grid", _stub)

    ta = TippingAnalysis(ds, Path("fake.ms"))
    ta.build_atm_grids()

    assert call_count["n"] == 1
    grids = list(ta._grids.values())
    assert len(grids) == n_scan
    assert all(id(g) == id(grids[0]) for g in grids)


def test_build_atm_grids_rebuilds_when_surface_pressure_exceeds_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two scans with identical profiles but |dP| > 0.2 hPa → two builds."""
    n_scan = 2
    atm_p_Pa = np.broadcast_to(
        np.array([85000, 70000, 50000, 30000], dtype=np.float64), (n_scan, 4)
    ).copy()
    ds = _make_ds(
        n_scan,
        atm_p_Pa=atm_p_Pa,
        surface_P_hPa=np.array([850.0, 850.21]),
    )

    call_count = {"n": 0}
    monkeypatch.setattr(
        "tipopac.atmgrid.build_pwv_grid",
        lambda *a, **kw: (
            call_count.__setitem__("n", call_count["n"] + 1) or _stub_grid()
        ),
    )

    TippingAnalysis(ds, Path("fake.ms")).build_atm_grids()
    assert call_count["n"] == 2


def test_build_atm_grids_reuses_at_exact_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """|dP| == 0.2 hPa is inclusive → one build."""
    n_scan = 2
    atm_p_Pa = np.broadcast_to(
        np.array([85000, 70000, 50000, 30000], dtype=np.float64), (n_scan, 4)
    ).copy()
    ds = _make_ds(
        n_scan,
        atm_p_Pa=atm_p_Pa,
        surface_P_hPa=np.array([850.0, 850.2]),
    )

    call_count = {"n": 0}
    monkeypatch.setattr(
        "tipopac.atmgrid.build_pwv_grid",
        lambda *a, **kw: (
            call_count.__setitem__("n", call_count["n"] + 1) or _stub_grid()
        ),
    )

    TippingAnalysis(ds, Path("fake.ms")).build_atm_grids()
    assert call_count["n"] == 1


def test_build_atm_grids_reuses_when_no_surface_pressure_data_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No surface_pressure_hPa data var + identical profiles → one build."""
    n_scan = 2
    atm_p_Pa = np.broadcast_to(
        np.array([85000, 70000, 50000, 30000], dtype=np.float64), (n_scan, 4)
    ).copy()
    ds = _make_ds(n_scan, atm_p_Pa=atm_p_Pa, surface_P_hPa=None)

    call_count = {"n": 0}
    monkeypatch.setattr(
        "tipopac.atmgrid.build_pwv_grid",
        lambda *a, **kw: (
            call_count.__setitem__("n", call_count["n"] + 1) or _stub_grid()
        ),
    )

    TippingAnalysis(ds, Path("fake.ms")).build_atm_grids()
    assert call_count["n"] == 1


def test_build_atm_grids_uses_full_band_span(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grid spans 1–51 GHz, not the observed spw range."""
    atm_p_Pa = np.array([[85000, 70000, 50000, 30000]], dtype=np.float64)
    ds = _make_ds(1, atm_p_Pa=atm_p_Pa, surface_P_hPa=None)

    seen: dict = {}

    def _stub(*args: object, **kwargs: object) -> PwvGrid:
        seen.update(kwargs)
        return _stub_grid()

    monkeypatch.setattr("tipopac.atmgrid.build_pwv_grid", _stub)

    TippingAnalysis(ds, Path("fake.ms")).build_atm_grids()

    assert seen["freq_min_Hz"] == pytest.approx(1e9)
    assert seen["freq_max_Hz"] == pytest.approx(51e9)


# ---------------------------------------------------------------------------
# Public mode defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "constant"),
    [
        ("spillover_model", "DEFAULT_SPILLOVER_MODEL"),
        ("group_duration_s", "DEFAULT_GROUP_DURATION_S"),
        ("min_airmass_span", "DEFAULT_MIN_AIRMASS_SPAN"),
    ],
)
def test_shared_defaults_come_from_one_object(param: str, constant: str) -> None:
    """Every signature declaring a shared default resolves to the same object.

    Re-inlining the literal at any one site puts a second object in the set
    and fails here, including at a backend the public API always forwards to
    explicitly (review L4).
    """
    import inspect

    from tipopac import defaults, fit, tcal
    from tipopac.api import TippingAnalysis, tipopac

    candidates = (tipopac, TippingAnalysis.fit, fit.fit_dataset, tcal.solve_tcal)
    found = {
        id(inspect.signature(f).parameters[param].default)
        for f in candidates
        if param in inspect.signature(f).parameters
    }
    assert len(found) == 1, f"{param} is declared from more than one object"
    assert found == {id(getattr(defaults, constant))}


def test_grid_step_defaults_come_from_atmgrid() -> None:
    """build_atm_grids must not restate the am-grid geometry literals."""
    import inspect

    from tipopac import atmgrid

    params = inspect.signature(TippingAnalysis.build_atm_grids).parameters
    assert params["pwv_step_mm"].default is atmgrid.DEFAULT_PWV_STEP_MM
    assert params["freq_step_Hz"].default is atmgrid.DEFAULT_FREQ_STEP_HZ


# ---------------------------------------------------------------------------
# API-reference coverage
# ---------------------------------------------------------------------------

# mkdocstrings omits members with no docstring, so an undocumented public
# member silently vanishes from docs/api.md rather than failing the build.


def _public_members(obj: type) -> list[str]:
    return [
        n
        for n in vars(obj)
        if not n.startswith("_")
        and (callable(getattr(obj, n)) or isinstance(vars(obj)[n], property))
    ]


@pytest.mark.parametrize("name", _public_members(TippingAnalysis))
def test_tipping_analysis_members_are_documented(name: str) -> None:
    assert getattr(TippingAnalysis, name).__doc__, f"{name} would be omitted"


def test_result_fields_are_documented() -> None:
    import dataclasses

    from tipopac.api import Result

    lines = inspect.getsource(Result).splitlines()
    for f in dataclasses.fields(Result):
        i = next(
            i for i, ln in enumerate(lines) if ln.strip().startswith(f"{f.name}: ")
        )
        assert lines[i + 1].strip().startswith('"""'), f"{f.name} would be omitted"


@pytest.mark.parametrize(
    "method", ["from_path", "apply_flags", "fit", "build_atm_grids"]
)
def test_staged_methods_share_tipopac_parameter_names(method: str) -> None:
    """The docstrings cross-reference tipopac() rather than restate it."""
    from tipopac.api import tipopac

    documented = set(inspect.signature(tipopac).parameters) | {
        "online",
        "file",
        "pwv_step_mm",
        "freq_step_Hz",
    }
    params = set(inspect.signature(getattr(TippingAnalysis, method)).parameters)
    undocumented = params - documented - {"self", "cls"}
    assert not undocumented, f"{method}: {sorted(undocumented)} not in tipopac()"
