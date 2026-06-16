"""Unit tests for tipopac.api (PwvGrid cache in build_atm_grids)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tipopac.api import TippingAnalysis
from tipopac.atmgrid import PwvGrid


def _toy_grid() -> PwvGrid:
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
        data_vars["surface_pressure_hPa"] = (("scan",), surface_P_hPa.astype(np.float64))

    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "scan": np.arange(1, n_scan + 1, dtype=np.intp),
            "frequency": (("spw",), np.array([22.2e9], dtype=np.float64)),
        },
        attrs={"atm_profile_source": "afgl_midlatitude_winter"},
    )


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
        return _toy_grid()

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
        lambda *a, **kw: (call_count.__setitem__("n", call_count["n"] + 1) or _toy_grid()),
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
        lambda *a, **kw: (call_count.__setitem__("n", call_count["n"] + 1) or _toy_grid()),
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
        lambda *a, **kw: (call_count.__setitem__("n", call_count["n"] + 1) or _toy_grid()),
    )

    TippingAnalysis(ds, Path("fake.ms")).build_atm_grids()
    assert call_count["n"] == 1
