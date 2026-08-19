"""Unit tests for the public entry points: ``tipopac()``, ``write_outputs``, ``Result``.

The one-shot function is exercised with every stage monkeypatched, so what is
under test is the orchestration — stage order and argument forwarding — not the
stages themselves. ``write_outputs`` runs for real on a synthetic fitted
dataset; only the caltable writers (which need CASA) are stubbed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from tests.factories import make_fitted_dataset
from tipopac import caltables
from tipopac.api import Result, TippingAnalysis, tipopac


def _output_ds() -> xr.Dataset:
    """Default-mode fitted dataset with every optional output var present."""
    return make_fitted_dataset(
        n_scan=2,
        n_ant=3,
        n_spw=2,
        with_am=True,
        with_atm=True,
        with_stage_c=True,
        with_pwv=True,
        mode="independent_tau",
    )


@pytest.fixture
def stub_stages(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Replace every pipeline stage with a recorder; return the call log."""
    calls: list[tuple[str, dict[str, Any]]] = []
    ds = _output_ds()

    class _StubReader:
        @classmethod
        def from_path(cls, path: Path, **kwargs: Any) -> "_StubReader":
            calls.append(("from_path", {"path": path, **kwargs}))
            return cls()

        def read(self) -> xr.Dataset:
            return ds

    monkeypatch.setattr("tipopac.api._detect_reader", lambda p: _StubReader)

    def _record(name: str) -> Any:
        def _stage(self: TippingAnalysis, **kwargs: Any) -> None:
            calls.append((name, kwargs))
            if name == "fit":
                self._mode = kwargs["mode"]

        return _stage

    for name in ("apply_flags", "fetch_atm_profile", "build_atm_grids", "fit"):
        monkeypatch.setattr(TippingAnalysis, name, _record(name))

    def _write_outputs(
        self: TippingAnalysis, output_dir: str | Path = Path("."), **kwargs: Any
    ) -> None:
        calls.append(("write_outputs", {"output_dir": output_dir, **kwargs}))

    monkeypatch.setattr(TippingAnalysis, "write_outputs", _write_outputs)
    return calls


# ---------------------------------------------------------------------------
# tipopac() orchestration
# ---------------------------------------------------------------------------


def test_tipopac_rejects_unknown_mode() -> None:
    """The mode guard fires before the reader runs, so no I/O is attempted."""
    with pytest.raises(ValueError, match="mode must be one of"):
        tipopac("/no/such/path.ms", mode="joint_tau")


def test_tipopac_runs_the_stages_in_order(
    stub_stages: list[tuple[str, dict[str, Any]]],
) -> None:
    tipopac("fake.ms", output_dir="out")

    assert [name for name, _ in stub_stages] == [
        "from_path",
        "apply_flags",
        "fetch_atm_profile",
        "build_atm_grids",
        "fit",
        "write_outputs",
    ]


def test_tipopac_forwards_every_argument_to_its_stage(
    stub_stages: list[tuple[str, dict[str, Any]]], tmp_path: Path
) -> None:
    """Each keyword reaches the stage that owns it, unaltered."""
    flag_file = tmp_path / "flags.txt"
    tipopac(
        "fake.ms",
        scans=[3, 5],
        bands=["K"],
        mode="independent_tau_solve",
        flags_online=False,
        flags_file=flag_file,
        atm_profile_source="afgl",
        afgl_climatology="midlatitude_winter",
        spillover_model=False,
        group_duration_s=None,
        min_airmass_span=0.7,
        n_workers=4,
        output_dir=tmp_path / "out",
        caltable_opacity=True,
        caltable_tcal=True,
    )
    by_name = dict(stub_stages)

    assert by_name["from_path"] == {
        "path": Path("fake.ms"),
        "scans": [3, 5],
        "bands": ["K"],
    }
    assert by_name["apply_flags"] == {"online": False, "file": flag_file}
    assert by_name["fetch_atm_profile"] == {
        "source": "afgl",
        "afgl_climatology": "midlatitude_winter",
    }
    assert by_name["build_atm_grids"] == {"n_workers": 4}
    assert by_name["fit"] == {
        "mode": "independent_tau_solve",
        "n_workers": 4,
        "spillover_model": False,
        "group_duration_s": None,
        "min_airmass_span": 0.7,
    }
    assert by_name["write_outputs"] == {
        "output_dir": tmp_path / "out",
        "caltable_opacity": True,
        "caltable_tcal": True,
    }


def test_tipopac_str_flag_file_becomes_a_path(
    stub_stages: list[tuple[str, dict[str, Any]]],
) -> None:
    tipopac("fake.ms", flags_file="flags.txt", output_dir=None)
    assert dict(stub_stages)["apply_flags"]["file"] == Path("flags.txt")


def test_tipopac_output_dir_none_writes_nothing(
    stub_stages: list[tuple[str, dict[str, Any]]],
) -> None:
    """Compute-only mode: a Result comes back, but no writer runs."""
    result = tipopac("fake.ms", output_dir=None)

    assert "write_outputs" not in [name for name, _ in stub_stages]
    assert isinstance(result, Result)
    assert result.mode == "independent_tau"
    assert result.input_path == Path("fake.ms")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_result_before_fit_raises() -> None:
    ta = TippingAnalysis(_output_ds(), Path("fake.ms"))
    with pytest.raises(RuntimeError, match="call fit"):
        _ = ta.result


def test_result_carries_the_dataset_and_provenance() -> None:
    ds = _output_ds()
    ds.attrs["source_format"] = "sdm"
    ta = TippingAnalysis(ds, Path("fake.sdm"))
    ta._mode = "independent_tau"

    result = ta.result

    assert result.dataset is ds
    assert result.input_format == "sdm"
    assert result.software_versions == ds.attrs["software_versions"]


def test_result_input_format_defaults_to_ms() -> None:
    """The readers always set ``source_format``; a hand-built dataset may not."""
    ta = TippingAnalysis(_output_ds(), Path("fake.ms"))
    ta._mode = "independent_tau"
    assert ta.result.input_format == "ms"


# ---------------------------------------------------------------------------
# write_outputs
# ---------------------------------------------------------------------------


def test_write_outputs_weblog_indexes_the_plots_it_wrote(tmp_path: Path) -> None:
    """The load-bearing ordering: plots are on disk before the weblog scans."""
    out = tmp_path / "run" / "outputs"
    TippingAnalysis(_output_ds(), Path("fake.ms")).write_outputs(out)

    body = (out / "index.html").read_text(encoding="utf-8")
    for rel in (
        "group_0/tau_vs_frequency.html",
        "group_0/summary.html",
        "group_0/tcal_fit_vs_frequency.html",
        "run_summary.html",
    ):
        assert f'"{rel}"' in body
        assert (out / rel).exists()


def test_write_outputs_writes_the_netcdf_and_both_tsvs(tmp_path: Path) -> None:
    ds = _output_ds()
    TippingAnalysis(ds, Path("fake.ms")).write_outputs(tmp_path)

    reopened = xr.open_dataset(tmp_path / "tipopac.nc")
    try:
        np.testing.assert_allclose(
            reopened["tau_zenith"].values, ds["tau_zenith"].values
        )
    finally:
        reopened.close()

    model = (tmp_path / "model_opacity.tsv").read_text().splitlines()
    assert model[0] == "group\tfrequency_Hz\ttau_model"
    assert len(model) > 1

    measured = (tmp_path / "measured_opacity.tsv").read_text().splitlines()
    assert measured[0].split("\t") == [
        "group",
        "scan",
        "spw",
        "band",
        "frequency_Hz",
        "tau_measured",
        "tau_err",
        "tau_model",
    ]
    # One row per (scan, spw) with a successful fit.
    assert len(measured) == 1 + ds.sizes["scan"] * ds.sizes["spw"]


def test_write_outputs_skips_caltables_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(caltables, "write_opacity", lambda *a, **k: calls.append("op"))
    monkeypatch.setattr(caltables, "write_tcal", lambda *a, **k: calls.append("tcal"))

    TippingAnalysis(_output_ds(), Path("fake.ms")).write_outputs(tmp_path)

    assert calls == []
    assert not (tmp_path / "tipopac.opacity").exists()


def test_write_outputs_caltable_flags_name_the_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        caltables, "write_opacity", lambda ds, p: written.append(("opacity", p))
    )
    monkeypatch.setattr(
        caltables, "write_tcal", lambda ds, p: written.append(("tcal", p))
    )

    TippingAnalysis(_output_ds(), Path("fake.ms")).write_outputs(
        tmp_path, caltable_opacity=True, caltable_tcal=True
    )

    assert written == [
        ("opacity", tmp_path / "tipopac.opacity"),
        ("tcal", tmp_path / "tipopac.tcal"),
    ]


def test_write_caltables_writes_only_what_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``None`` means "do not write" for each table independently."""
    written: list[str] = []
    monkeypatch.setattr(
        caltables, "write_opacity", lambda ds, p: written.append("opacity")
    )
    monkeypatch.setattr(caltables, "write_tcal", lambda ds, p: written.append("tcal"))
    ta = TippingAnalysis(_output_ds(), Path("fake.ms"))

    ta.write_caltables()
    assert written == []

    ta.write_caltables(tcal=tmp_path / "t.cal")
    assert written == ["tcal"]


def test_fit_forwards_n_workers_to_the_auto_grid_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The staged auto-build path sizes the am pool from `fit`'s own argument."""
    seen: dict[str, Any] = {}

    class _Stop(Exception):
        pass

    def _record(self: TippingAnalysis, **kwargs: Any) -> None:
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(TippingAnalysis, "build_atm_grids", _record)
    ta = TippingAnalysis(_output_ds(), Path("fake.ms"))
    with pytest.raises(_Stop):
        ta.fit(n_workers=5)
    assert seen == {"n_workers": 5}
