"""Unit tests for the on-disk writers on ``tipopac.api``.

Exercises ``_write_dataset_netcdf`` (with the gnarly attr coercion that
matters in production — list/dict/Path/None attrs and the object-dtype
``pwv_profile_source`` data var) and ``_write_model_opacity_tsv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from tipopac.api import _write_dataset_netcdf, _write_model_opacity_tsv


def _messy_dataset() -> xr.Dataset:
    """Synthetic dataset with the attr/var types that break naive ``to_netcdf``.

    These mirror what the real pipeline produces:
      - ``selected_scans``: list[int]
      - ``selected_bands``: list[str]
      - ``scans_requested``: ``"all"`` (string sentinel)
      - ``bands_requested``: ``"default_high_freq"`` (string sentinel)
      - ``source_path``: ``Path`` (not str)
      - ``open_meteo_query``: ``dict`` (the hardest case)
      - ``pwv_profile_source``: object-dtype string per-scan
    """
    freq = np.linspace(2e10, 4e10, 8)
    ds = xr.Dataset(
        data_vars={
            "am_freq_grid": (("frequency_dense",), freq.astype(np.float64)),
            "am_tau": (
                ("frequency_dense",),
                np.linspace(0.01, 0.05, 8).astype(np.float64),
            ),
            "pwv_profile_source": (
                ("scan",),
                np.array(["afgl_midlatitude_summer", "open_meteo"], dtype=object),
            ),
        },
    )
    ds.attrs["mode"] = "independent_tau_solve"
    ds.attrs["source_path"] = Path("/tmp/fake.ms")
    ds.attrs["source_format"] = "ms"
    ds.attrs["selected_scans"] = [1, 7]
    ds.attrs["selected_bands"] = ["K", "Ku"]
    ds.attrs["scans_requested"] = "all"
    ds.attrs["bands_requested"] = "default_high_freq"
    ds.attrs["atm_profile_source"] = "afgl_midlatitude_summer"
    ds.attrs["open_meteo_query"] = {
        "latitude": 34.0784,
        "longitude": -107.6184,
        "endpoint": "historical-forecast-api",
        "model": "gfs_hrrr",
    }
    return ds


def test_write_dataset_netcdf_roundtrip(tmp_path: Path) -> None:
    """Messy attrs and the object-dtype string var serialize and reopen cleanly."""
    ds = _messy_dataset()
    path = tmp_path / "tipopac.nc"

    _write_dataset_netcdf(ds, path)

    assert path.exists()
    # Caller's Dataset must be untouched (writer works on a copy).
    assert isinstance(ds.attrs["source_path"], Path)
    assert isinstance(ds.attrs["open_meteo_query"], dict)
    assert ds["pwv_profile_source"].dtype == np.dtype("O")

    reopened = xr.open_dataset(path)
    try:
        # Numeric data var round-trips bit-exact.
        np.testing.assert_array_equal(
            reopened["am_freq_grid"].values, ds["am_freq_grid"].values
        )
        np.testing.assert_array_equal(reopened["am_tau"].values, ds["am_tau"].values)
        # Object-dtype string var came through as a unicode array.
        rs_strings = [str(v) for v in reopened["pwv_profile_source"].values]
        assert rs_strings == ["afgl_midlatitude_summer", "open_meteo"]
        # Attr sanitization preserved the information we care about.
        assert reopened.attrs["mode"] == "independent_tau_solve"
        assert reopened.attrs["source_format"] == "ms"
        assert reopened.attrs["scans_requested"] == "all"
        assert reopened.attrs["bands_requested"] == "default_high_freq"
        # Path was stringified.
        assert reopened.attrs["source_path"] == "/tmp/fake.ms"
        # Lists came through as 1-D arrays of the right dtype.
        assert list(reopened.attrs["selected_scans"]) == [1, 7]
        assert sorted(str(b) for b in reopened.attrs["selected_bands"]) == ["K", "Ku"]
        # Dict round-trips via JSON.
        decoded = json.loads(reopened.attrs["open_meteo_query"])
        assert decoded["model"] == "gfs_hrrr"
        assert decoded["endpoint"] == "historical-forecast-api"
    finally:
        reopened.close()


def test_write_dataset_netcdf_handles_none_attr(tmp_path: Path) -> None:
    """``None`` attrs (legitimate in some pipeline states) must not crash ``to_netcdf``."""
    ds = _messy_dataset()
    ds.attrs["open_meteo_query"] = None
    path = tmp_path / "tipopac.nc"
    _write_dataset_netcdf(ds, path)
    reopened = xr.open_dataset(path)
    try:
        # ``None`` was coerced to empty string (NetCDF can't store None).
        assert reopened.attrs["open_meteo_query"] == ""
    finally:
        reopened.close()


def test_write_model_opacity_tsv_roundtrip(tmp_path: Path) -> None:
    """TSV is a header + N rows of ``frequency_Hz\\ttau_nepers``."""
    ds = _messy_dataset()
    path = tmp_path / "model_opacity.tsv"

    _write_model_opacity_tsv(ds, path)

    text = path.read_text()
    lines = text.strip().splitlines()
    assert lines[0] == "frequency_Hz\ttau_nepers"
    assert len(lines) == 1 + ds["am_freq_grid"].size

    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    np.testing.assert_allclose(data[:, 0], ds["am_freq_grid"].values, rtol=1e-6)
    np.testing.assert_allclose(data[:, 1], ds["am_tau"].values, rtol=1e-6)
