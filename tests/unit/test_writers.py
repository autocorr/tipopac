"""Unit tests for the on-disk writers on ``tipopac.api``.

Exercises ``_write_dataset_netcdf`` (with the gnarly attr coercion that
matters in production — list/dict/Path/None attrs and the object-dtype
``pwv_profile_source`` data var) and ``_write_tsv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from tipopac.api import _write_dataset_netcdf, _write_tsv
from tipopac.tables import model_opacity_table


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
                ("group", "frequency_dense"),
                np.linspace(0.01, 0.05, 8).astype(np.float64)[None, :],
            ),
            "pwv_profile_source": (
                ("scan",),
                np.array(["afgl_midlatitude_summer", "open_meteo"], dtype=object),
            ),
        },
    )
    ds.coords["group"] = np.array([0], dtype=np.int32)
    ds.coords["scan_group"] = (("scan",), np.zeros(2, dtype=np.int32))
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


def test_write_tsv_model_opacity_roundtrip(tmp_path: Path) -> None:
    """TSV is a header + N rows of ``group\\tfrequency_Hz\\ttau_model``."""
    ds = _messy_dataset()
    path = tmp_path / "model_opacity.tsv"

    _write_tsv(path, ds, model_opacity_table)

    text = path.read_text()
    lines = text.strip().splitlines()
    assert lines[0] == "group\tfrequency_Hz\ttau_model"
    assert len(lines) == 1 + ds["am_freq_grid"].size

    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    np.testing.assert_array_equal(data[:, 0], 0)
    np.testing.assert_allclose(data[:, 1], ds["am_freq_grid"].values, rtol=1e-6)
    np.testing.assert_allclose(data[:, 2], ds["am_tau"].values[0], rtol=1e-6)


def test_write_tsv_mixed_column_types(tmp_path: Path) -> None:
    """Ints and strings are written verbatim; floats in ``%.6e``."""
    path = tmp_path / "measured_opacity.tsv"
    ds = _messy_dataset()

    _write_tsv(path, ds, lambda _: (("scan", "band", "tau"), [(7, "Ka", 0.0325)]))

    assert path.read_text() == "group\tscan\tband\ttau\n0\t7\tKa\t3.250000e-02\n"


def test_write_tsv_concatenates_groups(tmp_path: Path) -> None:
    """Every group lands in one file behind a leading `group` column."""
    ds = _messy_dataset().drop_vars(["group", "am_tau"])
    ds["am_tau"] = (
        ("group", "frequency_dense"),
        np.stack([np.full(8, 0.02), np.full(8, 0.04)]),
    )
    ds.coords["group"] = np.array([0, 1], dtype=np.int32)
    ds.coords["scan_group"] = (("scan",), np.array([0, 1], dtype=np.int32))
    path = tmp_path / "model_opacity.tsv"

    _write_tsv(path, ds, model_opacity_table)

    data = np.loadtxt(path, delimiter="\t", skiprows=1)
    assert data.shape == (16, 3)
    np.testing.assert_array_equal(data[:8, 0], 0)
    np.testing.assert_array_equal(data[8:, 0], 1)
    np.testing.assert_allclose(data[:8, 2], 0.02)
    np.testing.assert_allclose(data[8:, 2], 0.04)
