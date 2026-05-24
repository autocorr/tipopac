"""Tests for SDMReader.

Fast tests exercise supports() and helper logic without loading the SDM.
Slow tests require data/tip_test.sdm and data/tip_test.ms, and include a
field-by-field parity check between the two readers on the shared observation
(DESIGN.md §4, lines 187–189).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


SDM_PATH = Path(__file__).parents[2] / "data" / "tip_test.sdm"
MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"


# ---------------------------------------------------------------------------
# Fast tests — no SDM required
# ---------------------------------------------------------------------------


def test_supports_rejects_plain_directory(tmp_path: Path) -> None:
    from tipopac.readers.sdm import SDMReader

    assert not SDMReader.supports(tmp_path)


def test_supports_rejects_ms_layout(tmp_path: Path) -> None:
    from tipopac.readers.sdm import SDMReader

    (tmp_path / "table.dat").touch()
    (tmp_path / "SYSPOWER").mkdir()
    assert not SDMReader.supports(tmp_path)


def test_supports_accepts_sdm_layout(tmp_path: Path) -> None:
    from tipopac.readers.sdm import SDMReader

    (tmp_path / "ASDM.xml").touch()
    assert SDMReader.supports(tmp_path)


def test_nearest_idx_basic() -> None:
    from tipopac.readers.sdm import _nearest_idx

    ref = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.array([0.4, 0.6, 1.9, 3.5])
    idx = _nearest_idx(ref, q)
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


def test_nearest_idx_exact_match() -> None:
    from tipopac.readers.sdm import _nearest_idx

    ref = np.array([10.0, 20.0, 30.0])
    q = np.array([10.0, 20.0, 30.0])
    idx = _nearest_idx(ref, q)
    np.testing.assert_array_equal(idx, [0, 1, 2])


# ---------------------------------------------------------------------------
# Slow tests — require data/tip_test.sdm (and data/tip_test.ms for parity)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_sdm():
    from tipopac.readers.sdm import SDMReader

    if not SDMReader.supports(SDM_PATH):
        pytest.skip(f"tip_test.sdm not found at {SDM_PATH}")
    return SDMReader.from_path(SDM_PATH).read()


@pytest.fixture(scope="module")
def ds_ms():
    from tipopac.readers.ms import MSReader

    if not MSReader.supports(MS_PATH):
        pytest.skip(f"tip_test.ms not found at {MS_PATH}")
    return MSReader.from_path(MS_PATH).read()


@pytest.mark.slow
def test_sdm_reader_schema_validates(ds_sdm) -> None:
    """SDMReader.read() on the real SDM must pass schema.validate()."""
    from tipopac import schema

    schema.validate(ds_sdm)


@pytest.mark.slow
def test_sdm_reader_source_format_attr(ds_sdm) -> None:
    assert ds_sdm.attrs["source_format"] == "sdm"


@pytest.mark.slow
def test_sdm_reader_flag_pad_invariant(ds_sdm) -> None:
    """flag must be True at every NaN-padded position."""
    nan_mask = np.isnan(ds_sdm["time_utc"].values)  # (scan, time)
    for i_scan in range(nan_mask.shape[0]):
        pad_start = int(np.argmax(nan_mask[i_scan]))
        if not nan_mask[i_scan, pad_start]:
            continue
        flag_at_pad = ds_sdm["flag"].values[i_scan, :, :, :, pad_start:]
        assert flag_at_pad.all(), (
            f"scan index {i_scan}: flag not True at NaN-pad positions"
        )


@pytest.mark.slow
def test_sdm_reader_no_data_in_flagged_cells(ds_sdm) -> None:
    """switched_diff must be NaN wherever flag is True."""
    diff = ds_sdm["switched_diff"].values
    fl = ds_sdm["flag"].values
    assert np.all(np.isnan(diff[fl])), "flagged cells contain non-NaN switched_diff"


@pytest.mark.slow
def test_sdm_ms_parity_coords(ds_ms, ds_sdm) -> None:
    """SDMReader and MSReader must produce identical coordinate values."""

    # antenna names and scan ids must be identical
    np.testing.assert_array_equal(
        ds_ms.coords["antenna"].values,
        ds_sdm.coords["antenna"].values,
        err_msg="antenna coord mismatch between MS and SDM readers",
    )
    np.testing.assert_array_equal(
        ds_ms.coords["scan"].values,
        ds_sdm.coords["scan"].values,
        err_msg="scan coord mismatch",
    )
    np.testing.assert_array_equal(
        ds_ms.coords["spw"].values,
        ds_sdm.coords["spw"].values,
        err_msg="spw coord mismatch",
    )
    np.testing.assert_array_equal(
        ds_ms.coords["polarization"].values,
        ds_sdm.coords["polarization"].values,
        err_msg="polarization coord mismatch",
    )

    # frequencies and bandwidths must match
    np.testing.assert_allclose(
        ds_ms.coords["frequency"].values,
        ds_sdm.coords["frequency"].values,
        rtol=1e-6,
        err_msg="frequency coord mismatch",
    )
    np.testing.assert_allclose(
        ds_ms.coords["bandwidth"].values,
        ds_sdm.coords["bandwidth"].values,
        rtol=1e-6,
        err_msg="bandwidth coord mismatch",
    )

    # antenna positions must match (ITRF metres)
    np.testing.assert_allclose(
        ds_ms.coords["antenna_position"].values,
        ds_sdm.coords["antenna_position"].values,
        atol=1.0,  # sub-metre agreement expected from same station list
        err_msg="antenna_position mismatch",
    )


def _common_time_indices(
    utc_ms: np.ndarray, utc_sdm: np.ndarray, tol: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """For two 1-D time arrays, return index pairs that match within `tol` seconds.

    The MS reader uses msmd integration-center times as scan boundaries, giving
    slightly fewer SysPower samples than the SDM reader which uses Scan.startTime/
    endTime.  Aligning on actual timestamps lets parity tests work despite the
    different n_time.
    """
    idx_ms: list[int] = []
    idx_sdm: list[int] = []
    for j_ms, t in enumerate(utc_ms):
        if not np.isfinite(t):
            continue
        j_sdm = int(np.searchsorted(utc_sdm, t))
        for cand in (j_sdm - 1, j_sdm, j_sdm + 1):
            if 0 <= cand < len(utc_sdm) and np.isfinite(utc_sdm[cand]):
                if abs(utc_sdm[cand] - t) <= tol:
                    idx_ms.append(j_ms)
                    idx_sdm.append(cand)
                    break
    return np.array(idx_ms, dtype=np.intp), np.array(idx_sdm, dtype=np.intp)


@pytest.mark.slow
def test_sdm_ms_parity_syspower(ds_ms, ds_sdm) -> None:
    """switched_diff and switched_sum must agree between readers to within 1e-4.

    The SDM reader may have slightly more samples than the MS reader (the MS
    reader bounds SYSPOWER on msmd integration-center times; the SDM reader uses
    the full Scan.startTime/endTime).  We compare only at matched timestamps.
    """
    n_scan = ds_ms.sizes["scan"]
    utc_ms = ds_ms["time_utc"].values    # (scan, time)
    utc_sdm = ds_sdm["time_utc"].values

    for var in ("switched_diff", "switched_sum"):
        ms_arr = ds_ms[var].values   # (scan, ant, spw, pol, time)
        sdm_arr = ds_sdm[var].values
        fl_ms = ds_ms["flag"].values
        fl_sdm = ds_sdm["flag"].values

        for i in range(n_scan):
            jj_ms, jj_sdm = _common_time_indices(utc_ms[i], utc_sdm[i])
            if len(jj_ms) == 0:
                continue
            ms_v = ms_arr[i, :, :, :, jj_ms]    # (ant, spw, pol, n_common)
            sdm_v = sdm_arr[i, :, :, :, jj_sdm]
            fl_v = fl_ms[i, :, :, :, jj_ms] | fl_sdm[i, :, :, :, jj_sdm]
            valid = ~fl_v
            if valid.sum() == 0:
                continue
            np.testing.assert_allclose(
                ms_v[valid],
                sdm_v[valid],
                rtol=1e-4,
                atol=1e-6,
                err_msg=f"{var} values diverge at scan index {i}",
            )


@pytest.mark.slow
def test_sdm_ms_parity_tcal_ref(ds_ms, ds_sdm) -> None:
    """tcal_ref must agree between MS and SDM readers to within 1%."""
    ms_tcal = ds_ms["tcal_ref"].values
    sdm_tcal = ds_sdm["tcal_ref"].values

    valid = np.isfinite(ms_tcal) & np.isfinite(sdm_tcal)
    np.testing.assert_allclose(
        ms_tcal[valid],
        sdm_tcal[valid],
        rtol=0.01,
        err_msg="tcal_ref diverges between MS and SDM readers",
    )


@pytest.mark.slow
def test_sdm_ms_parity_zenith_angle(ds_ms, ds_sdm) -> None:
    """zenith_angle must agree between MS and SDM readers to within 0.1 deg."""
    n_scan = ds_ms.sizes["scan"]
    utc_ms = ds_ms["time_utc"].values
    utc_sdm = ds_sdm["time_utc"].values
    ms_za = ds_ms["zenith_angle"].values    # (scan, ant, time)
    sdm_za = ds_sdm["zenith_angle"].values

    for i in range(n_scan):
        jj_ms, jj_sdm = _common_time_indices(utc_ms[i], utc_sdm[i])
        if len(jj_ms) == 0:
            continue
        ms_v = ms_za[i, :, jj_ms]
        sdm_v = sdm_za[i, :, jj_sdm]
        valid = np.isfinite(ms_v) & np.isfinite(sdm_v)
        if valid.sum() == 0:
            continue
        np.testing.assert_allclose(
            ms_v[valid],
            sdm_v[valid],
            atol=0.1,
            err_msg=f"zenith_angle diverges at scan index {i}",
        )


@pytest.mark.slow
def test_sdm_ms_parity_weather(ds_ms, ds_sdm) -> None:
    """weather_T, weather_P, weather_RH must agree between readers."""
    n_scan = ds_ms.sizes["scan"]
    utc_ms = ds_ms["time_utc"].values
    utc_sdm = ds_sdm["time_utc"].values

    for var, atol in [("weather_T", 1.0), ("weather_P", 200.0), ("weather_RH", 0.02)]:
        ms_arr = ds_ms[var].values    # (scan, time)
        sdm_arr = ds_sdm[var].values
        for i in range(n_scan):
            jj_ms, jj_sdm = _common_time_indices(utc_ms[i], utc_sdm[i])
            if len(jj_ms) == 0:
                continue
            ms_v = ms_arr[i, jj_ms]
            sdm_v = sdm_arr[i, jj_sdm]
            valid = np.isfinite(ms_v) & np.isfinite(sdm_v)
            if valid.sum() == 0:
                continue
            np.testing.assert_allclose(
                ms_v[valid],
                sdm_v[valid],
                atol=atol,
                err_msg=f"{var} diverges at scan index {i}",
            )
