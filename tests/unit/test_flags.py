"""Unit tests for tipopac.flags (DESIGN.md §8, §11.1).

The five interval-overlap cases confirm that one boolean expression
    (time_utc >= t_start) & (time_utc <= t_end)
subsumes v2.6's four-case block (task_tipopac.py:1116-1199).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tipopac.flags import (
    _apply_interval,
    _parse_command,
    _parse_user_line,
    _ymd_to_mjd_sec,
    apply,
)
from tipopac import schema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_flag_ds(n_time: int = 11) -> xr.Dataset:
    """Minimal dataset for flag tests.

    time_utc values are [0, 1, ..., n_time-1] (synthetic MJD-sec).
    All flags start as False (no flags).
    """
    n_scan, n_ant, n_spw, n_pol = 1, 2, 3, 2
    t = np.arange(n_time, dtype=np.float64)
    return xr.Dataset(
        data_vars={
            "switched_diff": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.ones((n_scan, n_ant, n_spw, n_pol, n_time), dtype=np.float32),
            ),
            "switched_sum": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.ones((n_scan, n_ant, n_spw, n_pol, n_time), dtype=np.float32),
            ),
            "zenith_angle": (
                ("scan", "antenna", "time"),
                np.full((n_scan, n_ant, n_time), 45.0, dtype=np.float32),
            ),
            "tcal_ref": (
                ("antenna", "spw", "polarization"),
                np.ones((n_ant, n_spw, n_pol), dtype=np.float32),
            ),
            "weather_T": (
                ("scan", "time"),
                np.full((n_scan, n_time), 280.0, dtype=np.float32),
            ),
            "weather_P": (
                ("scan", "time"),
                np.full((n_scan, n_time), 85000.0, dtype=np.float32),
            ),
            "weather_RH": (
                ("scan", "time"),
                np.full((n_scan, n_time), 0.3, dtype=np.float32),
            ),
            "flag": (
                ("scan", "antenna", "spw", "polarization", "time"),
                np.zeros((n_scan, n_ant, n_spw, n_pol, n_time), dtype=bool),
            ),
        },
        coords={
            "scan": np.array([1], dtype=np.intp),
            "antenna": ["ea01", "ea05"],
            "spw": np.array([0, 7, 15], dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), np.array([5e9, 15e9, 30e9])),
            "bandwidth": (("spw",), np.array([2e9, 2e9, 2e9])),
            "antenna_position": (("antenna", "xyz"), np.zeros((2, 3))),
            "scan_time_start": (("scan",), np.array([0.0])),
            "scan_time_end": (("scan",), np.array([float(n_time - 1)])),
            "time_utc": (("scan", "time"), t[np.newaxis, :]),
        },
        attrs={
            "source_path": "/fake/test.ms",
            "source_format": "ms",
            "observatory": "VLA",
        },
    )


# ---------------------------------------------------------------------------
# _apply_interval: five overlap cases
# ---------------------------------------------------------------------------


def _flag_vals(ds: xr.Dataset) -> np.ndarray:
    return ds["flag"].values  # (1, 2, 3, 2, n_time)


def test_no_overlap() -> None:
    """Flag interval entirely before scan data → no flags added."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "*", t_start=-20.0, t_end=-10.0)
    assert not flag.any(), "No flags should be set for a non-overlapping interval"


def test_fully_contained() -> None:
    """Flag interval strictly inside scan → exactly those samples are flagged."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "*", t_start=3.0, t_end=6.0)
    flagged_times = flag[0, 0, 0, 0, :]  # (n_time,)
    expected = np.array(
        [False, False, False, True, True, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_partial_left() -> None:
    """Flag starts before scan, ends inside → overlap portion flagged."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "*", t_start=-5.0, t_end=3.0)
    flagged_times = flag[0, 0, 0, 0, :]
    expected = np.array(
        [True, True, True, True, False, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_partial_right() -> None:
    """Flag starts inside scan, ends after → overlap portion flagged."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "*", t_start=7.0, t_end=20.0)
    flagged_times = flag[0, 0, 0, 0, :]
    expected = np.array(
        [False, False, False, False, False, False, False, True, True, True, True]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_spanning() -> None:
    """Flag spans entire scan → all time samples flagged."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "*", t_start=-5.0, t_end=100.0)
    assert flag.all(), "All samples should be flagged when interval spans the full scan"


# ---------------------------------------------------------------------------
# _apply_interval: antenna and spw selectivity
# ---------------------------------------------------------------------------


def test_antenna_selectivity() -> None:
    """Flagging ea01 must not affect ea05 flags."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "ea01", "*", t_start=0.0, t_end=10.0)
    # antenna index 0 = ea01, index 1 = ea05
    assert flag[:, 0, :, :, :].all(), "ea01 should be fully flagged"
    assert not flag[:, 1, :, :, :].any(), "ea05 must not be flagged"


def test_spw_selectivity() -> None:
    """Flagging spw=7 must not affect spw=0 or spw=15."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "*", "7", t_start=0.0, t_end=10.0)
    # spw index 1 = id 7
    assert flag[:, :, 1, :, :].all(), "spw=7 should be fully flagged"
    assert not flag[:, :, 0, :, :].any(), "spw=0 must not be flagged"
    assert not flag[:, :, 2, :, :].any(), "spw=15 must not be flagged"


def test_unknown_antenna_is_skipped() -> None:
    """Flagging an antenna not in the dataset silently does nothing."""
    ds = _make_flag_ds()
    flag = _flag_vals(ds)
    time_utc = ds["time_utc"].values
    ant = np.asarray(ds.coords["antenna"].values, dtype=str)
    spw = ds.coords["spw"].values

    _apply_interval(flag, time_utc, ant, spw, "ea99", "*", t_start=0.0, t_end=10.0)
    assert not flag.any()


# ---------------------------------------------------------------------------
# _parse_command: regex on FLAG_CMD COMMAND strings
# ---------------------------------------------------------------------------


def test_parse_command_basic() -> None:
    """Parse a standard VLA FLAG_CMD COMMAND string."""
    cmd = (
        "antenna='ea14&&*' timerange='2021/02/01/01:02:29.060~2021/02/01/01:02:45.969'"
    )
    result = _parse_command(cmd)
    assert result is not None
    antenna, t_start, t_end = result
    assert antenna == "ea14"
    assert t_end > t_start


def test_parse_command_with_mode_prefix() -> None:
    """COMMAND strings with a mode= prefix are parsed correctly."""
    cmd = "mode='manual' antenna='ea05&&*' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_command(cmd)
    assert result is not None
    assert result[0] == "ea05"


def test_parse_command_no_timerange_returns_none() -> None:
    """COMMAND strings without a timerange field return None."""
    result = _parse_command("antenna='ea01&&*'")
    assert result is None


def test_parse_command_bad_time_returns_none() -> None:
    """Unparseable time strings return None."""
    result = _parse_command("antenna='ea01&&*' timerange='badtime~badtime'")
    assert result is None


def test_parse_command_fractional_seconds() -> None:
    """Fractional seconds in timerange are handled."""
    cmd = (
        "antenna='ea01&&*' timerange='2021/02/01/01:02:29.500~2021/02/01/01:02:45.999'"
    )
    result = _parse_command(cmd)
    assert result is not None
    _, t_start, t_end = result
    assert t_start < t_end


# ---------------------------------------------------------------------------
# _parse_user_line: user flag file parsing
# ---------------------------------------------------------------------------


def test_parse_user_line_all_fields() -> None:
    """Standard user flag line with explicit antenna, spw, timerange."""
    line = "antenna='ea05' spw='7' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_user_line(line)
    assert result is not None
    antenna, spw, t_start, t_end = result
    assert antenna == "ea05"
    assert spw == "7"
    assert t_end > t_start


def test_parse_user_line_wildcard_antenna() -> None:
    """Wildcard '*' antenna → 'all' selection."""
    line = "antenna='*' spw='7' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_user_line(line)
    assert result is not None
    assert result[0] == "*"


def test_parse_user_line_missing_antenna_defaults_all() -> None:
    """Missing antenna field defaults to '*'."""
    line = "spw='7' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_user_line(line)
    assert result is not None
    assert result[0] == "*"


def test_parse_user_line_missing_spw_defaults_all() -> None:
    """Missing spw field defaults to '*'."""
    line = "antenna='ea01' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_user_line(line)
    assert result is not None
    assert result[1] == "*"


def test_parse_user_line_legacy_minus1() -> None:
    """Legacy v2.6 '-1' values are treated as all-select."""
    line = "antenna='-1' spw='-1' timerange='2021/02/01/00:00:00~2021/02/01/00:05:00'"
    result = _parse_user_line(line)
    assert result is not None
    assert result[0] == "*"
    assert result[1] == "*"


def test_parse_user_line_no_timerange_returns_none() -> None:
    """Line without timerange returns None."""
    result = _parse_user_line("antenna='ea01' spw='7'")
    assert result is None


# ---------------------------------------------------------------------------
# _ymd_to_mjd_sec
# ---------------------------------------------------------------------------


def test_ymd_to_mjd_sec_known_epoch() -> None:
    """MJD epoch (1858-11-17 00:00:00) maps to 0.0 seconds."""
    assert _ymd_to_mjd_sec("1858/11/17/00:00:00") == pytest.approx(0.0, abs=1e-3)


def test_ymd_to_mjd_sec_ordering() -> None:
    """Later timestamp produces a larger MJD-sec value."""
    t1 = _ymd_to_mjd_sec("2021/02/01/00:00:00")
    t2 = _ymd_to_mjd_sec("2021/02/01/01:00:00")
    assert t2 == pytest.approx(t1 + 3600.0, abs=1e-3)


# ---------------------------------------------------------------------------
# apply: user-file end-to-end
# ---------------------------------------------------------------------------


def test_apply_user_file_flags_correct_times(tmp_path: Path) -> None:
    """apply() with a user flag file sets the right time samples."""
    ds = _make_flag_ds(n_time=11)
    # time_utc is [0..10] synthetic MJD-sec; convert to real times for the flag file
    # Use dates that map to MJD-sec 3.0 and 6.0 via _ymd_to_mjd_sec
    from tipopac.flags import _MJD_EPOCH
    from datetime import timedelta

    def mjd_sec_to_ymd(mjd_sec: float) -> str:
        dt = _MJD_EPOCH + timedelta(seconds=mjd_sec)
        return dt.strftime("%Y/%m/%d/%H:%M:%S")

    # Set real time_utc values so our flag times make sense
    base = _ymd_to_mjd_sec("2021/02/01/01:00:00")
    time_utc = np.arange(11, dtype=np.float64) + base
    ds["time_utc"].values[:] = time_utc

    t_start_str = mjd_sec_to_ymd(base + 3.0)
    t_end_str = mjd_sec_to_ymd(base + 6.0)

    flag_file = tmp_path / "flags.txt"
    flag_file.write_text(
        f"antenna='ea01' spw='7' timerange='{t_start_str}~{t_end_str}'\n"
    )

    apply(ds, online=False, file=flag_file)

    flag = ds["flag"].values  # (1, 2, 3, 2, 11)
    spw_idx = int(np.where(ds.coords["spw"].values == 7)[0][0])  # index 1
    ant_idx = int(
        np.where(np.asarray(ds.coords["antenna"].values, dtype=str) == "ea01")[0][0]
    )

    # Times 3-6 relative to base should be flagged for ea01/spw=7
    flagged = flag[0, ant_idx, spw_idx, 0, :]
    expected = np.array(
        [False, False, False, True, True, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(flagged, expected)

    # Other antenna (ea05) should not be flagged
    ant_idx2 = int(
        np.where(np.asarray(ds.coords["antenna"].values, dtype=str) == "ea05")[0][0]
    )
    assert not flag[0, ant_idx2, spw_idx, 0, :].any()


def test_apply_user_file_comment_and_blank_lines(tmp_path: Path) -> None:
    """apply() skips comment and blank lines in the flag file."""
    ds = _make_flag_ds()
    flag_file = tmp_path / "flags.txt"
    flag_file.write_text("# this is a comment\n\n   \n")
    apply(ds, online=False, file=flag_file)
    assert not ds["flag"].values.any()


def test_apply_online_skipped_for_sdm(tmp_path: Path) -> None:
    """online=True is silently ignored for SDM-format datasets."""
    ds = _make_flag_ds()
    ds.attrs["source_format"] = "sdm"
    apply(ds, online=True, file=None)
    assert not ds["flag"].values.any()


def test_apply_preserves_existing_flags() -> None:
    """apply() ORs new flags in; pre-existing flags are not cleared."""
    ds = _make_flag_ds(n_time=11)
    # Pre-set a flag at time index 0
    ds["flag"].values[0, 0, 0, 0, 0] = True

    from tipopac.flags import _MJD_EPOCH
    from datetime import timedelta

    def mjd_sec_to_ymd(mjd_sec: float) -> str:
        dt = _MJD_EPOCH + timedelta(seconds=mjd_sec)
        return dt.strftime("%Y/%m/%d/%H:%M:%S")

    base = _ymd_to_mjd_sec("2021/02/01/01:00:00")
    time_utc = np.arange(11, dtype=np.float64) + base
    ds["time_utc"].values[:] = time_utc

    flag_file_path = Path("/tmp/_tipopac_test_flags.txt")
    t_start_str = mjd_sec_to_ymd(base + 5.0)
    t_end_str = mjd_sec_to_ymd(base + 7.0)
    flag_file_path.write_text(
        f"antenna='*' spw='*' timerange='{t_start_str}~{t_end_str}'\n"
    )

    apply(ds, online=False, file=flag_file_path)

    # Time 0 was pre-flagged → still flagged
    assert ds["flag"].values[0, 0, 0, 0, 0]
    # Times 5-7 are newly flagged
    assert ds["flag"].values[0, 0, 0, 0, 5]
    # Time 3 (not in either range) is not flagged
    assert not ds["flag"].values[0, 0, 0, 0, 3]
