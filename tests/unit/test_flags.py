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
    _apply_intervals,
    _parse_antenna_ids,
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
    _apply_interval(ds, "*", "*", t_start=-20.0, t_end=-10.0)
    assert not _flag_vals(ds).any(), (
        "No flags should be set for a non-overlapping interval"
    )


def test_fully_contained() -> None:
    """Flag interval strictly inside scan → exactly those samples are flagged."""
    ds = _make_flag_ds()
    _apply_interval(ds, "*", "*", t_start=3.0, t_end=6.0)
    flagged_times = _flag_vals(ds)[0, 0, 0, 0, :]  # (n_time,)
    expected = np.array(
        [False, False, False, True, True, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_partial_left() -> None:
    """Flag starts before scan, ends inside → overlap portion flagged."""
    ds = _make_flag_ds()
    _apply_interval(ds, "*", "*", t_start=-5.0, t_end=3.0)
    flagged_times = _flag_vals(ds)[0, 0, 0, 0, :]
    expected = np.array(
        [True, True, True, True, False, False, False, False, False, False, False]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_partial_right() -> None:
    """Flag starts inside scan, ends after → overlap portion flagged."""
    ds = _make_flag_ds()
    _apply_interval(ds, "*", "*", t_start=7.0, t_end=20.0)
    flagged_times = _flag_vals(ds)[0, 0, 0, 0, :]
    expected = np.array(
        [False, False, False, False, False, False, False, True, True, True, True]
    )
    np.testing.assert_array_equal(flagged_times, expected)


def test_spanning() -> None:
    """Flag spans entire scan → all time samples flagged."""
    ds = _make_flag_ds()
    _apply_interval(ds, "*", "*", t_start=-5.0, t_end=100.0)
    assert _flag_vals(ds).all(), (
        "All samples should be flagged when interval spans the full scan"
    )


# ---------------------------------------------------------------------------
# _apply_interval: antenna and spw selectivity
# ---------------------------------------------------------------------------


def test_antenna_selectivity() -> None:
    """Flagging ea01 must not affect ea05 flags."""
    ds = _make_flag_ds()
    _apply_interval(ds, "ea01", "*", t_start=0.0, t_end=10.0)
    flag = _flag_vals(ds)
    # antenna index 0 = ea01, index 1 = ea05
    assert flag[:, 0, :, :, :].all(), "ea01 should be fully flagged"
    assert not flag[:, 1, :, :, :].any(), "ea05 must not be flagged"


def test_spw_selectivity() -> None:
    """Flagging spw=7 must not affect spw=0 or spw=15."""
    ds = _make_flag_ds()
    _apply_interval(ds, "*", "7", t_start=0.0, t_end=10.0)
    flag = _flag_vals(ds)
    # spw index 1 = id 7
    assert flag[:, :, 1, :, :].all(), "spw=7 should be fully flagged"
    assert not flag[:, :, 0, :, :].any(), "spw=0 must not be flagged"
    assert not flag[:, :, 2, :, :].any(), "spw=15 must not be flagged"


def test_unknown_antenna_is_skipped() -> None:
    """Flagging an antenna not in the dataset silently does nothing."""
    ds = _make_flag_ds()
    _apply_interval(ds, "ea99", "*", t_start=0.0, t_end=10.0)
    assert not _flag_vals(ds).any()


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


def test_apply_online_skipped_when_flag_xml_missing(tmp_path: Path) -> None:
    """An SDM without Flag.xml skips online flags instead of raising."""
    ds = _make_flag_ds()
    ds.attrs["source_format"] = "sdm"
    ds.attrs["source_path"] = str(tmp_path)
    apply(ds, online=True, file=None)
    assert not ds["flag"].values.any()


def test_apply_online_skipped_for_unknown_format() -> None:
    """online=True is ignored for a dataset with no recognised source_format."""
    ds = _make_flag_ds()
    ds.attrs["source_format"] = "other"
    apply(ds, online=True, file=None)
    assert not ds["flag"].values.any()


# ---------------------------------------------------------------------------
# _parse_antenna_ids: ASDM antenna-id arrays
# ---------------------------------------------------------------------------


def test_parse_antenna_ids_single() -> None:
    """The common one-antenna form yields a single id."""
    assert _parse_antenna_ids("1 1 Antenna_26") == ["Antenna_26"]


def test_parse_antenna_ids_multiple() -> None:
    """numAntenna > 1 expands to one id per antenna."""
    assert _parse_antenna_ids("1 3 Antenna_0 Antenna_5 Antenna_9") == [
        "Antenna_0",
        "Antenna_5",
        "Antenna_9",
    ]


def test_parse_antenna_ids_malformed_returns_empty() -> None:
    """Fields too short or with a non-integer count yield no ids."""
    assert _parse_antenna_ids("1") == []
    assert _parse_antenna_ids("1 x Antenna_0") == []


# ---------------------------------------------------------------------------
# _apply_online_flags_sdm: Flag.xml end-to-end on a synthetic SDM
# ---------------------------------------------------------------------------


def _write_sdm(tmp_path: Path, flag_rows: str) -> Path:
    """Build a minimal SDM containing only Antenna and Flag tables."""
    sdm = tmp_path / "synth.sdm"
    sdm.mkdir()

    def table(name: str, n_rows: int) -> str:
        return (
            f"<Table><Name>{name}</Name><NumberRows>{n_rows}</NumberRows>"
            f'<Entity entityId="uid://x/{name}" entityIdEncrypted="na" '
            f'entityTypeName="{name}Table" schemaVersion="4" documentVersion="1"/>'
            "</Table>"
        )

    (sdm / "ASDM.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<ASDM>'
        '<Entity entityId="uid://x/asdm" entityIdEncrypted="na" '
        'entityTypeName="ASDM" schemaVersion="4" documentVersion="1"/>'
        "<TimeOfCreation>2021-02-01T01:00:56.000089</TimeOfCreation>"
        f"{table('Antenna', 2)}{table('Flag', 1)}</ASDM>"
    )
    (sdm / "Antenna.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<AntennaTable>'
        "<row><antennaId>Antenna_0</antennaId><name>ea01</name></row>"
        "<row><antennaId>Antenna_1</antennaId><name>ea05</name></row>"
        "</AntennaTable>"
    )
    (sdm / "Flag.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?>\n<FlagTable>{flag_rows}</FlagTable>'
    )
    return sdm


def _flag_row(ant: str, t0: float, t1: float, reason: str) -> str:
    """One Flag.xml row; t0/t1 are MJD-seconds, written out as ASDM nanoseconds."""
    return (
        f"<row><flagId>Flag_0</flagId>"
        f"<startTime>{int(t0 * 1e9)}</startTime>"
        f"<endTime>{int(t1 * 1e9)}</endTime>"
        f"<reason>{reason}</reason>"
        f"<numAntenna>1</numAntenna>"
        f"<antennaId>1 1 {ant}</antennaId></row>"
    )


def _sdm_flag_ds(tmp_path: Path, sdm: Path) -> xr.Dataset:
    ds = _make_flag_ds(n_time=11)
    ds.attrs["source_format"] = "sdm"
    ds.attrs["source_path"] = str(sdm)
    return ds


def test_sdm_flag_xml_applies_interval(tmp_path: Path) -> None:
    """A Flag.xml row flags the matching antenna over its ns-precision interval."""
    pytest.importorskip("sdmpy")
    sdm = _write_sdm(tmp_path, _flag_row("Antenna_0", 3.0, 6.0, "FOCUS_ERROR"))
    ds = _sdm_flag_ds(tmp_path, sdm)

    apply(ds, online=True, file=None)

    flag = ds["flag"].values
    expected = np.array(
        [False, False, False, True, True, True, True, False, False, False, False]
    )
    np.testing.assert_array_equal(flag[0, 0, 0, 0, :], expected)
    assert not flag[:, 1, :, :, :].any(), "ea05 must not be flagged"


def test_sdm_flag_xml_excludes_reasons(tmp_path: Path) -> None:
    """Rows whose reason is in the v2.6 exclusion set are not applied."""
    pytest.importorskip("sdmpy")
    rows = "".join(
        _flag_row("Antenna_0", 0.0, 10.0, r)
        for r in ("ANTENNA_NOT_ON_SOURCE", "SHADOW", "CLIP_ZERO_ALL")
    )
    ds = _sdm_flag_ds(tmp_path, _write_sdm(tmp_path, rows))

    apply(ds, online=True, file=None)

    assert not ds["flag"].values.any()


def test_sdm_flag_xml_multi_antenna_row(tmp_path: Path) -> None:
    """A single row naming several antennas flags each of them."""
    pytest.importorskip("sdmpy")
    row = (
        "<row><flagId>Flag_0</flagId>"
        f"<startTime>{int(3.0 * 1e9)}</startTime>"
        f"<endTime>{int(6.0 * 1e9)}</endTime>"
        "<reason>SUBREFLECTOR_ERROR</reason>"
        "<numAntenna>2</numAntenna>"
        "<antennaId>1 2 Antenna_0 Antenna_1</antennaId></row>"
    )
    ds = _sdm_flag_ds(tmp_path, _write_sdm(tmp_path, row))

    apply(ds, online=True, file=None)

    flag = ds["flag"].values
    for ant_idx in (0, 1):
        np.testing.assert_array_equal(
            flag[0, ant_idx, 0, 0, 3:7], np.array([True, True, True, True])
        )


def test_sdm_flag_xml_unknown_antenna_skipped(tmp_path: Path) -> None:
    """A row naming an antenna absent from the dataset is silently skipped."""
    pytest.importorskip("sdmpy")
    sdm = _write_sdm(tmp_path, _flag_row("Antenna_99", 0.0, 10.0, "FOCUS_ERROR"))
    ds = _sdm_flag_ds(tmp_path, sdm)

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


# ---------------------------------------------------------------------------
# _apply_intervals: batch path must match the per-command path
# ---------------------------------------------------------------------------


def test_apply_intervals_matches_per_command() -> None:
    """The batched online path is equivalent to repeated _apply_interval."""
    commands = [
        ("ea01", 3.0, 6.0),
        ("ea05", 0.0, 2.0),
        ("ea01", 7.0, 20.0),
        ("ea99", 0.0, 10.0),
        ("ea05", -5.0, 1.0),
    ]

    ds_batch = _make_flag_ds()
    _apply_intervals(ds_batch, commands)

    ds_loop = _make_flag_ds()
    for antenna, t_start, t_end in commands:
        _apply_interval(ds_loop, antenna, "*", t_start, t_end)

    np.testing.assert_array_equal(ds_batch["flag"].values, ds_loop["flag"].values)


def test_apply_intervals_wildcard_antenna() -> None:
    """antenna='*' in a batch flags every antenna."""
    ds = _make_flag_ds()
    _apply_intervals(ds, [("*", 3.0, 6.0)])
    flag = ds["flag"].values
    for ant_idx in (0, 1):
        np.testing.assert_array_equal(
            flag[0, ant_idx, 0, 0, :],
            np.array(
                [False, False, False, True, True, True, True]
                + [False, False, False, False]
            ),
        )


def test_apply_intervals_preserves_existing_flags() -> None:
    """The batch OR does not clear flags already set."""
    ds = _make_flag_ds()
    ds["flag"].values[0, 0, 0, 0, 0] = True
    _apply_intervals(ds, [("ea01", 5.0, 6.0)])
    assert ds["flag"].values[0, 0, 0, 0, 0]
    assert ds["flag"].values[0, 0, 0, 0, 5]


def test_apply_intervals_empty_is_noop() -> None:
    """An empty command batch leaves the flag array untouched."""
    ds = _make_flag_ds()
    _apply_intervals(ds, [])
    assert not ds["flag"].values.any()
