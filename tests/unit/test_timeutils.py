"""Unit tests for `tipopac.timeutils` — MJD conversion and scan grouping."""

from __future__ import annotations

import numpy as np
import pytest

from tipopac.timeutils import MJD_UNIX_EPOCH, assign_groups, mjd_s_to_unix_s

HOUR = 3600.0

# Scan start offsets (s) of TCAL0004_sb48127904 — six K+Ka pairs ~5 h apart.
TCAL0004_OFFSETS = np.array(
    [
        0.0,
        130.0,
        19087.0,
        19224.0,
        35813.0,
        35942.0,
        52600.0,
        52729.0,
        65041.0,
        65174.0,
        83768.0,
        83887.0,
    ]
)


def test_mjd_s_to_unix_s() -> None:
    assert mjd_s_to_unix_s(MJD_UNIX_EPOCH * 86400.0) == 0.0


# ---------------------------------------------------------------------------
# assign_groups
# ---------------------------------------------------------------------------


def test_tcal0004_pattern_gives_six_pairs() -> None:
    groups = assign_groups(TCAL0004_OFFSETS, HOUR)
    assert groups.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_none_duration_is_one_group() -> None:
    groups = assign_groups(TCAL0004_OFFSETS, None)
    assert groups.tolist() == [0] * TCAL0004_OFFSETS.size


def test_duration_longer_than_run_is_one_group() -> None:
    groups = assign_groups(TCAL0004_OFFSETS, 24 * HOUR)
    assert groups.tolist() == [0] * TCAL0004_OFFSETS.size


def test_cluster_straddling_a_window_edge_stays_together() -> None:
    """A pair either side of t0 + duration groups together, unlike fixed bins."""
    t = np.array([0.0, 0.99 * HOUR, 1.01 * HOUR])
    assert assign_groups(t, HOUR).tolist() == [0, 0, 1]


def test_evenly_spaced_scans_do_not_grow_unbounded() -> None:
    """The failure mode of gap-based clustering: 30-min cadence over 12 h."""
    t = np.arange(24) * 1800.0
    groups = assign_groups(t, HOUR)
    # Windows are inclusive at exactly `duration_s`, so each holds 3 scans.
    assert groups.max() == 7
    spans = [np.ptp(t[groups == g]) for g in range(groups.max() + 1)]
    assert max(spans) <= HOUR


def test_group_span_never_exceeds_duration() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0.0, 10 * HOUR, size=200))
    groups = assign_groups(t, HOUR)
    for g in range(groups.max() + 1):
        assert np.ptp(t[groups == g]) <= HOUR


def test_unsorted_input_is_grouped_in_time_order() -> None:
    t = np.array([2 * HOUR, 0.0, 2 * HOUR + 60.0, 60.0])
    assert assign_groups(t, HOUR).tolist() == [1, 0, 1, 0]


def test_single_scan() -> None:
    assert assign_groups(np.array([123.0]), HOUR).tolist() == [0]


def test_empty_input() -> None:
    assert assign_groups(np.array([]), HOUR).tolist() == []


def test_dtype_is_int32() -> None:
    assert assign_groups(TCAL0004_OFFSETS, HOUR).dtype == np.int32


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_duration_raises(bad: float) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        assign_groups(TCAL0004_OFFSETS, bad)
