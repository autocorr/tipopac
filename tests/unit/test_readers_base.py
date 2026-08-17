"""Unit tests for shared reader helpers in tipopac.readers.base."""

import logging

import numpy as np
import pytest

from tipopac.readers.base import _apply_selection, _drop_empty_scans, _nearest_idx


def _axis(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64)


def test_drop_empty_scans_passthrough() -> None:
    scan_ids = [3, 5]
    scan_times = [_axis(4), _axis(2)]
    out_ids, out_times = _drop_empty_scans(scan_ids, scan_times, None)
    assert out_ids is scan_ids
    assert out_times is scan_times


def test_drop_empty_scans_drops_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="tipopac.readers.base"):
        out_ids, out_times = _drop_empty_scans(
            [3, 5, 7], [_axis(4), _axis(0), _axis(2)], None
        )
    assert out_ids == [3, 7]
    assert [len(t) for t in out_times] == [4, 2]
    assert any("scan 5" in rec.getMessage() for rec in caplog.records)


def test_drop_empty_scans_raises_when_explicitly_requested() -> None:
    with pytest.raises(ValueError, match=r"scan\(s\) \[5\]"):
        _drop_empty_scans([3, 5], [_axis(4), _axis(0)], [3, 5])


def test_drop_empty_scans_raises_when_all_empty() -> None:
    with pytest.raises(ValueError, match="no scan has any SYSPOWER samples"):
        _drop_empty_scans([3, 5], [_axis(0), _axis(0)], None)


# ---------------------------------------------------------------------------
# _apply_selection — spw ids 0/1/2 are K, L, Ku; scan 1 has K+L, scan 2 only L
# ---------------------------------------------------------------------------

_SPW_BANDS = np.array(["K", "L", "Ku"], dtype="U4")


def _selection_args(
    scans_requested: list[int] | None = None,
    bands_requested: list[str] | None = None,
) -> tuple:
    return (
        [1, 2],
        {1: [0, 1], 2: [1]},
        {1: 100.0, 2: 200.0},
        {1: 150.0, 2: 250.0},
        _SPW_BANDS,
        scans_requested,
        bands_requested,
    )


def test_apply_selection_keeps_only_the_requested_band() -> None:
    scan_ids, scan_spws, t_start, t_end, tip_spws = _apply_selection(
        *_selection_args(bands_requested=["K"])
    )
    assert scan_ids == [1]
    assert scan_spws == {1: [0]}
    assert t_start == {1: 100.0} and t_end == {1: 150.0}
    assert tip_spws == [0]


def test_apply_selection_default_bands_drop_a_low_band_only_scan() -> None:
    """Scan 2 is L-only, so the default high-frequency selection drops it."""
    scan_ids, scan_spws, _, _, tip_spws = _apply_selection(*_selection_args())
    assert scan_ids == [1]
    assert scan_spws == {1: [0]}
    assert tip_spws == [0]


def test_apply_selection_raises_when_no_spw_matches_the_bands() -> None:
    with pytest.raises(ValueError, match=r"no SPWs match bands=\['Q'\]"):
        _apply_selection(*_selection_args(bands_requested=["Q"]))


def test_apply_selection_raises_when_a_requested_scan_loses_every_spw() -> None:
    """An explicitly named scan dropped by the band filter is a contract violation."""
    with pytest.raises(ValueError, match=r"requested scan\(s\) \[2\] have no SPWs"):
        _apply_selection(
            *_selection_args(scans_requested=[1, 2], bands_requested=["K"])
        )


# ---------------------------------------------------------------------------
# _nearest_idx
# ---------------------------------------------------------------------------


def test_nearest_idx_picks_the_closest_sample() -> None:
    ref = np.array([0.0, 10.0, 20.0])
    query = np.array([-5.0, 4.0, 6.0, 14.0, 25.0])
    np.testing.assert_array_equal(_nearest_idx(ref, query), [0, 0, 1, 1, 2])


def test_nearest_idx_breaks_ties_to_the_right() -> None:
    ref = np.array([0.0, 10.0])
    np.testing.assert_array_equal(_nearest_idx(ref, np.array([5.0])), [1])


def test_nearest_idx_clamps_at_both_edges() -> None:
    ref = np.array([0.0, 10.0, 20.0])
    np.testing.assert_array_equal(
        _nearest_idx(ref, np.array([-1e6, 1e6])), [0, ref.size - 1]
    )
