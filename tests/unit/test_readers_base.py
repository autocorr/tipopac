"""Unit tests for shared reader helpers in tipopac.readers.base."""

import logging

import numpy as np
import pytest

from tipopac.readers.base import _drop_empty_scans


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
