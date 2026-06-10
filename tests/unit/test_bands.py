"""Unit tests for `tipopac.bands` — band table and selection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tipopac.bands import (
    HIGH_FREQ_DEFAULT,
    VLA_BANDS,
    band_for_frequency,
    normalize_bands,
    select_spws_by_band,
    validate_scan_selection,
)


# ---------------------------------------------------------------------------
# band_for_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "freq_Hz, expected",
    [
        (1.5e9, "L"),
        (3.0e9, "S"),
        (6.0e9, "C"),
        (10.0e9, "X"),
        (14.0e9, "Ku"),
        (22.0e9, "K"),
        (33.0e9, "Ka"),
        (45.0e9, "Q"),
    ],
)
def test_band_for_frequency_typical(freq_Hz: float, expected: str) -> None:
    assert band_for_frequency(freq_Hz) == expected


def test_band_for_frequency_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="outside any VLA band"):
        band_for_frequency(100.0e9)


def test_band_for_frequency_below_range_raises() -> None:
    with pytest.raises(ValueError, match="outside any VLA band"):
        band_for_frequency(1.0e6)


# ---------------------------------------------------------------------------
# normalize_bands
# ---------------------------------------------------------------------------


def test_normalize_bands_default() -> None:
    assert normalize_bands(None) == HIGH_FREQ_DEFAULT
    assert normalize_bands(None) == ("Ku", "K", "Ka", "Q")


def test_normalize_bands_case_insensitive_and_dedupe() -> None:
    # input order preserved after canonical mapping; duplicates dropped
    assert normalize_bands(["ka", "KU", "Ka"]) == ("Ka", "Ku")


def test_normalize_bands_low_band_opt_in() -> None:
    assert normalize_bands(["X"]) == ("X",)
    assert normalize_bands(["l", "s", "c"]) == ("L", "S", "C")


def test_normalize_bands_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown band"):
        normalize_bands(["bogus"])


def test_normalize_bands_empty_raises() -> None:
    with pytest.raises(ValueError, match="explicit empty"):
        normalize_bands([])


# ---------------------------------------------------------------------------
# validate_scan_selection
# ---------------------------------------------------------------------------


def test_validate_scan_selection_none_returns_all() -> None:
    assert validate_scan_selection(None, [1, 2, 3]) == [1, 2, 3]


def test_validate_scan_selection_subset_preserves_available_order() -> None:
    # request scans in a different order; result keeps the order from
    # `available` (which is the sorted DO_SKYDIP scan list)
    assert validate_scan_selection([3, 1], [1, 2, 3, 4]) == [1, 3]


def test_validate_scan_selection_missing_raises() -> None:
    with pytest.raises(ValueError, match=r"99"):
        validate_scan_selection([1, 99], [1, 2, 3])


def test_validate_scan_selection_empty_raises() -> None:
    with pytest.raises(ValueError, match="explicit empty"):
        validate_scan_selection([], [1, 2, 3])


# ---------------------------------------------------------------------------
# select_spws_by_band
# ---------------------------------------------------------------------------


def test_select_spws_by_band_filters_to_allowed() -> None:
    # spw 0=L, 1=Ka, 2=Q, 3=Ku
    spw_freq = np.array([1.5e9, 33.0e9, 45.0e9, 14.0e9])
    tip_spws = [0, 1, 2, 3]
    kept = select_spws_by_band(tip_spws, spw_freq, ("Ka", "Q"))
    assert kept == [1, 2]


def test_select_spws_by_band_preserves_input_order() -> None:
    spw_freq = np.array([14.0e9, 33.0e9, 22.0e9])
    tip_spws = [2, 0, 1]  # K, Ku, Ka in input order
    kept = select_spws_by_band(tip_spws, spw_freq, HIGH_FREQ_DEFAULT)
    assert kept == [2, 0, 1]


def test_select_spws_by_band_can_return_empty() -> None:
    # all SPWs are L-band; caller is responsible for raising the
    # zero-match error.
    spw_freq = np.array([1.5e9, 1.7e9])
    assert select_spws_by_band([0, 1], spw_freq, ("Ka",)) == []


def test_vla_bands_no_gaps_within_known_ranges() -> None:
    # Sanity check: every band's `lo <= hi`.
    for name, (lo, hi) in VLA_BANDS.items():
        assert lo < hi, f"band {name!r} has lo>=hi"
