"""Unit tests for `tipopac.bands` — SPW NAME parsing and selection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tipopac.bands import (
    HIGH_FREQ_DEFAULT,
    VLA_BAND_LABELS,
    band_for_spw_name,
    normalize_bands,
    select_spws_by_band,
    validate_scan_selection,
)


# ---------------------------------------------------------------------------
# band_for_spw_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        # SPECTRAL_WINDOW.NAME form (upper-case band token, as seen in
        # tip_test.{ms,sdm})
        ("EVLA_K#A0C0#0", "K"),
        ("EVLA_KA#A0C0#16", "Ka"),
        ("EVLA_KU#A0C0#48", "Ku"),
        ("EVLA_Q#A0C0#80", "Q"),
        # Receiver.xml `<frequencyBand>` form (mixed-case) is bare but
        # still accepted — same parser applies once the prefix is split.
        ("EVLA_Ka", "Ka"),
        ("EVLA_Ku", "Ku"),
        # low bands round-trip too
        ("EVLA_L#…", "L"),
        ("EVLA_S#…", "S"),
        ("EVLA_C#…", "C"),
        ("EVLA_X#…", "X"),
    ],
)
def test_band_for_spw_name_typical(name: str, expected: str) -> None:
    assert band_for_spw_name(name) == expected


def test_band_for_spw_name_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        band_for_spw_name("")


def test_band_for_spw_name_no_prefix_raises() -> None:
    with pytest.raises(ValueError, match=r"EVLA_<BAND>"):
        band_for_spw_name("WIDE_0#A0C0#0")


def test_band_for_spw_name_unknown_band_raises() -> None:
    # `EVLA_` prefix present but the token isn't a known VLA band
    with pytest.raises(ValueError, match=r"unknown band token"):
        band_for_spw_name("EVLA_Z#A0C0#0")


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
    spw_bands = np.array(["L", "Ka", "Q", "Ku"])
    tip_spws = [0, 1, 2, 3]
    kept = select_spws_by_band(tip_spws, spw_bands, ("Ka", "Q"))
    assert kept == [1, 2]


def test_select_spws_by_band_preserves_input_order() -> None:
    spw_bands = np.array(["Ku", "Ka", "K"])
    tip_spws = [2, 0, 1]  # K, Ku, Ka in input order
    kept = select_spws_by_band(tip_spws, spw_bands, HIGH_FREQ_DEFAULT)
    assert kept == [2, 0, 1]


def test_select_spws_by_band_can_return_empty() -> None:
    # all SPWs are L-band; caller is responsible for raising the
    # zero-match error.
    spw_bands = np.array(["L", "L"])
    assert select_spws_by_band([0, 1], spw_bands, ("Ka",)) == []


def test_vla_band_labels_match_high_freq_default() -> None:
    # Sanity check: the high-freq default is a subset of the canonical
    # label tuple.
    assert set(HIGH_FREQ_DEFAULT) <= set(VLA_BAND_LABELS)
