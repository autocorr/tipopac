"""Tests for MSReader.

The slow test (marked `slow`) requires data/tip_test.ms and validates the full
schema contract against a real Measurement Set.  The fast tests exercise helper
logic without any CASA dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fast tests — no MS required
# ---------------------------------------------------------------------------


def test_supports_rejects_plain_directory(tmp_path: Path) -> None:
    from tipopac.readers.ms import MSReader

    assert not MSReader.supports(tmp_path)


def test_supports_rejects_missing_syspower(tmp_path: Path) -> None:
    from tipopac.readers.ms import MSReader

    (tmp_path / "table.dat").touch()
    assert not MSReader.supports(tmp_path)


def test_supports_accepts_ms_layout(tmp_path: Path) -> None:
    from tipopac.readers.ms import MSReader

    (tmp_path / "table.dat").touch()
    (tmp_path / "SYSPOWER").mkdir()
    assert MSReader.supports(tmp_path)


def test_nearest_idx_basic() -> None:
    from tipopac.readers.ms import _nearest_idx

    ref = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.array([0.4, 0.6, 1.9, 3.5])
    idx = _nearest_idx(ref, q)
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])


def test_nearest_idx_exact_match() -> None:
    from tipopac.readers.ms import _nearest_idx

    ref = np.array([10.0, 20.0, 30.0])
    q = np.array([10.0, 20.0, 30.0])
    idx = _nearest_idx(ref, q)
    np.testing.assert_array_equal(idx, [0, 1, 2])


def test_constructor_stores_selection_unchanged(tmp_path: Path) -> None:
    """`scans` / `bands` are stored raw and validated only at read()."""
    from tipopac.readers.ms import MSReader

    r = MSReader.from_path(tmp_path, scans=[12, 18], bands=["Ka", "Q"])
    assert r._scans_requested == [12, 18]
    assert r._bands_requested == ["Ka", "Q"]
    r2 = MSReader.from_path(tmp_path)
    assert r2._scans_requested is None
    assert r2._bands_requested is None


def test_apply_selection_band_filters_low_bands() -> None:
    """Default bands filter drops L/C/X SPWs and Ku/K/Ka/Q survive."""
    from tipopac.readers.ms import _apply_selection

    spw_freq = np.array([1.5e9, 6.0e9, 14.0e9, 22.0e9, 33.0e9, 45.0e9])
    scan_ids = [1, 2]
    scan_spws = {1: [0, 1, 2, 3], 2: [4, 5]}
    scan_t_start = {1: 0.0, 2: 100.0}
    scan_t_end = {1: 90.0, 2: 190.0}

    out_scan_ids, out_spws, _, _, tip_spws = _apply_selection(
        scan_ids, scan_spws, scan_t_start, scan_t_end, spw_freq, None, None
    )
    assert out_scan_ids == [1, 2]
    assert tip_spws == [2, 3, 4, 5]  # Ku, K, Ka, Q kept
    assert out_spws[1] == [2, 3]
    assert out_spws[2] == [4, 5]


def test_apply_selection_drops_scan_with_no_surviving_spws() -> None:
    """A scan whose SPWs are all band-rejected is dropped."""
    from tipopac.readers.ms import _apply_selection

    # scan 1 has only an L SPW; scan 2 has Ka and Q
    spw_freq = np.array([1.5e9, 33.0e9, 45.0e9])
    scan_ids = [1, 2]
    scan_spws = {1: [0], 2: [1, 2]}
    scan_t_start = {1: 0.0, 2: 100.0}
    scan_t_end = {1: 90.0, 2: 190.0}

    out_scan_ids, out_spws, out_t_start, out_t_end, tip_spws = _apply_selection(
        scan_ids, scan_spws, scan_t_start, scan_t_end, spw_freq, None, None
    )
    assert out_scan_ids == [2]
    assert 1 not in out_spws
    assert 1 not in out_t_start and 1 not in out_t_end
    assert tip_spws == [1, 2]


def test_apply_selection_zero_match_raises() -> None:
    """All-L data with default high-freq selection raises with band names."""
    from tipopac.readers.ms import _apply_selection

    spw_freq = np.array([1.5e9, 1.7e9])
    with pytest.raises(ValueError, match=r"observed bands"):
        _apply_selection([1], {1: [0, 1]}, {1: 0.0}, {1: 90.0}, spw_freq, None, None)


def test_apply_selection_scan_subset() -> None:
    """User-specified scans narrow the resolved set."""
    from tipopac.readers.ms import _apply_selection

    spw_freq = np.array([33.0e9])  # one Ka spw
    scan_ids = [1, 2, 3]
    scan_spws = {1: [0], 2: [0], 3: [0]}
    scan_t_start = {1: 0.0, 2: 100.0, 3: 200.0}
    scan_t_end = {1: 90.0, 2: 190.0, 3: 290.0}

    out_scan_ids, _, _, _, _ = _apply_selection(
        scan_ids, scan_spws, scan_t_start, scan_t_end, spw_freq, [2], None
    )
    assert out_scan_ids == [2]


def test_apply_selection_explicit_scan_dropped_by_band_filter_raises() -> None:
    """`scans=[X]` where X has only band-rejected SPWs must raise, not drop."""
    from tipopac.readers.ms import _apply_selection

    # scan 1 has an L SPW (rejected by default Ku/K/Ka/Q), scan 2 has Ka
    spw_freq = np.array([1.5e9, 33.0e9])
    scan_ids = [1, 2]
    scan_spws = {1: [0], 2: [1]}
    scan_t_start = {1: 0.0, 2: 100.0}
    scan_t_end = {1: 90.0, 2: 190.0}

    with pytest.raises(ValueError, match=r"requested scan\(s\) \[1\]"):
        _apply_selection(
            scan_ids, scan_spws, scan_t_start, scan_t_end, spw_freq, [1, 2], None
        )


# ---------------------------------------------------------------------------
# Slow test — requires data/tip_test.ms
# ---------------------------------------------------------------------------

MS_PATH = Path(__file__).parents[2] / "data" / "tip_test.ms"


@pytest.mark.slow
def test_ms_reader_schema_validates() -> None:
    """MSReader.read() on the real MS must pass schema.validate()."""
    from tipopac import schema
    from tipopac.readers.ms import MSReader

    assert MSReader.supports(MS_PATH), f"tip_test.ms not found at {MS_PATH}"
    reader = MSReader.from_path(MS_PATH)
    ds = reader.read()
    schema.validate(ds)  # raises SchemaError on failure


@pytest.mark.slow
def test_ms_reader_dims_match_reference() -> None:
    """Dataset dimensions must match the reference.json captured from v2.6.

    v2.6 processed every DO_SKYDIP SPW with no band filter, so this
    comparison reads with all VLA bands enabled.
    """
    import json

    from tipopac.bands import VLA_BANDS
    from tipopac.readers.ms import MSReader

    ref_path = (
        Path(__file__).parents[2]
        / "tests"
        / "integration"
        / "reference"
        / "v26"
        / "tau_per_antenna"
        / "reference.json"
    )
    ref = json.loads(ref_path.read_text())

    reader = MSReader.from_path(MS_PATH, bands=list(VLA_BANDS))
    ds = reader.read()

    assert list(ds.coords["antenna"].values) == ref["coords"]["antenna"]
    assert len(ds.coords["scan"]) == len(ref["coords"]["scan"])
    assert list(ds.coords["spw"].values) == ref["coords"]["spw"]
    assert list(ds.coords["polarization"].values) == ["R", "L"]


@pytest.mark.slow
def test_ms_reader_flag_pad_invariant() -> None:
    """flag must be True at every NaN-padded position (advisor requirement)."""
    from tipopac.readers.ms import MSReader

    reader = MSReader.from_path(MS_PATH)
    ds = reader.read()

    # Every time index where time_utc is NaN must be fully flagged
    nan_mask = np.isnan(ds["time_utc"].values)  # (scan, time)
    for i_scan in range(nan_mask.shape[0]):
        pad_start = int(np.argmax(nan_mask[i_scan]))
        if not nan_mask[i_scan, pad_start]:
            continue  # no padding for this scan
        flag_at_pad = ds["flag"].values[i_scan, :, :, :, pad_start:]
        assert flag_at_pad.all(), (
            f"scan index {i_scan}: flag not True at NaN-pad positions"
        )


@pytest.mark.slow
def test_ms_reader_no_data_in_flagged_cells() -> None:
    """switched_diff must be NaN wherever flag is True for all (ant, spw, pol, t)."""
    from tipopac.readers.ms import MSReader

    reader = MSReader.from_path(MS_PATH)
    ds = reader.read()

    diff = ds["switched_diff"].values
    fl = ds["flag"].values
    # All flagged cells must be NaN
    assert np.all(np.isnan(diff[fl])), "flagged cells contain non-NaN switched_diff"


@pytest.mark.slow
def test_ms_reader_default_keeps_only_high_freq_bands() -> None:
    """Default `bands=None` keeps only Ku/K/Ka/Q SPWs."""
    from tipopac.readers.ms import MSReader

    ds = MSReader.from_path(MS_PATH).read()
    bands = set(ds.coords["band"].values.tolist())
    assert bands <= {"Ku", "K", "Ka", "Q"}, f"unexpected low band(s) survived: {bands}"


@pytest.mark.slow
def test_ms_reader_explicit_band_filter_narrows() -> None:
    """`bands=["Ka"]` keeps only Ka-band SPWs (covers low-Ka + high-Ka)."""
    from tipopac.readers.ms import MSReader

    ds_default = MSReader.from_path(MS_PATH).read()
    ds_ka = MSReader.from_path(MS_PATH, bands=["Ka"]).read()
    assert set(ds_ka.coords["band"].values.tolist()) == {"Ka"}
    assert ds_ka.sizes["spw"] <= ds_default.sizes["spw"]


@pytest.mark.slow
def test_ms_reader_scan_subset_keeps_only_requested() -> None:
    """`scans=[first]` returns a dataset with exactly that scan."""
    from tipopac.readers.ms import MSReader

    ds_all = MSReader.from_path(MS_PATH).read()
    first_scan = int(ds_all.coords["scan"].values[0])
    ds_one = MSReader.from_path(MS_PATH, scans=[first_scan]).read()
    assert ds_one.sizes["scan"] == 1
    assert int(ds_one.coords["scan"].values[0]) == first_scan


@pytest.mark.slow
def test_ms_reader_invalid_scan_raises() -> None:
    """Requesting a non-DO_SKYDIP scan number raises `ValueError`."""
    from tipopac.readers.ms import MSReader

    with pytest.raises(ValueError, match="not DO_SKYDIP scans"):
        MSReader.from_path(MS_PATH, scans=[999_999]).read()


@pytest.mark.slow
def test_ms_reader_no_match_band_raises() -> None:
    """`bands=["L"]` on an MS with no L SPWs raises with band hint."""
    from tipopac.readers.ms import MSReader

    with pytest.raises(ValueError, match="observed bands"):
        MSReader.from_path(MS_PATH, bands=["L"]).read()


@pytest.mark.slow
def test_ms_reader_sets_provenance_attrs() -> None:
    """Selection attrs are set by the reader itself, not the api layer."""
    from tipopac.readers.ms import MSReader

    ds = MSReader.from_path(MS_PATH).read()
    assert ds.attrs["scans_requested"] == "all"
    assert ds.attrs["bands_requested"] == "default_high_freq"
    assert ds.attrs["selected_scans"] == [int(s) for s in ds.coords["scan"].values]
    assert ds.attrs["selected_bands"] == sorted(
        {str(b) for b in ds.coords["band"].values.tolist()}
    )

    first = int(ds.coords["scan"].values[0])
    ds2 = MSReader.from_path(MS_PATH, scans=[first], bands=["K"]).read()
    assert ds2.attrs["scans_requested"] == [first]
    assert ds2.attrs["bands_requested"] == ["K"]
    assert ds2.attrs["selected_bands"] == ["K"]
    assert ds2.attrs["selected_scans"] == [first]
