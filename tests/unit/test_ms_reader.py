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
    """Dataset dimensions must match the reference.json captured from v2.6."""
    import json

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

    reader = MSReader.from_path(MS_PATH)
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
