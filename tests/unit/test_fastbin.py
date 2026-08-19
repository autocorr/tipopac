"""Tests for the fast ASDM binary readers (`readers/_fastbin.py`).

The slow tests require ``data/tip_test.sdm`` and check byte-for-byte parity
against sdmpy's own ``.data`` unpack on the full SysPower / Pointing tables.
The fast guard test needs no fixture.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tests.conftest import SDM_PATH


# ---------------------------------------------------------------------------
# Fast tests — no SDM required
# ---------------------------------------------------------------------------


def _syspower_row(antenna: str = "Antenna_0", spw: str = "SpectralWindow_0") -> bytes:
    """One well-formed SysPower row, all three optional arrays present."""
    out = struct.pack(">i", len(antenna)) + antenna.encode()
    out += struct.pack(">i", len(spw)) + spw.encode()
    out += struct.pack(">i", 0)  # feedId
    out += struct.pack(">q", 1_000_000_000)  # timeMid
    out += struct.pack(">q", 500_000_000)  # interval
    out += struct.pack(">i", 2)  # numReceptor
    for value in (1.0, 2.0, 3.0):
        out += b"\x01" + struct.pack(">i", 2) + struct.pack(">2f", value, value)
    return out


class _FakeUnpacker:
    def __init__(self, columns: list, pos0: int) -> None:
        self.columns = columns
        self._pos0 = pos0


class _FakeBinTable:
    """Minimal stand-in for an sdmpy SDMBinaryTable payload."""

    def __init__(self, name: str, columns: list, payload: bytes) -> None:
        self.name = name
        self._data = payload
        self._doffs = 0
        self._dsize = len(payload)
        self._unpacker = _FakeUnpacker(columns, 0)


def test_syspower_reads_a_synthetic_payload() -> None:
    """The synthetic row builder matches the reader's expected layout."""
    from tipopac.readers import _fastbin

    payload = _syspower_row() * 3 + b"\n"
    out = _fastbin.unpack_syspower(
        _FakeBinTable("SysPower", _fastbin._SYSPOWER_COLUMNS, payload)
    )

    assert out.shape[0] == 3
    assert out["antennaId"][0] == "Antenna_0"
    np.testing.assert_array_equal(out["switchedPowerSum"][0], [2.0, 2.0])


def test_syspower_truncated_payload_raises() -> None:
    """A row cut short raises instead of silently returning a short table."""
    from tipopac.readers import _fastbin

    payload = _syspower_row() * 3
    truncated = payload[: len(payload) - 20]

    with pytest.raises(_fastbin.FastBinLayoutError, match="bytes unread"):
        _fastbin.unpack_syspower(
            _FakeBinTable("SysPower", _fastbin._SYSPOWER_COLUMNS, truncated)
        )


def test_syspower_corrupt_string_length_raises() -> None:
    """A bogus antenna-name length is caught, not decoded into a short table."""
    from tipopac.readers import _fastbin

    row = _syspower_row()
    payload = row + struct.pack(">i", 2**30) + row[4:]

    with pytest.raises(_fastbin.FastBinLayoutError):
        _fastbin.unpack_syspower(
            _FakeBinTable("SysPower", _fastbin._SYSPOWER_COLUMNS, payload)
        )


def test_layout_guard_rejects_drifted_columns() -> None:
    """A table whose column layout differs raises FastBinLayoutError."""
    from tipopac.readers import _fastbin

    class _FakeUnpacker:
        columns = [("antennaId", False, "S32", ())]  # truncated / wrong

    class _FakeTable:
        name = "SysPower"
        _unpacker = _FakeUnpacker()

    with pytest.raises(_fastbin.FastBinLayoutError):
        _fastbin.unpack_syspower(_FakeTable())
    with pytest.raises(_fastbin.FastBinLayoutError):
        _fastbin.unpack_pointing(_FakeTable())


# ---------------------------------------------------------------------------
# Slow tests — require data/tip_test.sdm
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sdm():
    sdmpy = pytest.importorskip("sdmpy")
    return sdmpy.SDM(str(SDM_PATH), use_xsd=False)


@pytest.mark.slow
def test_syspower_parity(sdm) -> None:
    from tipopac.readers import _fastbin

    fast = _fastbin.unpack_syspower(sdm["SysPower"])
    ref = sdm["SysPower"].data

    assert fast.shape[0] == ref.shape[0]
    for field in (
        "antennaId",
        "spectralWindowId",
        "timeMid",
        "interval",
        "switchedPowerDifference",
        "switchedPowerSum",
    ):
        np.testing.assert_array_equal(
            fast[field], ref[field], err_msg=f"SysPower.{field} mismatch"
        )


@pytest.mark.slow
def test_pointing_parity(sdm) -> None:
    from tipopac.readers import _fastbin

    fast = _fastbin.unpack_pointing(sdm["Pointing"])
    ref = sdm["Pointing"].data

    assert fast.shape[0] == ref.shape[0]
    for field in ("antennaId", "timeMid", "encoder"):
        np.testing.assert_array_equal(
            fast[field], ref[field], err_msg=f"Pointing.{field} mismatch"
        )
