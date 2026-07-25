"""Tests for the fast ASDM binary readers (`readers/_fastbin.py`).

The slow tests require ``data/tip_test.sdm`` and check byte-for-byte parity
against sdmpy's own ``.data`` unpack on the full SysPower / Pointing tables.
The fast guard test needs no fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

SDM_PATH = Path(__file__).parents[2] / "data" / "tip_test.sdm"


# ---------------------------------------------------------------------------
# Fast tests — no SDM required
# ---------------------------------------------------------------------------


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
    if not (SDM_PATH / "SysPower.bin").exists():
        pytest.skip(f"tip_test.sdm not found at {SDM_PATH}")
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
