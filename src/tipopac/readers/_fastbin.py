"""Fast readers for the two ASDM binary tables tipopac needs.

sdmpy unpacks *every* column of *every* row of the ``SysPower`` and
``Pointing`` binary tables in pure Python (``sdmpy/bintab.py``), which is
~97 % of a typical SDM read. tipopac uses
only 6 of SysPower's 9 columns and 3 of Pointing's 19. These functions walk
the binary payload, extract just the needed columns, and return a numpy
structured array that drops in for ``sdm[name].data`` — measured ~37× faster
with byte-for-byte parity.

We reuse sdmpy's cheap container work (MIME split + entity-header skip): the
``SDMBinaryTable`` exposes ``_data`` (file bytes), ``_doffs`` (payload
offset), ``_dsize`` (payload byte budget) and ``_unpacker._pos0`` (first-row
offset) without triggering the slow ``.unpack()``. Only the row loop is
replaced.

The ASDM header records ``nrows = -1`` (unknown) and rows are variable-length
(leading length-prefixed id strings + optional fields), so the row count is
not known a priori and there is no fixed stride. Each table is therefore
walked twice: a cheap count pass (advance the cursor only), then an
exact-size fill pass — no over-allocation, no incremental growth. A row that
does not decode, or two passes that disagree on the row count, raise
``FastBinLayoutError`` so the caller falls back to sdmpy; the alternative is a
short read that looks like a scan with missing samples.

Row format (``sdmpy/bintab.py`` ``_get_val`` + the unpacker ``columns``):
big-endian; string = 4-byte length prefix + bytes; optional field = 1
presence byte then the value if present (absent numeric → sdmpy fills 0);
array field carries inline int32 dims then the elements.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FastBinLayoutError", "unpack_syspower", "unpack_pointing"]


class FastBinLayoutError(RuntimeError):
    """Raised when a table's sdmpy column layout is not the expected VLA one."""


# Expected column layouts, copied from sdmpy's SysPowerUnpacker /
# PointingUnpacker (bintab.py:195, :214). If sdmpy changes these the byte
# offsets below no longer hold, so we refuse the fast path and fall back.
_SYSPOWER_COLUMNS = [
    ("antennaId", False, "S32", ()),
    ("spectralWindowId", False, "S32", ()),
    ("feedId", False, "i4", ()),
    ("timeMid", False, "i8", ()),
    ("interval", False, "i8", ()),
    ("numReceptor", False, "i4", ()),
    ("switchedPowerDifference", True, "f4", (2,)),
    ("switchedPowerSum", True, "f4", (2,)),
    ("requantizerGain", True, "f4", (2,)),
]
_POINTING_COLUMNS = [
    ("antennaId", False, "S32", ()),
    ("timeMid", False, "i8", ()),
    ("interval", False, "i8", ()),
    ("numSample", False, "i4", ()),
    ("encoder", False, "f8", (1, 2)),
    ("pointingTracking", False, "b", ()),
    ("usePolynomials", False, "b", ()),
    ("timeOrigin", False, "i8", ()),
    ("numTerm", False, "i4", ()),
    ("pointingDirection", False, "f8", (1, 2)),
    ("target", False, "f8", (1, 2)),
    ("offset", False, "f8", (1, 2)),
    ("pointingModelId", False, "i4", ()),
    ("overTheTop", True, "b", ()),
    ("sourceOffset", True, "f8", (1, 2)),
    ("sourceOffsetReferenceCode", True, "i4", ()),
    ("sourceOffsetEquinox", True, "i4", ()),
    ("sampledTimeInterval", True, "i4", ()),
    ("atmosphericCorrection", True, "f8", (1, 2)),
]

_I4 = struct.Struct(">i")
_I8 = struct.Struct(">q")
_F8x2 = struct.Struct(">2d")
_F4x2 = struct.Struct(">2f")

# Fixed byte width of Pointing columns 6..13 (pointingTracking..pointingModelId):
# b + b + i8 + i4 + 3×[8 dim + 16 data] + i4 = 1+1+8+4+24+24+24+4 = 90.
_POINTING_FIXED_TAIL = 90
# Optional Pointing columns 14..19: value width if present (excludes presence byte).
_POINTING_OPT_WIDTHS = (1, 24, 4, 4, 4, 24)


def _payload_bounds(table: Any) -> tuple[bytes, int, int]:
    """Return (buffer, first_row_offset, end_offset) from the sdmpy table.

    Uses sdmpy internals but does not trigger the slow ``.unpack()``.
    """
    buf = table._data
    start = table._doffs + table._unpacker._pos0
    end = table._doffs + table._dsize
    return buf, start, end


def _truncated(name: str, row: int, p: int, end: int) -> FastBinLayoutError:
    """Error for a row walk that ran off its payload before the end."""
    return FastBinLayoutError(
        f"{name}: row {row} does not decode at byte {p}; "
        f"{end - p} of {end} payload bytes unread — fast reader disabled"
    )


def _check_count(name: str, n_count: int, n_fill: int) -> None:
    """Both walks must agree; a short fill pass would leave arrays uninitialised."""
    if n_fill != n_count:
        raise FastBinLayoutError(
            f"{name}: fill pass read {n_fill} rows, count pass {n_count} "
            "— fast reader disabled"
        )


def _check_layout(table: Any, expected: list) -> None:
    if list(table._unpacker.columns) != expected:
        raise FastBinLayoutError(
            f"{table.name}: unexpected sdmpy column layout; "
            "fast reader disabled (sdmpy version drift?)"
        )


def unpack_syspower(table: Any) -> np.ndarray:
    """Read the SysPower binary table into a structured array (6 columns).

    Fields: ``antennaId`` (<U32), ``spectralWindowId`` (<U32), ``timeMid``
    (i8, ns), ``interval`` (i8, ns), ``switchedPowerDifference`` (f4, (2,)),
    ``switchedPowerSum`` (f4, (2,)). Matches ``sdm['SysPower'].data`` on
    these fields; absent optionals read back as 0 (sdmpy convention).
    """
    _check_layout(table, _SYSPOWER_COLUMNS)
    buf, start, end = _payload_bounds(table)

    n = _walk_syspower(buf, start, end, None)
    cols = _SysPowerCols(
        antennaId=np.empty(n, "<U32"),
        spectralWindowId=np.empty(n, "<U32"),
        timeMid=np.empty(n, "i8"),
        interval=np.empty(n, "i8"),
        # zero-filled so absent optionals read back as 0 (sdmpy convention)
        switchedPowerDifference=np.zeros((n, 2), "f4"),
        switchedPowerSum=np.zeros((n, 2), "f4"),
    )
    _check_count("SysPower", n, _walk_syspower(buf, start, end, cols))

    out = np.empty(
        n,
        dtype=[
            ("antennaId", "<U32"),
            ("spectralWindowId", "<U32"),
            ("timeMid", "i8"),
            ("interval", "i8"),
            ("switchedPowerDifference", "f4", (2,)),
            ("switchedPowerSum", "f4", (2,)),
        ],
    )
    out["antennaId"] = cols.antennaId
    out["spectralWindowId"] = cols.spectralWindowId
    out["timeMid"] = cols.timeMid
    out["interval"] = cols.interval
    out["switchedPowerDifference"] = cols.switchedPowerDifference
    out["switchedPowerSum"] = cols.switchedPowerSum
    return out


@dataclass
class _SysPowerCols:
    antennaId: np.ndarray
    spectralWindowId: np.ndarray
    timeMid: np.ndarray
    interval: np.ndarray
    switchedPowerDifference: np.ndarray
    switchedPowerSum: np.ndarray


def _walk_syspower(buf: bytes, p: int, end: int, cols: _SysPowerCols | None) -> int:
    """Walk SysPower rows from ``p`` to ``end``; return the row count.

    ``cols is None`` counts only (advance the cursor, no decode). Otherwise
    fills the exact-size column arrays in ``cols`` (sized by the count pass).
    Raises FastBinLayoutError if a row does not decode.
    """
    n = 0
    fill = cols is not None
    while p < end - 4:
        row_start = p
        try:
            length = _I4.unpack_from(buf, p)[0]
            p += 4
            if fill:
                a = buf[p : p + length].decode()
            p += length  # antennaId bytes
            length = _I4.unpack_from(buf, p)[0]
            p += 4
            if fill:
                s = buf[p : p + length].decode()
            p += length  # spectralWindowId bytes
            p += 4  # feedId (skip)
            tm = _I8.unpack_from(buf, p)[0]
            p += 8
            iv = _I8.unpack_from(buf, p)[0]
            p += 8
            p += 4  # numReceptor (skip)
            # switchedPowerDifference / Sum / requantizerGain: optional f4[2],
            # each = 1 presence byte, then [int32 dim + 2 floats] if present.
            d = None
            if buf[p]:
                p += 1 + 4
                if fill:
                    d = _F4x2.unpack_from(buf, p)
                p += 8
            else:
                p += 1
            sm = None
            if buf[p]:
                p += 1 + 4
                if fill:
                    sm = _F4x2.unpack_from(buf, p)
                p += 8
            else:
                p += 1
            if buf[p]:  # requantizerGain (skip)
                p += 1 + 4 + 8
            else:
                p += 1
        except (struct.error, IndexError, UnicodeDecodeError) as exc:
            raise _truncated("SysPower", n, row_start, end) from exc

        if fill:
            cols.antennaId[n] = a
            cols.spectralWindowId[n] = s
            cols.timeMid[n] = tm
            cols.interval[n] = iv
            if d is not None:
                cols.switchedPowerDifference[n] = d
            if sm is not None:
                cols.switchedPowerSum[n] = sm
        n += 1

    return n


def unpack_pointing(table: Any) -> np.ndarray:
    """Read the Pointing binary table into a structured array (3 columns).

    Fields: ``antennaId`` (<U32), ``timeMid`` (i8, ns), ``encoder``
    (f8, (1, 2)). Only these are extracted; the other 16 columns are skipped
    by advancing the byte cursor. Matches ``sdm['Pointing'].data`` on these
    fields.
    """
    _check_layout(table, _POINTING_COLUMNS)
    buf, start, end = _payload_bounds(table)

    n = _walk_pointing(buf, start, end, None)
    cols = _PointingCols(
        antennaId=np.empty(n, "<U32"),
        timeMid=np.empty(n, "i8"),
        encoder=np.zeros((n, 1, 2), "f8"),
    )
    _check_count("Pointing", n, _walk_pointing(buf, start, end, cols))

    out = np.empty(
        n,
        dtype=[
            ("antennaId", "<U32"),
            ("timeMid", "i8"),
            ("encoder", "f8", (1, 2)),
        ],
    )
    out["antennaId"] = cols.antennaId
    out["timeMid"] = cols.timeMid
    out["encoder"] = cols.encoder
    return out


@dataclass
class _PointingCols:
    antennaId: np.ndarray
    timeMid: np.ndarray
    encoder: np.ndarray


def _walk_pointing(buf: bytes, p: int, end: int, cols: _PointingCols | None) -> int:
    """Walk Pointing rows from ``p`` to ``end``; return the row count.

    ``cols is None`` counts only (no decode). Otherwise fills the exact-size
    column arrays in ``cols``. Raises FastBinLayoutError if a row does not
    decode.
    """
    n = 0
    fill = cols is not None
    while p < end - 4:
        row_start = p
        try:
            length = _I4.unpack_from(buf, p)[0]
            p += 4
            if fill:
                a = buf[p : p + length].decode()
            p += length  # antennaId bytes
            tm = _I8.unpack_from(buf, p)[0]
            p += 8
            p += 12  # interval (i8) + numSample (i4)
            # encoder: inline dims (1, 2) then 2 doubles [az, el].
            d0 = _I4.unpack_from(buf, p)[0]
            d1 = _I4.unpack_from(buf, p + 4)[0]
            if (d0, d1) != (1, 2):
                raise FastBinLayoutError(
                    f"Pointing.encoder dims {(d0, d1)} != (1, 2); non-VLA data?"
                )
            p += 8
            if fill:
                azel = _F8x2.unpack_from(buf, p)
            p += 16
            p += _POINTING_FIXED_TAIL  # cols 6..13 (fixed width)
            for w in _POINTING_OPT_WIDTHS:  # cols 14..19 (optional)
                if buf[p]:
                    p += 1 + w
                else:
                    p += 1
        except (struct.error, IndexError, UnicodeDecodeError) as exc:
            raise _truncated("Pointing", n, row_start, end) from exc

        if fill:
            cols.antennaId[n] = a
            cols.timeMid[n] = tm
            cols.encoder[n, 0, 0] = azel[0]
            cols.encoder[n, 0, 1] = azel[1]
        n += 1

    return n
