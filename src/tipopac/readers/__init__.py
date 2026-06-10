"""Reader registry and dispatch — pick MSReader or SDMReader for a given path."""

from __future__ import annotations

from pathlib import Path

_READERS: list = []


def _get_readers() -> list:
    """Return the registered reader classes, importing on first use.

    Deferred import keeps `casatools` / `sdmpy` off the import path until
    a reader is actually needed.
    """
    global _READERS
    if not _READERS:
        from tipopac.readers.ms import MSReader
        from tipopac.readers.sdm import SDMReader

        _READERS = [MSReader, SDMReader]
    return _READERS


def detect_reader(path: Path):
    """Return the reader class whose `supports()` matches `path`."""
    for R in _get_readers():
        if R.supports(path):
            return R
    raise ValueError(
        f"{path} is not a recognised MS or SDM path "
        f"(no reader's supports() returned True)"
    )


__all__ = ["detect_reader"]
