"""TippingReader Protocol — the interface both MSReader and SDMReader implement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import xarray as xr


class TippingReader(Protocol):
    """Parse a tipping-data source into the canonical xarray.Dataset (DESIGN.md §5)."""

    @classmethod
    def supports(cls, path: Path) -> bool: ...

    @classmethod
    def from_path(cls, path: Path) -> "TippingReader": ...

    def read(self) -> xr.Dataset: ...


@dataclass(frozen=True)
class SkydipScanInfo:
    """Lightweight DO_SKYDIP scan record used by `summarize_skydip_scans`.

    Populated by ``MSReader.list_skydip_scans`` /
    ``SDMReader.list_skydip_scans`` without invoking the full dataset
    read.
    """

    scan_id: int
    start_mjd_s: float
    spw_ids: tuple[int, ...]
    bands: tuple[str, ...]
