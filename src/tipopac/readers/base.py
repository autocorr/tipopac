"""TippingReader Protocol — the interface both MSReader and SDMReader implement."""

from __future__ import annotations

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
