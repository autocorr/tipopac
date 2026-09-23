"""SDM metadata for the synthesized caltable subtables (design.md §9.2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["CaltableMeta", "from_sdm"]

ANTENNA_MOUNT = "ALT-AZ"
ANTENNA_TYPE = "GROUND-BASED"
MEAS_FREQ_REF = 5

_NET_SIDEBAND = {"NOSB": 0, "LSB": 1, "USB": 2, "DSB": 3}


@dataclass(frozen=True)
class CaltableMeta:
    """Full-length ANTENNA and SPECTRAL_WINDOW payloads, indexed by row id."""

    source_name: str
    telescope: str
    ant_name: np.ndarray
    ant_station: np.ndarray
    ant_position: np.ndarray
    ant_offset: np.ndarray
    ant_dish_diameter: np.ndarray
    spw_name: np.ndarray
    spw_ref_frequency: np.ndarray
    spw_total_bandwidth: np.ndarray
    spw_net_sideband: np.ndarray

    @property
    def n_antenna(self) -> int:
        """Number of ANTENNA rows."""
        return len(self.ant_name)

    @property
    def n_spw(self) -> int:
        """Number of SPECTRAL_WINDOW rows."""
        return len(self.spw_name)


def _parse_vector(text: str) -> np.ndarray:
    """Parse an SDM `'1 N v0 v1 …'` one-dimensional array literal."""
    parts = str(text).split()
    return np.array([float(v) for v in parts[2:]], dtype=np.float64)


def _check_spw_ids(spw_ids: list[str]) -> None:
    """Raise unless every `SpectralWindow_N` sits at row `N`."""
    for i, spw_id in enumerate(spw_ids):
        suffix = int(spw_id.split("_")[1])
        if suffix != i:
            raise ValueError(
                f"SpectralWindow.xml row {i} is {spw_id!r}; the caltable writers "
                "index SPECTRAL_WINDOW by row position and require the two to agree"
            )


def _read_antenna(sdm: Any) -> dict[str, np.ndarray]:
    """Return the ANTENNA payload from the Antenna and Station tables."""
    stations = {str(s.stationId): s for s in sdm["Station"]}
    rows = list(sdm["Antenna"])

    name, station, position, offset, diameter = [], [], [], [], []
    for a in rows:
        st = stations[str(a.stationId)]
        name.append(str(a.name))
        station.append(str(st.name))
        position.append(_parse_vector(st.position))
        offset.append(_parse_vector(a.offset))
        diameter.append(float(a.dishDiameter))

    return {
        "ant_name": np.array(name, dtype="U"),
        "ant_station": np.array(station, dtype="U"),
        "ant_position": np.array(position, dtype=np.float64),
        "ant_offset": np.array(offset, dtype=np.float64),
        "ant_dish_diameter": np.array(diameter, dtype=np.float64),
    }


def _read_spectral_window(sdm: Any) -> dict[str, Any]:
    """Return the SPECTRAL_WINDOW payload, one entry per spw id."""
    rows = list(sdm["SpectralWindow"])
    _check_spw_ids([str(s.spectralWindowId) for s in rows])

    name, ref_freq, total_bw, sideband = [], [], [], []
    for s in rows:
        name.append(str(s.name))
        ref_freq.append(float(s.refFreq))
        total_bw.append(float(s.totBandwidth))
        sideband.append(_NET_SIDEBAND.get(str(s.netSideband), 0))

    return {
        "spw_name": np.array(name, dtype="U"),
        "spw_ref_frequency": np.array(ref_freq, dtype=np.float64),
        "spw_total_bandwidth": np.array(total_bw, dtype=np.float64),
        "spw_net_sideband": np.array(sideband, dtype=np.int32),
    }


def _read_telescope(sdm: Any) -> str:
    """Return `ExecBlock.telescopeName`, or an empty string if absent."""
    for row in sdm["ExecBlock"]:
        return str(row.telescopeName)
    return ""


def from_sdm(path: str | Path) -> CaltableMeta:
    """Read the caltable subtable metadata out of the SDM at `path`."""
    import sdmpy

    path = Path(path)
    sdm = sdmpy.SDM(str(path), use_xsd=False)
    return CaltableMeta(
        source_name=path.name,
        telescope=_read_telescope(sdm),
        **_read_antenna(sdm),
        **_read_spectral_window(sdm),
    )
