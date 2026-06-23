"""TippingReader Protocol and shared MS/SDM reader helpers.

Both ``MSReader`` and ``SDMReader`` parse their format-specific tables into
the *same* canonical :class:`xarray.Dataset` (DESIGN.md §5). The selection,
nearest-sample, and final-assembly logic is identical between them and lives
here so the two readers cannot drift — the SDM↔MS parity contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import xarray as xr

from tipopac import schema
from tipopac.bands import normalize_bands, select_spws_by_band, validate_scan_selection


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


def _apply_selection(
    scan_ids: list[int],
    scan_spws: dict[int, list[int]],
    scan_t_start: dict[int, float],
    scan_t_end: dict[int, float],
    spw_bands: np.ndarray,
    scans_requested: Sequence[int] | None,
    bands_requested: Sequence[str] | None,
) -> tuple[
    list[int],
    dict[int, list[int]],
    dict[int, float],
    dict[int, float],
    list[int],
]:
    """Apply the user's scan + band selection to reader metadata.

    Shared by both readers. Raises `ValueError` if the resulting set is empty.
    """
    scan_ids = validate_scan_selection(scans_requested, scan_ids)
    scan_spws = {sc: scan_spws[sc] for sc in scan_ids}
    scan_t_start = {sc: scan_t_start[sc] for sc in scan_ids}
    scan_t_end = {sc: scan_t_end[sc] for sc in scan_ids}

    tip_spws_all = sorted({s for spws in scan_spws.values() for s in spws})

    allowed_bands = normalize_bands(bands_requested)
    tip_spws = select_spws_by_band(tip_spws_all, spw_bands, allowed_bands)
    if not tip_spws:
        observed = sorted({str(spw_bands[s]) for s in tip_spws_all})
        raise ValueError(
            f"no SPWs match bands={list(allowed_bands)!r}; "
            f"observed bands in this dataset: {observed}"
        )

    keep_set = set(tip_spws)
    kept_scan_ids: list[int] = []
    for sc in scan_ids:
        scan_spws[sc] = [s for s in scan_spws[sc] if s in keep_set]
        if scan_spws[sc]:
            kept_scan_ids.append(sc)
    if not kept_scan_ids:
        raise ValueError(
            f"no scans retain any SPW after bands={list(allowed_bands)!r} "
            "filter; widen the band selection or pick different scans"
        )

    dropped = [sc for sc in scan_ids if sc not in kept_scan_ids]
    # When the user named scans explicitly, a band-filter drop of any of
    # those is a contract violation — match the "raise on miss" behavior
    # of `validate_scan_selection`.
    if scans_requested is not None and dropped:
        raise ValueError(
            f"requested scan(s) {dropped} have no SPWs in "
            f"bands={list(allowed_bands)!r}; either widen the band "
            "selection or drop these scans from the request"
        )
    for sc in dropped:
        scan_spws.pop(sc, None)
        scan_t_start.pop(sc, None)
        scan_t_end.pop(sc, None)

    return kept_scan_ids, scan_spws, scan_t_start, scan_t_end, tip_spws


def _nearest_idx(ref_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    """Return indices into `ref_times` nearest to each value in `query_times`."""
    idx = np.searchsorted(ref_times, query_times)
    idx = np.clip(idx, 0, len(ref_times) - 1)
    left = np.clip(idx - 1, 0, len(ref_times) - 1)
    use_left = np.abs(query_times - ref_times[left]) < np.abs(
        query_times - ref_times[idx]
    )
    idx[use_left] = left[use_left]
    return idx


def build_canonical_dataset(
    *,
    path: Path,
    source_format: str,
    scan_ids: list[int],
    ant_names: np.ndarray | list[str],
    tip_spws: list[int],
    ant_positions: np.ndarray,
    spw_freq: np.ndarray,
    spw_bw: np.ndarray,
    spw_bands: np.ndarray,
    switched_diff: np.ndarray,
    switched_sum: np.ndarray,
    zenith_angle: np.ndarray,
    tcal_ref: np.ndarray,
    weather_T: np.ndarray,
    weather_P: np.ndarray,
    weather_RH: np.ndarray,
    exposure_time: np.ndarray,
    flag: np.ndarray,
    scan_time_start: np.ndarray,
    scan_time_end: np.ndarray,
    time_utc: np.ndarray,
) -> xr.Dataset:
    """Assemble already-built reader arrays into the canonical schema dataset.

    The single source of truth for the dataset layout both readers must emit
    (the SDM↔MS parity contract). ``spw_freq``/``spw_bw``/``spw_bands`` are the
    full per-spw arrays; they are indexed by ``tip_spws`` here.
    """
    return xr.Dataset(
        data_vars={
            "switched_diff": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_diff,
            ),
            "switched_sum": (
                ("scan", "antenna", "spw", "polarization", "time"),
                switched_sum,
            ),
            "zenith_angle": (("scan", "antenna", "time"), zenith_angle),
            "tcal_ref": (("antenna", "spw", "polarization"), tcal_ref),
            "weather_T": (("scan", "time"), weather_T),
            "weather_P": (("scan", "time"), weather_P),
            "weather_RH": (("scan", "time"), weather_RH),
            "exposure_time": (("scan", "time"), exposure_time),
            "flag": (("scan", "antenna", "spw", "polarization", "time"), flag),
        },
        coords={
            "scan": np.array(scan_ids, dtype=np.intp),
            "antenna": ant_names,
            "spw": np.array(tip_spws, dtype=np.intp),
            "polarization": list(schema.POL_VALUES),
            "xyz": ["X", "Y", "Z"],
            "frequency": (("spw",), spw_freq[tip_spws].astype(np.float64)),
            "bandwidth": (("spw",), spw_bw[tip_spws].astype(np.float64)),
            "band": (("spw",), spw_bands[tip_spws].astype("U4")),
            "antenna_position": (("antenna", "xyz"), ant_positions.astype(np.float64)),
            "scan_time_start": (("scan",), scan_time_start),
            "scan_time_end": (("scan",), scan_time_end),
            "time_utc": (("scan", "time"), time_utc),
        },
        attrs={
            "source_path": str(path),
            "source_format": source_format,
            "observatory": "VLA",
        },
    )
