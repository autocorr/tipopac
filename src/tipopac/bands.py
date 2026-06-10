"""VLA receiver-band table and selection helpers.

Used by `tipopac.api` and the MS/SDM readers to filter scans and
spectral windows at read time. The default `bands=None` resolves to the
high-frequency receivers (`Ku, K, Ka, Q`) where tipping-curve opacity
fits are well-conditioned; low bands (`L, S, C, X`) are excluded by
default but available on explicit request.

The band table covers the full VLA receiver suite; frequency edges are
the standard receiver coverage. SPWs that span no band raise — real
VLA data should never miss.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

__all__ = [
    "HIGH_FREQ_DEFAULT",
    "VLA_BANDS",
    "attach_selection_attrs",
    "band_for_frequency",
    "normalize_bands",
    "select_spws_by_band",
    "validate_scan_selection",
]


# VLA receiver bands in Hz. Edges chosen per the standard band labels
# (e.g. https://science.nrao.edu/facilities/vla/docs/manuals/oss).
# Multiple SPWs may share a label (e.g. low-Ka and high-Ka in the same
# scan are both "Ka") — the label is not a unique key for an SPW.
VLA_BANDS: dict[str, tuple[float, float]] = {
    "4": (58.0e6, 84.0e6),
    "P": (224.0e6, 480.0e6),
    "L": (1.0e9, 2.0e9),
    "S": (2.0e9, 4.0e9),
    "C": (4.0e9, 8.0e9),
    "X": (8.0e9, 12.0e9),
    "Ku": (12.0e9, 18.0e9),
    "K": (18.0e9, 26.5e9),
    "Ka": (26.5e9, 40.0e9),
    "Q": (40.0e9, 50.0e9),
}

HIGH_FREQ_DEFAULT: tuple[str, ...] = ("Ku", "K", "Ka", "Q")

# Case-insensitive lookup: lowercase → canonical key.
_BAND_BY_LOWER: dict[str, str] = {k.lower(): k for k in VLA_BANDS}


def band_for_frequency(freq_Hz: float) -> str:
    """Return the VLA band label for a frequency in Hz.

    Raises `ValueError` if the frequency falls outside every band — real
    VLA tipping-scan SPWs should never hit this path.
    """
    for name, (lo, hi) in VLA_BANDS.items():
        if lo <= freq_Hz <= hi:
            return name
    raise ValueError(
        f"frequency {freq_Hz:.3e} Hz falls outside any VLA band "
        f"(known bands: {tuple(VLA_BANDS)})"
    )


def normalize_bands(bands: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a user-supplied band selection.

    - `None` → `HIGH_FREQ_DEFAULT` (`Ku, K, Ka, Q`).
    - Case-insensitive match against `VLA_BANDS` keys; preserves input
      order and dedupes.
    - Empty sequence raises `ValueError` (explicit empty ≠ default).
    - Unknown band names raise `ValueError`, naming the offender and
      listing the known bands.
    """
    if bands is None:
        return HIGH_FREQ_DEFAULT
    if len(bands) == 0:
        raise ValueError(
            "bands=[] is an explicit empty selection. Pass `bands=None` "
            "for the default Ku/K/Ka/Q set."
        )
    seen: dict[str, None] = {}
    bad: list[str] = []
    for token in bands:
        canonical = _BAND_BY_LOWER.get(str(token).lower())
        if canonical is None:
            bad.append(str(token))
            continue
        seen.setdefault(canonical, None)
    if bad:
        raise ValueError(f"unknown band(s) {bad!r}; known bands: {tuple(VLA_BANDS)}")
    return tuple(seen)


def validate_scan_selection(
    requested: Sequence[int] | None,
    available: Sequence[int],
) -> list[int]:
    """Resolve a user-supplied scan selection against the DO_SKYDIP set.

    - `None` → return `list(available)` (all DO_SKYDIP scans).
    - Empty sequence raises `ValueError`.
    - Any requested id not in `available` raises `ValueError` naming
      the offenders.
    - Otherwise returns the requested ids in the order they appear in
      `available` (so downstream code sees them sorted as the reader
      already sorted them).
    """
    if requested is None:
        return list(available)
    if len(requested) == 0:
        raise ValueError(
            "scans=[] is an explicit empty selection. Pass `scans=None` "
            "to include all DO_SKYDIP scans."
        )
    requested_set = {int(s) for s in requested}
    available_set = set(int(s) for s in available)
    missing = sorted(requested_set - available_set)
    if missing:
        raise ValueError(
            f"requested scan(s) {missing} are not DO_SKYDIP scans; "
            f"available DO_SKYDIP scans: {sorted(available_set)}"
        )
    return [int(s) for s in available if int(s) in requested_set]


def attach_selection_attrs(
    ds: xr.Dataset,
    scans_requested: Sequence[int] | None,
    bands_requested: Sequence[str] | None,
) -> None:
    """Record scan / band selection provenance on `ds.attrs` in place.

    Writes four attrs (see DESIGN.md §4):
      - ``scans_requested``: ``"all"`` or ``list[int]`` (raw user input).
      - ``bands_requested``: ``"default_high_freq"`` or ``list[str]``.
      - ``selected_scans``: resolved DO_SKYDIP scan ids on ``ds``.
      - ``selected_bands``: sorted unique band labels present on ``ds``.
    """
    ds.attrs["scans_requested"] = (
        "all" if scans_requested is None else [int(s) for s in scans_requested]
    )
    ds.attrs["bands_requested"] = (
        "default_high_freq"
        if bands_requested is None
        else [str(b) for b in bands_requested]
    )
    ds.attrs["selected_scans"] = [int(s) for s in ds.coords["scan"].values]
    ds.attrs["selected_bands"] = sorted(
        {str(b) for b in ds.coords["band"].values.tolist()}
    )


def select_spws_by_band(
    tip_spws: Sequence[int],
    spw_freq: np.ndarray,
    allowed_bands: Sequence[str],
) -> list[int]:
    """Return SPW ids whose ref frequency falls in `allowed_bands`.

    `tip_spws` are the candidate SPW indices (into `spw_freq`); the
    result preserves the input order.
    """
    allowed = set(allowed_bands)
    return [
        int(s)
        for s in tip_spws
        if band_for_frequency(float(spw_freq[int(s)])) in allowed
    ]
