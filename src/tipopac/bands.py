"""VLA receiver-band labels and selection helpers.

Used by `tipopac.api` and the MS/SDM readers to filter scans and
spectral windows at read time. The default `bands=None` resolves to the
high-frequency receivers (`Ku, K, Ka, Q`) where tipping-curve opacity
fits are well-conditioned; low bands (`L, S, C, X`) are excluded by
default but available on explicit request.

The band label of each SPW is read from the authoritative SPW NAME
string carried by both the MS (`SPECTRAL_WINDOW.NAME`) and the SDM
(`SpectralWindow.xml` `<name>`), in the form ``EVLA_<BAND>#…`` (e.g.
``EVLA_KA#A0C0#16``). The casing of `<BAND>` is not consistent across
sources (`Receiver.xml` uses ``EVLA_Ka`` while `SpectralWindow` uses
``EVLA_KA``), so parsing is case-insensitive and returns the canonical
mixed-case label.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

__all__ = [
    "HIGH_FREQ_DEFAULT",
    "VLA_BAND_LABELS",
    "attach_selection_attrs",
    "band_for_spw_name",
    "normalize_bands",
    "select_spws_by_band",
    "validate_scan_selection",
]


# Canonical VLA receiver band labels in order of increasing frequency.
VLA_BAND_LABELS: tuple[str, ...] = (
    "4",
    "P",
    "L",
    "S",
    "C",
    "X",
    "Ku",
    "K",
    "Ka",
    "Q",
)

HIGH_FREQ_DEFAULT: tuple[str, ...] = ("Ku", "K", "Ka", "Q")

# Case-insensitive lookup: lowercase → canonical label.
_BAND_BY_LOWER: dict[str, str] = {b.lower(): b for b in VLA_BAND_LABELS}


def band_for_spw_name(name: str) -> str:
    """Return the canonical VLA band label parsed from an SPW NAME.

    Expects the receiver-set string used by both the MS and the SDM:
    ``EVLA_<BAND>#<BASEBAND>#<INDEX>`` (e.g. ``EVLA_KA#A0C0#16``). The
    `<BAND>` token is matched case-insensitively against the known VLA
    labels.

    Raises `ValueError` if `name` is empty, lacks the ``EVLA_`` prefix,
    or carries an unknown band token — the SPW NAME is the source of
    truth; an unparseable value should never be silently mapped.
    """
    token = str(name).strip()
    if not token:
        raise ValueError("SPW NAME is empty; cannot identify VLA band")
    head = token.split("#", 1)[0]
    if not head.startswith("EVLA_"):
        raise ValueError(
            f"SPW NAME {token!r} is not in the expected ``EVLA_<BAND>#…`` "
            f"form; cannot identify VLA receiver band"
        )
    band_token = head[len("EVLA_") :]
    canonical = _BAND_BY_LOWER.get(band_token.lower())
    if canonical is None:
        raise ValueError(
            f"SPW NAME {token!r} carries unknown band token "
            f"{band_token!r}; known bands: {VLA_BAND_LABELS}"
        )
    return canonical


def normalize_bands(bands: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize a user-supplied band selection.

    - `None` → `HIGH_FREQ_DEFAULT` (`Ku, K, Ka, Q`).
    - Case-insensitive match against `VLA_BAND_LABELS`; preserves input
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
        raise ValueError(f"unknown band(s) {bad!r}; known bands: {VLA_BAND_LABELS}")
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
    spw_bands: np.ndarray,
    allowed_bands: Sequence[str],
) -> list[int]:
    """Return SPW ids whose band label is in `allowed_bands`.

    `tip_spws` are the candidate SPW indices (into `spw_bands`);
    `spw_bands[i]` is the canonical band label for SPW `i` (as produced
    by `band_for_spw_name` at read time). The result preserves the
    input order.
    """
    allowed = set(allowed_bands)
    return [int(s) for s in tip_spws if str(spw_bands[int(s)]) in allowed]
