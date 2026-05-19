"""Geometry helpers for tipopac (DESIGN.md §6.2)."""

from __future__ import annotations

import numpy as np

__all__ = ["zenith_angle"]


def zenith_angle(el_encoder_rad: np.ndarray) -> np.ndarray:
    """Convert AZELGEO elevation encoder (rad) to zenith angle (deg).

    No refraction correction is applied — flat-earth assumption matches v2.6.
    """
    return 90.0 - np.rad2deg(el_encoder_rad)
