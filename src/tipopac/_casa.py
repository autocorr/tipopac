"""Deferred `casatools` import with its per-process log file suppressed."""

from __future__ import annotations

import os
from types import ModuleType


def silence_casa_log() -> None:
    """Route the CASA log sink to os.devnull; call before importing casatools.

    `casatools` opens `casa-<timestamp>.log` in the working directory at
    import time. Leaving `logfile` unset (`None`) also drops the file, but
    then casaconfig's start-up messages go to the terminal instead.
    """
    from casaconfig import config

    config.logfile = os.devnull  # ty: ignore[unresolved-attribute]


def import_casatools() -> ModuleType:
    """Return the `casatools` module, imported without creating a log file."""
    silence_casa_log()
    import casatools

    return casatools
