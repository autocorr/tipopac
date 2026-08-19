"""Shared fixtures for the slow tests.

Reading `data/tip_test.ms` takes ~3 s and several slow tests need the same
default-selection dataset, so the reads are session-scoped and each test
gets a copy. Non-default selections (explicit `scans`/`bands`) still read
inline — they cannot share a cached dataset.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import xarray as xr

from tipopac._casa import silence_casa_log

silence_casa_log()

DATA_DIR = Path(
    os.environ.get("TIPOPAC_TEST_DATA_DIR") or Path(__file__).parents[1] / "data"
)
MS_PATH = DATA_DIR / "tip_test.ms"
SDM_PATH = DATA_DIR / "tip_test.sdm"


def n_test_workers() -> int:
    """Stage-A fit parallelism for the slow tests, overridable per run.

    ``TIPOPAC_TEST_WORKERS=1`` restores serial fitting for debugging.
    """
    v = os.environ.get("TIPOPAC_TEST_WORKERS")
    if v:
        return int(v)
    return min(16, os.cpu_count() or 1)


@pytest.fixture(scope="session")
def n_workers() -> int:
    """Stage-A fit parallelism for the slow tests."""
    return n_test_workers()


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Fail the run outright if slow tests are selected without their data.

    A test declares its fixture by requesting ``ds_ms``/``ds_sdm`` or by
    carrying ``needs_ms``/``needs_sdm``, so an MS-only checkout can still
    run the MS-only slow tests.
    """
    required: set[Path] = set()
    for item in items:
        if item.get_closest_marker("slow") is None:
            continue
        names = getattr(item, "fixturenames", ())
        if "ds_ms" in names or item.get_closest_marker("needs_ms"):
            required.add(MS_PATH)
        if "ds_sdm" in names or item.get_closest_marker("needs_sdm"):
            required.add(SDM_PATH)
    missing = sorted(str(p) for p in required if not p.exists())
    if missing:
        raise pytest.UsageError(
            "slow tests selected but the validation data is missing: "
            + ", ".join(missing)
            + " — see docs/installation.md"
        )


@pytest.fixture(scope="session")
def ms_dataset_source() -> xr.Dataset:
    """Default-selection MS dataset, read once per session. Do not mutate."""
    from tipopac.readers.ms import MSReader

    return MSReader.from_path(MS_PATH).read()


@pytest.fixture(scope="session")
def sdm_dataset_source() -> xr.Dataset:
    """Default-selection SDM dataset, read once per session. Do not mutate."""
    from tipopac.readers.sdm import SDMReader

    return SDMReader.from_path(SDM_PATH).read()


@pytest.fixture
def ds_ms(ms_dataset_source: xr.Dataset) -> xr.Dataset:
    """Fresh copy of the default-selection MS dataset, safe to mutate."""
    return ms_dataset_source.copy(deep=True)


@pytest.fixture
def ds_sdm(sdm_dataset_source: xr.Dataset) -> xr.Dataset:
    """Fresh copy of the default-selection SDM dataset, safe to mutate."""
    return sdm_dataset_source.copy(deep=True)
