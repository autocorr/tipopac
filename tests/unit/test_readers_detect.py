"""Unit tests for reader dispatch (tipopac.readers.detect_reader)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tipopac.readers import detect_reader


def _fake_ms(root: Path) -> Path:
    """Directory with the two entries ``MSReader.supports`` looks for."""
    path = root / "fake.ms"
    (path / "SYSPOWER").mkdir(parents=True)
    (path / "table.dat").write_bytes(b"")
    return path


def _fake_sdm(root: Path) -> Path:
    path = root / "fake.sdm"
    path.mkdir()
    (path / "ASDM.xml").write_text("<ASDM/>", encoding="utf-8")
    return path


def test_detect_reader_picks_the_ms_reader(tmp_path: Path) -> None:
    from tipopac.readers.ms import MSReader

    assert detect_reader(_fake_ms(tmp_path)) is MSReader


def test_detect_reader_picks_the_sdm_reader(tmp_path: Path) -> None:
    from tipopac.readers.sdm import SDMReader

    assert detect_reader(_fake_sdm(tmp_path)) is SDMReader


def test_detect_reader_rejects_an_ms_without_syspower(tmp_path: Path) -> None:
    """A calibrated MS with no SYSPOWER subtable carries no tipping data."""
    path = tmp_path / "plain.ms"
    path.mkdir()
    (path / "table.dat").write_bytes(b"")
    with pytest.raises(ValueError, match="not a recognised MS or SDM path"):
        detect_reader(path)


def test_detect_reader_rejects_an_unrelated_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a recognised MS or SDM path"):
        detect_reader(tmp_path / "nothing_here")
