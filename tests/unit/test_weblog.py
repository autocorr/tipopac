"""Unit tests for tipopac.weblog (DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path

from tipopac.weblog import build_weblog


def _touch(path: Path) -> None:
    path.write_text("<html></html>", encoding="utf-8")


def test_build_weblog_writes_index_html(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    out = build_weblog(tmp_path)
    assert out == tmp_path / "index.html"
    assert out.exists()


def test_build_weblog_index_is_self_contained(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # Self-contained == no external CSS / JS references.
    assert "<link" not in body
    assert 'src="http' not in body
    # Has inline style + script.
    assert "<style>" in body
    assert "<script>" in body


def test_build_weblog_lists_present_aggregates_only(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    _touch(tmp_path / "tcal_ref_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "tau_vs_frequency.html" in body
    assert "tcal_ref_vs_frequency.html" in body
    # Missing aggregates must not appear in the dropdown.
    assert "tcal_fit_vs_frequency.html" not in body
    assert 'value="c_vs_frequency.html"' not in body


def test_build_weblog_offers_elevation_when_tippingcurve_files_present(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / "tippingcurve_spw_0_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert 'value="elevation"' in body
    assert "Elevation curve" in body


def test_build_weblog_no_elevation_without_tippingcurve_files(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert 'value="elevation"' not in body


def test_build_weblog_scan_and_antenna_dropdowns_populated_from_filenames(
    tmp_path: Path,
) -> None:
    for name in (
        "tippingcurve_spw_0_ea01_scan_4.html",
        "tippingcurve_spw_3_ea02_scan_10.html",
        "tippingcurve_spw_3_ea01_scan_4.html",
    ):
        _touch(tmp_path / name)
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '<select id="scan">' in body
    assert '<select id="antenna">' in body
    assert '<option value="4">4</option>' in body
    assert '<option value="10">10</option>' in body
    assert '<option value="ea01">ea01</option>' in body
    assert '<option value="ea02">ea02</option>' in body


def test_build_weblog_spw_dropdown_starts_empty(tmp_path: Path) -> None:
    """spw options are added by JS once a scan is picked, not at render time."""
    _touch(tmp_path / "tippingcurve_spw_0_ea01_scan_4.html")
    _touch(tmp_path / "tippingcurve_spw_3_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # Initial spw select only carries the placeholder — no static spw values.
    assert '<select id="spw"><option value="">—</option></select>' in body


def test_build_weblog_embeds_scan_to_spws_map(tmp_path: Path) -> None:
    """The per-scan spw filter is driven by an embedded SCAN_TO_SPWS map."""
    for name in (
        "tippingcurve_spw_0_ea01_scan_4.html",
        "tippingcurve_spw_3_ea01_scan_4.html",
        "tippingcurve_spw_13_ea01_scan_10.html",
    ):
        _touch(tmp_path / name)
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "SCAN_TO_SPWS" in body
    # Scan 4 saw spws 0 and 3; scan 10 saw spw 13 only.
    assert '"4": [0, 3]' in body
    assert '"10": [13]' in body


def test_build_weblog_embeds_available_set(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    _touch(tmp_path / "tippingcurve_spw_0_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The JS-side existence check uses an embedded set.
    assert "AVAILABLE = new Set(" in body
    assert '"tau_vs_frequency.html"' in body
    assert '"tippingcurve_spw_0_ea01_scan_4.html"' in body


def test_build_weblog_announces_missing_plot_string(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The user-visible string the GUI shows when a requested file is absent.
    assert "Plot not found:" in body


def test_build_weblog_lists_fit_quality_heatmap_when_present(tmp_path: Path) -> None:
    _touch(tmp_path / "fit_quality_heatmap.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "fit_quality_heatmap.html" in body
    assert "Fit quality heatmap" in body


def test_build_weblog_ignores_existing_index_html(tmp_path: Path) -> None:
    _touch(tmp_path / "tau_vs_frequency.html")
    _touch(tmp_path / "index.html")  # stale index from a prior run
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The set must not list index.html itself.
    assert '"index.html"' not in body


def test_build_weblog_empty_directory(tmp_path: Path) -> None:
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "(no plots found in this directory)" in body
