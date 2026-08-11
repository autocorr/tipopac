"""Unit tests for tipopac.weblog (DESIGN.md §9.3)."""

from __future__ import annotations

from pathlib import Path

from tipopac.weblog import build_weblog


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<html></html>", encoding="utf-8")


def _group(root: Path, k: int = 0) -> Path:
    return root / f"group_{k}"


def test_build_weblog_writes_index_html(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    out = build_weblog(tmp_path)
    assert out == tmp_path / "index.html"
    assert out.exists()


def test_build_weblog_index_is_self_contained(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # Self-contained == no external CSS / JS references.
    assert "<link" not in body
    assert 'src="http' not in body
    # Has inline style + script.
    assert "<style>" in body
    assert "<script>" in body


# ---------------------------------------------------------------------------
# Group discovery
# ---------------------------------------------------------------------------


def test_build_weblog_lists_one_option_per_group_dir(tmp_path: Path) -> None:
    for k in (0, 1, 2):
        _touch(_group(tmp_path, k) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '<select id="group">' in body
    for k in (0, 1, 2):
        assert f'<option value="{k}">Group {k}</option>' in body


def test_build_weblog_group_options_are_numerically_sorted(tmp_path: Path) -> None:
    for k in (10, 2, 1):
        _touch(_group(tmp_path, k) / "summary.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    order = [body.index(f'<option value="{k}">Group {k}</option>') for k in (1, 2, 10)]
    assert order == sorted(order)


def test_build_weblog_paths_are_group_scoped(tmp_path: Path) -> None:
    _touch(_group(tmp_path, 1) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"group_1/tau_vs_frequency.html"' in body


def test_build_weblog_run_summary_sits_outside_the_group_dirs(tmp_path: Path) -> None:
    _touch(tmp_path / "run_summary.html")
    _touch(_group(tmp_path) / "summary.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"run_summary.html"' in body
    assert '"group_0/run_summary.html"' not in body
    assert "Run summary" in body


def test_build_weblog_without_run_summary(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "summary.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "RUN_SUMMARY = null" in body


# ---------------------------------------------------------------------------
# Per-group plot-type menu
# ---------------------------------------------------------------------------


def test_build_weblog_lists_present_aggregates_only(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    _touch(_group(tmp_path) / "tcal_ref_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "tau_vs_frequency.html" in body
    assert "tcal_ref_vs_frequency.html" in body
    # Missing aggregates must not appear in the dropdown.
    assert "tcal_fit_vs_frequency.html" not in body
    assert "c_vs_frequency.html" not in body


def test_build_weblog_kinds_are_per_group(tmp_path: Path) -> None:
    """A plot present in one group must not be offered for another."""
    _touch(_group(tmp_path, 0) / "atmospheric_profile.html")
    _touch(_group(tmp_path, 1) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"0": [["atmospheric_profile.html", "Atmospheric profile"]]' in body
    assert '"1": [["tau_vs_frequency.html", "Opacity vs frequency"]]' in body


def test_build_weblog_offers_elevation_when_tippingcurve_files_present(
    tmp_path: Path,
) -> None:
    _touch(_group(tmp_path) / "tippingcurve_spw_0_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"elevation"' in body
    assert "Elevation curve" in body


def test_build_weblog_no_elevation_without_tippingcurve_files(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "Elevation curve" not in body


def test_build_weblog_lists_heatmaps_and_tables_when_present(tmp_path: Path) -> None:
    for name in (
        "fit_quality_heatmap.html",
        "residual_rms_heatmap.html",
        "model_opacity_table.html",
        "measured_opacity_table.html",
    ):
        _touch(_group(tmp_path) / name)
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    for label in (
        "Fit quality heatmap",
        "Residual RMS heatmap",
        "Model opacity table",
        "Measured opacity table",
    ):
        assert label in body


def test_build_weblog_omits_opacity_tables_when_absent(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "model_opacity_table.html" not in body
    assert "measured_opacity_table.html" not in body


# ---------------------------------------------------------------------------
# Elevation selectors, scoped to the group
# ---------------------------------------------------------------------------


def test_build_weblog_scan_and_antenna_maps_are_per_group(tmp_path: Path) -> None:
    _touch(_group(tmp_path, 0) / "tippingcurve_spw_0_ea01_scan_4.html")
    _touch(_group(tmp_path, 1) / "tippingcurve_spw_3_ea02_scan_10.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"0": [4]' in body
    assert '"1": [10]' in body
    assert '"0": ["ea01"]' in body
    assert '"1": ["ea02"]' in body


def test_build_weblog_selectors_start_empty(tmp_path: Path) -> None:
    """Scan/antenna/spw options are filled by JS from the group maps."""
    _touch(_group(tmp_path) / "tippingcurve_spw_0_ea01_scan_4.html")
    _touch(_group(tmp_path) / "tippingcurve_spw_3_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    for name in ("scan", "antenna", "spw"):
        assert f'<select id="{name}"><option value="">—</option></select>' in body


def test_build_weblog_embeds_per_group_scan_to_spws_map(tmp_path: Path) -> None:
    for name in (
        "tippingcurve_spw_0_ea01_scan_4.html",
        "tippingcurve_spw_3_ea01_scan_4.html",
        "tippingcurve_spw_13_ea01_scan_10.html",
    ):
        _touch(_group(tmp_path, 2) / name)
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "SPWS = " in body
    # Scan 4 saw spws 0 and 3; scan 10 saw spw 13 only — both under group 2.
    assert '"2": {"4": [0, 3], "10": [13]}' in body


# ---------------------------------------------------------------------------
# Availability set and degenerate inputs
# ---------------------------------------------------------------------------


def test_build_weblog_embeds_available_set(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    _touch(_group(tmp_path) / "tippingcurve_spw_0_ea01_scan_4.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The JS-side existence check uses an embedded set.
    assert "AVAILABLE = new Set(" in body
    assert '"group_0/tau_vs_frequency.html"' in body
    assert '"group_0/tippingcurve_spw_0_ea01_scan_4.html"' in body


def test_build_weblog_announces_missing_plot_string(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The user-visible string the GUI shows when a requested file is absent.
    assert "Plot not found:" in body


def test_build_weblog_ignores_existing_index_html(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    _touch(tmp_path / "index.html")  # stale index from a prior run
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    # The set must not list index.html itself.
    assert '"index.html"' not in body


def test_build_weblog_ignores_non_group_subdirectories(tmp_path: Path) -> None:
    _touch(_group(tmp_path) / "tau_vs_frequency.html")
    _touch(tmp_path / "scratch" / "tau_vs_frequency.html")
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert '"scratch/tau_vs_frequency.html"' not in body


def test_build_weblog_empty_directory(tmp_path: Path) -> None:
    body = build_weblog(tmp_path).read_text(encoding="utf-8")
    assert "(no group_* directories found)" in body
