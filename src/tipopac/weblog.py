"""Self-contained GUI weblog for the tipopac plot directory.

``build_weblog(plot_dir)`` scans ``plot_dir`` for the ``group_{k}/``
subdirectories :meth:`tipopac.plot.PlotData.save_all` writes, matches the
hard-coded plot-naming patterns inside each, and emits an ``index.html``
with inline CSS + JS that lets the reader pick a time group, then a plot
type (and, for elevation curves, a scan / antenna / spw). The scan,
antenna, and spw menus are scoped to the selected group. If the user
requests a plot whose file isn't present, the GUI says so instead of
loading a broken iframe.

The page is independent of the xarray dataset — only filenames drive
the available options. Run as a pipeline step *after* the plots have
been written.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

__all__ = ["build_weblog"]

_log = logging.getLogger(__name__)

# Hard-coded naming patterns (mirror plot.PlotData.save_all).
_ELEVATION_RE = re.compile(r"^tippingcurve_spw_(\d+)_(\w+)_scan_(\d+)\.html$")
_GROUP_RE = re.compile(r"^group_(\d+)$")
# Run-global page, outside the group dirs — the weblog's landing view.
_RUN_SUMMARY: tuple[str, str] = ("run_summary.html", "Run summary")
_SUMMARY_PLOT: tuple[str, str] = ("summary.html", "Summary")
_AGGREGATE_PLOTS: tuple[tuple[str, str], ...] = (
    ("tau_vs_frequency.html", "Opacity vs frequency"),
    ("tcal_fit_vs_frequency.html", "T_cal (fit) vs frequency"),
    ("tcal_ref_vs_frequency.html", "T_cal (ref) vs frequency"),
    ("c_vs_frequency.html", "c = T_cal,fit / T_cal,ref"),
    ("atmospheric_profile.html", "Atmospheric profile"),
    ("fit_quality_heatmap.html", "Fit quality heatmap"),
    ("residual_rms_heatmap.html", "Residual RMS heatmap"),
    ("model_opacity_table.html", "Model opacity table"),
    ("measured_opacity_table.html", "Measured opacity table"),
)
_ELEVATION_LABEL = "Elevation curve"


def build_weblog(plot_dir: str | Path) -> Path:
    """Write a self-contained ``index.html`` GUI into ``plot_dir``."""
    plot_dir = Path(plot_dir)
    groups = sorted(
        int(m.group(1))
        for p in plot_dir.iterdir()
        if p.is_dir() and (m := _GROUP_RE.match(p.name))
    )

    available: list[str] = []
    if (plot_dir / _RUN_SUMMARY[0]).is_file():
        available.append(_RUN_SUMMARY[0])

    # Per-group option and selector data, keyed by group index as a string
    # so it survives the JSON round-trip into the page.
    kinds: dict[str, list[list[str]]] = {}
    scans_by_group: dict[str, list[int]] = {}
    antennas_by_group: dict[str, list[str]] = {}
    spws_by_group: dict[str, dict[str, list[int]]] = {}

    for k in groups:
        gdir = plot_dir / f"group_{k}"
        names = sorted(p.name for p in gdir.glob("*.html"))
        available.extend(f"group_{k}/{name}" for name in names)
        present = set(names)

        options: list[list[str]] = []
        if _SUMMARY_PLOT[0] in present:
            options.append([_SUMMARY_PLOT[0], _SUMMARY_PLOT[1]])
        triples = [
            (int(m.group(1)), m.group(2), int(m.group(3)))
            for name in names
            if (m := _ELEVATION_RE.match(name))
        ]
        if triples:
            options.append(["elevation", _ELEVATION_LABEL])
        options.extend([fn, label] for fn, label in _AGGREGATE_PLOTS if fn in present)
        kinds[str(k)] = options

        scan_to_spws: dict[str, list[int]] = {}
        for spw, _ant, scan in triples:
            spws = scan_to_spws.setdefault(str(scan), [])
            if spw not in spws:
                spws.append(spw)
        for spws in scan_to_spws.values():
            spws.sort()

        scans_by_group[str(k)] = sorted({t[2] for t in triples})
        antennas_by_group[str(k)] = sorted({t[1] for t in triples})
        spws_by_group[str(k)] = scan_to_spws

    index_path = plot_dir / "index.html"
    index_path.write_text(
        _render_html(
            groups=groups,
            has_run_summary=_RUN_SUMMARY[0] in available,
            kinds=kinds,
            scans_by_group=scans_by_group,
            antennas_by_group=antennas_by_group,
            spws_by_group=spws_by_group,
            available=available,
        ),
        encoding="utf-8",
    )
    _log.info("weblog written: %s (%d group(s))", index_path, len(groups))
    return index_path


def _render_html(
    *,
    groups: list[int],
    has_run_summary: bool,
    kinds: dict[str, list[list[str]]],
    scans_by_group: dict[str, list[int]],
    antennas_by_group: dict[str, list[str]],
    spws_by_group: dict[str, dict[str, list[int]]],
    available: list[str],
) -> str:
    group_options = "".join(f'<option value="{k}">Group {k}</option>' for k in groups)
    if not group_options:
        group_options = '<option value="">(no group_* directories found)</option>'

    run_summary_option = (
        f'<option value="{_RUN_SUMMARY[0]}">{_RUN_SUMMARY[1]}</option>'
        if has_run_summary
        else ""
    )

    def _select(select_id: str) -> str:
        return f'<select id="{select_id}"><option value="">—</option></select>'

    available_json = json.dumps(available)
    kinds_json = json.dumps(kinds)
    scans_json = json.dumps(scans_by_group)
    antennas_json = json.dumps(antennas_by_group)
    spws_json = json.dumps(spws_by_group)
    run_summary_json = json.dumps(_RUN_SUMMARY[0] if has_run_summary else None)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tipopac plots</title>
<style>
  html, body {{ height: 100%; }}
  body {{
    margin: 0; padding: 1em; box-sizing: border-box;
    font-family: -apple-system, system-ui, sans-serif;
    display: flex; flex-direction: column;
  }}
  h1 {{
    margin: 0 0 0.5em; font-size: 1.3em;
    border-bottom: 1px solid #ccc; padding-bottom: 0.2em;
  }}
  .controls {{
    display: flex; flex-wrap: wrap; gap: 1.2em; align-items: center;
    margin-bottom: 0.4em;
  }}
  .controls label {{ display: flex; align-items: center; gap: 0.35em; }}
  .controls select {{ padding: 0.15em 0.3em; }}
  #elev {{ display: flex; gap: 1.2em; align-items: center; }}
  #status {{ color: #b00; min-height: 1.2em; margin-bottom: 0.4em; }}
  #frame {{ flex: 1; border: 1px solid #ccc; width: 100%; background: #fff; }}
  [hidden] {{ display: none !important; }}
</style>
</head>
<body>
<div class="controls">
  <label>Group:
    <select id="group">{group_options}</select>
  </label>
  <label>Plot type:
    <select id="kind">{run_summary_option}</select>
  </label>
  <div id="elev" hidden>
    <label>Scan: {_select("scan")}</label>
    <label>Antenna: {_select("antenna")}</label>
    <label>spw: {_select("spw")}</label>
  </div>
</div>
<div id="status"></div>
<iframe id="frame" src="about:blank"></iframe>
<script>
  const AVAILABLE = new Set({available_json});
  const KINDS = {kinds_json};
  const SCANS = {scans_json};
  const ANTENNAS = {antennas_json};
  const SPWS = {spws_json};
  const RUN_SUMMARY = {run_summary_json};
  const group = document.getElementById("group");
  const kind = document.getElementById("kind");
  const elev = document.getElementById("elev");
  const scan = document.getElementById("scan");
  const antenna = document.getElementById("antenna");
  const spw = document.getElementById("spw");
  const status = document.getElementById("status");
  const frame = document.getElementById("frame");

  function fill(select, values, labeller) {{
    const previous = select.value;
    const opts = ['<option value="">—</option>'];
    for (const v of values) opts.push(`<option value="${{v}}">${{labeller(v)}}</option>`);
    select.innerHTML = opts.join("");
    select.value = values.map(String).includes(previous) ? previous : "";
  }}

  // The plot set, and the scan/antenna/spw menus, belong to one group.
  function refreshGroup() {{
    const entries = KINDS[group.value] || [];
    const previous = kind.value;
    const opts = [];
    if (RUN_SUMMARY) opts.push(`<option value="${{RUN_SUMMARY}}">Run summary</option>`);
    for (const [value, label] of entries) {{
      opts.push(`<option value="${{value}}">${{label}}</option>`);
    }}
    if (!opts.length) opts.push('<option value="">(no plots for this group)</option>');
    kind.innerHTML = opts.join("");
    if ([...kind.options].some((o) => o.value === previous)) kind.value = previous;
    fill(scan, SCANS[group.value] || [], (v) => v);
    fill(antenna, ANTENNAS[group.value] || [], (v) => v);
    refreshSpws();
  }}

  function refreshSpws() {{
    const perScan = SPWS[group.value] || {{}};
    fill(spw, perScan[scan.value] || [], (v) => v);
  }}

  function pathFor() {{
    if (!kind.value) return null;
    // Run-global pages sit outside the group directories.
    if (RUN_SUMMARY && kind.value === RUN_SUMMARY) return RUN_SUMMARY;
    if (!group.value) return null;
    if (kind.value === "elevation") {{
      if (!scan.value || !antenna.value || !spw.value) return null;
      return `group_${{group.value}}/tippingcurve_spw_${{spw.value}}_${{antenna.value}}_scan_${{scan.value}}.html`;
    }}
    return `group_${{group.value}}/${{kind.value}}`;
  }}

  function update() {{
    const isElev = kind.value === "elevation";
    elev.hidden = !isElev;
    const path = pathFor();
    if (path === null) {{
      frame.src = "about:blank";
      status.textContent = isElev ? "Pick scan, antenna, and spw above." : "";
      return;
    }}
    if (AVAILABLE.has(path)) {{
      if (frame.getAttribute("src") !== path) frame.src = path;
      status.textContent = "";
    }} else {{
      frame.src = "about:blank";
      status.textContent = "Plot not found: " + path;
    }}
  }}

  group.addEventListener("change", refreshGroup);
  scan.addEventListener("change", refreshSpws);
  document.addEventListener("change", update);
  refreshGroup();
  update();
</script>
</body>
</html>
"""
