"""Self-contained GUI weblog for the tipopac plot directory.

``build_weblog(plot_dir)`` scans ``plot_dir`` for files matching the
hard-coded plot-naming patterns produced by
:meth:`tipopac.plot.PlotData.save_all` and emits an ``index.html`` with
inline CSS + JS that lets the reader pick a plot type from a dropdown
(and, for elevation curves, type ``scan`` / ``antenna`` / ``spw`` into
text boxes). If the user requests a plot whose file isn't present, the
GUI says so instead of loading a broken iframe.

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
_AGGREGATE_PLOTS: tuple[tuple[str, str], ...] = (
    ("tau_vs_frequency.html", "τ vs frequency"),
    ("tcal_fit_vs_frequency.html", "T_cal (fit) vs frequency"),
    ("tcal_ref_vs_frequency.html", "T_cal (ref) vs frequency"),
    ("c_vs_frequency.html", "c = T_cal,fit / T_cal,ref"),
    ("atmospheric_profile.html", "Atmospheric profile"),
)
_ELEVATION_LABEL = "Elevation curve"


def build_weblog(plot_dir: str | Path) -> Path:
    """Write a self-contained ``index.html`` GUI into ``plot_dir``."""
    plot_dir = Path(plot_dir)
    files = sorted(p.name for p in plot_dir.glob("*.html") if p.name != "index.html")
    files_set = set(files)

    aggregates = [(fn, label) for fn, label in _AGGREGATE_PLOTS if fn in files_set]
    triples = [
        (int(m.group(1)), m.group(2), int(m.group(3)))
        for name in files
        if (m := _ELEVATION_RE.match(name))
    ]
    antennas = sorted({t[1] for t in triples})
    scans = sorted({t[2] for t in triples})
    # Per-scan spws — spws observed vary by scan (one band per scan).
    scan_to_spws: dict[int, list[int]] = {s: [] for s in scans}
    for spw, _ant, scan in triples:
        if spw not in scan_to_spws[scan]:
            scan_to_spws[scan].append(spw)
    for s in scan_to_spws:
        scan_to_spws[s].sort()

    index_path = plot_dir / "index.html"
    index_path.write_text(
        _render_html(
            aggregates=aggregates,
            has_elevation=bool(triples),
            scans=scans,
            antennas=antennas,
            scan_to_spws=scan_to_spws,
            available=files,
        ),
        encoding="utf-8",
    )
    _log.info("weblog written: %s", index_path)
    return index_path


def _render_html(
    *,
    aggregates: list[tuple[str, str]],
    has_elevation: bool,
    scans: list[int],
    antennas: list[str],
    scan_to_spws: dict[int, list[int]],
    available: list[str],
) -> str:
    options: list[str] = []
    if has_elevation:
        options.append(f'<option value="elevation">{_ELEVATION_LABEL}</option>')
    for fn, label in aggregates:
        options.append(f'<option value="{fn}" data-file="{fn}">{label}</option>')
    if not options:
        options.append('<option value="">(no plots found in this directory)</option>')

    def _select(select_id: str, values: list[str]) -> str:
        opts = '<option value="">—</option>' + "".join(
            f'<option value="{v}">{v}</option>' for v in values
        )
        return f'<select id="{select_id}">{opts}</select>'

    scan_select = _select("scan", [str(s) for s in scans])
    antenna_select = _select("antenna", antennas)
    # spw select starts empty; JS rebuilds it from SCAN_TO_SPWS when scan changes.
    spw_select = '<select id="spw"><option value="">—</option></select>'

    available_json = json.dumps(available)
    scan_to_spws_json = json.dumps({str(k): v for k, v in scan_to_spws.items()})
    elev_hidden = "" if has_elevation else " hidden"

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
  <label>Plot type:
    <select id="kind">{"".join(options)}</select>
  </label>
  <div id="elev"{elev_hidden}>
    <label>Scan: {scan_select}</label>
    <label>Antenna: {antenna_select}</label>
    <label>spw: {spw_select}</label>
  </div>
</div>
<div id="status"></div>
<iframe id="frame" src="about:blank"></iframe>
<script>
  const AVAILABLE = new Set({available_json});
  const SCAN_TO_SPWS = {scan_to_spws_json};
  const kind = document.getElementById("kind");
  const elev = document.getElementById("elev");
  const scan = document.getElementById("scan");
  const antenna = document.getElementById("antenna");
  const spw = document.getElementById("spw");
  const status = document.getElementById("status");
  const frame = document.getElementById("frame");

  function refreshSpws() {{
    const valid = SCAN_TO_SPWS[scan.value] || [];
    const previous = spw.value;
    const opts = ['<option value="">—</option>'];
    for (const s of valid) opts.push(`<option value="${{s}}">${{s}}</option>`);
    spw.innerHTML = opts.join("");
    spw.value = valid.includes(Number(previous)) ? previous : "";
  }}

  function pathFor() {{
    const opt = kind.selectedOptions[0];
    if (!opt || !opt.value) return null;
    if (opt.value === "elevation") {{
      if (!scan.value || !antenna.value || !spw.value) return null;
      return `tippingcurve_spw_${{spw.value}}_${{antenna.value}}_scan_${{scan.value}}.html`;
    }}
    return opt.dataset.file;
  }}

  function update() {{
    const isElev = kind.value === "elevation";
    elev.hidden = !isElev;
    const path = pathFor();
    if (path === null) {{
      frame.src = "about:blank";
      status.textContent = isElev
        ? "Pick scan, antenna, and spw above."
        : "";
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

  scan.addEventListener("change", refreshSpws);
  document.addEventListener("change", update);
  update();
</script>
</body>
</html>
"""
