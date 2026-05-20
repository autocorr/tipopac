"""Capture legacy tipopac_v2.6 output for v1 acceptance comparison.

Runs the v2.6 task on data/tip_test.ms once per fit mode and writes a
structured JSON reference under tests/integration/reference/v26/<mode>/
alongside the raw caltables, casalog, and plot PNGs. Designed to be invoked
inside a CASA shell (whose Python env lacks xarray):

    casa --nologger --nogui --log2term -c tests/integration/run_legacy.py [opts]

The reference.json structure is {attrs, coords, data_vars} where each data_var
is {dims, dtype, data}; the rewrite's integration test reads it in the uv venv
and reconstructs an xarray.Dataset (or just uses the raw arrays directly).

See DESIGN.md sec 11.2 / 11.3 / 12 for the role this output plays in the
rewrite's acceptance comparison.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from casatasks import casalog
from casatools import table as _table
import casatools

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PKG = REPO_ROOT / "tipopac_v2.6" / "lastversion"
sys.path.insert(0, str(LEGACY_PKG))
from tipping import tipopac as legacy_tipopac  # noqa: E402

DEFAULT_MS = REPO_ROOT / "data" / "tip_test.ms"
DEFAULT_OUT = REPO_ROOT / "tests" / "integration" / "reference" / "v26"

MODES: dict[str, dict[str, bool]] = {
    "tau_per_antenna": {"tauPerAnt": True, "calcTcals": False},
    "global_tau": {"tauPerAnt": False, "calcTcals": False},
    "tcal_solve": {"tauPerAnt": False, "calcTcals": True},
}

PER_ANT_FIT = re.compile(
    r"\s+scan (\d+), (\w+), spw (\d+) - tau0: ([-\d.]+), "
    r"Tae \(K\): ([-\d.]+) \(R\), ([-\d.]+) \(L\)"
)
GLOBAL_FIT = re.compile(
    r"Fit attempt: (\d+)\. Scan (\d+) opacity for spw (\d+) at "
    r"[\d.]+ GHz: ([-\d.]+) pm ([\d.]+)"
)
NEG_TAU_RESCUE = re.compile(
    r"(?:After fit attempt: \d+ )?Scan (\d+) opacity for spw (\d+) at "
    r"[\d.]+ GHz computed per antenna without fitting Tcal"
)
ANTENNAS_USED = re.compile(
    r"Scan (\d+): Antennas used for spw (\d+) to get tau:\[(.*?)\]"
)
TOO_FEW_ANT = re.compile(
    r"Not enought unflagged data to fit antenna (\d+) at scan (\d+)"
)
TOO_FEW_SPW = re.compile(r"Not enought unflagged data to fit spw (\d+) at scan (\d+)")
TIPOPAC_BANNER = re.compile(r"--> tipopac version (\S+)")


def read_antenna_names(ms_dir: Path) -> list[str]:
    tb = _table()
    tb.open(str(ms_dir / "ANTENNA"))
    names = list(tb.getcol("NAME"))
    tb.close()
    return [str(n) for n in names]


def read_caltable_z(z_path: Path) -> dict[str, np.ndarray]:
    tb = _table()
    tb.open(str(z_path))
    out = {
        "fparam": tb.getcol("FPARAM")[0, 0, :],
        "paramerr": tb.getcol("PARAMERR")[0, 0, :],
        "snr": tb.getcol("SNR")[0, 0, :],
        "flag": tb.getcol("FLAG")[0, 0, :],
        "scan": tb.getcol("SCAN_NUMBER"),
        "ant": tb.getcol("ANTENNA1"),
        "spw": tb.getcol("SPECTRAL_WINDOW_ID"),
        "time": tb.getcol("TIME"),
    }
    tb.close()
    return out


def read_caltable_t(t_path: Path) -> dict[str, np.ndarray]:
    tb = _table()
    tb.open(str(t_path))
    out = {
        "noise_cal": tb.getcol("NOISE_CAL"),  # (n_load=2, n_pol=2, n_rows)
        "ant": tb.getcol("ANTENNA_ID"),
        "spw": tb.getcol("SPECTRAL_WINDOW_ID"),
        "time": tb.getcol("TIME"),
    }
    tb.close()
    return out


def read_caldevice_ref(ms_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tb = _table()
    tb.open(str(ms_dir / "CALDEVICE"))
    nc = tb.getcol("NOISE_CAL")  # (n_load, n_pol, n_rows)
    ant = tb.getcol("ANTENNA_ID")
    spw = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    return (
        nc[0, :, :],
        ant,
        spw,
    )  # row 0 is the noise tube; row 1 is the unused solar slot


def parse_casalog(text: str) -> dict[str, object]:
    per_ant: list[tuple[int, str, int, float, float, float]] = []
    global_fit: list[tuple[int, int, int, float, float]] = []
    neg_rescue: set[tuple[int, int]] = set()
    ants_used: dict[tuple[int, int], list[str]] = {}
    too_few_ant: list[tuple[int, int]] = []
    too_few_spw: list[tuple[int, int]] = []
    version = "unknown"

    for line in text.splitlines():
        if (m := PER_ANT_FIT.search(line)) is not None:
            s, an, w, t, t0r, t0l = m.groups()
            per_ant.append((int(s), an, int(w), float(t), float(t0r), float(t0l)))
            continue
        if (m := GLOBAL_FIT.search(line)) is not None:
            v, s, w, t, te = m.groups()
            global_fit.append((int(v), int(s), int(w), float(t), float(te)))
            continue
        if (m := NEG_TAU_RESCUE.search(line)) is not None:
            s, w = m.groups()
            neg_rescue.add((int(s), int(w)))
            continue
        if (m := ANTENNAS_USED.search(line)) is not None:
            s, w, names_str = m.groups()
            names = (
                [n.strip().strip("'\"") for n in names_str.split(",")]
                if names_str.strip()
                else []
            )
            ants_used[(int(s), int(w))] = names
            continue
        if (m := TOO_FEW_ANT.search(line)) is not None:
            a, s = m.groups()
            too_few_ant.append((int(a), int(s)))
            continue
        if (m := TOO_FEW_SPW.search(line)) is not None:
            w, s = m.groups()
            too_few_spw.append((int(w), int(s)))
            continue
        if (m := TIPOPAC_BANNER.search(line)) is not None:
            version = m.group(1)

    return {
        "per_ant": per_ant,
        "global_fit": global_fit,
        "neg_rescue": neg_rescue,
        "ants_used": ants_used,
        "too_few_ant": too_few_ant,
        "too_few_spw": too_few_spw,
        "tipopac_version": version,
    }


def build_dataset(
    z_path: Path,
    t_path: Path | None,
    casalog_path: Path,
    mode: str,
    ms_dir: Path,
) -> dict:
    antenna_names = read_antenna_names(ms_dir)
    name_to_idx = {n: i for i, n in enumerate(antenna_names)}
    cz = read_caltable_z(z_path)
    parsed = parse_casalog(casalog_path.read_text())

    scans = np.unique(cz["scan"]).astype(np.int32)
    spws = np.unique(cz["spw"]).astype(np.int32)
    n_s, n_a, n_w = len(scans), len(antenna_names), len(spws)
    s_idx = {int(s): i for i, s in enumerate(scans)}
    w_idx = {int(w): i for i, w in enumerate(spws)}

    tau_caltable = np.full((n_s, n_a, n_w), np.nan, dtype=np.float32)
    flag_caltable = np.ones((n_s, n_a, n_w), dtype=bool)
    paramerr_caltable = np.full((n_s, n_a, n_w), np.nan, dtype=np.float32)
    snr_caltable = np.full((n_s, n_a, n_w), np.nan, dtype=np.float32)
    for k in range(len(cz["scan"])):
        i = s_idx[int(cz["scan"][k])]
        a = int(cz["ant"][k])
        w = w_idx[int(cz["spw"][k])]
        if not bool(cz["flag"][k]):
            tau_caltable[i, a, w] = cz["fparam"][k]
            flag_caltable[i, a, w] = False
            paramerr_caltable[i, a, w] = cz["paramerr"][k]
            snr_caltable[i, a, w] = cz["snr"][k]

    tau_log = np.full((n_s, n_a, n_w), np.nan, dtype=np.float32)
    tau_err_log = np.full((n_s, n_a, n_w), np.nan, dtype=np.float32)
    # T0 is only logged per-(scan, ant, spw, pol) in tau_per_antenna mode.
    # In global modes the matching casalog.post line at task_tipopac.py:1619 is
    # commented out, so T0 lives only in v2.6's in-memory dataTae array and is
    # unrecoverable without patching the legacy code (out of scope per plan).
    T0 = np.full((n_s, n_a, n_w, 2), np.nan, dtype=np.float32)

    if mode == "tau_per_antenna":
        for s, an, w, t, t0r, t0l in parsed["per_ant"]:
            if s in s_idx and w in w_idx and an in name_to_idx:
                i, a, k = s_idx[s], name_to_idx[an], w_idx[w]
                tau_log[i, a, k] = t
                T0[i, a, k, 0] = t0r
                T0[i, a, k, 1] = t0l
    else:
        for _v, s, w, t, te in parsed["global_fit"]:
            if s in s_idx and w in w_idx:
                tau_log[s_idx[s], :, w_idx[w]] = t
                tau_err_log[s_idx[s], :, w_idx[w]] = te

    version_fit = np.zeros((n_s, n_w), dtype=np.int8)
    for v, s, w, _t, _te in parsed["global_fit"]:
        if s in s_idx and w in w_idx:
            version_fit[s_idx[s], w_idx[w]] = v

    neg_tau = np.zeros((n_s, n_w), dtype=bool)
    for s, w in parsed["neg_rescue"]:
        if s in s_idx and w in w_idx:
            neg_tau[s_idx[s], w_idx[w]] = True

    ants_used_count = np.zeros((n_s, n_w), dtype=np.int16)
    for (s, w), names in parsed["ants_used"].items():
        if s in s_idx and w in w_idx:
            ants_used_count[s_idx[s], w_idx[w]] = len(names)

    # `besta` rescue heuristic: v2.6 silently substitutes the lowest-σ antenna
    # at task_tipopac.py:1480-1492 when AntArr is empty — no log marker. We
    # flag (scan, spw) cells where antennas_used_count == 1 in a multi-antenna
    # array; this may have false positives but errs on the conservative side
    # (excludes more cells from the v1 acceptance comparison than strictly
    # necessary, which is the safe direction).
    besta_rescue = (ants_used_count == 1) & (n_a > 1)

    # The "Not enought unflagged data to fit antenna A at scan I" line carries
    # no spw context (task_tipopac.py:1314); we mark all spws for that
    # (scan, antenna) pair.
    too_few_per_ant = np.zeros((n_s, n_a, n_w), dtype=bool)
    for a, s in parsed["too_few_ant"]:
        if s in s_idx and 0 <= a < n_a:
            too_few_per_ant[s_idx[s], a, :] = True
    too_few_per_spw = np.zeros((n_s, n_w), dtype=bool)
    for w, s in parsed["too_few_spw"]:
        if s in s_idx and w in w_idx:
            too_few_per_spw[s_idx[s], w_idx[w]] = True

    data_vars: dict[str, tuple] = {
        "tau_caltable": (("scan", "antenna", "spw"), tau_caltable),
        "tau_log": (("scan", "antenna", "spw"), tau_log),
        "tau_err_log": (("scan", "antenna", "spw"), tau_err_log),
        "paramerr_caltable": (("scan", "antenna", "spw"), paramerr_caltable),
        "snr_caltable": (("scan", "antenna", "spw"), snr_caltable),
        "T0": (("scan", "antenna", "spw", "polarization"), T0),
        "version_fit": (("scan", "spw"), version_fit),
        "negative_tau_rescue": (("scan", "spw"), neg_tau),
        "besta_rescue": (("scan", "spw"), besta_rescue),
        "antennas_used_count": (("scan", "spw"), ants_used_count),
        "too_few_samples_per_antenna": (("scan", "antenna", "spw"), too_few_per_ant),
        "too_few_samples_per_spw": (("scan", "spw"), too_few_per_spw),
        "caltable_flag": (("scan", "antenna", "spw"), flag_caltable),
    }

    if t_path is not None and t_path.exists():
        ct = read_caltable_t(t_path)
        nrows = ct["noise_cal"].shape[2]
        expected = n_s * n_a * n_w
        tcal_fit = np.full((n_s, n_a, n_w, 2), np.nan, dtype=np.float32)
        if nrows == expected:
            # caltableT lacks a SCAN_NUMBER column; v2.6 writes rows in the
            # order (scan, antenna, spw) at task_tipopac.py:1047-1059, so we
            # recover the scan index by row-index arithmetic.
            for k in range(nrows):
                i = k // (n_a * n_w)
                a = (k // n_w) % n_a
                w = k % n_w
                tcal_fit[i, a, w, 0] = ct["noise_cal"][0, 0, k]
                tcal_fit[i, a, w, 1] = ct["noise_cal"][0, 1, k]
        else:
            casalog.post(
                f"WARNING: caltableT has {nrows} rows, expected {expected}; "
                "tcal_fit indexing may be wrong",
                "WARN",
            )

        ref_nc, ref_ant, ref_spw = read_caldevice_ref(ms_dir)
        tcal_ref = np.full((n_a, n_w, 2), np.nan, dtype=np.float32)
        for r in range(len(ref_ant)):
            a = int(ref_ant[r])
            w = w_idx.get(int(ref_spw[r]))
            if w is not None and 0 <= a < n_a:
                tcal_ref[a, w, 0] = ref_nc[0, r]
                tcal_ref[a, w, 1] = ref_nc[1, r]

        with np.errstate(invalid="ignore", divide="ignore"):
            tcal_pct_change = (tcal_fit / tcal_ref[None, :, :, :] - 1.0) * 100.0

        data_vars["tcal_fit"] = (("scan", "antenna", "spw", "polarization"), tcal_fit)
        data_vars["tcal_ref"] = (("antenna", "spw", "polarization"), tcal_ref)
        data_vars["tcal_pct_change"] = (
            ("scan", "antenna", "spw", "polarization"),
            tcal_pct_change.astype(np.float32),
        )

    excluded: set[tuple[int, int]] = set()
    for i in range(n_s):
        for w in range(n_w):
            if (
                version_fit[i, w] > 1
                or neg_tau[i, w]
                or besta_rescue[i, w]
                or too_few_per_spw[i, w]
                or flag_caltable[i, :, w].all()
            ):
                excluded.add((int(scans[i]), int(spws[w])))

    casa_version = getattr(casatools, "__version__", "unknown")
    attrs: dict[str, str] = {
        "mode": mode,
        "msname": str(ms_dir),
        "tipopac_version": parsed["tipopac_version"],
        "casa_version": str(casa_version),
        "task_args": json.dumps(MODES[mode]),
        "capture_time": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "casalog_path": casalog_path.name,
        "caltable_z_path": z_path.name,
        "caltable_t_path": t_path.name if t_path else "",
        "acceptance_excluded_cells": ";".join(f"{s},{w}" for s, w in sorted(excluded)),
    }

    return {
        "attrs": attrs,
        "coords": {
            "scan": scans.tolist(),
            "antenna": list(antenna_names),
            "spw": spws.tolist(),
            "polarization": ["R", "L"],
        },
        "data_vars": {
            name: {
                "dims": list(dims),
                "dtype": str(arr.dtype),
                "data": _array_to_json(arr),
            }
            for name, (dims, arr) in data_vars.items()
        },
    }


def _array_to_json(arr: np.ndarray) -> list:
    if arr.dtype.kind == "b":
        return arr.tolist()
    if arr.dtype.kind in "iu":
        return arr.astype(int).tolist()
    # float — replace NaN with None so json.dump(allow_nan=False) accepts it.
    out = arr.astype(object)
    out[np.isnan(arr)] = None
    return out.tolist()


def save_reference(data: dict, path: Path) -> None:
    with path.open("w") as f:
        json.dump(data, f, allow_nan=False)


def run_mode(ms: Path, mode: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    casalog_path = out_dir / "casalog.txt"
    if casalog_path.exists():
        casalog_path.unlink()
    casalog.setlogfile(str(casalog_path))

    z_path = out_dir / "caltable.zopac"
    t_path = out_dir / "caltable.tcal"
    for p in (z_path, t_path):
        if p.exists():
            shutil.rmtree(p)

    use_tcal = MODES[mode]["calcTcals"]
    legacy_tipopac(
        msname=str(ms),
        caltableZ=str(z_path),
        caltableT=str(t_path) if use_tcal else "",
        cmdFlag=True,
        usrFlag=False,
        flagFile="",
        caltable=True,
        doPlot=True,
        doModel=False,
        **MODES[mode],
    )

    # v2.6 writes plots into <msname>.tipping.plots/ next to the MS; relocate
    # so the three modes do not collide and the fixture stays self-contained.
    src_plots = ms.parent / (ms.name + ".tipping.plots")
    if src_plots.exists():
        dst_plots = out_dir / "plots"
        if dst_plots.exists():
            shutil.rmtree(dst_plots)
        shutil.move(str(src_plots), str(dst_plots))

    data = build_dataset(z_path, t_path if use_tcal else None, casalog_path, mode, ms)
    save_reference(data, out_dir / "reference.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy tipopac_v2.6 and capture all relevant quantities."
    )
    parser.add_argument("--mode", choices=["all", *MODES.keys()], default="all")
    parser.add_argument("--ms", type=Path, default=DEFAULT_MS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    # casa launcher prepends its own argv; if invoked via `casa -c script.py
    # --foo bar`, drop everything up to and including the script path.
    if "-c" in sys.argv:
        i = sys.argv.index("-c") + 2
        args = parser.parse_args(sys.argv[i:])
    else:
        args = parser.parse_args()

    modes_to_run = list(MODES) if args.mode == "all" else [args.mode]
    for m in modes_to_run:
        print(f"=== Running v2.6 tipopac mode={m} ===", flush=True)
        run_mode(args.ms.resolve(), m, (args.out / m).resolve())
    print("=== Done ===", flush=True)


if __name__ == "__main__":
    main()
