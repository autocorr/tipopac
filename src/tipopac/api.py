"""Public API for tipopac — one-shot function and staged class (DESIGN.md §2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from tipopac.atmgrid import PwvGrid

_READERS: list = []  # populated on first call to avoid import-time casatools load

# Public Stage A+B modes (independent τ fit + per-antenna PWV anchor;
# `design/independent_tau_fit.md`). The values are the Stage-A backend
# mode in :func:`tipopac.fit.fit_dataset`. The legacy modes
# (``tau_per_antenna``, ``global_tau``, ``tcal_solve``) are still
# exposed directly and skip Stage B.
_INDEPENDENT_TO_BACKEND: dict[str, str] = {
    "independent_tau": "tau_per_antenna",
    "independent_tau_solve": "tcal_solve",
}

# VLA coordinates for open-meteo lookup. Repeated from atmosphere.py because the
# api layer needs them too; harmless duplication for two lines.
_VLA_LAT: float = 34.0784
_VLA_LON: float = -107.6177


def _get_readers() -> list:
    global _READERS
    if not _READERS:
        from tipopac.readers.ms import MSReader
        from tipopac.readers.sdm import SDMReader

        _READERS = [MSReader, SDMReader]
    return _READERS


def _detect_reader(path: Path):
    for R in _get_readers():
        if R.supports(path):
            return R
    raise ValueError(
        f"{path} is not a recognised MS or SDM path "
        f"(no reader's supports() returned True)"
    )


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        import casatools

        versions["casatools"] = str(getattr(casatools, "__version__", "unknown"))
    except Exception:
        versions["casatools"] = "unavailable"
    try:
        import sdmpy

        versions["sdmpy"] = str(getattr(sdmpy, "__version__", "unknown"))
    except Exception:
        versions["sdmpy"] = "unavailable"
    try:
        import amwrap

        versions["amwrap"] = str(getattr(amwrap, "__version__", "unknown"))
    except Exception:
        versions["amwrap"] = "unavailable"
    try:
        import subprocess

        r = subprocess.run(["am", "--version"], capture_output=True, text=True)
        line = (
            (r.stdout or r.stderr).splitlines()[0] if r.returncode == 0 else "unknown"
        )
        versions["am"] = line
    except Exception:
        versions["am"] = "unavailable"
    try:
        import importlib.metadata

        versions["tipopac"] = importlib.metadata.version("tipopac")
    except Exception:
        versions["tipopac"] = "unknown"
    return versions


@dataclass(frozen=True)
class Result:
    """Return value of `tipopac()` and `TippingAnalysis.result`."""

    dataset: xr.Dataset
    mode: str
    input_path: Path
    input_format: Literal["ms", "sdm"]
    software_versions: dict[str, str]


def tipopac(
    path: str | Path,
    *,
    mode: str = "independent_tau_solve",
    flags_online: bool = True,
    flags_file: str | Path | None = None,
    atm_model: bool = True,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "midlatitude_summer",
    n_workers: int | None = None,
    plot_dir: str | Path | None = None,
    caltable_opacity: str | Path | None = None,
    caltable_tcal: str | Path | None = None,
) -> Result:
    """Run the full tipping-curve pipeline and return a :class:`Result`.

    Parameters
    ----------
    path:
        Path to an MS or SDM (auto-detected).
    mode:
        Fit mode. Defaults to ``"independent_tau_solve"`` — per-(scan, spw)
        Stage-A Tcal-solve fit followed by a per-antenna PWV anchor (Stage
        B). Other Stage-A+B mode: ``"independent_tau"`` (per-(scan, ant,
        spw) opacity). Legacy single-stage modes also exposed:
        ``"tau_per_antenna"``, ``"global_tau"``, ``"tcal_solve"`` — these
        use the v2.6 Bevis ``T_mean`` heuristic and skip Stage B.
    flags_online:
        Apply FLAG_CMD online flags (MS only; SDM has no equivalent).
    flags_file:
        Path to a user flag file (one ``antenna/spw/timerange`` line per row).
    atm_model:
        For legacy modes only: run am + open-meteo atmospheric
        extrapolation. Ignored for the Stage-A+B modes — Stage B already
        anchors against the precomputed am grid.
    atm_profile_source:
        ``"open-meteo"`` (default) or ``"afgl"``. Used by the grid build
        (Stage A+B) and the legacy extrapolate.
    afgl_climatology:
        AFGL climatology name used as fallback or when ``atm_profile_source="afgl"``.
    n_workers:
        Stage-A fit parallelism. ``None`` runs serially. Higher values
        dispatch via a process pool with single-threaded BLAS per worker.
    plot_dir:
        If set, write per-(scan, antenna, spw) diagnostic PNGs here.
    caltable_opacity:
        If set, write a CASA TOpac caltable to this path.
    caltable_tcal:
        If set, write a CALDEVICE-style Tcal caltable to this path.
    """
    ta = TippingAnalysis.from_path(path)
    ta.apply_flags(
        online=flags_online, file=None if flags_file is None else Path(flags_file)
    )

    if mode in _INDEPENDENT_TO_BACKEND:
        ta.build_atm_grids(
            atm_profile_source=atm_profile_source,
            afgl_climatology=afgl_climatology,
        )
        ta.fit(mode=mode, n_workers=n_workers)
    else:
        ta.fit(mode=mode, n_workers=n_workers)
        if atm_model:
            ta.extrapolate(
                atm_profile_source=atm_profile_source,
                afgl_climatology=afgl_climatology,
            )

    if plot_dir is not None:
        ta.plot(out_dir=Path(plot_dir))
    if caltable_opacity is not None or caltable_tcal is not None:
        ta.write_caltables(
            opacity=None if caltable_opacity is None else Path(caltable_opacity),
            tcal=None if caltable_tcal is None else Path(caltable_tcal),
        )
    return ta.result


class TippingAnalysis:
    """Staged pipeline for notebook / interactive use.

    Each stage mutates ``self._ds`` in place; ``result`` is available once
    ``fit()`` has been called.
    """

    def __init__(self, ds: xr.Dataset, path: Path) -> None:
        self._ds = ds
        self._path = path
        self._mode: str | None = None
        self._versions = _software_versions()
        self._grids: dict[int, PwvGrid] = {}

    @classmethod
    def from_path(cls, path: str | Path) -> "TippingAnalysis":
        p = Path(path)
        R = _detect_reader(p)
        ds = R.from_path(p).read()
        return cls(ds, p)

    def apply_flags(
        self,
        *,
        online: bool = True,
        file: Path | None = None,
    ) -> None:
        from tipopac import flags

        self._ds = flags.apply(self._ds, online=online, file=file)

    def build_atm_grids(
        self,
        *,
        atm_profile_source: str = "open-meteo",
        afgl_climatology: str = "midlatitude_summer",
        pwv_step_mm: float = 0.5,
        freq_step_Hz: float = 100e6,
        n_workers: int | None = None,
    ) -> None:
        """Build per-scan :class:`PwvGrid` objects.

        Populates ``self._grids[scan_id] = PwvGrid`` for every scan and writes
        ``ds.attrs["pwv_profile_source"][scan_id]`` for provenance. Used by
        the post-fit atmospheric anchor (see ``design/independent_tau_fit.md``);
        not consumed by :meth:`fit`.
        """
        import astropy.units as u

        from tipopac.atmgrid import build_pwv_grid
        from tipopac.atmosphere import fetch_profile

        freqs = self._ds.coords["frequency"].values
        freq_min_Hz = float(freqs.min()) * 0.95
        freq_max_Hz = float(freqs.max()) * 1.05

        scan_ids = self._ds.coords["scan"].values
        scan_times = self._ds.coords["scan_time_start"].values
        # Per-scan surface pressure (Pa) from the WEATHER table. Median of
        # finite samples; missing scans get NaN → no clipping (full column).
        weather_P_Pa = self._ds["weather_P"].values  # (scan, time), Pa
        sources: dict[int, str] = {}
        surface_pressures_hPa: dict[int, float] = {}
        for i, scan_id in enumerate(scan_ids):
            obs_time_mjd_s = float(scan_times[i])
            p_samples = weather_P_Pa[i][np.isfinite(weather_P_Pa[i])]
            if p_samples.size:
                p_surf_hPa = float(np.median(p_samples)) / 100.0
                surface_pressure: u.Quantity | None = p_surf_hPa * u.hPa
                surface_pressures_hPa[int(scan_id)] = p_surf_hPa
            else:
                surface_pressure = None
            pressure, temperature, h2o_vmr, source_label, _meta = fetch_profile(
                _VLA_LAT,
                _VLA_LON,
                obs_time_mjd_s,
                source=atm_profile_source,
                afgl_climatology=afgl_climatology,
                surface_pressure=surface_pressure,
            )
            grid = build_pwv_grid(
                pressure,
                temperature,
                h2o_vmr,
                freq_min_Hz=freq_min_Hz,
                freq_max_Hz=freq_max_Hz,
                profile_source=source_label,
                pwv_step_mm=pwv_step_mm,
                freq_step_Hz=freq_step_Hz,
                n_workers=n_workers,
            )
            self._grids[int(scan_id)] = grid
            sources[int(scan_id)] = source_label

        self._ds.attrs["pwv_profile_source"] = sources
        if surface_pressures_hPa:
            self._ds.attrs["surface_pressure_hPa"] = surface_pressures_hPa

    def fit(
        self,
        mode: str = "independent_tau_solve",
        *,
        n_workers: int | None = None,
    ) -> None:
        from tipopac import fit

        if mode in _INDEPENDENT_TO_BACKEND:
            # Stage A + Stage B. Build grids if not done already; the
            # grid drives both the Stage A T_mean input and the Stage B
            # PWV anchor against τ_z(ν).
            from tipopac.anchor import anchor_pwv, compute_t_mean_grid

            if not self._grids:
                self.build_atm_grids()

            freqs_Hz = self._ds.coords["frequency"].values
            # `_grids` is keyed by the scan_id *value* (matches the rest
            # of the codebase); `anchor` and `compute_t_mean_grid` want
            # positional indices aligned with array axes.  Remap here.
            scan_ids = self._ds.coords["scan"].values
            grids_by_pos = {
                i: self._grids[int(sid)]
                for i, sid in enumerate(scan_ids)
                if int(sid) in self._grids
            }
            t_mean = compute_t_mean_grid(
                grids_by_pos, freqs_Hz, n_scan=int(scan_ids.size)
            )

            fit.fit_dataset(
                self._ds,
                mode=_INDEPENDENT_TO_BACKEND[mode],
                t_mean=t_mean,
                n_workers=n_workers,
            )

            pwv, pwv_err = anchor_pwv(
                self._ds["tau_zenith"].values,
                self._ds["tau_err"].values,
                grids_by_pos,
                freqs_Hz,
            )
            self._ds["pwv"] = (("antenna",), pwv.astype(np.float32))
            self._ds["pwv_err"] = (("antenna",), pwv_err.astype(np.float32))
            self._ds.attrs["mode"] = mode  # public mode label, not backend
        else:
            fit.fit_dataset(self._ds, mode=mode, n_workers=n_workers)
        self._mode = mode

    def extrapolate(
        self,
        *,
        atm_profile_source: str = "open-meteo",
        afgl_climatology: str = "midlatitude_summer",
    ) -> None:
        from tipopac import atmosphere

        atmosphere.extrapolate(
            self._ds,
            atm_profile_source=atm_profile_source,
            afgl_climatology=afgl_climatology,
        )

    def plot(self, out_dir: str | Path) -> None:
        from tipopac.plot import PlotData

        PlotData(self._ds).save_all(out_dir=Path(out_dir))

    def write_caltables(
        self,
        *,
        opacity: Path | None = None,
        tcal: Path | None = None,
    ) -> None:
        from tipopac import caltables

        if opacity is not None:
            caltables.write_opacity(self._ds, opacity)
        if tcal is not None:
            caltables.write_tcal(self._ds, tcal)

    @property
    def result(self) -> Result:
        if self._mode is None:
            raise RuntimeError("call fit() before accessing result")
        fmt: Literal["ms", "sdm"] = self._ds.attrs.get("source_format", "ms")
        return Result(
            dataset=self._ds,
            mode=self._mode,
            input_path=self._path,
            input_format=fmt,
            software_versions=self._versions,
        )

    @property
    def dataset(self) -> xr.Dataset:
        return self._ds
