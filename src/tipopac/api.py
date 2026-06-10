"""Public API for tipopac — one-shot function and staged class (DESIGN.md §2)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from tipopac.atmgrid import PwvGrid
from tipopac.readers import detect_reader as _detect_reader

# Public Stage A+B modes (independent τ fit + per-antenna PWV anchor;
# `design/independent_tau_fit.md`). The values are the Stage-A backend
# mode in :func:`tipopac.fit.fit_dataset`.
_INDEPENDENT_TO_BACKEND: dict[str, str] = {
    "independent_tau": "tau_per_antenna",
    "independent_tau_solve": "tcal_solve",
}


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
    scans: Sequence[int] | None = None,
    bands: Sequence[str] | None = None,
    mode: str = "independent_tau_solve",
    flags_online: bool = True,
    flags_file: str | Path | None = None,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "auto",
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
    scans:
        DO_SKYDIP scan numbers to keep. ``None`` (default) keeps every
        DO_SKYDIP scan in the input.
    bands:
        VLA receiver bands to keep (e.g. ``["Ku", "K"]``; case-
        insensitive). ``None`` (default) keeps the high-frequency
        receivers ``("Ku", "K", "Ka", "Q")`` where tipping-curve fits
        are well-conditioned; pass ``bands=["L", ...]`` to opt into low
        bands explicitly.
    mode:
        Fit mode. Defaults to ``"independent_tau_solve"`` — per-(scan, spw)
        Stage-A Tcal-solve fit followed by a per-antenna PWV anchor (Stage
        B). The other accepted value is ``"independent_tau"`` — per-(scan,
        ant, spw) opacity Stage-A fit with the same Stage-B anchor.
    flags_online:
        Apply FLAG_CMD online flags (MS only; SDM has no equivalent).
    flags_file:
        Path to a user flag file (one ``antenna/spw/timerange`` line per row).
    atm_profile_source:
        ``"open-meteo"`` (default) or ``"afgl"``. Drives the single
        :meth:`TippingAnalysis.fetch_atm_profile` call; downstream
        consumers read the profile off the dataset.
    afgl_climatology:
        AFGL climatology name used on open-meteo fallback or when
        ``atm_profile_source="afgl"``. Default ``"auto"`` picks
        ``midlatitude_summer`` / ``midlatitude_winter`` from the
        observation's month.
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
    if mode not in _INDEPENDENT_TO_BACKEND:
        raise ValueError(
            f"mode must be one of {tuple(_INDEPENDENT_TO_BACKEND)!r}, got {mode!r}"
        )

    ta = TippingAnalysis.from_path(path, scans=scans, bands=bands)
    ta.apply_flags(
        online=flags_online, file=None if flags_file is None else Path(flags_file)
    )
    ta.fetch_atm_profile(
        source=atm_profile_source,
        afgl_climatology=afgl_climatology,
    )
    ta.build_atm_grids()
    ta.fit(mode=mode, n_workers=n_workers)

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
    def from_path(
        cls,
        path: str | Path,
        *,
        scans: Sequence[int] | None = None,
        bands: Sequence[str] | None = None,
    ) -> "TippingAnalysis":
        p = Path(path)
        R = _detect_reader(p)
        ds = R.from_path(p, scans=scans, bands=bands).read()
        return cls(ds, p)

    def apply_flags(
        self,
        *,
        online: bool = True,
        file: Path | None = None,
    ) -> None:
        from tipopac import flags

        self._ds = flags.apply(self._ds, online=online, file=file)

    def fetch_atm_profile(
        self,
        *,
        source: str = "open-meteo",
        afgl_climatology: str = "auto",
    ) -> None:
        """Fetch the atmospheric profile once and attach it to the dataset.

        Idempotent: re-running on a dataset that already has
        ``atm_pressure`` is a no-op.

        Adds ``atm_pressure(atm_level)``, ``atm_temperature(scan,
        atm_level)``, ``atm_h2o_vmr(scan, atm_level)``. Writes attrs
        ``atm_profile_source``, ``open_meteo_query``,
        ``surface_pressure_hPa``.

        ``source``:
            ``"open-meteo"`` (default) — one HTTP call covering the obs
            date range, per-scan closest hourly slice; AFGL fallback on
            error.  ``"afgl"`` — skip the network call entirely.
        ``afgl_climatology``:
            ``"auto"`` (default) picks summer/winter from the obs month.
        """
        if "atm_pressure" in self._ds.data_vars:
            return
        from tipopac.atmosphere import attach_profile

        attach_profile(self._ds, source=source, afgl_climatology=afgl_climatology)

    def build_atm_grids(
        self,
        *,
        pwv_step_mm: float = 0.5,
        freq_step_Hz: float = 100e6,
        n_workers: int | None = None,
    ) -> None:
        """Build per-scan :class:`PwvGrid` objects.

        Auto-calls :meth:`fetch_atm_profile` with defaults if the profile
        is not yet on the dataset. Populates ``self._grids[scan_id] =
        PwvGrid`` for every scan and writes
        ``ds.attrs["pwv_profile_source"][scan_id]`` for provenance. Used
        by the post-fit atmospheric anchor (see
        ``design/independent_tau_fit.md``); not consumed by :meth:`fit`.
        """
        import astropy.units as u

        from tipopac.atmgrid import build_pwv_grid

        if "atm_pressure" not in self._ds.data_vars:
            self.fetch_atm_profile()

        freqs = self._ds.coords["frequency"].values
        freq_min_Hz = float(freqs.min()) * 0.95
        freq_max_Hz = float(freqs.max()) * 1.05

        scan_ids = self._ds.coords["scan"].values
        atm_source = str(self._ds.attrs.get("atm_profile_source", "unknown"))
        pressure_Pa = self._ds["atm_pressure"].values  # (atm_level,)
        pressure_q = pressure_Pa * u.Pa
        temp_K = self._ds["atm_temperature"].values  # (scan, atm_level)
        vmr = self._ds["atm_h2o_vmr"].values  # (scan, atm_level)

        sources: dict[int, str] = {}
        for i, scan_id in enumerate(scan_ids):
            temperature_q = temp_K[i].astype(np.float64) * u.K
            h2o_q = vmr[i].astype(np.float64) * u.dimensionless_unscaled
            grid = build_pwv_grid(
                pressure_q,
                temperature_q,
                h2o_q,
                freq_min_Hz=freq_min_Hz,
                freq_max_Hz=freq_max_Hz,
                profile_source=atm_source,
                pwv_step_mm=pwv_step_mm,
                freq_step_Hz=freq_step_Hz,
                n_workers=n_workers,
            )
            self._grids[int(scan_id)] = grid
            sources[int(scan_id)] = atm_source

        self._ds.attrs["pwv_profile_source"] = sources

    def fit(
        self,
        mode: str = "independent_tau_solve",
        *,
        n_workers: int | None = None,
    ) -> None:
        if mode not in _INDEPENDENT_TO_BACKEND:
            raise ValueError(
                f"mode must be one of {tuple(_INDEPENDENT_TO_BACKEND)!r}, got {mode!r}"
            )

        from tipopac import fit
        from tipopac.anchor import anchor_pwv, compute_t_mean_grid, write_am_curve

        # Stage A + Stage B. Build grids if not done already; the grid
        # drives both the Stage A T_mean input and the Stage B PWV anchor
        # against τ_z(ν).
        if not self._grids:
            self.build_atm_grids()

        freqs_Hz = self._ds.coords["frequency"].values
        # `_grids` is keyed by the scan_id *value* (matches the rest of
        # the codebase); `anchor` and `compute_t_mean_grid` want positional
        # indices aligned with array axes.  Remap here.
        scan_ids = self._ds.coords["scan"].values
        grids_by_pos = {
            i: self._grids[int(sid)]
            for i, sid in enumerate(scan_ids)
            if int(sid) in self._grids
        }
        t_mean = compute_t_mean_grid(grids_by_pos, freqs_Hz, n_scan=int(scan_ids.size))

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
        write_am_curve(self._ds, grids_by_pos, pwv)
        self._ds.attrs["mode"] = mode  # public mode label, not backend
        self._mode = mode

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
