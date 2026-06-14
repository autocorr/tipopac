"""Public API for tipopac — one-shot function and staged class (DESIGN.md §2)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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


def _coerce_attr_for_netcdf(value: Any) -> Any:
    """Map a Dataset attr to a NetCDF-serializable value.

    NetCDF attrs accept strings, numbers, and 1-D numeric/string arrays.
    Dicts (e.g. ``open_meteo_query``) and ``None`` are JSON-encoded;
    ``Path`` is stringified; lists are upcast to ``np.ndarray`` when
    homogeneously numeric or string, else JSON-encoded.
    """
    if value is None:
        return ""
    if isinstance(value, str | bytes | int | float | np.ndarray):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        if all(isinstance(v, bool | np.bool_) for v in value):
            return np.asarray(value, dtype=np.int8)
        if all(isinstance(v, int | np.integer) for v in value):
            return np.asarray(value, dtype=np.int64)
        if all(isinstance(v, float | np.floating) for v in value):
            return np.asarray(value, dtype=np.float64)
        if all(isinstance(v, str) for v in value):
            return np.asarray(value, dtype="U")
        return json.dumps(list(value), default=str)
    if isinstance(value, dict):
        return json.dumps(value, default=str)
    return repr(value)


def _write_dataset_netcdf(ds: xr.Dataset, path: Path) -> None:
    """Write ``ds`` to NetCDF, sanitizing attrs/vars NetCDF cannot encode.

    Works on a shallow copy so the caller's in-memory Dataset is not
    mutated. Coerces ``pwv_profile_source`` from object dtype to a
    fixed-width unicode array and runs every Dataset attr through
    :func:`_coerce_attr_for_netcdf`.
    """
    to_write = ds.copy()
    if "pwv_profile_source" in to_write.data_vars and to_write[
        "pwv_profile_source"
    ].dtype == np.dtype("O"):
        vals = to_write["pwv_profile_source"].values
        to_write["pwv_profile_source"] = (
            to_write["pwv_profile_source"].dims,
            np.asarray([str(v) for v in vals], dtype="U"),
        )
    to_write.attrs = {k: _coerce_attr_for_netcdf(v) for k, v in to_write.attrs.items()}
    to_write.to_netcdf(path)


def _write_model_opacity_tsv(ds: xr.Dataset, path: Path) -> None:
    """Stage-B model atmospheric opacity τ(ν) as a two-column TSV.

    Reads ``am_freq_grid`` (Hz) and ``am_tau`` (nepers) from ``ds`` —
    both 1-D over ``frequency_dense`` — and writes
    ``frequency_Hz\\ttau_nepers`` rows.
    """
    freq_Hz = np.asarray(ds["am_freq_grid"].values, dtype=np.float64)
    tau = np.asarray(ds["am_tau"].values, dtype=np.float64)
    with path.open("w") as f:
        f.write("frequency_Hz\ttau_nepers\n")
        for nu, t in zip(freq_Hz, tau, strict=True):
            f.write(f"{nu:.6e}\t{t:.6e}\n")


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
    output_dir: str | Path | None = Path("."),
    caltable_opacity: bool = False,
    caltable_tcal: bool = False,
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
    output_dir:
        Directory for all on-disk outputs (created if missing). Default
        ``Path(".")`` writes into the current working directory. ``None``
        is compute-only mode — return the :class:`Result` without writing
        anything. When set, every run produces ``tipopac.nc`` (full
        Dataset), ``model_opacity.tsv`` (Stage-B τ(ν) at the
        representative PWV), the interactive ``.html`` plots, and the
        ``index.html`` weblog.
    caltable_opacity:
        Opt-in: write a CASA TOpac caltable to ``output_dir/tipopac.opacity``.
        No effect when ``output_dir is None``.
    caltable_tcal:
        Opt-in: write a CALDEVICE-style Tcal caltable to
        ``output_dir/tipopac.tcal``. No effect when ``output_dir is None``.
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

    if output_dir is not None:
        ta.write_outputs(
            output_dir,
            caltable_opacity=caltable_opacity,
            caltable_tcal=caltable_tcal,
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
        atm_level)``, ``atm_h2o_vmr(scan, atm_level)``,
        ``surface_pressure_hPa(scan,)`` (omitted when no scan has finite
        weather_P). Writes attrs ``atm_profile_source``,
        ``open_meteo_query``.

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
        PwvGrid`` for every scan and writes the ``pwv_profile_source(scan,)``
        data var for provenance. Used by the post-fit atmospheric anchor
        (see ``design/independent_tau_fit.md``); not consumed by :meth:`fit`.
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

        sources_arr = np.full(scan_ids.size, "", dtype=object)
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
            sources_arr[i] = atm_source

        self._ds["pwv_profile_source"] = (("scan",), sources_arr)

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

    def weblog(self, plot_dir: str | Path) -> None:
        from tipopac.weblog import build_weblog

        build_weblog(Path(plot_dir))

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

    def write_outputs(
        self,
        output_dir: str | Path = Path("."),
        *,
        caltable_opacity: bool = False,
        caltable_tcal: bool = False,
    ) -> None:
        """Write every artifact for this analysis into ``output_dir``.

        Creates ``output_dir`` if missing, then writes the full Dataset
        (``tipopac.nc``), the Stage-B τ(ν) table (``model_opacity.tsv``),
        every diagnostic plot, and the weblog ``index.html``. Caltables
        are opt-in via the boolean flags and land in the same directory
        as ``tipopac.opacity`` / ``tipopac.tcal``.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_netcdf(self._ds, out_dir / "tipopac.nc")
        _write_model_opacity_tsv(self._ds, out_dir / "model_opacity.tsv")
        self.plot(out_dir=out_dir)
        self.weblog(plot_dir=out_dir)
        if caltable_opacity or caltable_tcal:
            self.write_caltables(
                opacity=out_dir / "tipopac.opacity" if caltable_opacity else None,
                tcal=out_dir / "tipopac.tcal" if caltable_tcal else None,
            )

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
