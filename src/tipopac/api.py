"""Public API for tipopac — one-shot function and staged class (design.md §2)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr

from tipopac.atmgrid import (
    DEFAULT_FREQ_STEP_HZ,
    DEFAULT_N_WORKERS,
    DEFAULT_PWV_STEP_MM,
    PwvGrid,
    grid_freq_span,
)
from tipopac.defaults import (
    DEFAULT_GROUP_DURATION_S,
    DEFAULT_MIN_AIRMASS_SPAN,
    DEFAULT_SPILLOVER_MODEL,
)
from tipopac.readers import detect_reader as _detect_reader

_log = logging.getLogger(__name__)

# Public Stage A+B modes (design.md §2.1); values are the Stage-A backend
# mode in :func:`tipopac.fit.fit_dataset`.
_INDEPENDENT_TO_BACKEND: dict[str, str] = {
    "independent_tau": "tau_per_antenna",
    "independent_tau_solve": "tcal_solve",
}


def _module_version(name: str) -> str:
    """Return the module's version, or a sentinel if import/lookup fails.

    ``__version__`` first, then the installed distribution metadata —
    ``casatools`` and ``sdmpy`` carry no ``__version__``.
    """
    import importlib
    import importlib.metadata

    from tipopac._casa import import_casatools

    try:
        mod = (
            import_casatools() if name == "casatools" else importlib.import_module(name)
        )
    except Exception:
        return "unavailable"
    declared = getattr(mod, "__version__", None)
    if declared is not None:
        return str(declared)
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _software_versions() -> dict[str, str]:
    versions = {n: _module_version(n) for n in ("casatools", "sdmpy", "amwrap")}
    try:
        import amwrap

        # am-serial is the executable the grid workers run (atmgrid:285).
        versions["am"] = str(amwrap.AM_SERIAL.version)
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
    """Write ``ds`` to NetCDF, sanitizing attrs NetCDF cannot encode.

    Works on a shallow copy so the caller's in-memory Dataset is not
    mutated. Object-dtype string vars (``fit_reason``,
    ``pwv_profile_source``) need no help — xarray encodes them itself.
    """
    to_write = ds.copy()
    to_write.attrs = {k: _coerce_attr_for_netcdf(v) for k, v in to_write.attrs.items()}
    to_write.to_netcdf(path)


def _write_tsv(
    path: Path,
    ds: xr.Dataset,
    builder: Callable[[xr.Dataset], tuple[Sequence[str], Sequence[Sequence[Any]]]],
) -> None:
    """Write one TSV over every time group, with a leading ``group`` column.

    The per-group weblog page renders `builder`'s rows for its own group, so
    the page stays a filtered view of this file rather than a second rendering
    that could disagree with it (§9).
    """
    from tipopac.schema import select_group

    header_written = False
    with path.open("w") as f:
        for k in range(int(ds.sizes["group"])):
            columns, rows = builder(select_group(ds, k))
            if not header_written:
                f.write("\t".join(("group", *columns)) + "\n")
                header_written = True
            for row in rows:
                cells = (f"{v:.6e}" if isinstance(v, float) else str(v) for v in row)
                f.write("\t".join((str(k), *cells)) + "\n")


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
    mode: str = "independent_tau",
    flags_online: bool = True,
    flags_file: str | Path | None = None,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "auto",
    spillover_model: bool = DEFAULT_SPILLOVER_MODEL,
    group_duration_s: float | None = DEFAULT_GROUP_DURATION_S,
    min_airmass_span: float = DEFAULT_MIN_AIRMASS_SPAN,
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
        Fit mode. Defaults to ``"independent_tau"`` — per-(scan, ant, spw)
        opacity Stage-A fit at ``c ≡ 1``, a per-antenna PWV anchor (Stage B),
        and Stage C, which estimates the Tcal scale against that anchor in
        closed form. The other accepted value is
        ``"independent_tau_solve"`` — per-(scan, spw) Stage-A fit solving τ
        and a per-antenna Tcal gain jointly. It is **legacy**: over the
        sampled airmass the data cannot separate a change in τ from a gain,
        so the free gain buys no fit quality while biasing ``tau_zenith``
        high against an independent atmospheric prediction, and it runs no
        Stage C.
    flags_online:
        Apply online flags (MS ``FLAG_CMD`` / SDM ``Flag.xml``).
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
    spillover_model:
        When ``True`` (default), model instrumental ground pickup as a
        ``η(ν)·k2nt(T_surf,ν)·airmass`` term inside the Stage-A Tsys forward
        model, so ``tau_zenith`` is fit spillover-free and PWV anchors on it
        directly. ``False`` reproduces the pre-spillover fit for
        parity/repro. This replaces the retired flat post-hoc δτ de-bias; the
        old ``0.0036`` behaviour is intentionally not preserved.
    group_duration_s:
        Stage-B time grouping. Scans are partitioned into greedy sequential
        windows of at most this many seconds and one PWV per antenna is fit
        within each; default 7200 s. ``None`` puts every scan in one group,
        reproducing the pre-grouping pooled anchor. A group can never span
        more than the duration, so a whole-day execution block no longer
        collapses to a single PWV.
    min_airmass_span:
        Stage-C leverage floor on ``max(airmass) − min(airmass)``; default 0.3.
        Cells below it get NaN ``tcal_fit``/``sigma_tcal``.
    n_workers:
        Process-pool size for both the am grid build and the Stage-A fit.
        ``None`` or ``≤ 1`` runs both serially.
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
    ta.build_atm_grids(n_workers=n_workers)
    ta.fit(
        mode=mode,
        n_workers=n_workers,
        spillover_model=spillover_model,
        group_duration_s=group_duration_s,
        min_airmass_span=min_airmass_span,
    )

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
        ds.attrs["software_versions"] = self._versions
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

        Adds ``atm_pressure(scan, atm_level)``, ``atm_temperature(scan,
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
        pwv_step_mm: float = DEFAULT_PWV_STEP_MM,
        freq_step_Hz: float = DEFAULT_FREQ_STEP_HZ,
        n_workers: int | None = DEFAULT_N_WORKERS,
    ) -> None:
        """Build per-scan :class:`PwvGrid` objects.

        Auto-calls :meth:`fetch_atm_profile` with defaults if the profile
        is not yet on the dataset. Populates ``self._grids[scan_id] =
        PwvGrid`` for every scan and writes the ``pwv_profile_source(scan,)``
        and ``pwv_model(scan,)`` data vars for provenance. The grids feed
        Stage A's ``T_mean`` and the Stage-B anchor (design.md §6).
        ``n_workers`` sizes the am process pool; ``None`` or ``≤ 1``
        builds the grids serially.
        """
        import astropy.units as u

        from tipopac.atmgrid import build_pwv_grid

        if "atm_pressure" not in self._ds.data_vars:
            self.fetch_atm_profile()

        freqs = self._ds.coords["frequency"].values
        freq_min_Hz, freq_max_Hz = grid_freq_span(
            float(freqs.min()), float(freqs.max()), freq_step_Hz
        )

        scan_ids = self._ds.coords["scan"].values
        atm_source = str(self._ds.attrs.get("atm_profile_source", "unknown"))
        pressure_Pa = self._ds["atm_pressure"].values  # (scan, atm_level)
        temp_K = self._ds["atm_temperature"].values  # (scan, atm_level)
        vmr = self._ds["atm_h2o_vmr"].values  # (scan, atm_level)
        surface_P_hPa = (
            self._ds["surface_pressure_hPa"].values
            if "surface_pressure_hPa" in self._ds.data_vars
            else None
        )

        # Within a single call, scans sharing the same am inputs (and surface
        # pressures within 0.2 hPa of the first builder) reuse the same
        # PwvGrid. PwvGrid is frozen + read-only downstream — sharing is safe.
        cache: list[tuple[bytes, float | None, PwvGrid]] = []

        sources_arr = np.full(scan_ids.size, "", dtype=object)
        pwv_model = np.full(scan_ids.size, np.nan, dtype=np.float32)
        for i, scan_id in enumerate(scan_ids):
            p_row = pressure_Pa[i].astype(np.float64)
            t_row = temp_K[i].astype(np.float64)
            h_row = vmr[i].astype(np.float64)
            keep = np.isfinite(p_row)
            p_keep = p_row[keep]
            t_keep = t_row[keep]
            h_keep = h_row[keep]
            # Skip level 0 (each scan's interpolated surface value, which
            # tracks the per-scan surface clip from atmosphere.py). Upper
            # levels must match exactly; the 0.2 hPa gate below handles the
            # surface-level differences.
            key = (
                p_keep[1:].tobytes()
                + t_keep[1:].tobytes()
                + h_keep[1:].tobytes()
                + np.asarray(
                    [freq_min_Hz, freq_max_Hz, freq_step_Hz, pwv_step_mm],
                    dtype=np.float64,
                ).tobytes()
            )
            p_scan_hPa: float | None = (
                float(surface_P_hPa[i])
                if surface_P_hPa is not None and np.isfinite(surface_P_hPa[i])
                else None
            )

            grid: PwvGrid | None = None
            for cached_key, anchor_P, cached_grid in cache:
                if cached_key != key:
                    continue
                if p_scan_hPa is None and anchor_P is None:
                    grid = cached_grid
                    break
                if (
                    p_scan_hPa is not None
                    and anchor_P is not None
                    # 1e-9 slack handles float repr near the inclusive boundary
                    # (e.g. 850.2 - 850.0 returns 0.20000000000000018).
                    and abs(p_scan_hPa - anchor_P) <= 0.2 + 1e-9
                ):
                    grid = cached_grid
                    break

            if grid is None:
                grid = build_pwv_grid(
                    p_keep * u.Pa,
                    t_keep * u.K,
                    h_keep * u.dimensionless_unscaled,
                    freq_min_Hz=freq_min_Hz,
                    freq_max_Hz=freq_max_Hz,
                    profile_source=atm_source,
                    pwv_step_mm=pwv_step_mm,
                    freq_step_Hz=freq_step_Hz,
                    n_workers=n_workers,
                )
                cache.append((key, p_scan_hPa, grid))

            self._grids[int(scan_id)] = grid
            sources_arr[i] = atm_source
            pwv_model[i] = grid.pwv_unscaled_mm

        _log.info(
            "PwvGrid cache: built %d unique grid(s) for %d scan(s)",
            len(cache),
            scan_ids.size,
        )

        self._ds["pwv_profile_source"] = (("scan",), sources_arr)
        self._ds["pwv_model"] = (("scan",), pwv_model)

    def fit(
        self,
        mode: str = "independent_tau",
        *,
        n_workers: int | None = None,
        spillover_model: bool = DEFAULT_SPILLOVER_MODEL,
        group_duration_s: float | None = DEFAULT_GROUP_DURATION_S,
        min_airmass_span: float = DEFAULT_MIN_AIRMASS_SPAN,
    ) -> None:
        if mode not in _INDEPENDENT_TO_BACKEND:
            raise ValueError(
                f"mode must be one of {tuple(_INDEPENDENT_TO_BACKEND)!r}, got {mode!r}"
            )

        from tipopac import fit
        from tipopac.anchor import attach_stage_b, compute_t_mean_grid

        # Stage A + Stage B. Build grids if not done already; the grid
        # drives both the Stage A T_mean input and the Stage B PWV anchor
        # against τ_z(ν).
        if not self._grids:
            self.build_atm_grids(n_workers=n_workers)

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
            spillover_model=spillover_model,
        )

        attach_stage_b(
            self._ds,
            grids_by_pos,
            freqs_Hz,
            group_duration_s=group_duration_s,
        )

        # Stage C. Skipped under independent_tau_solve, whose tau_zenith the
        # anchor is fit to — pinning c to it would fold that mode's own
        # opacity inflation into c.
        if mode == "independent_tau":
            from tipopac.tcal import solve_tcal

            solve_tcal(self._ds, min_airmass_span=min_airmass_span)

        self._ds.attrs["mode"] = mode  # public mode label, not backend
        self._ds.attrs["group_duration_s"] = (
            "none" if group_duration_s is None else float(group_duration_s)
        )
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
        (``tipopac.nc``), the Stage-B model τ(ν) curve
        (``model_opacity.tsv``), the fitted and model τ at the spw centres
        (``measured_opacity.tsv``), every diagnostic plot, and the weblog
        ``index.html``. Caltables are opt-in via the boolean flags and land
        in the same directory as ``tipopac.opacity`` / ``tipopac.tcal``.
        """
        from tipopac.tables import measured_opacity_table, model_opacity_table

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_dataset_netcdf(self._ds, out_dir / "tipopac.nc")
        _write_tsv(out_dir / "model_opacity.tsv", self._ds, model_opacity_table)
        _write_tsv(out_dir / "measured_opacity.tsv", self._ds, measured_opacity_table)
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
