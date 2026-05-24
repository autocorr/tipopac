"""Public API for tipopac — one-shot function and staged class (DESIGN.md §2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import xarray as xr

_READERS: list = []  # populated on first call to avoid import-time casatools load


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
        line = (r.stdout or r.stderr).splitlines()[0] if r.returncode == 0 else "unknown"
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
    mode: str = "tcal_solve",
    flags_online: bool = True,
    flags_file: str | Path | None = None,
    atm_model: bool = True,
    atm_profile_source: str = "open-meteo",
    afgl_climatology: str = "midlatitude_summer",
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
        Fit mode — ``"tau_per_antenna"``, ``"global_tau"``, or ``"tcal_solve"``.
    flags_online:
        Apply FLAG_CMD online flags (MS only; SDM has no equivalent).
    flags_file:
        Path to a user flag file (one ``antenna/spw/timerange`` line per row).
    atm_model:
        Run am + open-meteo atmospheric extrapolation.
    atm_profile_source:
        ``"open-meteo"`` (default) or ``"afgl"``.
    afgl_climatology:
        AFGL climatology name used as fallback or when ``atm_profile_source="afgl"``.
    plot_dir:
        If set, write per-(scan, antenna, spw) diagnostic PNGs here.
    caltable_opacity:
        If set, write a CASA TOpac caltable to this path.
    caltable_tcal:
        If set, write a CALDEVICE-style Tcal caltable to this path.
    """
    ta = TippingAnalysis.from_path(path)
    ta.apply_flags(online=flags_online, file=None if flags_file is None else Path(flags_file))
    ta.fit(mode=mode)
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

    def fit(self, mode: str = "tcal_solve") -> None:
        from tipopac import fit

        fit.fit_dataset(self._ds, mode=mode)
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
        from tipopac import plot

        plot.plot_dataset(self._ds, out_dir=Path(out_dir))

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
