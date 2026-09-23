"""Frozen casacore table descriptions for the caltable writers (design.md §9.2)."""

from __future__ import annotations

import copy
from typing import Any

__all__ = ["table_desc", "TOPAC_INFO"]

TOPAC_INFO = {"type": "Calibration", "subType": "TOpac", "readme": ""}

_EPOCH_MEASINFO = {"Ref": "UTC", "type": "epoch"}
_POSITION_MEASINFO = {"Ref": "ITRF", "type": "position"}
_DIRECTION_MEASINFO = {"Ref": "J2000", "type": "direction"}
_FREQUENCY_MEASINFO = {
    "TabRefCodes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 64],
    "TabRefTypes": [
        "REST",
        "LSRK",
        "LSRD",
        "BARY",
        "GEO",
        "TOPO",
        "GALACTO",
        "LGROUP",
        "CMB",
        "Undefined",
    ],
    "VarRefCol": "MEAS_FREQ_REF",
    "type": "frequency",
}


def _col(
    value_type: str,
    *,
    ndim: int | None = None,
    shape: list[int] | None = None,
    option: int = 0,
    keywords: dict[str, Any] | None = None,
    comment: str = "",
    group: str = "MSMTAB",
) -> dict[str, Any]:
    """Build one column description in casacore's `getdesc` form."""
    desc: dict[str, Any] = {
        "valueType": value_type,
        "dataManagerType": "StandardStMan",
        "dataManagerGroup": group,
        "option": option,
        "maxlen": 0,
        "comment": comment,
        "keywords": dict(keywords or {}),
    }
    if ndim is not None:
        desc["ndim"] = ndim
    if shape is not None:
        desc["shape"] = shape
    return desc


_DESCS: dict[str, dict[str, Any]] = {
    "MAIN": {
        "ANTENNA1": _col("int", option=5),
        "ANTENNA2": _col("int", option=5),
        "FIELD_ID": _col("int", option=5),
        "FLAG": _col("boolean", ndim=-1),
        "FPARAM": _col("float", ndim=-1),
        "INTERVAL": _col("double", option=5, keywords={"QuantumUnits": ["s"]}),
        "OBSERVATION_ID": _col("int", option=5),
        "PARAMERR": _col("float", ndim=-1),
        "SCAN_NUMBER": _col("int", option=5),
        "SNR": _col("float", ndim=-1),
        "SPECTRAL_WINDOW_ID": _col("int", option=5),
        "TIME": _col(
            "double",
            option=5,
            keywords={"MEASINFO": _EPOCH_MEASINFO, "QuantumUnits": ["s"]},
        ),
        "WEIGHT": _col("float", ndim=-1),
    },
    "ANTENNA": {
        "DISH_DIAMETER": _col(
            "double",
            keywords={"QuantumUnits": ["m"]},
            comment="Physical diameter of dish",
        ),
        "FLAG_ROW": _col("boolean", comment="Flag for this row"),
        "MOUNT": _col("string", comment="Mount type e.g. alt-az, equatorial, etc."),
        "NAME": _col("string", comment="Antenna name, e.g. VLA22, CA03"),
        "OFFSET": _col(
            "double",
            ndim=1,
            shape=[3],
            option=5,
            keywords={"MEASINFO": _POSITION_MEASINFO, "QuantumUnits": ["m", "m", "m"]},
            comment="Axes offset of mount to FEED REFERENCE point",
        ),
        "POSITION": _col(
            "double",
            ndim=1,
            shape=[3],
            option=5,
            keywords={"MEASINFO": _POSITION_MEASINFO, "QuantumUnits": ["m", "m", "m"]},
            comment="Antenna X,Y,Z phase reference position",
        ),
        "STATION": _col("string", comment="Station (antenna pad) name"),
        "TYPE": _col("string", comment="Antenna type (e.g. SPACE-BASED)"),
    },
    "SPECTRAL_WINDOW": {
        "CHAN_FREQ": _col(
            "double",
            ndim=1,
            keywords={"MEASINFO": _FREQUENCY_MEASINFO, "QuantumUnits": ["Hz"]},
            comment="Center frequencies for each channel in the data matrix",
        ),
        "CHAN_WIDTH": _col(
            "double",
            ndim=1,
            keywords={"QuantumUnits": ["Hz"]},
            comment="Channel width for each channel",
        ),
        "EFFECTIVE_BW": _col(
            "double",
            ndim=1,
            keywords={"QuantumUnits": ["Hz"]},
            comment="Effective noise bandwidth of each channel",
        ),
        "FLAG_ROW": _col("boolean", comment="Row flag"),
        "FREQ_GROUP": _col("int", comment="Frequency group"),
        "FREQ_GROUP_NAME": _col("string", comment="Frequency group name"),
        "IF_CONV_CHAIN": _col("int", comment="The IF conversion chain number"),
        "MEAS_FREQ_REF": _col("int", comment="Frequency Measure reference"),
        "NAME": _col("string", comment="Spectral window name"),
        "NET_SIDEBAND": _col("int", comment="Net sideband"),
        "NUM_CHAN": _col("int", comment="Number of spectral channels"),
        "REF_FREQUENCY": _col(
            "double",
            keywords={"MEASINFO": _FREQUENCY_MEASINFO, "QuantumUnits": ["Hz"]},
            comment="The reference frequency",
        ),
        "RESOLUTION": _col(
            "double",
            ndim=1,
            keywords={"QuantumUnits": ["Hz"]},
            comment="The effective noise bandwidth for each channel",
        ),
        "TOTAL_BANDWIDTH": _col(
            "double",
            keywords={"QuantumUnits": ["Hz"]},
            comment="The total bandwidth for this window",
        ),
    },
    "FIELD": {
        "CODE": _col(
            "string",
            comment="Special characteristics of field, e.g. Bandpass calibrator",
        ),
        "DELAY_DIR": _col(
            "double",
            ndim=2,
            keywords={"MEASINFO": _DIRECTION_MEASINFO, "QuantumUnits": ["rad", "rad"]},
            comment="Direction of delay center (e.g. RA, DEC)as polynomial in time.",
        ),
        "FLAG_ROW": _col("boolean", comment="Row Flag"),
        "NAME": _col("string", comment="Name of this field"),
        "NUM_POLY": _col("int", comment="Polynomial order of _DIR columns"),
        "PHASE_DIR": _col(
            "double",
            ndim=2,
            keywords={"MEASINFO": _DIRECTION_MEASINFO, "QuantumUnits": ["rad", "rad"]},
            comment="Direction of phase center (e.g. RA, DEC).",
        ),
        "REFERENCE_DIR": _col(
            "double",
            ndim=2,
            keywords={"MEASINFO": _DIRECTION_MEASINFO, "QuantumUnits": ["rad", "rad"]},
            comment="Direction of REFERENCE center (e.g. RA, DEC).as polynomial in time.",
        ),
        "SOURCE_ID": _col("int", comment="Source id"),
        "TIME": _col(
            "double",
            keywords={"MEASINFO": _EPOCH_MEASINFO, "QuantumUnits": ["s"]},
            comment="Time origin for direction and rate",
        ),
    },
    "OBSERVATION": {
        "FLAG_ROW": _col("boolean", comment="Row flag"),
        "LOG": _col("string", ndim=1, comment="Observing log"),
        "OBSERVER": _col("string", comment="Name of observer(s)"),
        "PROJECT": _col("string", comment="Project identification string"),
        "RELEASE_DATE": _col(
            "double",
            keywords={"MEASINFO": _EPOCH_MEASINFO, "QuantumUnits": ["s"]},
            comment="Release date when data becomes public",
        ),
        "SCHEDULE": _col("string", ndim=1, comment="Observing schedule"),
        "SCHEDULE_TYPE": _col("string", comment="Observing schedule type"),
        "TELESCOPE_NAME": _col("string", comment="Telescope Name (e.g. WSRT, VLBA)"),
        "TIME_RANGE": _col(
            "double",
            ndim=1,
            shape=[2],
            option=5,
            keywords={"MEASINFO": _EPOCH_MEASINFO, "QuantumUnits": ["s"]},
            comment="Start and end of observation",
        ),
    },
    "HISTORY": {
        "APPLICATION": _col("string", comment="Application name"),
        "APP_PARAMS": _col("string", ndim=1, comment="Application parameters"),
        "CLI_COMMAND": _col("string", ndim=1, comment="CLI command sequence"),
        "MESSAGE": _col("string", comment="Log message"),
        "OBJECT_ID": _col("int", comment="Originating ObjectID"),
        "OBSERVATION_ID": _col(
            "int", comment="Observation id (index in OBSERVATION table)"
        ),
        "ORIGIN": _col(
            "string", comment="(Source code) origin from which message originated"
        ),
        "PRIORITY": _col("string", comment="Message priority"),
        "TIME": _col(
            "double",
            keywords={"MEASINFO": _EPOCH_MEASINFO, "QuantumUnits": ["s"]},
            comment="Timestamp of message",
        ),
    },
    "CALDEVICE": {
        "ANTENNA_ID": _col(
            "int", comment="Antenna's identifier", group="StandardStMan"
        ),
        "CAL_EFF": _col(
            "float",
            ndim=-1,
            comment="Calibration efficiencies (one per receptor per load).",
            group="StandardStMan",
        ),
        "CAL_LOAD_NAMES": _col(
            "string", ndim=-1, comment="Calibration load names.", group="StandardStMan"
        ),
        "FEED_ID": _col("int", comment="Feed's index", group="StandardStMan"),
        "INTERVAL": _col(
            "double", comment="Interval of measurement.", group="StandardStMan"
        ),
        "NOISE_CAL": _col(
            "float",
            ndim=-1,
            comment="Equivalent temperatures of the noise sources (TCAL for EVLA).",
            group="StandardStMan",
        ),
        "NUM_CAL_LOAD": _col(
            "int", comment="Number of calibration loads.", group="StandardStMan"
        ),
        "NUM_RECEPTOR": _col(
            "int", comment="Number of receptors.", group="StandardStMan"
        ),
        "SPECTRAL_WINDOW_ID": _col(
            "int", comment="Spectral window identifier.", group="StandardStMan"
        ),
        "TEMPERATURE_LOAD": _col(
            "double", ndim=-1, comment="Physical.", group="StandardStMan"
        ),
        "TIME": _col(
            "double", comment="Midpoint of time measurement.", group="StandardStMan"
        ),
    },
}


def table_desc(name: str) -> dict[str, Any]:
    """Return a fresh copy of the frozen description for `name`."""
    return copy.deepcopy(_DESCS[name])
