"""Shared defaults for the Stage A/B/C fit behaviour.

These are declared once here because more than one entry point takes them
— the public :func:`tipopac.api.tipopac` and :class:`TippingAnalysis`
signatures, and the backends they forward to. Values must not be restated
at a call site.

Scope: fit behaviour only. The am-grid geometry defaults (``DEFAULT_PWV_*``,
``DEFAULT_FREQ_STEP_HZ``, ``DEFAULT_N_WORKERS``) live in
:mod:`tipopac.atmgrid` next to the builder that owns them. This module
imports nothing, so it can back a signature default anywhere without
pulling scipy into ``import tipopac``.
"""

DEFAULT_SPILLOVER_MODEL: bool = True
DEFAULT_GROUP_DURATION_S: float | None = 7200.0
DEFAULT_MIN_AIRMASS_SPAN: float = 0.3
