"""V1-B clinical extension: reserved schema-only interface (V1-A gate).

V1-A ships with this empty reserved boundary. It contains no patient values,
no clinical measurements, and no adapter logic. ``clinical_status()`` returns
``not_started_no_patient_cohort`` until a compliant cohort is supplied; an
empty module is a valid state and never changes V1-A graphs, checkpoints,
folds, or metrics (V1_IMPLEMENTATION_PLAN section 11).
"""

from __future__ import annotations

CLINICAL_FEATURE_IDS = (
    "CLIN:SUA",
    "CLIN:LDL_C",
    "CLIN:HS_CRP",
    "CLIN:IVUS_PLAQUE_BURDEN",
    "CLIN:IVUS_MIN_LUMEN_AREA",
)

CLINICAL_LONG_FORMAT_COLUMNS = (
    "patient_id",
    "visit_id",
    "site_id",
    "visit_datetime",
    "disease_id",
    "feature_id",
    "value",
    "unit",
    "missing_reason",
    "source_record_id",
)


def clinical_status() -> str:
    """V1-A ships without a clinical cohort; report the reserved state."""
    return "not_started_no_patient_cohort"


__all__ = [
    "CLINICAL_FEATURE_IDS",
    "CLINICAL_LONG_FORMAT_COLUMNS",
    "clinical_status",
]
