# -*- coding: utf-8 -*-
"""
models/validation.py
=====================

Shared building blocks used by every validation rule, ported verbatim from
`codigo/djo_validation_core.ipynb` (the "Shared models / helpers" section).
No behavior changed — just moved out of the notebook so `djo/validator.py`,
`ace18/rule_engine.py`, and `validation/*` can all import the same
definitions instead of each notebook cell redefining them.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class ValidationStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    MANUAL_VERIFICATION = "MANUAL_VERIFICATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def make_result(rule: str, status: ValidationStatus, message: str, **extra: Any) -> Dict[str, Any]:
    """Builds one entry of validation_result["validations"]. Always stores
    the plain string value of `status` so the whole result stays directly
    json.dumps-able without a custom encoder."""
    result: Dict[str, Any] = {"rule": rule, "status": status.value, "message": message}
    result.update(extra)
    return result


# The literal placeholder the DJO normalization pipeline preserves verbatim
# inside material-table cells (see djo/normalizer.py). Top-level scalar
# fields of json_djo already collapse this placeholder to "" during
# normalization, but table cells do not — validators must check both forms
# explicitly (see `is_missing_or_not_informado` below).
NAO_INFORMADO = "[Não Informado]"


def is_blank(value: Any) -> bool:
    """True for None or an empty/whitespace-only string. Does NOT treat the
    literal "[Não Informado]" placeholder as blank by itself — see
    `is_missing_or_not_informado` for the table-cell case where that
    placeholder is preserved verbatim instead of being collapsed to ""."""
    return value is None or (isinstance(value, str) and value.strip() == "")


def is_missing_or_not_informado(value: Any) -> bool:
    """Used for material-table cell validation, where the normalization
    pipeline preserves "[Não Informado]" as a literal string instead of
    collapsing it to ""."""
    return is_blank(value) or (isinstance(value, str) and value.strip() == NAO_INFORMADO)


def parse_percentage(value: Optional[str]) -> Optional[float]:
    """Parses a Brazilian-formatted percentage string ('13,13%', '0%',
    '100%') into a float. Returns None (never 0.0) when the value is
    blank, '[Não Informado]', or otherwise unparsable — callers must treat
    None as "cannot validate this value", never silently as zero."""
    if is_missing_or_not_informado(value):
        return None
    cleaned = str(value).strip().replace("%", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None
