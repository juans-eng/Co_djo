# -*- coding: utf-8 -*-
"""
validation/material_validator.py
==================================

Rules 6 & 7 — material-table validation. Ported verbatim from
`codigo/djo_validation_core.ipynb`.

Rule 6 (table_completeness): `originarios_estado_parte_produtor` is the
ONLY table where "NCM/SH" and "Valor (US$)" may be "[Não Informado]"/empty;
every other column, and every column in the other 3 tables, must be
present and not "[Não Informado]". A table with zero Items is treated as
"does not exist in this DJO" — NOT_APPLICABLE, not a failure.

Rule 7 (table_percentage): for every table that has data, the sum of its
items' "% s/Valor FOB" must numerically match the table's own declared
"Somatório", within PERCENTAGE_TOLERANCE percentage points.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.models.validation import ValidationStatus, make_result, is_blank, is_missing_or_not_informado, parse_percentage, NAO_INFORMADO
from src.models.djo import MATERIAL_TABLE_NAMES, ITEM_COLUMNS

# Tolerance (percentage points) used when comparing a table's calculated
# '% s/Valor FOB' sum against its declared 'Somatório' (Rule 7). Absorbs
# rounding drift across several line items without masking real mismatches.
PERCENTAGE_TOLERANCE = 0.5

# Rule 6.1: only these two columns, and only in
# "originarios_estado_parte_produtor", may be "[Não Informado]"/empty.
COLUMNS_ALLOWED_NOT_INFORMADO = {"NCM/SH", "Valor (US$)"}


def _validate_table_item(table_name: str, item: Dict[str, str], item_index: int) -> List[str]:
    """Returns human-readable issues for one item; an empty list means the
    item is fully valid for `table_name`'s rules."""
    issues = []
    for column in ITEM_COLUMNS:
        value = item.get(column)
        if is_missing_or_not_informado(value):
            if table_name == "originarios_estado_parte_produtor" and column in COLUMNS_ALLOWED_NOT_INFORMADO:
                continue  # explicitly allowed by Rule 6.1
            issues.append(f"item {item_index}: '{column}' is missing or '{NAO_INFORMADO}'")
    return issues


def validate_table_completeness(materiales: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 6."""
    details: Dict[str, Any] = {}
    any_fail = False
    any_checked = False

    for table_name in MATERIAL_TABLE_NAMES:
        table = materiales.get(table_name) or {}
        items = table.get("Items", [])

        if not items:
            details[table_name] = {"status": ValidationStatus.NOT_APPLICABLE.value, "issues": []}
            continue

        any_checked = True
        issues: List[str] = []
        for idx, item in enumerate(items):
            issues.extend(_validate_table_item(table_name, item, idx))

        if issues:
            any_fail = True
            details[table_name] = {"status": ValidationStatus.FAIL.value, "issues": issues}
        else:
            details[table_name] = {"status": ValidationStatus.PASS.value, "issues": []}

    if not any_checked:
        overall_status, message = ValidationStatus.NOT_APPLICABLE, "No material tables with data were present in this DJO."
    elif any_fail:
        overall_status, message = ValidationStatus.FAIL, "One or more material tables have missing/invalid required columns."
    else:
        overall_status, message = ValidationStatus.PASS, "All existing material tables have complete required columns."

    return make_result("table_completeness", overall_status, message, details=details)


def validate_table_percentages(materiales: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 7."""
    details: Dict[str, Any] = {}
    any_fail = False
    any_manual = False
    any_checked = False

    for table_name in MATERIAL_TABLE_NAMES:
        table = materiales.get(table_name) or {}
        items = table.get("Items", [])
        somatorio_raw = table.get("Somatório", "")

        if not items or is_blank(somatorio_raw):
            details[table_name] = {"status": ValidationStatus.NOT_APPLICABLE.value}
            continue

        somatorio = parse_percentage(somatorio_raw)
        item_percentages = [parse_percentage(item.get("% s/Valor FOB")) for item in items]

        if somatorio is None or any(p is None for p in item_percentages):
            # A value couldn't be parsed at all — a data problem, not a
            # numeric mismatch. Flag for manual review instead of treating
            # the unparsable value as 0 (which would silently mask it).
            any_checked = True
            any_manual = True
            details[table_name] = {
                "status": ValidationStatus.MANUAL_VERIFICATION.value,
                "declared_somatorio": somatorio_raw,
            }
            continue

        any_checked = True
        calculated_sum = round(sum(item_percentages), 2)
        difference = round(abs(calculated_sum - somatorio), 2)
        passed = difference <= PERCENTAGE_TOLERANCE

        details[table_name] = {
            "status": ValidationStatus.PASS.value if passed else ValidationStatus.FAIL.value,
            "calculated_sum": calculated_sum,
            "declared_somatorio": somatorio,
            "difference": difference,
        }
        if not passed:
            any_fail = True

    if not any_checked:
        overall_status = ValidationStatus.NOT_APPLICABLE
        message = "No material tables with a declared Somatório were present."
    elif any_fail:
        overall_status = ValidationStatus.FAIL
        message = "The sum of '% s/Valor FOB' does not match the declared Somatório for one or more tables."
    elif any_manual:
        overall_status = ValidationStatus.MANUAL_VERIFICATION
        message = "One or more percentage values could not be parsed; manual verification required."
    else:
        overall_status = ValidationStatus.PASS
        message = "The sum of '% s/Valor FOB' matches the declared Somatório for all applicable tables."

    return make_result("table_percentage", overall_status, message, details=details)
