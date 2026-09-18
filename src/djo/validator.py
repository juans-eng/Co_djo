# -*- coding: utf-8 -*-
"""
djo/validator.py
==================

DJO-level validation rules — the ones that only need `json_djo`'s own
top-level fields (not the ACE 18 rule, not the material tables). Ported
verbatim from `codigo/djo_validation_core.ipynb`:

    Rule 1 -> validate_djo_approval
    Rule 2 -> validate_producer_information
    Rule 4 -> validate_mandatory_fields
    Rule 8 -> validate_preco_fob

Rules 3 (NCM/ACE 18), 5 (process materials), 6-7 (material tables) live in
`ace18/` and `validation/` respectively, since they need more than just
`json_djo`'s top-level fields. `validation/validation_service.py` is what
combines all 8 into the final `validate_djo(...)` result.
"""
from __future__ import annotations

from typing import Any, Dict

from src.models.validation import ValidationStatus, make_result, is_blank, is_missing_or_not_informado, parse_percentage
from src.models.djo import MATERIAL_TABLE_NAMES

# Rule 4's required top-level fields.
MANDATORY_TOP_LEVEL_FIELDS = ["Valor FOB (USD)", "Unidade de medida", "Descrição do processo produtivo"]

# The one material table whose exclusive presence exempts "Valor FOB (USD)"
# from Rule 4 — see _only_producer_originating_table_present below.
_FOB_EXEMPT_TABLE = "originarios_estado_parte_produtor"


def _only_producer_originating_table_present(materiales: Dict[str, Any]) -> bool:
    """True when 'originarios_estado_parte_produtor' is the ONLY material
    table with items among MATERIAL_TABLE_NAMES (the shared table list —
    see models/djo.py) — i.e. every other table
    ('originarios_outros_estados_partes', 'nao_originarios',
    'terceiros_paises_ptc') is empty. This is the one configuration under
    which "Valor FOB (USD)" is allowed to be blank (see
    validate_mandatory_fields)."""
    has_producer_table = bool((materiales.get(_FOB_EXEMPT_TABLE) or {}).get("Items"))
    if not has_producer_table:
        return False
    return not any(
        (materiales.get(table) or {}).get("Items")
        for table in MATERIAL_TABLE_NAMES
        if table != _FOB_EXEMPT_TABLE
    )


def validate_djo_approval(json_djo: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 1 — informational finding, not a pass/fail judgement on the
    DJO's validity: a DJO that hasn't been approved yet is the normal
    state for a first submission, so that case is NOT_APPLICABLE rather
    than FAIL."""
    codigo = json_djo.get("Código da aprovação DJO")
    data_apresentacao = json_djo.get("Data de apresentação")

    if not is_blank(codigo) and not is_blank(data_apresentacao):
        return make_result(
            "djo_approval",
            ValidationStatus.PASS,
            "The DJO has already been approved.",
            details={"codigo_aprovacao": codigo, "data_apresentacao": data_apresentacao},
        )

    return make_result(
        "djo_approval",
        ValidationStatus.NOT_APPLICABLE,
        "The DJO does not have both an approval code and a presentation date yet; it has not been approved.",
    )


def validate_producer_information(json_djo: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 2 — "Razão social do produtor" and "Domicílio legal e parque
    industrial do produtor" are nested dicts (the DJO form splits them
    into sub-fields); we check the sub-field that carries each field's
    essence ("Razão social", "Endereço") — CNPJ/CPF, Inscrição Estadual,
    Tel and E-mail are secondary and their absence alone doesn't fail this
    rule. Documented decision, not a silent guess."""
    razao_social = (json_djo.get("Razão social do produtor") or {}).get("Razão social")
    endereco = (json_djo.get("Domicílio legal e parque industrial do produtor") or {}).get("Endereço")

    missing = []
    if is_blank(razao_social):
        missing.append("Razão social do produtor")
    if is_blank(endereco):
        missing.append("Domicílio legal e parque industrial do produtor")

    if missing:
        return make_result(
            "producer_information",
            ValidationStatus.FAIL,
            "Missing required producer information: " + ", ".join(missing) + ".",
            details={"missing_fields": missing},
        )

    return make_result(
        "producer_information",
        ValidationStatus.PASS,
        "Producer identity and domicile are present.",
    )


def validate_mandatory_fields(json_djo: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 4 — "Valor FOB (USD)", "Unidade de medida",
    "Descrição do processo produtivo" must be present.

    Exception: when 'originarios_estado_parte_produtor' is the ONLY
    material table with items (no other material table has any),
    "Valor FOB (USD)" is not required — that configuration represents a
    DJO with no cross-border materials to price in USD. Any other table
    configuration (including that table combined with another) still
    requires "Valor FOB (USD)", exactly as before."""
    materiales = json_djo.get("Materiales") or {}
    fields_to_check = MANDATORY_TOP_LEVEL_FIELDS
    if _only_producer_originating_table_present(materiales):
        fields_to_check = [f for f in MANDATORY_TOP_LEVEL_FIELDS if f != "Valor FOB (USD)"]

    missing = [f for f in fields_to_check if is_missing_or_not_informado(json_djo.get(f))]

    if missing:
        return make_result(
            "mandatory_fields",
            ValidationStatus.FAIL,
            "Missing required fields: " + ", ".join(missing) + ".",
            details={"missing_fields": missing},
        )

    return make_result(
        "mandatory_fields",
        ValidationStatus.PASS,
        "All mandatory fields are present.",
    )


def validate_preco_fob(materiales: Dict[str, Any]) -> Dict[str, Any]:
    """Rule 8 — json_djo["Materiales"]["Preço FOB"] must not exceed 100%.
    A missing value is MANUAL_VERIFICATION, never silently treated as 0
    (which would read as an automatic PASS) — UNLESS
    'originarios_estado_parte_produtor' is the only material table with
    items (see _only_producer_originating_table_present, shared with
    Rule 4's "Valor FOB (USD)" exception: same underlying scenario, no
    cross-border materials to compute a FOB percentage from), in which
    case a blank "Preço FOB" is NOT_APPLICABLE rather than
    MANUAL_VERIFICATION."""
    raw_value = materiales.get("Preço FOB", "")

    if is_missing_or_not_informado(raw_value):
        if _only_producer_originating_table_present(materiales):
            return make_result(
                "preco_fob",
                ValidationStatus.NOT_APPLICABLE,
                "\"Preço FOB\" is empty, but 'originarios_estado_parte_produtor' is the only material table "
                "present; there is nothing to validate against the 100% ceiling.",
            )
        return make_result(
            "preco_fob",
            ValidationStatus.MANUAL_VERIFICATION,
            "\"Preço FOB\" is missing; cannot validate against the 100% ceiling.",
        )

    value = parse_percentage(raw_value)
    if value is None:
        return make_result(
            "preco_fob",
            ValidationStatus.MANUAL_VERIFICATION,
            f"\"Preço FOB\" value '{raw_value}' could not be parsed as a percentage.",
        )

    if value > 100:
        return make_result(
            "preco_fob",
            ValidationStatus.FAIL,
            f"\"Preço FOB\" is {value}%, which exceeds the 100% ceiling.",
            details={"value": value},
        )

    return make_result(
        "preco_fob",
        ValidationStatus.PASS,
        f"\"Preço FOB\" is {value}%, within the allowed range.",
        details={"value": value},
    )
