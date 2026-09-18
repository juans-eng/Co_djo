# -*- coding: utf-8 -*-
"""
validation/validation_service.py
==================================

Orchestrates all 8 business rules into the single `validate_djo(...)`
result, ported verbatim from `codigo/djo_validation_core.ipynb`'s
"Orchestration — run every rule" section.

    Rule 1 -> djo.validator.validate_djo_approval
    Rule 2 -> djo.validator.validate_producer_information
    Rule 3 -> ace18.ncm_service.validate_ncm_ace18
    Rule 4 -> djo.validator.validate_mandatory_fields
    Rule 5 -> validation.process_validator.validate_process_materials
    Rule 6 -> validation.material_validator.validate_table_completeness
    Rule 7 -> validation.material_validator.validate_table_percentages
    Rule 8 -> djo.validator.validate_preco_fob

`overall_status` = FAIL if any rule failed, else MANUAL_VERIFICATION if any
rule needs manual review, else PASS. NOT_APPLICABLE results never affect it.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.models.validation import ValidationStatus
from src import config
from src.djo.validator import (
    validate_djo_approval,
    validate_producer_information,
    validate_mandatory_fields,
    validate_preco_fob,
)
from src.validation.process_validator import validate_process_materials
from src.validation.material_validator import validate_table_completeness, validate_table_percentages
from src.ace18 import ncm_service
from src.ace18.ncm_service import ACE18Service, normalize_agreement, validate_ncm_ace18
from src.llm.base import LLMProvider


def default_agreement_services() -> Dict[str, Callable[[], ACE18Service]]:
    """The out-of-the-box agreement registry, pointed at the ACE 18
    PDF/cache paths from `config.py`. Pass a different registry to
    `validate_djo(...)` (e.g. in tests, pointed at a fixture cache) to
    override this."""
    return ncm_service.build_agreement_services(config.ACE18_PDF_PATH, config.ACE18_CACHE_PATH)


def _overall_status(validations: List[Dict[str, Any]]) -> ValidationStatus:
    statuses = {v["status"] for v in validations}
    if ValidationStatus.FAIL.value in statuses:
        return ValidationStatus.FAIL
    if ValidationStatus.MANUAL_VERIFICATION.value in statuses:
        return ValidationStatus.MANUAL_VERIFICATION
    return ValidationStatus.PASS


def validate_djo(
    json_djo: Dict[str, Any],
    agreement: str = "ACE_18",
    agreement_services: Optional[Dict[str, Callable[[], ACE18Service]]] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Runs all 8 business rules against `json_djo` and returns the
    structured validation_result:
        {"overall_status": "PASS"|"FAIL"|"MANUAL_VERIFICATION",
         "agreement": "ACE_18", "validations": [ {rule, status, message, ...}, ... ]}

    `agreement` selects which trade-agreement service backs Rule 3
    (`agreement_services`, default `default_agreement_services()`) — the
    extension point for supporting agreements beyond ACE 18 later, without
    changing any other rule. `llm_provider` (see `llm/base.py`) backs
    Rule 5 only — defaults to Gemini via env vars when omitted.
    """
    agreement = normalize_agreement(agreement)
    if agreement_services is None:
        agreement_services = default_agreement_services()
    materiales = json_djo.get("Materiales") or {}

    validations = [
        validate_djo_approval(json_djo),
        validate_producer_information(json_djo),
        validate_ncm_ace18(json_djo, agreement, agreement_services),
        validate_mandatory_fields(json_djo),
        validate_process_materials(json_djo, llm_provider),
        validate_table_completeness(materiales),
        validate_table_percentages(materiales),
        validate_preco_fob(materiales),
    ]

    return {
        "overall_status": _overall_status(validations).value,
        "agreement": agreement,
        "validations": validations,
    }
