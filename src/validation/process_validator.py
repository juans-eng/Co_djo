# -*- coding: utf-8 -*-
"""
validation/process_validator.py
=================================

Rule 5 — process-description materials vs. material-table descriptions.
Ported verbatim from `codigo/djo_validation_core.ipynb`. The only rule that
genuinely needs language understanding: does every input named in the
free-text "Descrição do processo produtivo" show up (exactly or with
reasonable semantic equivalence) in one of the material tables' Descrição
cells? Delegated to whichever `LLMProvider` is configured (see
`llm/base.py`) — this module has no knowledge of Gemini/Groq specifically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models.validation import ValidationStatus, make_result, is_blank
from src.models.djo import MATERIAL_TABLE_NAMES
from src.llm.base import LLMProvider, LLMProviderError
from src.llm.service import create_provider


def collect_material_descriptions(materiales: Dict[str, Any]) -> List[Dict[str, str]]:
    """Flattens every material table's Items into a
    [{"table": <table_name>, "descricao": <Descrição value>}, ...] list,
    the input Rule 5's Gemini prompt is built from."""
    collected = []
    for table_name in MATERIAL_TABLE_NAMES:
        table = materiales.get(table_name) or {}
        for item in table.get("Items", []):
            descricao = item.get("Descrição", "")
            if not is_blank(descricao):
                collected.append({"table": table_name, "descricao": descricao})
    return collected


def validate_process_materials(
    json_djo: Dict[str, Any],
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Rule 5. If the process description is empty, Rule 4 already reports
    that as a missing mandatory field — this reports NOT_APPLICABLE
    instead of a second, redundant failure. If the LLM call itself fails
    (any provider), the result is MANUAL_VERIFICATION, never an assumed
    PASS/FAIL. `llm_provider` defaults to Gemini (via env vars) when
    omitted, exactly like the previous Gemini-only signature."""
    descricao_processo = json_djo.get("Descrição do processo produtivo")
    if is_blank(descricao_processo):
        return make_result(
            "process_materials",
            ValidationStatus.NOT_APPLICABLE,
            "\"Descrição do processo produtivo\" is empty; nothing to analyze.",
        )

    materiales = json_djo.get("Materiales") or {}
    material_items = collect_material_descriptions(materiales)
    llm_provider = llm_provider or create_provider()

    try:
        llm_result = llm_provider.match_process_materials(descricao_processo, material_items)
    except LLMProviderError as exc:
        return make_result(
            "process_materials",
            ValidationStatus.MANUAL_VERIFICATION,
            f"Could not run automated material matching ({llm_provider.name}): {exc}",
        )

    all_found = bool(llm_result.get("all_inputs_found"))
    not_found = [i["input"] for i in llm_result.get("process_inputs", []) if not i.get("found")]

    status = ValidationStatus.PASS if all_found else ValidationStatus.FAIL
    message = (
        "All process inputs are represented in the material tables."
        if all_found
        else "Some process inputs were not found in the material tables: " + ", ".join(not_found) + "."
    )

    return make_result("process_materials", status, message, details=llm_result)
