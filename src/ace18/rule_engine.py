# -*- coding: utf-8 -*-
"""
ace18/rule_engine.py
======================

Evaluates a parsed ACE 18 rule (`rule_parser.parse_rule_conditions`'s
output) against a specific product + its non-originating/third-country
material items. Ported verbatim from `codigo/djo_validation_core.ipynb`'s
origin-rule engine section.

Deterministic conditions (MP / MSP / MAXMNO / EXCLUDED_POSITIONS) are
evaluated first; the configured `LLMProvider` (see `llm/base.py`) is
invoked ONLY for the remaining free-text (UNKNOWN) conditions — e.g.
"Reação química", "Processo biotecnológico" — and only when the
deterministic ones couldn't already decide the outcome alone
(short-circuit: an OR with a deterministic True, or an AND with a
deterministic False, never touches the LLM). All remaining UNKNOWN
segments are bundled into a single LLM call rather than one call per
segment. This module has no knowledge of which provider (Gemini/Groq) is
actually configured.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.validation import parse_percentage
from src.llm.base import LLMProvider, LLMProviderError
from src.llm.service import create_provider


def _first_n_digits(ncm: Any, n: int) -> Optional[str]:
    digits = re.sub(r"\D", "", str(ncm or ""))
    return digits[:n] if len(digits) >= n else None


def evaluate_mp_condition(product_ncm: str, tables_items: List[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """MP (Mudança de Partida / change of tariff heading): the product's
    4-digit heading must differ from the 4-digit heading of every
    non-originating/third-country material."""
    produto_4 = _first_n_digits(product_ncm, 4)
    if produto_4 is None:
        return None, "Código NCM do produto inválido/ausente; não é possível validar MP."

    conflitos = [item.get("NCM/SH") for item in tables_items if _first_n_digits(item.get("NCM/SH"), 4) == produto_4]
    inconclusivos = [item.get("NCM/SH") for item in tables_items if _first_n_digits(item.get("NCM/SH"), 4) is None]

    if inconclusivos:
        return None, f"NCM(s) inválido(s)/ausente(s) nos materiais não originários: {inconclusivos}; MP não pôde ser validada com certeza."
    if conflitos:
        return False, f"Material(is) não originário(s) compartilham a partida (4 dígitos) do produto ({produto_4}): {conflitos}."
    return True, f"Nenhum material não originário compartilha a partida (4 dígitos) do produto ({produto_4})."


def evaluate_msp_condition(product_ncm: str, tables_items: List[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """MSP (Mudança de Subpartida / change of tariff subheading): same as
    MP but comparing 6-digit subheadings."""
    produto_6 = _first_n_digits(product_ncm, 6)
    if produto_6 is None:
        return None, "Código NCM do produto inválido/ausente; não é possível validar MSP."

    conflitos = [item.get("NCM/SH") for item in tables_items if _first_n_digits(item.get("NCM/SH"), 6) == produto_6]
    inconclusivos = [item.get("NCM/SH") for item in tables_items if _first_n_digits(item.get("NCM/SH"), 6) is None]

    if inconclusivos:
        return None, f"NCM(s) inválido(s)/ausente(s) nos materiais não originários: {inconclusivos}; MSP não pôde ser validada com certeza."
    if conflitos:
        return False, f"Material(is) não originário(s) compartilham a subpartida (6 dígitos) do produto ({produto_6}): {conflitos}."
    return True, f"Nenhum material não originário compartilha a subpartida (6 dígitos) do produto ({produto_6})."


def evaluate_maxmno_condition(max_percentage: float, tables_items: List[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """MaxMNO x%: the sum of '% s/Valor FOB' across the non-originating /
    third-country tables must not exceed `max_percentage`."""
    valores = [parse_percentage(item.get("% s/Valor FOB")) for item in tables_items]
    if any(v is None for v in valores):
        return None, "Um ou mais valores de '% s/Valor FOB' não puderam ser interpretados; não foi possível calcular o total para MaxMNO."
    total = round(sum(valores), 2)
    if total <= max_percentage:
        return True, f"Soma de '% s/Valor FOB' dos materiais não originários ({total}%) está dentro do limite MaxMNO ({max_percentage}%)."
    return False, f"Soma de '% s/Valor FOB' dos materiais não originários ({total}%) excede o limite MaxMNO ({max_percentage}%)."


def evaluate_excluded_positions_condition(range_start: int, range_end: int, tables_items: List[Dict[str, Any]]) -> Tuple[Optional[bool], str]:
    """'exceto das posições X a Y': none of the non-originating/third-country
    materials' 4-digit headings may fall inside [range_start, range_end]."""
    inconclusivos = [item.get("NCM/SH") for item in tables_items if _first_n_digits(item.get("NCM/SH"), 4) is None]
    if inconclusivos:
        return None, f"NCM(s) inválido(s)/ausente(s): {inconclusivos}; a exclusão de posições não pôde ser validada com certeza."

    violacoes = [
        item.get("NCM/SH") for item in tables_items
        if range_start <= int(_first_n_digits(item.get("NCM/SH"), 4)) <= range_end
    ]
    if violacoes:
        return False, f"Material(is) não originário(s) pertencem à faixa excluída {range_start}-{range_end}: {violacoes}."
    return True, f"Nenhum material não originário pertence à faixa excluída de posições {range_start}-{range_end}."


def evaluate_unknown_condition_with_llm(
    condition_texts: List[str],
    descricao_processo: str,
    tables_items: List[Dict[str, Any]],
    llm_provider: Optional[LLMProvider] = None,
) -> Tuple[Optional[bool], str]:
    """Delegates one or more UNKNOWN (free-text) rule conditions to the
    configured LLM provider. Returns (result, reason); result is
    True/False/None (None = the provider call failed or itself returned
    an inconclusive answer)."""
    material_items_for_prompt = [
        {
            "table": item.get("_table", "?"),
            "ncm": item.get("NCM/SH", ""),
            "pais_origem": item.get("País origem", ""),
            "descricao": item.get("Descrição", ""),
        }
        for item in tables_items
    ]
    llm_provider = llm_provider or create_provider()
    try:
        resultado = llm_provider.evaluate_rule_condition(
            condition_texts, descricao_processo, material_items_for_prompt,
        )
        return resultado.get("result"), resultado.get("reasoning", "")
    except LLMProviderError as exc:
        return None, f"Falha ao consultar o provedor LLM ({llm_provider.name}) para avaliação semântica: {exc}"


def evaluate_rule_conditions(
    parsed_rule: Dict[str, Any],
    product_ncm: str,
    tables_items: List[Dict[str, Any]],
    descricao_processo: str,
    llm_provider: Optional[LLMProvider] = None,
) -> Tuple[Optional[bool], List[Dict[str, Any]]]:
    """Evaluates every condition in `parsed_rule`, combined via its
    operator. Returns (result, condition_traces): result is True/False/None
    (None = inconclusive -> MANUAL_VERIFICATION upstream). Each entry in
    condition_traces is {"condition", "type", "result", "reason", "used_llm"}."""
    operator = parsed_rule["operator"]
    conditions = parsed_rule["conditions"]

    traces: List[Dict[str, Any]] = []
    deterministic_results: List[Optional[bool]] = []

    for cond in conditions:
        if cond["type"] == "MP":
            result, reason = evaluate_mp_condition(product_ncm, tables_items)
        elif cond["type"] == "MSP":
            result, reason = evaluate_msp_condition(product_ncm, tables_items)
        elif cond["type"] == "MAXMNO":
            result, reason = evaluate_maxmno_condition(cond["max_percentage"], tables_items)
        elif cond["type"] == "EXCLUDED_POSITIONS":
            result, reason = evaluate_excluded_positions_condition(cond["range_start"], cond["range_end"], tables_items)
        else:
            continue  # UNKNOWN: handled below, only if the operator still needs it
        traces.append({"condition": cond["text"], "type": cond["type"], "result": result, "reason": reason, "used_llm": False})
        deterministic_results.append(result)

    # Short-circuit: can the operator already be decided from the
    # deterministic conditions alone, without touching UNKNOWN/Gemini ones?
    if operator == "OR" and any(r is True for r in deterministic_results):
        return True, traces
    if operator in ("AND", "SINGLE") and any(r is False for r in deterministic_results):
        return False, traces

    unknown_conditions = [c for c in conditions if c["type"] == "UNKNOWN"]

    if not unknown_conditions:
        if any(r is None for r in deterministic_results):
            return None, traces
        return (any(deterministic_results) if operator == "OR" else all(deterministic_results)), traces

    llm_result, llm_reason = evaluate_unknown_condition_with_llm(
        [c["text"] for c in unknown_conditions], descricao_processo, tables_items, llm_provider,
    )
    traces.append({
        "condition": " | ".join(c["text"] for c in unknown_conditions),
        "type": "UNKNOWN", "result": llm_result, "reason": llm_reason, "used_llm": True,
    })

    if llm_result is None or any(r is None for r in deterministic_results):
        return None, traces

    if operator == "OR":
        combined = any(deterministic_results) or llm_result
    else:  # AND / SINGLE
        combined = all(deterministic_results) and llm_result

    return combined, traces
