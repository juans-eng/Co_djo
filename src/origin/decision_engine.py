# -*- coding: utf-8 -*-
"""
origin/decision_engine.py
============================

The ACE 18 origin-rule decision engine — implements the official
"Fluxograma Regime de Origem do ACE N° 18 (MERCOSUL)" flowchart on top of
(not instead of) `validation.validation_service.validate_djo`. Ported
verbatim from `codigo/djo_validation_core.ipynb`'s origin-rule engine
section.

Three places where the flowchart's own wording had to be mapped onto the
structured `json_djo` fields — documented here explicitly rather than left
implicit:

1. "O produto é totalmente elaborado... (Art. 5°)?" / "...elaborado
   exclusivamente a partir de materiais originários?" — json_djo has no
   field describing the production process against Art. 5 wording, so this
   pair of questions is answered from which originating-material tables
   are non-empty (see `determine_origin_rule_simple`): only
   `originarios_estado_parte_produtor` -> Rule A; both
   `originarios_estado_parte_produtor` and
   `originarios_outros_estados_partes` -> Rule B; any other combination ->
   MANUAL_VERIFICATION (never guessed).
2. "O produto possui apenas as operações... insuficientes (Art. 8°)?" — no
   direct field either; an empty "Descrição do processo produtivo" is
   treated as equivalent to failing this check -> DOES_NOT_CONFER_ORIGIN.
3. "O produto cumpre com a regra de 'De minimis' (Art. 6°)?" — requires a
   numeric tolerance threshold that appears NOWHERE in the flowchart or
   the ACE 18 rule cache; only a boolean (`de_minimis_applies`, from
   `ace18.rule_parser.parse_rule_conditions`) is available. If the rule
   text explicitly excludes de minimis -> DOES_NOT_CONFER_ORIGIN;
   otherwise the question genuinely cannot be answered deterministically
   -> MANUAL_VERIFICATION (never assumed to pass). A real gap in the
   available data, not a design shortcut.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from src.models.validation import is_blank
from src.models.origin_rule import PRODUCT_CATEGORIES
from src.ace18.ncm_service import ACE18Service, normalize_agreement
from src.ace18.rule_parser import parse_rule_conditions
from src.ace18.rule_engine import evaluate_rule_conditions
from src.llm.base import LLMProvider
from src.origin.decision_trace import DecisionTraceBuilder, build_origin_result
from src.validation.validation_service import validate_djo, default_agreement_services


def determine_origin_rule_simple(materiales: Dict[str, Any]) -> Tuple[str, Optional[str], str]:
    """No third-country materials: Rule A/B is decided purely from which
    originating-material tables are non-empty (see module docstring)."""
    tem_estado_parte = len((materiales.get("originarios_estado_parte_produtor") or {}).get("Items", [])) > 0
    tem_outros_estados = len((materiales.get("originarios_outros_estados_partes") or {}).get("Items", [])) > 0

    if tem_estado_parte and not tem_outros_estados:
        return "ORIGIN_RULE_A", "A", "Somente 'originarios_estado_parte_produtor' contém materiais -> Norma de Origem A."
    if tem_estado_parte and tem_outros_estados:
        return "ORIGIN_RULE_B", "B", "'originarios_estado_parte_produtor' e 'originarios_outros_estados_partes' contêm materiais -> Norma de Origem B."
    return "MANUAL_VERIFICATION", None, (
        f"Combinação de tabelas originárias não prevista (estado_parte_produtor="
        f"{'presente' if tem_estado_parte else 'vazia'}, outros_estados_partes="
        f"{'presente' if tem_outros_estados else 'vazia'}) — verificação manual necessária."
    )


def evaluate_origin_rule(
    json_djo: Dict[str, Any],
    product_category: str,
    article14_compliant: Optional[bool] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    agreement: str = "ACE_18",
    agreement_services: Optional[Dict[str, Callable[[], ACE18Service]]] = None,
    llm_provider: Optional[LLMProvider] = None,
) -> Dict[str, Any]:
    """Runs the ACE 18 origin-rule flowchart on top of the existing DJO
    validation. `product_category` (industrial|automotive|game) and, for
    games, `article14_compliant` (True/False/None) are supplied externally
    — never inferred. Pass a pre-computed `validation_result` to avoid
    recomputing it (see `services/djo_service.py`). `llm_provider` is the
    provider-agnostic `LLMProvider` (Gemini/Groq/...) used for Rule 5 (if
    `validation_result` isn't already provided) and for any free-text
    ACE 18 rule condition the deterministic parser can't classify;
    defaults to Gemini via env vars when omitted.
    """
    if agreement_services is None:
        agreement_services = default_agreement_services()
    if llm_provider is None:
        from src.llm.service import create_provider

        llm_provider = create_provider()

    builder = DecisionTraceBuilder()

    # Category is checked FIRST: automotive is entirely out of scope for
    # this flowchart, so it short-circuits before even running the base
    # DJO validation — running (Gemini-backed) validation for a product
    # this flowchart never applies to would be wasted work and would
    # otherwise surface a nonsensical CORRECT_DJO/MANUAL_VERIFICATION for
    # an out-of-scope product.
    if product_category not in PRODUCT_CATEGORIES:
        raise ValueError(f"product_category inválido: {product_category!r}. Esperado um de {PRODUCT_CATEGORIES}.")
    builder.add_step("Qual a categoria do produto?", {"product_category": product_category}, product_category,
                      "Categoria informada externamente (não inferida).")
    if product_category == "automotive":
        return build_origin_result(None, "NOT_APPLICABLE", None, builder.trace,
                                    "Produtos automotivos estão fora do escopo deste fluxograma.")

    if validation_result is None:
        validation_result = validate_djo(json_djo, agreement=agreement, agreement_services=agreement_services, llm_provider=llm_provider)

    builder.add_step(
        "A Declaração Juramentada de Origem contém as informações obrigatórias (validação existente)?",
        {"overall_status": validation_result["overall_status"]},
        validation_result["overall_status"] == "PASS",
        f"Status geral da validação existente: {validation_result['overall_status']}.",
    )
    if validation_result["overall_status"] == "FAIL":
        return build_origin_result(validation_result, "CORRECT_DJO", None, builder.trace,
                                    "A DJO não passou na validação obrigatória existente; corrija a declaração antes de prosseguir.")
    if validation_result["overall_status"] == "MANUAL_VERIFICATION":
        return build_origin_result(validation_result, "MANUAL_VERIFICATION", None, builder.trace,
                                    "A validação existente da DJO exige verificação manual antes de prosseguir.")

    materiales = json_djo.get("Materiales") or {}
    nao_originarios_items = (materiales.get("nao_originarios") or {}).get("Items", [])
    terceiros_items = (materiales.get("terceiros_paises_ptc") or {}).get("Items", [])
    tem_terceiros = bool(nao_originarios_items) or bool(terceiros_items)

    builder.add_step(
        "O produto possui materiais originários de terceiros países ('nao_originarios' / 'terceiros_paises_ptc')?",
        {"nao_originarios_items": len(nao_originarios_items), "terceiros_paises_ptc_items": len(terceiros_items)},
        tem_terceiros,
        f"'nao_originarios' com {len(nao_originarios_items)} item(ns), 'terceiros_paises_ptc' com {len(terceiros_items)} item(ns).",
    )

    if not tem_terceiros:
        status, regra, motivo = determine_origin_rule_simple(materiales)
        builder.add_step(
            "Qual tabela de materiais originários está preenchida?",
            {"originarios_estado_parte_produtor_items": len((materiales.get("originarios_estado_parte_produtor") or {}).get("Items", [])),
             "originarios_outros_estados_partes_items": len((materiales.get("originarios_outros_estados_partes") or {}).get("Items", []))},
            regra, motivo,
        )
        return build_origin_result(validation_result, status, regra, builder.trace, motivo)

    descricao_processo = json_djo.get("Descrição do processo produtivo")
    tem_descricao = not is_blank(descricao_processo)
    builder.add_step(
        "A 'Descrição do processo produtivo' está preenchida?", {"descricao_presente": tem_descricao}, tem_descricao,
        "Sem a descrição do processo produtivo não é possível avaliar se as operações são suficientes para conferir origem (Art. 8°).",
    )
    if not tem_descricao:
        return build_origin_result(validation_result, "DOES_NOT_CONFER_ORIGIN", None, builder.trace,
                                    "Produto não confere origem: descrição do processo produtivo ausente.")

    ncm = json_djo.get("Código NCM")
    ncm_query = None
    if not is_blank(ncm):
        service = agreement_services[normalize_agreement(agreement)]()
        ncm_query = service.query_ncm(ncm)
    ncm_encontrada = bool(ncm_query and ncm_query.get("found"))

    builder.add_step(
        "A NCM está presente na lista de itens sujeitos a requisitos específicos de origem (base ACE 18)?",
        {"Código NCM": ncm}, ncm_encontrada,
        (f"Regra Mercosul encontrada: {ncm_query['mercosul_rule']['raw_rule']!r}." if ncm_encontrada
         else "NCM ausente na DJO ou não encontrada na base ACE 18 (Apêndice II)."),
    )
    if not ncm_encontrada:
        return build_origin_result(validation_result, "DOES_NOT_CONFER_ORIGIN", None, builder.trace,
                                    "Produto não confere origem: NCM não localizada na base de regras do ACE 18.", ncm_rule_info=ncm_query)

    mercosul_rule = ncm_query["mercosul_rule"]
    raw_rule = mercosul_rule["raw_rule"]

    if product_category == "game":
        builder.add_step("O produto cumpre com a regra de Jogos e Sortidos, definida no Artigo 14°?",
                          {"article14_compliant": article14_compliant}, article14_compliant,
                          "Resposta fornecida externamente pelo usuário." if article14_compliant is not None else "Resposta ainda não fornecida pelo usuário.")
        if article14_compliant is None:
            return build_origin_result(validation_result, "MANUAL_VERIFICATION", None, builder.trace,
                                        "Produto classificado como jogo/sortido: informe se cumpre a regra do Artigo 14° antes de prosseguir.",
                                        ncm_rule_info=ncm_query)
        if article14_compliant is False:
            return build_origin_result(validation_result, "DOES_NOT_CONFER_ORIGIN", None, builder.trace,
                                        "Produto não confere origem: não cumpre a regra de Jogos e Sortidos do Artigo 14°.",
                                        ncm_rule_info=ncm_query)

    parsed_rule = parse_rule_conditions(raw_rule)
    tagged_items = (
        [{"_table": "nao_originarios", **item} for item in nao_originarios_items] +
        [{"_table": "terceiros_paises_ptc", **item} for item in terceiros_items]
    )
    complies, condition_traces = evaluate_rule_conditions(parsed_rule, ncm, tagged_items, descricao_processo, llm_provider)

    for ct in condition_traces:
        builder.add_step(
            f"A condição '{ct['condition']}' (operador {parsed_rule['operator']}) é satisfeita?",
            {"raw_rule": raw_rule}, ct["result"],
            ct["reason"] + (f" [avaliado via {llm_provider.name}]" if ct["used_llm"] else ""),
        )

    builder.add_step(
        "O produto cumpre com o requisito de origem (combinação das condições acima)?",
        {"raw_rule": raw_rule, "operator": parsed_rule["operator"]}, complies,
        "Resultado combinado das condições acima segundo o operador lógico identificado." if complies is not None
        else "Uma ou mais condições não puderam ser avaliadas deterministicamente nem via Gemini.",
    )

    if complies is True:
        return build_origin_result(validation_result, "ORIGIN_RULE_C", "C", builder.trace,
                                    "Produto confere origem pela Norma C: cumpre o requisito específico de origem do ACE 18 para sua NCM.",
                                    ncm_rule_info=ncm_query)
    if complies is None:
        return build_origin_result(validation_result, "MANUAL_VERIFICATION", None, builder.trace,
                                    "Não foi possível determinar (nem deterministicamente, nem via Gemini) se o produto cumpre o requisito de origem.",
                                    ncm_rule_info=ncm_query)

    de_minimis_applies = parsed_rule["de_minimis_applies"]
    builder.add_step(
        "O produto cumpre com a regra de 'De minimis' definida no Artigo 6°?",
        {"de_minimis_applies_per_rule_text": de_minimis_applies}, None if de_minimis_applies else False,
        ("A regra explicitamente declara que 'de minimis' não se aplica." if not de_minimis_applies else
         "A regra não exclui 'de minimis', mas não há um percentual de tolerância disponível nos dados para avaliar esta condição deterministicamente."),
    )
    if not de_minimis_applies:
        return build_origin_result(validation_result, "DOES_NOT_CONFER_ORIGIN", None, builder.trace,
                                    "Produto não confere origem: não cumpre o requisito específico de origem, e a regra aplicável exclui explicitamente a tolerância de minimis (Art. 6°).",
                                    ncm_rule_info=ncm_query)
    return build_origin_result(validation_result, "MANUAL_VERIFICATION", None, builder.trace,
                                "Produto não cumpre o requisito específico de origem; a tolerância de minimis (Art. 6°) poderia se aplicar, mas o percentual de tolerância não está disponível nos dados — verificação manual necessária.",
                                ncm_rule_info=ncm_query)
