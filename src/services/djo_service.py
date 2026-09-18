# -*- coding: utf-8 -*-
"""
services/djo_service.py
=========================

The single high-level entry point the frontend/future API should call.
Orchestrates the complete pipeline end to end:

    PDF
     -> djo.extractor            (OCR)
     -> djo.normalizer           (structured json_djo)
     -> validation.validation_service.validate_djo   (Rules 1-8)
     -> origin.decision_engine.evaluate_origin_rule   (ACE 18 flowchart)
     -> llm.service (Gemini or Groq)                  (summaries)
     -> structured, JSON-serializable response

No internal Python objects, Docling objects, DataFrames, or exceptions are
exposed to the caller — every code path either returns the response dict
below or raises nothing (errors are captured into
`response["error"]` / `status: PROCESSING_ERROR`, see `validate_djo_pdf`).

This module is also where the LLM provider is selected (`llm_provider` /
`llm_model` string arguments — "gemini"/"groq" and an optional model
name) and turned into a concrete `LLMProvider` instance (see
`llm/service.py`) that's threaded through the rest of the pipeline. This
is the only place a provider *name* is resolved; everything downstream
(Rule 5, the origin-rule engine) only ever sees the `LLMProvider`
interface.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.djo.extractor import extract_text_from_pdf, DjoExtractionError, ExtractorConfig
from src.djo import normalizer
from src.models.validation import is_blank
from src.models.origin_rule import PRODUCT_CATEGORIES
from src.validation.validation_service import validate_djo as run_validation, default_agreement_services
from src.origin.decision_engine import evaluate_origin_rule
from src.llm.base import LLMProvider, LLMProviderError
from src.llm.service import create_provider, DEFAULT_PROVIDER


def _looks_like_failed_extraction(json_djo: Dict[str, Any]) -> bool:
    """Heuristic for "the PDF probably isn't a DJO, or OCR/normalization
    produced nothing usable": every field a validation rule could latch
    onto is blank. Distinct from a DJO that's just missing ONE field
    (which correctly surfaces as FAIL via the normal Rule 2/3/4 path) —
    this only fires when there is essentially nothing to validate at all.
    """
    razao_social = (json_djo.get("Razão social do produtor") or {}).get("Razão social")
    return (
        is_blank(json_djo.get("Código NCM"))
        and is_blank(razao_social)
        and is_blank(json_djo.get("Descrição do processo produtivo"))
    )


def _generate_validation_summary(validation_result: Dict[str, Any], llm_provider: LLMProvider) -> str:
    """Same graceful-fallback pattern used throughout: never lets an LLM
    failure (whichever provider is configured) break the response, falls
    back to a short placeholder."""
    try:
        return llm_provider.summarize_validation_result(validation_result)
    except LLMProviderError as exc:
        return f"(Resumo de validação indisponível via {llm_provider.name}: {exc})"


def _generate_origin_summary(origin_result: Dict[str, Any], llm_provider: LLMProvider) -> str:
    try:
        return llm_provider.summarize_origin_rule_result(origin_result)
    except LLMProviderError as exc:
        return f"(Resumo da norma de origem indisponível via {llm_provider.name}: {exc})\n\n{origin_result['simple_explanation']}"


def _map_top_level_status(json_djo: Dict[str, Any], origin_result: Dict[str, Any]) -> str:
    if _looks_like_failed_extraction(json_djo):
        return "INVALID_DJO"
    final_status = origin_result["final_status"]
    if final_status.startswith("ORIGIN_RULE_"):
        return "VALID"
    return final_status  # CORRECT_DJO | DOES_NOT_CONFER_ORIGIN | MANUAL_VERIFICATION | NOT_APPLICABLE


def _build_response(
    status: str,
    json_djo: Optional[Dict[str, Any]] = None,
    validation_result: Optional[Dict[str, Any]] = None,
    origin_result: Optional[Dict[str, Any]] = None,
    summary: str = "",
    error: Optional[Dict[str, str]] = None,
    product_category: Optional[str] = None,
    agreement: Optional[str] = None,
    llm_provider_name: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    ncm_rule_info = (origin_result or {}).get("ncm_rule_info")
    decision_trace = (origin_result or {}).get("decision_trace", [])

    return {
        "status": status,
        "product_category": product_category,
        "agreement": agreement,
        "llm_provider": llm_provider_name,
        "llm_model": llm_model,
        "djo": json_djo,
        "djo_validation": validation_result,
        "ncm": {
            "code": (json_djo or {}).get("Código NCM", ""),
            "found_in_ace18": bool(ncm_rule_info and ncm_rule_info.get("found")),
            "match_info": (ncm_rule_info or {}).get("match_info"),
        } if json_djo is not None else None,
        "mercosul_rule": (ncm_rule_info or {}).get("mercosul_rule"),
        "origin_rule": {
            "applicable_rule": (origin_result or {}).get("applicable_origin_rule"),
            "status": (origin_result or {}).get("final_status"),
            "decision_trace": decision_trace,
            "explanation": (origin_result or {}).get("simple_explanation"),
        } if origin_result is not None else None,
        "decision_trace": decision_trace,
        "summary": summary,
        "error": error,
    }


def validate_djo_pdf(
    pdf_path: str,
    product_category: str,
    article14_compliant: Optional[bool] = None,
    agreement: str = "ACE_18",
    extractor_config: Optional[ExtractorConfig] = None,
    llm_provider: str = DEFAULT_PROVIDER,
    llm_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs the complete DJO validation pipeline for one PDF file.

    Args:
        pdf_path: path to the DJO PDF to validate.
        product_category: "industrial" | "game" | "automotive" — required,
            never inferred.
        article14_compliant: only meaningful when product_category=="game";
            True/False (the user's answer to the Games & Assortments /
            Art. 14 question) or None if not yet answered (the response's
            origin_rule.status will then be MANUAL_VERIFICATION, asking
            for it).
        agreement: which trade agreement backs the NCM lookup — only
            "ACE_18" (and aliases) is implemented today.
        llm_provider: "gemini" | "groq" (case-insensitive) — which LLM
            backs Rule 5, any free-text ACE 18 rule condition, and both
            summaries. Defaults to Gemini. The corresponding API key
            (GEMINI_API_KEY / GROQ_API_KEY) is read from the environment
            only — never accepted here or exposed in the response.
        llm_model: optional model override for the selected provider
            (e.g. "llama-3.3-70b-versatile" for Groq). Defaults to that
            provider's own default model when omitted.

    Returns a JSON-serializable dict (see module docstring for shape).
    Never raises: any failure — including an unrecognized `llm_provider`
    or `llm_model` — is captured into `status: "PROCESSING_ERROR"` with a
    clean, human-readable `error` object — no raw exception or traceback
    reaches the caller.
    """
    if product_category not in PRODUCT_CATEGORIES:
        return _build_response(
            "PROCESSING_ERROR",
            product_category=product_category,
            agreement=agreement,
            error={
                "type": "InvalidProductCategory",
                "message": f"product_category must be one of {sorted(PRODUCT_CATEGORIES)}, got {product_category!r}.",
            },
        )

    try:
        provider = create_provider(llm_provider, llm_model)
    except LLMProviderError as exc:
        return _build_response(
            "PROCESSING_ERROR",
            product_category=product_category,
            agreement=agreement,
            llm_provider_name=llm_provider,
            llm_model=llm_model,
            error={"type": "InvalidLlmProvider", "message": str(exc)},
        )

    try:
        extracted_text = extract_text_from_pdf(pdf_path, config=extractor_config)
    except DjoExtractionError as exc:
        return _build_response(
            "PROCESSING_ERROR",
            product_category=product_category,
            agreement=agreement,
            llm_provider_name=provider.name,
            llm_model=provider.model,
            error={"type": "DjoExtractionError", "message": str(exc)},
        )

    try:
        json_djo, _debug_blocks = normalizer.normalize(extracted_text)
    except Exception as exc:  # normalization is pure text processing; any failure here is unexpected
        return _build_response(
            "PROCESSING_ERROR",
            product_category=product_category,
            agreement=agreement,
            llm_provider_name=provider.name,
            llm_model=provider.model,
            error={"type": "DjoNormalizationError", "message": f"Failed to interpret the extracted text: {exc}"},
        )

    try:
        agreement_services = default_agreement_services()

        origin_result = evaluate_origin_rule(
            json_djo,
            product_category=product_category,
            article14_compliant=article14_compliant,
            agreement=agreement,
            agreement_services=agreement_services,
            llm_provider=provider,
        )
        validation_result = origin_result["validation_result"]

        if validation_result is not None:
            validation_summary = _generate_validation_summary(validation_result, provider)
            origin_summary = _generate_origin_summary(origin_result, provider)
            summary = f"{validation_summary}\n\n{origin_summary}"
        else:
            # automotive: base validation never ran (out of scope)
            validation_summary = "(Resumo não gerado: fluxograma não aplicável a este produto.)"
            summary = f"{validation_summary}\n\n{origin_result['simple_explanation']}"

        top_status = _map_top_level_status(json_djo, origin_result)

        return _build_response(
            top_status,
            json_djo=json_djo,
            validation_result=validation_result,
            origin_result=origin_result,
            summary=summary,
            product_category=product_category,
            agreement=agreement,
            llm_provider_name=provider.name,
            llm_model=provider.model,
        )
    except Exception as exc:  # last-resort safety net: never let a raw exception/traceback escape
        return _build_response(
            "PROCESSING_ERROR",
            json_djo=json_djo,
            product_category=product_category,
            agreement=agreement,
            llm_provider_name=provider.name,
            llm_model=provider.model,
            error={"type": type(exc).__name__, "message": f"Unexpected error while validating the DJO: {exc}"},
        )
