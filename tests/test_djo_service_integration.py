# -*- coding: utf-8 -*-
"""src/services/djo_service.py — the top-level orchestrator.

Most tests here mock extraction (no PDF/OCR/network needed) and Rule 5's
Gemini call, so the suite runs offline by default. One real integration
test is marked `@pytest.mark.integration` and skipped automatically unless
LLMWHISPERER_API_KEY and GEMINI_API_KEY are set — run it explicitly with:
    pytest -m integration
"""
import json
import os
from unittest.mock import patch

import pytest

from src.services.djo_service import validate_djo_pdf
from src.djo.extractor import DjoExtractionError


def test_invalid_product_category_returns_processing_error(sample_json_djo):
    result = validate_djo_pdf("irrelevant.pdf", product_category="bicycle")
    assert result["status"] == "PROCESSING_ERROR"
    assert result["error"]["type"] == "InvalidProductCategory"
    # never a raw traceback / exception object
    assert isinstance(result["error"]["message"], str)


def test_extraction_failure_returns_processing_error_not_a_crash():
    with patch("src.services.djo_service.extract_text_from_pdf", side_effect=DjoExtractionError("boom")):
        result = validate_djo_pdf("missing.pdf", product_category="industrial")
    assert result["status"] == "PROCESSING_ERROR"
    assert result["error"]["type"] == "DjoExtractionError"
    assert result["djo"] is None


def test_full_response_is_json_serializable(sample_json_djo):
    """The whole point of the response contract: no Python objects,
    exceptions, or non-serializable types anywhere in the output."""
    # No GEMINI_API_KEY in the test environment -> graceful fallback text,
    # no network call (see LLMProvider docstrings).
    with patch("src.services.djo_service.extract_text_from_pdf", return_value="fake ocr text"), \
         patch("src.services.djo_service.normalizer.normalize", return_value=(sample_json_djo, [])):
        result = validate_djo_pdf("fake.pdf", product_category="industrial")

    # Raises if anything in the tree isn't JSON-serializable.
    serialized = json.dumps(result, ensure_ascii=False)
    assert isinstance(serialized, str)

    assert result["status"] in {"VALID", "CORRECT_DJO", "DOES_NOT_CONFER_ORIGIN", "MANUAL_VERIFICATION", "INVALID_DJO"}
    assert result["djo"]["Número da DJO"] == sample_json_djo["Número da DJO"]
    assert result["origin_rule"] is not None
    assert "decision_trace" in result
    assert result["error"] is None
    assert result["llm_provider"] == "gemini"
    assert result["llm_model"]


def test_unsupported_llm_provider_returns_processing_error_before_extraction():
    with patch("src.services.djo_service.extract_text_from_pdf") as mock_extract:
        result = validate_djo_pdf("irrelevant.pdf", product_category="industrial", llm_provider="chatgpt")

    assert result["status"] == "PROCESSING_ERROR"
    assert result["error"]["type"] == "InvalidLlmProvider"
    mock_extract.assert_not_called()  # fails fast, before any OCR cost


def test_groq_provider_selection_is_reflected_in_response(sample_json_djo):
    # No GROQ_API_KEY in the test environment -> graceful fallback text,
    # no network call — same behavior as the default Gemini path.
    with patch("src.services.djo_service.extract_text_from_pdf", return_value="fake ocr text"), \
         patch("src.services.djo_service.normalizer.normalize", return_value=(sample_json_djo, [])):
        result = validate_djo_pdf("fake.pdf", product_category="industrial", llm_provider="groq", llm_model="openai/gpt-oss-120b")

    assert result["llm_provider"] == "groq"
    assert result["llm_model"] == "openai/gpt-oss-120b"
    assert "groq" in result["summary"]  # graceful-fallback text names the provider that failed


def test_automotive_category_never_reaches_validation(sample_json_djo):
    with patch("src.services.djo_service.extract_text_from_pdf", return_value="fake ocr text"), \
         patch("src.services.djo_service.normalizer.normalize", return_value=(sample_json_djo, [])):
        result = validate_djo_pdf("fake.pdf", product_category="automotive")

    assert result["status"] == "NOT_APPLICABLE"
    assert result["djo_validation"] is None


def test_looks_like_failed_extraction_maps_to_invalid_djo():
    empty_djo = {
        "Código NCM": "",
        "Razão social do produtor": {"Razão social": ""},
        "Descrição do processo produtivo": "",
        "Materiales": {
            "originarios_estado_parte_produtor": {"Items": [], "Somatório": ""},
            "originarios_outros_estados_partes": {"Items": [], "Somatório": ""},
            "nao_originarios": {"Items": [], "Somatório": ""},
            "terceiros_paises_ptc": {"Items": [], "Somatório": ""},
            "Preço FOB": "",
        },
    }
    with patch("src.services.djo_service.extract_text_from_pdf", return_value="garbage"), \
         patch("src.services.djo_service.normalizer.normalize", return_value=(empty_djo, [])):
        result = validate_djo_pdf("fake.pdf", product_category="industrial")

    assert result["status"] == "INVALID_DJO"


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.environ.get("LLMWHISPERER_API_KEY") and os.environ.get("GEMINI_API_KEY")),
    reason="requires real LLMWHISPERER_API_KEY and GEMINI_API_KEY",
)
def test_real_end_to_end_pipeline_against_sample_pdf():
    """The one test in this suite that actually calls the real LLM
    Whisperer + Gemini APIs, against a real sample DJO PDF. Skipped by
    default; run with `pytest -m integration` once .env is configured."""
    pdf_path = os.path.join(
        os.path.dirname(__file__), "..", "dados", "djos", "DJO 3 (1).pdf",
    )
    result = validate_djo_pdf(pdf_path, product_category="industrial")

    assert result["status"] != "PROCESSING_ERROR", result.get("error")
    assert result["djo"]["Número da DJO"]
    json.dumps(result, ensure_ascii=False)  # must stay fully serializable
