# -*- coding: utf-8 -*-
"""src/gemini/service.py — error handling with mocks. No real API key or
network access required for any test in this file."""
import json
from unittest.mock import MagicMock, patch

import pytest

from src.gemini import service as gemini_service
from src.gemini.service import GeminiConfig, GeminiServiceError


def test_missing_api_key_raises_gemini_service_error():
    config = GeminiConfig(api_key=None)
    with pytest.raises(GeminiServiceError, match="GEMINI_API_KEY is not set"):
        gemini_service.summarize_validation_result({"overall_status": "PASS", "validations": []}, config=config)


def test_match_process_materials_parses_valid_response():
    fake_response = MagicMock()
    fake_response.text = json.dumps({
        "process_inputs": [{"input": "tecido", "found": True, "matched_description": "TECIDO X", "table": "nao_originarios"}],
        "all_inputs_found": True,
    })
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    config = GeminiConfig(api_key="fake-key", min_interval_seconds=0)
    with patch("src.gemini.service.genai.Client", return_value=fake_client):
        result = gemini_service.match_process_materials("processo de teste", [{"table": "nao_originarios", "descricao": "TECIDO X"}], config=config)

    assert result["all_inputs_found"] is True


def test_malformed_json_response_raises_gemini_service_error():
    fake_response = MagicMock()
    fake_response.text = "not json at all"
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    config = GeminiConfig(api_key="fake-key", min_interval_seconds=0)
    with patch("src.gemini.service.genai.Client", return_value=fake_client):
        with pytest.raises(GeminiServiceError, match="not valid JSON"):
            gemini_service.match_process_materials("processo", [], config=config)


def test_missing_required_keys_raises_gemini_service_error():
    fake_response = MagicMock()
    fake_response.text = json.dumps({"unexpected": "shape"})
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = fake_response

    config = GeminiConfig(api_key="fake-key", min_interval_seconds=0)
    with patch("src.gemini.service.genai.Client", return_value=fake_client):
        with pytest.raises(GeminiServiceError, match="missing required keys"):
            gemini_service.match_process_materials("processo", [], config=config)


def test_transient_error_is_retried_then_succeeds():
    fake_response = MagicMock()
    fake_response.text = json.dumps({"result": True, "reasoning": "ok"})
    fake_client = MagicMock()
    # First call raises a transient 503, second call succeeds.
    fake_client.models.generate_content.side_effect = [Exception("503 UNAVAILABLE"), fake_response]

    config = GeminiConfig(api_key="fake-key", min_interval_seconds=0)
    with patch("src.gemini.service.genai.Client", return_value=fake_client), \
         patch("src.gemini.service.time.sleep", return_value=None):  # skip the real backoff delay
        result = gemini_service.evaluate_rule_condition(["Reação química"], "processo", [], config=config)

    assert result["result"] is True
    assert fake_client.models.generate_content.call_count == 2


def test_non_transient_error_is_not_retried():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = Exception("401 Unauthorized")

    config = GeminiConfig(api_key="fake-key", min_interval_seconds=0)
    with patch("src.gemini.service.genai.Client", return_value=fake_client):
        with pytest.raises(GeminiServiceError):
            gemini_service.summarize_validation_result({"overall_status": "PASS", "validations": []}, config=config)

    assert fake_client.models.generate_content.call_count == 1


def test_summarize_origin_rule_result_falls_back_gracefully_on_failure(sample_json_djo):
    """Exercises the caller-side fallback pattern used throughout the
    backend (services/djo_service.py, origin engine's LLM step): an
    LLMProviderError must never propagate as a raw exception."""
    from src.services.djo_service import _generate_origin_summary
    from src.llm.gemini import GeminiProvider

    origin_result = {
        "final_status": "ORIGIN_RULE_C",
        "applicable_origin_rule": "C",
        "decision_trace": [],
        "simple_explanation": "fallback text",
    }
    provider = GeminiProvider(GeminiConfig(api_key=None))  # forces LLMProviderError, no network
    summary = _generate_origin_summary(origin_result, provider)
    assert "fallback text" in summary
