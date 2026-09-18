# -*- coding: utf-8 -*-
"""LLM provider abstraction — src/llm/*.py. No real network calls; Groq's
HTTP layer is mocked via unittest.mock.patch("requests.post"/"requests.get").
Gemini's own SDK-level behavior is already covered by test_gemini_service.py
— here we only verify GeminiProvider correctly adapts it to LLMProvider."""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.llm.base import LLMProvider, LLMProviderError
from src.llm.service import create_provider, list_available_models, SUPPORTED_PROVIDERS
from src.llm.gemini import GeminiProvider
from src.llm.groq import GroqProvider, GroqConfig, list_groq_models, FALLBACK_MODELS


# ---------------------------------------------------------------------------
# Factory (llm/service.py)
# ---------------------------------------------------------------------------

def test_create_provider_defaults_to_gemini():
    provider = create_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.name == "gemini"


def test_create_provider_is_case_insensitive():
    provider = create_provider("GROQ")
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"


def test_create_provider_applies_model_override():
    provider = create_provider("groq", model="openai/gpt-oss-20b")
    assert provider.model == "openai/gpt-oss-20b"


def test_create_provider_rejects_unsupported_name():
    with pytest.raises(LLMProviderError):
        create_provider("chatgpt")


def test_supported_providers_are_gemini_and_groq():
    assert set(SUPPORTED_PROVIDERS) == {"gemini", "groq"}


def test_list_available_models_unknown_provider_returns_empty_list():
    assert list_available_models("chatgpt") == []


def test_list_available_models_gemini_returns_static_list():
    models = list_available_models("gemini")
    assert "gemini-3.6-flash" in models


# ---------------------------------------------------------------------------
# Every concrete provider implements the full interface
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [create_provider("gemini"), create_provider("groq")])
def test_providers_implement_llm_provider_interface(provider):
    assert isinstance(provider, LLMProvider)
    assert provider.name in ("gemini", "groq")
    assert provider.model


# ---------------------------------------------------------------------------
# GeminiProvider — translates GeminiServiceError -> LLMProviderError
# ---------------------------------------------------------------------------

def test_gemini_provider_wraps_service_error_as_llm_provider_error():
    from src.gemini.service import GeminiConfig

    provider = GeminiProvider(GeminiConfig(api_key=None))  # forces a GeminiServiceError, no network
    with pytest.raises(LLMProviderError):
        provider.summarize_validation_result({"overall_status": "PASS", "validations": []})


# ---------------------------------------------------------------------------
# GroqProvider
# ---------------------------------------------------------------------------

def test_groq_provider_raises_without_api_key():
    provider = GroqProvider(GroqConfig(api_key=None))
    with pytest.raises(LLMProviderError, match="GROQ_API_KEY"):
        provider.summarize_validation_result({"overall_status": "PASS", "validations": []})


def _mock_groq_response(content: str, status_code: int = 200) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.text = content
    response.json.return_value = {"choices": [{"message": {"content": content}}]}
    return response


def test_groq_provider_summarize_validation_result_returns_text():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response("Resumo de teste.")):
        result = provider.summarize_validation_result({"overall_status": "PASS", "validations": []})
    assert result == "Resumo de teste."


def test_groq_provider_match_process_materials_parses_json():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    payload = '{"process_inputs": [], "all_inputs_found": true}'
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response(payload)):
        result = provider.match_process_materials("processo de teste", [])
    assert result["all_inputs_found"] is True


def test_groq_provider_derives_all_inputs_found_when_model_omits_it():
    # Regression: Groq's json_object mode only guarantees syntactically
    # valid JSON, not schema adherence — some models (observed with Llama)
    # return process_inputs but drop the redundant all_inputs_found field.
    # This must be derived, never raised as a missing-key error.
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    payload = '{"process_inputs": [{"input": "tecido", "found": true, "matched_description": "TECIDO X", "table": "nao_originarios"}]}'
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response(payload)):
        result = provider.match_process_materials("processo de teste", [])
    assert result["all_inputs_found"] is True

    payload_with_miss = '{"process_inputs": [{"input": "tecido", "found": false, "matched_description": null, "table": null}]}'
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response(payload_with_miss)):
        result = provider.match_process_materials("processo de teste", [])
    assert result["all_inputs_found"] is False


def test_groq_provider_raises_when_process_inputs_itself_is_missing():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response('{"all_inputs_found": true}')):
        with pytest.raises(LLMProviderError):
            provider.match_process_materials("processo de teste", [])


def test_groq_provider_malformed_json_raises_llm_provider_error():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response("not json")):
        with pytest.raises(LLMProviderError):
            provider.match_process_materials("processo", [])


def test_groq_provider_http_error_raises_llm_provider_error_without_retry_when_not_transient():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    with patch("src.llm.groq.requests.post", return_value=_mock_groq_response("invalid model", status_code=400)) as mock_post:
        with pytest.raises(LLMProviderError, match="400"):
            provider.summarize_validation_result({"overall_status": "PASS", "validations": []})
    assert mock_post.call_count == 1  # 400 is not transient -> no retry


def test_groq_provider_retries_on_transient_error_then_succeeds():
    provider = GroqProvider(GroqConfig(api_key="fake-key", min_interval_seconds=0))
    responses = [_mock_groq_response("rate limited", status_code=429), _mock_groq_response("Resumo ok.")]
    with patch("src.llm.groq.requests.post", side_effect=responses), patch("src.llm.groq.time.sleep"):
        result = provider.summarize_validation_result({"overall_status": "PASS", "validations": []})
    assert result == "Resumo ok."


def test_list_groq_models_falls_back_when_no_api_key():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GROQ_API_KEY", None)
        assert list_groq_models(api_key=None) == list(FALLBACK_MODELS)


def test_list_groq_models_falls_back_on_request_failure():
    with patch("src.llm.groq.requests.get", side_effect=ConnectionError("no network")):
        assert list_groq_models(api_key="fake-key") == list(FALLBACK_MODELS)


def test_list_groq_models_parses_live_response():
    response = MagicMock()
    response.json.return_value = {"data": [{"id": "model-b"}, {"id": "model-a"}]}
    with patch("src.llm.groq.requests.get", return_value=response):
        assert list_groq_models(api_key="fake-key") == ["model-a", "model-b"]
