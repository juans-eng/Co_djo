# -*- coding: utf-8 -*-
"""
llm/groq.py
============

`LLMProvider` implementation backed by the Groq API (OpenAI-compatible
chat-completions endpoint). This is the only module that talks to Groq —
nothing else in `src/` imports `requests` for this purpose.

Reuses the EXACT same prompt templates as `gemini/service.py` (imported,
not copy-pasted) for the four semantic tasks, so switching providers
changes only which model answers the prompt, never the prompt itself or
the expected response shape — the same principle that keeps Rule 5 / the
origin-rule Gemini fallback / both summaries provider-agnostic all the
way up to `services/djo_service.py`.

Configuration is read from environment variables (see `.env.example`):

    GROQ_API_KEY               - required to actually call the API
    GROQ_MODEL                 - optional, defaults to DEFAULT_MODEL below
    GROQ_MIN_INTERVAL_SECONDS  - optional, defaults to 2.0 (rate pacing)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from src.gemini.service import (
    MATERIAL_MATCH_PROMPT_TEMPLATE,
    SUMMARY_PROMPT_TEMPLATE,
    ORIGIN_RULE_SUMMARY_PROMPT_TEMPLATE,
    RULE_CONDITION_PROMPT_TEMPLATE,
    _format_materials_for_prompt,
    _format_rule_condition_materials,
)
from src.llm.base import LLMProvider, LLMProviderError, RateLimiter, is_transient_error_text

GROQ_API_BASE = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_MIN_INTERVAL_SECONDS = 2.0
MAX_RETRIES_ON_TRANSIENT_ERROR = 2
REQUEST_TIMEOUT_SECONDS = 60

# Static fallback for the Streamlit model dropdown when live discovery
# (`list_groq_models`) is unavailable (no key yet, offline, API error) —
# adding/removing a Groq model only ever means editing this list, never
# touching the validation engine. Kept short and current as of this
# backend's last verification against the live Groq /models endpoint;
# the live list is preferred whenever reachable, since Groq retires/adds
# models over time (e.g. the Llama 3.x chat models used here previously
# have since been retired in favor of the models below).
FALLBACK_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
    "groq/compound-mini",
    "qwen/qwen3.8-27b",
]


@dataclass
class GroqConfig:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "GroqConfig":
        return cls(
            api_key=os.environ.get("GROQ_API_KEY") or None,
            model=os.environ.get("GROQ_MODEL", DEFAULT_MODEL),
            min_interval_seconds=float(os.environ.get("GROQ_MIN_INTERVAL_SECONDS", DEFAULT_MIN_INTERVAL_SECONDS)),
        )


def list_groq_models(api_key: Optional[str] = None) -> List[str]:
    """Live model discovery via GET /models. Returns `FALLBACK_MODELS` if
    no key is configured or the request fails for any reason — this is a
    convenience for populating the UI dropdown, never something the
    validation engine depends on."""
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return list(FALLBACK_MODELS)
    try:
        response = requests.get(
            f"{GROQ_API_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        model_ids = sorted(m["id"] for m in data.get("data", []) if m.get("id"))
        return model_ids or list(FALLBACK_MODELS)
    except Exception:
        return list(FALLBACK_MODELS)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self, config: Optional[GroqConfig] = None) -> None:
        self.config = config or GroqConfig.from_env()
        self.model = self.config.model
        self._rate_limiter = RateLimiter(self.config.min_interval_seconds)

    # -----------------------------------------------------------------
    # Shared call path (mirrors gemini/service.py's _generate_content:
    # pacing + bounded retry-with-backoff on transient errors, always
    # raises LLMProviderError - never the raw requests exception)
    # -----------------------------------------------------------------
    def _chat_completion(self, prompt: str, *, json_mode: bool, temperature: float) -> str:
        if not self.config.api_key:
            raise LLMProviderError(
                "GROQ_API_KEY is not set. Add it to your .env file "
                "(see .env.example) before requesting a Groq-backed step."
            )

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Optional[Exception] = None
        for attempt in range(MAX_RETRIES_ON_TRANSIENT_ERROR + 1):
            self._rate_limiter.wait()
            try:
                response = requests.post(
                    f"{GROQ_API_BASE}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
                if response.status_code >= 400:
                    error_text = response.text[:500]
                    if attempt < MAX_RETRIES_ON_TRANSIENT_ERROR and (
                        response.status_code in (429, 503) or is_transient_error_text(error_text)
                    ):
                        time.sleep(2 ** attempt)
                        continue
                    raise LLMProviderError(
                        f"Groq request failed (HTTP {response.status_code}): {error_text}"
                    )
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except LLMProviderError:
                raise
            except Exception as exc:  # network/timeout/parsing errors
                last_exc = exc
                if attempt < MAX_RETRIES_ON_TRANSIENT_ERROR and is_transient_error_text(str(exc)):
                    time.sleep(2 ** attempt)
                    continue
                raise LLMProviderError(f"Groq request failed: {exc}") from exc

        raise LLMProviderError(f"Groq request failed: {last_exc}")

    # -----------------------------------------------------------------
    # The four LLMProvider methods — same prompts/response contracts as
    # GeminiProvider, imported from gemini/service.py.
    # -----------------------------------------------------------------
    def match_process_materials(self, descricao_processo, material_items):
        prompt = MATERIAL_MATCH_PROMPT_TEMPLATE.format(
            descricao_processo=descricao_processo.strip(),
            materiais_listados=_format_materials_for_prompt(material_items),
        )
        text = self._chat_completion(prompt, json_mode=True, temperature=0).strip()
        data = self._parse_json(text)
        if "process_inputs" not in data:
            raise LLMProviderError(f"Groq response is missing required keys. Got: {list(data.keys())}")
        if "all_inputs_found" not in data:
            # Groq's json_object mode only guarantees syntactically valid
            # JSON, not adherence to the schema described in the prompt —
            # some models (observed with Llama models) omit this
            # redundant field even though process_inputs (the actual
            # per-input data) came back correctly. Derive it deterministically
            # from process_inputs rather than fail the whole match on a
            # field that's fully computable from data already present.
            data["all_inputs_found"] = all(bool(i.get("found")) for i in data["process_inputs"])
        return data

    def summarize_validation_result(self, validation_result):
        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            validation_result_json=json.dumps(validation_result, ensure_ascii=False, indent=2)
        )
        text = self._chat_completion(prompt, json_mode=False, temperature=0.2).strip()
        if not text:
            raise LLMProviderError("Groq returned an empty summary.")
        return text

    def summarize_origin_rule_result(self, origin_result):
        payload = {
            "final_status": origin_result.get("final_status"),
            "applicable_origin_rule": origin_result.get("applicable_origin_rule"),
            "ncm_rule_info": origin_result.get("ncm_rule_info"),
            "decision_trace": origin_result.get("decision_trace"),
        }
        prompt = ORIGIN_RULE_SUMMARY_PROMPT_TEMPLATE.format(
            origin_result_json=json.dumps(payload, ensure_ascii=False, indent=2)
        )
        text = self._chat_completion(prompt, json_mode=False, temperature=0.2).strip()
        if not text:
            raise LLMProviderError("Groq returned an empty origin-rule summary.")
        return text

    def evaluate_rule_condition(self, condition_texts, descricao_processo, material_items):
        prompt = RULE_CONDITION_PROMPT_TEMPLATE.format(
            condition_texts="\n".join(f"- {c}" for c in condition_texts),
            descricao_processo=descricao_processo.strip(),
            materiais_listados=_format_rule_condition_materials(material_items),
        )
        text = self._chat_completion(prompt, json_mode=True, temperature=0).strip()
        data = self._parse_json(text)
        if "result" not in data:
            raise LLMProviderError(f"Groq response is missing 'result' key. Got: {list(data.keys())}")
        return data

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                f"Groq response was not valid JSON: {exc}. Raw response: {text[:500]}"
            ) from exc
