# -*- coding: utf-8 -*-
"""
llm/gemini.py
==============

`LLMProvider` implementation backed by `gemini/service.py` (the only
module that touches the `google.genai` SDK). This class does no Gemini
work itself — it is a thin adapter translating the provider-agnostic
`LLMProvider` interface onto the four existing, already-tested Gemini
functions, and translating `GeminiServiceError` into the
provider-agnostic `LLMProviderError` so callers never need to know which
provider they're talking to.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.gemini import service as gemini_service
from src.llm.base import LLMProvider, LLMProviderError


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, config: Optional["gemini_service.GeminiConfig"] = None) -> None:
        self.config = config or gemini_service.GeminiConfig.from_env()
        self.model = self.config.model

    def match_process_materials(self, descricao_processo, material_items):
        try:
            return gemini_service.match_process_materials(descricao_processo, material_items, config=self.config)
        except gemini_service.GeminiServiceError as exc:
            raise LLMProviderError(str(exc)) from exc

    def summarize_validation_result(self, validation_result):
        try:
            return gemini_service.summarize_validation_result(validation_result, config=self.config)
        except gemini_service.GeminiServiceError as exc:
            raise LLMProviderError(str(exc)) from exc

    def summarize_origin_rule_result(self, origin_result):
        try:
            return gemini_service.summarize_origin_rule_result(origin_result, config=self.config)
        except gemini_service.GeminiServiceError as exc:
            raise LLMProviderError(str(exc)) from exc

    def evaluate_rule_condition(self, condition_texts, descricao_processo, material_items):
        try:
            return gemini_service.evaluate_rule_condition(
                condition_texts, descricao_processo, material_items, config=self.config
            )
        except gemini_service.GeminiServiceError as exc:
            raise LLMProviderError(str(exc)) from exc


# A small, stable set of Gemini models this backend is known to work with.
# Used by the Streamlit UI's model dropdown — Gemini has no lightweight
# public "list models" endpoint suitable for a per-request UI call, so
# unlike Groq this list is static rather than fetched live.
AVAILABLE_MODELS: List[str] = [
    "gemini-3.6-flash",
    "gemini-3.6-pro",
    "gemini-2.5-flash",
]
