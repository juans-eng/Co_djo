# -*- coding: utf-8 -*-
"""
llm/service.py
===============

Factory for `LLMProvider` instances. This is the ONLY place in the
backend that knows the mapping from a provider name string
("gemini"/"groq") to a concrete class — every other module (validation,
origin engine, djo_service) depends only on `LLMProvider` (see
`llm/base.py`). Adding a third provider later means adding one branch
here (and one new module implementing `LLMProvider`) — nothing else in
`src/` changes.
"""
from __future__ import annotations

from typing import List, Optional

from src.llm.base import LLMProvider, LLMProviderError

SUPPORTED_PROVIDERS = ("gemini", "groq")
DEFAULT_PROVIDER = "gemini"


def create_provider(provider_name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    """Builds an `LLMProvider` for `provider_name` ("gemini"/"groq",
    case-insensitive; defaults to DEFAULT_PROVIDER when omitted/blank),
    optionally overriding its model. Reads API keys from the environment
    only — never accepts one as an argument, so a key can never
    accidentally flow through from a frontend request.

    Raises LLMProviderError immediately for an unrecognized provider
    name. Does NOT validate the API key or model here — those are only
    discovered on the first actual call, exactly like the existing
    Gemini-only behavior, so constructing a provider is always cheap and
    never itself makes a network call.
    """
    name = (provider_name or DEFAULT_PROVIDER).strip().lower()

    if name == "gemini":
        from src.llm.gemini import GeminiProvider
        from src.gemini.service import GeminiConfig

        config = GeminiConfig.from_env()
        if model:
            config.model = model
        return GeminiProvider(config)

    if name == "groq":
        from src.llm.groq import GroqProvider, GroqConfig

        config = GroqConfig.from_env()
        if model:
            config.model = model
        return GroqProvider(config)

    raise LLMProviderError(
        f"Unsupported LLM provider: {provider_name!r}. Supported providers: {list(SUPPORTED_PROVIDERS)}."
    )


def list_available_models(provider_name: str) -> List[str]:
    """Models to offer in a UI dropdown for `provider_name`. Gemini uses a
    static, curated list (see `llm/gemini.py`); Groq discovers live from
    its API when a key is available, falling back to a static list
    otherwise (see `llm/groq.py`). Never raises — an unrecognized
    provider name returns an empty list."""
    name = (provider_name or "").strip().lower()
    if name == "gemini":
        from src.llm.gemini import AVAILABLE_MODELS

        return list(AVAILABLE_MODELS)
    if name == "groq":
        from src.llm.groq import list_groq_models

        return list_groq_models()
    return []
