# -*- coding: utf-8 -*-
"""
llm/base.py
============

Provider-agnostic contract for the "semantic layer" of this backend — the
handful of genuinely language-understanding tasks the deterministic
business logic delegates out (see module docstrings in
`validation/process_validator.py`, `ace18/rule_engine.py`,
`services/djo_service.py` for exactly which ones, and why only those).

Every LLM provider (Gemini, Groq, ...) implements `LLMProvider` with these
four methods. Callers throughout `src/` — `validate_process_materials`,
`evaluate_rule_conditions`, `validate_djo_pdf` — depend ONLY on this
interface, never on a specific provider's SDK or config type. This is
what makes "which provider generated this explanation" purely a
presentation-layer choice (surfaced to the frontend as
`llm_provider`/`llm_model` in the response) rather than something the
validation engine has any awareness of.

IMPORTANT — this abstraction governs presentation/semantic-matching only.
It is not, and must never become, a second place where PASS/FAIL/A/B/C
decisions get made: every method here either narrates an
already-decided result, or answers one bounded yes/no/inconclusive
question that deterministic logic could not resolve on its own. See
section headers in `ace18/rule_engine.py` and `origin/decision_engine.py`
for where the line is drawn.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProviderError(Exception):
    """Raised by any LLMProvider method whenever its result cannot be
    trusted: no API key configured, a request/network failure, an
    unsupported/invalid model, or a response that doesn't parse into the
    expected shape. Callers throughout this backend already catch this
    (provider-agnostically) and fall back to MANUAL_VERIFICATION or a
    placeholder summary — never crash, never silently guess a result."""


class LLMProvider(ABC):
    """Common interface every LLM provider must implement. `name`/`model`
    are for display only (surfaced in the response's `llm_provider`/
    `llm_model` fields) — the validation engine never branches on them."""

    name: str = "unknown"
    model: str = ""

    @abstractmethod
    def match_process_materials(
        self, descricao_processo: str, material_items: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Rule 5: decide whether the inputs named in `descricao_processo`
        are represented in `material_items`. Returns
        {"process_inputs": [...], "all_inputs_found": bool}. Raises
        LLMProviderError on failure."""

    @abstractmethod
    def summarize_validation_result(self, validation_result: Dict[str, Any]) -> str:
        """Human-readable (Portuguese) narration of an already-computed
        validation_result. Purely presentational. Raises LLMProviderError
        on failure."""

    @abstractmethod
    def summarize_origin_rule_result(self, origin_result: Dict[str, Any]) -> str:
        """Human-readable (Portuguese) narration of an already-computed
        origin-rule result, same tone/style as
        `summarize_validation_result` so the two read as one summary when
        concatenated. Raises LLMProviderError on failure."""

    @abstractmethod
    def evaluate_rule_condition(
        self,
        condition_texts: List[str],
        descricao_processo: str,
        material_items: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Origin-rule engine fallback: decide whether the product
        satisfies at least one free-text `mercosul_rule` condition that
        the deterministic parser couldn't classify. Returns
        {"result": true|false|null, "reasoning": str}. Raises
        LLMProviderError on failure."""


class RateLimiter:
    """Sleeps just enough to keep consecutive calls at least
    `min_interval_seconds` apart. One instance per provider (not shared
    across providers), so pacing a Groq provider never throttles a
    concurrently-configured Gemini one, and vice-versa."""

    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._last_call_at: Optional[float] = None

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.monotonic()
        if self._last_call_at is not None:
            elapsed = now - self._last_call_at
            remaining = self.min_interval_seconds - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call_at = time.monotonic()


def is_transient_error_text(text: str) -> bool:
    """Shared heuristic for 'worth a bounded retry with backoff' across
    providers: rate-limited / temporarily overloaded, not a hard failure
    (bad key, bad model, malformed request)."""
    return any(marker in text for marker in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate_limit"))
