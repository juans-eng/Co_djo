# -*- coding: utf-8 -*-
"""
djo/extractor.py
==================

DJO extraction: turns a PDF into raw OCR text via LLM Whisperer
(layout_preserving mode), matching `codigo/extraction_djo.ipynb`'s cell 1
exactly (same client, same call parameters). This module does NOT
interpret the text at all — that's `djo/normalizer.py`'s job — it only
gets the text out of the PDF.

Security note (this is the one behavioral change from the notebook
reference, requested explicitly — see project instructions section 15):
the notebook has the LLM Whisperer API key hard-coded inline. Here it is
read from the `LLMWHISPERER_API_KEY` environment variable instead and
never hard-coded, never logged, and never included in any returned
structure.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from unstract.llmwhisperer import LLMWhispererClientV2
from unstract.llmwhisperer.client_v2 import LLMWhispererClientException

DEFAULT_WAIT_TIMEOUT = 200


class DjoExtractionError(Exception):
    """Raised when the PDF could not be OCR'd — missing API key, a
    LLMWhisperer API failure, or a missing/unreadable file. Callers
    (djo_service) must turn this into a structured PROCESSING_ERROR
    response rather than letting a raw exception/traceback reach the
    frontend."""


@dataclass
class ExtractorConfig:
    api_key: Optional[str] = None
    wait_timeout: int = DEFAULT_WAIT_TIMEOUT

    @classmethod
    def from_env(cls) -> "ExtractorConfig":
        return cls(
            api_key=os.environ.get("LLMWHISPERER_API_KEY") or None,
            wait_timeout=int(os.environ.get("LLMWHISPERER_WAIT_TIMEOUT", DEFAULT_WAIT_TIMEOUT)),
        )


def extract_text_from_pdf(pdf_path: str, config: Optional[ExtractorConfig] = None) -> str:
    """Runs LLM Whisperer OCR (layout_preserving, form mode — the client's
    default, matching the notebook) on `pdf_path` and returns the raw
    extracted text, unmodified. Raises DjoExtractionError on any failure.
    """
    config = config or ExtractorConfig.from_env()

    if not config.api_key:
        raise DjoExtractionError(
            "LLMWHISPERER_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before extracting a DJO PDF."
        )

    if not os.path.isfile(pdf_path):
        raise DjoExtractionError(f"DJO PDF not found: {pdf_path!r}")

    client = LLMWhispererClientV2(api_key=config.api_key)

    try:
        result = client.whisper(
            file_path=pdf_path,
            wait_for_completion=True,
            wait_timeout=config.wait_timeout,
        )
    except LLMWhispererClientException as exc:
        raise DjoExtractionError(
            f"LLM Whisperer API error while processing {os.path.basename(pdf_path)!r}: "
            f"{exc.message} (status code {exc.status_code})"
        ) from exc
    except Exception as exc:  # network/timeout/unexpected client errors
        raise DjoExtractionError(
            f"Unexpected error while extracting {os.path.basename(pdf_path)!r}: {exc}"
        ) from exc

    try:
        return result["extraction"]["result_text"]
    except (KeyError, TypeError) as exc:
        raise DjoExtractionError(
            f"LLM Whisperer returned an unexpected response shape for {os.path.basename(pdf_path)!r}."
        ) from exc
