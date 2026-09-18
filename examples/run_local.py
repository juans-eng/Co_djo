# -*- coding: utf-8 -*-
"""
examples/run_local.py
=======================

Minimal example of running the complete DJO validation pipeline locally,
end to end, exactly the way a future API endpoint would call it. See
ARCHITECTURE.md for the full response schema and integration guide.

Usage (from the project root, with the venv activated and .env configured
with GEMINI_API_KEY + LLMWHISPERER_API_KEY):

    python examples/run_local.py "dados/djos/DJO 3 (1).pdf" industrial
    python examples/run_local.py "dados/djos/DJO 1 (1).pdf" game --article14 true
    python examples/run_local.py "dados/djos/DJO 4.pdf" automotive
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `import src...` work when this script is run directly (not installed
# as a package) — same pattern used by tests/conftest.py's pytest.ini.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()  # loads GEMINI_API_KEY / LLMWHISPERER_API_KEY / etc. from .env

from src.services.djo_service import validate_djo_pdf


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "sim")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete DJO validation pipeline on one PDF.")
    parser.add_argument("pdf_path", help="Path to the DJO PDF to validate.")
    parser.add_argument("product_category", choices=["industrial", "game", "automotive"])
    parser.add_argument(
        "--article14", type=_parse_bool, default=None,
        help="Only used when product_category=game: true/false answer to the "
             "Games & Assortments (Art. 14) question. Omit if not yet known.",
    )
    parser.add_argument(
        "--llm-provider", choices=["gemini", "groq"], default="gemini",
        help="Which LLM backs Rule 5, the ACE 18 rule-condition fallback, and both summaries. Defaults to gemini.",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help="Optional model override for the selected provider (e.g. openai/gpt-oss-120b for Groq).",
    )
    args = parser.parse_args()

    result = validate_djo_pdf(
        pdf_path=args.pdf_path,
        product_category=args.product_category,
        article14_compliant=args.article14,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("status:", result["status"])
    if result["ncm"]:
        print("NCM:", result["ncm"]["code"])
    if result["origin_rule"]:
        print("applicable origin rule:", result["origin_rule"]["applicable_rule"])
        print("origin rule status:", result["origin_rule"]["status"])
    print("llm provider / model:", result.get("llm_provider"), "/", result.get("llm_model"))
    print("=" * 60)
    print(result["summary"])


if __name__ == "__main__":
    main()
