# -*- coding: utf-8 -*-
"""
ace18/ncm_service.py
======================

ACE 18 (MERCOSUL) NCM lookup, ported verbatim from `codigo/ace18_service.py`
(itself a refactor of `codigo/estraction_ac18.ipynb`'s prototype logic).

    - Docling/OCR of the ACE 18 PDF runs ONLY ONCE, the first time the JSON
      cache doesn't exist yet.
    - Every later query reads the cached JSON — Docling is imported lazily
      (inside `extract_and_build_cache`) so importing this module, or
      querying an already-cached NCM table, never pays Docling's cost.
    - NCM matching priority: EXACT (8-digit) > RANGE > CHAPTER.

`ACE18Service.query_ncm(...)` additionally normalizes the result into a
`mercosul_rule` object carrying both `matched_ncm_expression` and
`raw_rule` together (what Rule 3 / the origin-rule engine need), while
preserving the fuller parsed-rule detail under `mercosul_rule["parsed_details"]`.

This module also carries the Rule-3 validation function
(`validate_ncm_ace18`) and the trade-agreement registry
(`AGREEMENT_SERVICES` / `normalize_agreement`) — the extension point for
supporting agreements beyond ACE 18 later without touching any other rule.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from src.models.validation import ValidationStatus, make_result, is_blank


# ---------------------------------------------------------------------------
# Cache building / loading
# ---------------------------------------------------------------------------
def extract_and_build_cache(pdf_path: str, cache_json_path: str) -> List[Dict[str, Any]]:
    """Processes the ACE 18 PDF with Docling ONCE and persists the resulting
    NCM/Mercosul-rule rows to `cache_json_path`. Only called when the cache
    file does not exist yet (see `load_rules_from_cache`)."""
    from docling.document_converter import DocumentConverter
    import pandas as pd

    print("Processing ACE 18 PDF with Docling (this only happens once per PDF)...")
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document

    all_rows: List[Dict[str, Any]] = []

    for num, table in enumerate(doc.tables):
        df = table.export_to_dataframe()

        col_ncm = None
        col_mercosul = None
        for col in df.columns:
            col_str = str(col).strip()
            if "NCM" in col_str:
                col_ncm = col
            elif "MERCOSUL" in col_str.upper():
                col_mercosul = col

        if col_ncm and col_mercosul:
            for _, row in df.iterrows():
                ncm_val = row[col_ncm]
                mercosul_val = row[col_mercosul]
                if pd.notna(ncm_val) and pd.notna(mercosul_val):
                    all_rows.append({
                        "table_num": num,
                        "ncm_raw": str(ncm_val).strip(),
                        "mercosul_raw": str(mercosul_val).strip(),
                    })

    os.makedirs(os.path.dirname(cache_json_path) or ".", exist_ok=True)
    with open(cache_json_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    print(f"Extraction complete. Cached {len(all_rows)} rows to: {cache_json_path}")
    return all_rows


def load_rules_from_cache(pdf_path: str, cache_json_path: str) -> List[Dict[str, Any]]:
    """Loads the ACE 18 rows from the JSON cache if it exists; otherwise
    builds the cache first (one-time Docling run)."""
    if not os.path.exists(cache_json_path):
        return extract_and_build_cache(pdf_path, cache_json_path)

    with open(cache_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# NCM cell parsing (which NCMs a given ACE 18 table row covers)
# ---------------------------------------------------------------------------
def extract_ncms_from_text(raw_text: str) -> List[str]:
    """Finds valid NCM patterns (XXXX.XX.XX or 8 plain digits) in a cell's
    text before any further cleanup is applied."""
    text = str(raw_text).strip()
    ncm_matches = re.findall(r"\b\d{4}\.\d{2}\.\d{2}\b|\b\d{8}\b", text)
    return [re.sub(r"[^\d]", "", m) for m in ncm_matches]


def expand_ncm_pattern(raw_text: str) -> List[Dict[str, Any]]:
    """Classifies an NCM cell strictly, avoiding false ranges caused by
    free-text descriptions in the same cell."""
    text = str(raw_text).strip()

    match_range = re.search(
        r"(\d{4}\.\d{2}\.\d{2}|\d{8})\s+a\s+(\d{4}\.\d{2}\.\d{2}|\d{8})",
        text,
        re.IGNORECASE,
    )
    if match_range:
        start_digits = re.sub(r"[^\d]", "", match_range.group(1))
        end_digits = re.sub(r"[^\d]", "", match_range.group(2))
        return [{
            "type": "RANGE",
            "start": int(start_digits),
            "end": int(end_digits),
            "raw": text,
        }]

    match_cap = re.search(r"Cap[íi]tulo\s*(\d{1,2})", text, re.IGNORECASE)
    if match_cap:
        cap_num = match_cap.group(1).zfill(2)
        return [{"type": "CHAPTER", "chapter": cap_num, "raw": text}]

    explicit_ncms = extract_ncms_from_text(text)
    if explicit_ncms:
        return [{"type": "EXACT", "ncm": ncm_code, "raw": text} for ncm_code in explicit_ncms]

    match_partida = re.search(r"\b\d{4}\.\d{2}\b|\b\d{4}\b", text)
    if match_partida:
        cleaned = re.sub(r"[^\d]", "", match_partida.group(0))
        start = int(cleaned.ljust(8, "0"))
        end = int(cleaned.ljust(8, "9"))
        return [{"type": "RANGE", "start": start, "end": end, "raw": text}]

    return [{"type": "OTHER", "raw": text}]


def search_ncm_fast(search_ncm: str, rules_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Searches the cached ACE 18 rows for the tariff hierarchy match with
    the strictest priority: EXACT (8-digit) > RANGE > CHAPTER."""
    target_clean = re.sub(r"[^\d]", "", str(search_ncm))
    if len(target_clean) != 8:
        raise ValueError("The NCM to search must have exactly 8 digits.")

    target_int = int(target_clean)
    target_chapter = target_clean[:2]

    matches = []
    for row in rules_data:
        # rule_parser.expand_ncm_pattern lives here to avoid a circular
        # import between ncm_service and rule_parser; see rule_parser.py
        # for the mercosul_raw-side counterpart (parse_mercosul_rule).
        patterns = expand_ncm_pattern(row["ncm_raw"])
        for p in patterns:
            if p["type"] == "EXACT" and p["ncm"] == target_clean:
                matches.append({"priority": 1, "row": row, "pattern": p})
            elif p["type"] == "RANGE" and p["start"] <= target_int <= p["end"]:
                matches.append({"priority": 2, "row": row, "pattern": p})
            elif p["type"] == "CHAPTER" and p["chapter"] == target_chapter:
                matches.append({"priority": 3, "row": row, "pattern": p})

    if not matches:
        return {
            "search_query": {"ncm": search_ncm, "chapter": target_chapter},
            "status": "NOT_FOUND",
        }

    best_match = sorted(matches, key=lambda x: x["priority"])[0]
    matched_row = best_match["row"]

    from src.ace18.rule_parser import parse_mercosul_rule  # local import: avoids a module-load cycle

    return {
        "search_query": {"ncm": search_ncm, "chapter": target_chapter},
        "match_info": {
            "match_type": best_match["pattern"]["type"],
            "matched_ncm_expression": matched_row["ncm_raw"],
            "table_index": matched_row["table_num"],
        },
        "mercosul_rule": parse_mercosul_rule(matched_row["mercosul_raw"]),
    }


# ---------------------------------------------------------------------------
# Service facade
# ---------------------------------------------------------------------------
class ACE18Service:
    """Thin service facade around the cached ACE 18 table.

    Usage:
        service = ACE18Service(pdf_path, cache_path)
        result = service.query_ncm("9404.90.00")
    """

    def __init__(self, pdf_path: str, cache_path: str):
        self.pdf_path = pdf_path
        self.cache_path = cache_path
        self._rules_db: Optional[List[Dict[str, Any]]] = None

    def _ensure_loaded(self) -> None:
        if self._rules_db is None:
            self._rules_db = load_rules_from_cache(self.pdf_path, self.cache_path)

    def query_ncm(self, ncm: str) -> Dict[str, Any]:
        """Returns:
            {
                "found": bool,
                "search_query": {...},
                "match_info": {...} | None when not found,
                "mercosul_rule": {
                    "matched_ncm_expression": str,
                    "raw_rule": str,
                    "parsed_details": {...}   # full parse_mercosul_rule() output
                } | None when not found,
            }
        """
        self._ensure_loaded()
        raw_result = search_ncm_fast(ncm, self._rules_db)

        if raw_result.get("status") == "NOT_FOUND":
            return {
                "found": False,
                "search_query": raw_result["search_query"],
                "match_info": None,
                "mercosul_rule": None,
            }

        match_info = raw_result["match_info"]
        parsed_rule = raw_result["mercosul_rule"]

        return {
            "found": True,
            "search_query": raw_result["search_query"],
            "match_info": match_info,
            "mercosul_rule": {
                "matched_ncm_expression": match_info["matched_ncm_expression"],
                "raw_rule": parsed_rule["raw_rule"],
                "parsed_details": parsed_rule,
            },
        }


# ---------------------------------------------------------------------------
# Trade-agreement registry (extension point for future agreements) + Rule 3
# ---------------------------------------------------------------------------
_ace18_service_singleton: Optional[ACE18Service] = None


def get_ace18_service(pdf_path: str, cache_path: str) -> ACE18Service:
    """Returns a process-wide singleton ACE18Service so the JSON cache is
    only read from disk once per process, no matter how many NCM queries
    are made."""
    global _ace18_service_singleton
    if _ace18_service_singleton is None:
        _ace18_service_singleton = ACE18Service(pdf_path, cache_path)
    return _ace18_service_singleton


def reset_ace18_service_singleton() -> None:
    """Test helper: forces the next get_ace18_service() call to build a
    fresh instance (e.g. to point at a different cache file)."""
    global _ace18_service_singleton
    _ace18_service_singleton = None


# FUTURE: register more agreements here (e.g. {"ACE_XX": build_some_other_service})
# — anything exposing the same .query_ncm(ncm) -> {"found", "match_info",
# "mercosul_rule"} shape as ACE18Service works without changing Rule 3 or
# the origin-rule engine.
def build_agreement_services(ace18_pdf_path: str, ace18_cache_path: str) -> Dict[str, Callable[[], ACE18Service]]:
    return {"ACE_18": lambda: get_ace18_service(ace18_pdf_path, ace18_cache_path)}


AGREEMENT_ALIASES = {
    "ACE_18": "ACE_18",
    "ACE18": "ACE_18",
    "ACE 18": "ACE_18",
    "MERCOSUL": "ACE_18",
    "MERCOSUR": "ACE_18",
}


def normalize_agreement(agreement: str) -> str:
    key = AGREEMENT_ALIASES.get(str(agreement).strip().upper())
    if key is None:
        raise ValueError(
            f"Unsupported agreement '{agreement}'. Supported: {sorted(set(AGREEMENT_ALIASES.values()))}"
        )
    return key


def validate_ncm_ace18(
    json_djo: Dict[str, Any],
    agreement: str,
    agreement_services: Dict[str, Callable[[], ACE18Service]],
) -> Dict[str, Any]:
    """Rule 3 — queries json_djo["Código NCM"] against the cached ACE 18
    table. If the NCM is not found: MANUAL_VERIFICATION, never an assumed
    valid/invalid (the NCM's validity is not assumed either way)."""
    ncm = json_djo.get("Código NCM")
    if is_blank(ncm):
        return make_result(
            "ncm_ace18",
            ValidationStatus.MANUAL_VERIFICATION,
            "\"Código NCM\" is missing from the DJO; cannot query the trade agreement.",
            mercosul_rule=None,
        )

    service = agreement_services[agreement]()

    try:
        result = service.query_ncm(ncm)
    except ValueError as exc:
        return make_result(
            "ncm_ace18",
            ValidationStatus.MANUAL_VERIFICATION,
            f"Could not query {agreement} for NCM '{ncm}': {exc}",
            mercosul_rule=None,
        )

    if not result["found"]:
        return make_result(
            "ncm_ace18",
            ValidationStatus.MANUAL_VERIFICATION,
            f"NCM '{ncm}' was not found in {agreement}. Manual verification required — "
            f"the NCM's validity is not assumed either way.",
            mercosul_rule=None,
        )

    return make_result(
        "ncm_ace18",
        ValidationStatus.PASS,
        f"NCM '{ncm}' matched {agreement} rule: {result['mercosul_rule']['raw_rule']}",
        mercosul_rule={
            "matched_ncm_expression": result["mercosul_rule"]["matched_ncm_expression"],
            "raw_rule": result["mercosul_rule"]["raw_rule"],
        },
    )
