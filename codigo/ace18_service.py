# -*- coding: utf-8 -*-
"""
ace18_service.py
=================

Reusable service for querying the Acordo de Complementação Econômica nº 18
(ACE 18 / Mercosur) rules of origin table by NCM code.

This module is a refactor of the logic prototyped in `estraction_ac18.ipynb`
into an importable, dependency-light service:

    - The Docling/OCR extraction of the ACE 18 PDF runs ONLY ONCE, the first
      time a cache file does not yet exist on disk.
    - Every subsequent query (in this process or a new one) reads the cached
      JSON instead of re-processing the PDF, so validating many DJOs does
      not re-run Docling for each one.

Behavior/format preserved from the notebook (do not change without also
checking `estraction_ac18.ipynb`, which remains the source of truth for the
raw parsing rules):
    - cache row shape: {"table_num", "ncm_raw", "mercosul_raw"}
    - `parse_mercosul_rule(...)` output shape (criteria, max_mno_percentage,
      chemical_processes, operator, de_minimis_applies, raw_rule)
    - NCM matching priority: EXACT > RANGE > CHAPTER

On top of that, `ACE18Service.query_ncm(...)` additionally normalizes the
result into a `mercosul_rule` object that carries both
`matched_ncm_expression` and `raw_rule` together, since that is the shape
the DJO validation core (Rule 3) needs to hand to the frontend. The fuller
parsed-rule detail (criteria, max_mno_percentage, ...) is preserved too,
under `mercosul_rule["parsed_details"]`, so nothing from the original
notebook's output is lost.

Docling is imported lazily (only inside `extract_and_build_cache`) so that
importing this module - and running validations against an already-cached
NCM table - never pays Docling's import cost.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Cache building / loading
# ---------------------------------------------------------------------------
def extract_and_build_cache(pdf_path: str, cache_json_path: str) -> List[Dict[str, Any]]:
    """Processes the ACE 18 PDF with Docling ONCE and persists the resulting
    NCM/Mercosul-rule rows to `cache_json_path`. Only called when the cache
    file does not exist yet (see `load_rules_from_cache`)."""
    # Imported lazily: Docling is only needed the very first time a given
    # ACE 18 PDF is cached; every later query just reads the JSON cache.
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
# NCM pattern parsing (unchanged from estraction_ac18.ipynb)
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

    # 1. Explicit range: "0401.10.10 a 0401.40.10"
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

    # 2. Chapter: "Capítulo 30", "ex Capítulo 15"
    match_cap = re.search(r"Cap[íi]tulo\s*(\d{1,2})", text, re.IGNORECASE)
    if match_cap:
        cap_num = match_cap.group(1).zfill(2)
        return [{"type": "CHAPTER", "chapter": cap_num, "raw": text}]

    # 3. Explicit NCM codes present in the cell (supports "e", "ex", commas)
    explicit_ncms = extract_ncms_from_text(text)
    if explicit_ncms:
        return [{"type": "EXACT", "ncm": ncm_code, "raw": text} for ncm_code in explicit_ncms]

    # 4. Isolated heading (4 digits) or subheading (6 digits)
    match_partida = re.search(r"\b\d{4}\.\d{2}\b|\b\d{4}\b", text)
    if match_partida:
        cleaned = re.sub(r"[^\d]", "", match_partida.group(0))
        start = int(cleaned.ljust(8, "0"))
        end = int(cleaned.ljust(8, "9"))
        return [{"type": "RANGE", "start": start, "end": end, "raw": text}]

    return [{"type": "OTHER", "raw": text}]


def parse_mercosul_rule(rule_str: str) -> Dict[str, Any]:
    """Extracts tariff criteria (MP, MS, MSP, CC), the MaxMNO percentage,
    named processes, operator and de-minimis applicability from a raw
    Mercosul-rule cell."""
    rule_str = str(rule_str).strip()
    parsed: Dict[str, Any] = {
        "raw_rule": rule_str,
        "criteria": [],
        "max_mno_percentage": None,
        "chemical_processes": [],
        "operator": "SINGLE",
        "de_minimis_applies": True,
    }

    mno_match = re.search(r"MaxMNO\s*(\d+)%", rule_str, re.IGNORECASE)
    if mno_match:
        parsed["max_mno_percentage"] = int(mno_match.group(1))

    if re.search(r"\bMSP\b", rule_str):
        parsed["criteria"].append("MSP")
    else:
        if re.search(r"\bMP\b", rule_str):
            parsed["criteria"].append("MP")
        if re.search(r"\bMS\b", rule_str):
            parsed["criteria"].append("MS")

    if re.search(r"\bCC\b", rule_str):
        parsed["criteria"].append("CC")

    processes = [
        "Reação química", "Purificação", "Separação isomérica",
        "Mudança de tamanho de partícula", "Produção de materiais padronizados",
        "Processo biotecnológico",
    ]
    for proc in processes:
        if proc.lower() in rule_str.lower():
            parsed["chemical_processes"].append(proc)

    if " ou " in rule_str.lower():
        parsed["operator"] = "OR"
    elif " mais " in rule_str.lower():
        parsed["operator"] = "AND"

    if "não se aplica de minimis" in rule_str.lower():
        parsed["de_minimis_applies"] = False

    return parsed


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
# Service wrapper (used by djo_validation_core)
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
                "match_info": {...} | absent when not found,
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
