# -*- coding: utf-8 -*-
"""
ace18/rule_parser.py
======================

Parses ACE 18 `mercosul_rule.raw_rule` text (never evaluates it against a
product — see `rule_engine.py` for that). Two layers, both ported verbatim:

    parse_mercosul_rule(rule_str)   — from codigo/ace18_service.py: extracts
        tariff criteria (MP/MS/MSP/CC), MaxMNO %, named processes, the
        OR/AND operator and de-minimis applicability as loose metadata.

    parse_rule_conditions(raw_rule) — from codigo/djo_validation_core.ipynb's
        origin-rule engine: splits the same raw_rule into atomic,
        individually-evaluable conditions (MP / MSP / MAXMNO /
        EXCLUDED_POSITIONS / UNKNOWN) + the operator joining them. This is
        what `rule_engine.evaluate_rule_conditions` actually consumes.

    Validated against all 25 distinct raw_rule texts found in the real
    reos_ace18_cache.json (584 rows).
"""
from __future__ import annotations

import re
from typing import Any, Dict


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


def parse_rule_conditions(raw_rule: str) -> Dict[str, Any]:
    """Splits a mercosul_rule raw_rule string into atomic conditions plus
    the operator joining them, e.g.:

        "MP ou MaxMNO 45%"
            -> operator "OR", conditions [MP, MAXMNO(45)]
        "MP, exceto das posições 7206 a 7217"
            -> operator "AND", conditions [MP, EXCLUDED_POSITIONS(7206-7217)]
        "MP mais MaxMNO 45%. Não se aplica de minimis..."
            -> operator "AND", conditions [MP, MAXMNO(45)], de_minimis_applies=False

    Known limitation: only ONE operator level is detected per rule (first
    "ou", else "mais", else a comma for the "MP, exceto..." exception
    pattern) — no rule in the real ACE 18 corpus mixes operators, so this
    single-level split is sufficient for the observed data; a rule mixing
    "ou" and "mais" together would not be split correctly.

    Returns {"operator": "SINGLE"|"OR"|"AND", "conditions": [...],
    "de_minimis_applies": bool}.
    """
    texto = raw_rule.strip()

    # "Não se aplica de minimis..." is metadata about Art. 6 applicability,
    # not a pass/fail condition of the origin requirement itself (reuses
    # the same phrase parse_mercosul_rule above already looks for) — strip
    # it before splitting into conditions.
    de_minimis_applies = "não se aplica de minimis" not in texto.lower()
    texto_sem_minimis = re.split(r"\.\s*Não se aplica de minimis", texto, flags=re.IGNORECASE)[0].strip()

    if re.search(r"\bou\b", texto_sem_minimis, re.IGNORECASE):
        operator = "OR"
        segmentos = re.split(r"\bou\b", texto_sem_minimis, flags=re.IGNORECASE)
    elif re.search(r"\bmais\b", texto_sem_minimis, re.IGNORECASE):
        operator = "AND"
        segmentos = re.split(r"\bmais\b", texto_sem_minimis, flags=re.IGNORECASE)
    elif "," in texto_sem_minimis:
        operator = "AND"
        segmentos = texto_sem_minimis.split(",", 1)
    else:
        operator = "SINGLE"
        segmentos = [texto_sem_minimis]

    condicoes = [_classify_condition(seg.strip(" .")) for seg in segmentos if seg.strip(" .")]
    return {"operator": operator, "conditions": condicoes, "de_minimis_applies": de_minimis_applies}


def _classify_condition(segment: str) -> Dict[str, Any]:
    """Classifies one already-split segment into a known deterministic
    condition type, or UNKNOWN (free text — handled later via Gemini)."""
    seg = segment.strip()
    seg_upper = seg.upper()

    if seg_upper == "MSP":
        return {"type": "MSP", "text": seg}
    if seg_upper == "MP":
        return {"type": "MP", "text": seg}

    m = re.match(r"^MaxMNO\s*(\d+(?:[.,]\d+)?)\s*%$", seg, re.IGNORECASE)
    if m:
        return {"type": "MAXMNO", "text": seg, "max_percentage": float(m.group(1).replace(",", "."))}

    m = re.match(r"^exceto\s+das\s+posições\s+(\d{4})\s+a\s+(\d{4})$", seg, re.IGNORECASE)
    if m:
        return {"type": "EXCLUDED_POSITIONS", "text": seg,
                "range_start": int(m.group(1)), "range_end": int(m.group(2))}

    return {"type": "UNKNOWN", "text": seg}
