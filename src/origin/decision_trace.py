# -*- coding: utf-8 -*-
"""
origin/decision_trace.py
==========================

Small helpers for building and narrating the ACE 18 origin-rule engine's
`decision_trace` — one entry per flowchart question, in the order they
were evaluated. Ported from `codigo/djo_validation_core.ipynb`'s origin
engine (the closure-based `add_step` there is promoted here to a small
reusable class; behavior is otherwise identical).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.models.origin_rule import STATUS_LABELS


class DecisionTraceBuilder:
    """Accumulates one `decision_trace` entry per flowchart question, each
    numbered sequentially: {"step", "question", "input_used", "result", "reason"}."""

    def __init__(self) -> None:
        self._trace: List[Dict[str, Any]] = []

    def add_step(self, question: str, input_used: Any, result: Any, reason: str) -> Dict[str, Any]:
        entry = {
            "step": len(self._trace) + 1,
            "question": question,
            "input_used": input_used,
            "result": result,
            "reason": reason,
        }
        self._trace.append(entry)
        return entry

    @property
    def trace(self) -> List[Dict[str, Any]]:
        return self._trace


def build_simple_explanation(trace: List[Dict[str, Any]], final_status: str, closing_sentence: str) -> str:
    """Deterministic (non-Gemini) plain-language narration of the decision
    trace. Kept deterministic — not Gemini-generated — specifically so it
    can never contradict the trace it is built from."""
    linhas = []
    for e in trace:
        r = e["result"]
        r_str = "Sim" if r is True else "Não" if r is False else "Indeterminado" if r is None else str(r)
        linhas.append(f"{e['step']}. {e['question']} -> {r_str}. {e['reason']}")
    corpo = "\n".join(linhas)
    rotulo = STATUS_LABELS.get(final_status, final_status)
    return f"Sequência de decisões do fluxograma ACE 18:\n{corpo}\n\nConclusão: {rotulo}. {closing_sentence}"


def build_origin_result(
    validation_result: Optional[Dict[str, Any]],
    final_status: str,
    applicable_rule: Optional[str],
    trace: List[Dict[str, Any]],
    closing_sentence: str,
    ncm_rule_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "validation_result": validation_result,
        "ncm_rule_info": ncm_rule_info,
        "applicable_origin_rule": applicable_rule,
        "final_status": final_status,
        "decision_trace": trace,
        "simple_explanation": build_simple_explanation(trace, final_status, closing_sentence),
    }
