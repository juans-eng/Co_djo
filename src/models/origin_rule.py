# -*- coding: utf-8 -*-
"""
models/origin_rule.py
======================

Constants for the ACE 18 origin-rule decision engine (see
`origin/decision_engine.py`), ported verbatim from
`codigo/djo_validation_core.ipynb`.
"""

# Products the ACE 18 origin-rule flowchart applies to (spec section 2):
# only industrial and game/assortment products; automotive is explicitly
# out of scope.
PRODUCT_CATEGORIES = {"industrial", "automotive", "game"}

# Every possible value of origin_result["final_status"].
FINAL_STATUSES = {
    "ORIGIN_RULE_A",
    "ORIGIN_RULE_B",
    "ORIGIN_RULE_C",
    "DOES_NOT_CONFER_ORIGIN",
    "CORRECT_DJO",
    "MANUAL_VERIFICATION",
    "NOT_APPLICABLE",
}

STATUS_LABELS = {
    "ORIGIN_RULE_A": "Norma de Origem A (produto elaborado exclusivamente a partir de materiais originários / Art. 5°)",
    "ORIGIN_RULE_B": "Norma de Origem B (produto elaborado no território dos Estados Partes a partir de materiais originários)",
    "ORIGIN_RULE_C": "Norma de Origem C (produto cumpre o requisito específico de origem do ACE 18)",
    "DOES_NOT_CONFER_ORIGIN": "o produto NÃO confere origem",
    "CORRECT_DJO": "é necessário corrigir a declaração (DJO) antes de prosseguir",
    "MANUAL_VERIFICATION": "é necessária verificação manual",
    "NOT_APPLICABLE": "este fluxograma não se aplica a este produto",
}
