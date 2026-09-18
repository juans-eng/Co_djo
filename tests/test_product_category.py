# -*- coding: utf-8 -*-
"""Product-category handling — spec section 6: industrial/game/automotive,
never inferred. No network required (automotive short-circuits before any
validation runs; the game cases below use fake_pass_validation_result to
stay isolated from Rule 5's Gemini dependency)."""
import pytest

from src.origin.decision_engine import evaluate_origin_rule


def test_automotive_is_not_applicable_and_skips_validation_entirely(sample_json_djo):
    result = evaluate_origin_rule(sample_json_djo, "automotive")
    assert result["final_status"] == "NOT_APPLICABLE"
    assert result["validation_result"] is None  # base validation never ran


def test_invalid_category_raises_value_error(sample_json_djo):
    with pytest.raises(ValueError):
        evaluate_origin_rule(sample_json_djo, "bicycle")


def test_game_without_article14_answer_is_manual_verification(sample_json_djo, fake_pass_validation_result):
    result = evaluate_origin_rule(sample_json_djo, "game", article14_compliant=None, validation_result=fake_pass_validation_result)
    assert result["final_status"] == "MANUAL_VERIFICATION"
    last_step = result["decision_trace"][-1]
    assert "Artigo 14" in last_step["question"]


def test_game_with_article14_false_does_not_confer_origin(sample_json_djo, fake_pass_validation_result):
    result = evaluate_origin_rule(sample_json_djo, "game", article14_compliant=False, validation_result=fake_pass_validation_result)
    assert result["final_status"] == "DOES_NOT_CONFER_ORIGIN"


def test_game_with_article14_true_proceeds_to_rule_c(sample_json_djo, fake_pass_validation_result):
    result = evaluate_origin_rule(sample_json_djo, "game", article14_compliant=True, validation_result=fake_pass_validation_result)
    assert result["final_status"] == "ORIGIN_RULE_C"


def test_industrial_skips_article14_gate_entirely(sample_json_djo, fake_pass_validation_result):
    result = evaluate_origin_rule(sample_json_djo, "industrial", validation_result=fake_pass_validation_result)
    questions = [step["question"] for step in result["decision_trace"]]
    assert not any("Artigo 14" in q for q in questions)
