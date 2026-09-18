# -*- coding: utf-8 -*-
"""ACE 18 origin-rule flowchart — src/origin/decision_engine.py. Uses the
real cached ACE 18 table (no network); the `fake_pass_validation_result`
fixture is used to exercise the origin engine in isolation from Rule 5's
Gemini dependency (Rule 5 itself is covered separately, with mocks, in
test_gemini_service.py)."""
import copy

from src.origin.decision_engine import evaluate_origin_rule


def test_rule_a_when_only_estado_parte_table_has_materials(sample_json_djo, fake_pass_validation_result):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"] = {"Items": [], "Somatório": ""}
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "ORIGIN_RULE_A"
    assert result["applicable_origin_rule"] == "A"


def test_rule_b_when_both_originating_tables_have_materials(sample_json_djo, fake_pass_validation_result):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"] = {"Items": [], "Somatório": ""}
    d["Materiales"]["originarios_outros_estados_partes"] = {
        "Items": [{"NCM/SH": "1111.11.11", "Descrição": "X", "País origem": "ARGENTINA",
                   "Valor (US$)": "1", "% s/Valor FOB": "1%", "Fornecedor/Fabricante": "Y"}],
        "Somatório": "1%",
    }
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "ORIGIN_RULE_B"
    assert result["applicable_origin_rule"] == "B"


def test_unexpected_table_combination_is_manual_verification(sample_json_djo, fake_pass_validation_result):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"] = {"Items": [], "Somatório": ""}
    d["Materiales"]["originarios_estado_parte_produtor"] = {"Items": [], "Somatório": ""}
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "MANUAL_VERIFICATION"


def test_rule_c_when_mp_condition_passes(sample_json_djo, fake_pass_validation_result):
    # NCM 9404.90.00 -> real ACE18 rule "MP ou MaxMNO 45%"; fixture's
    # non-originating materials (headings 5208/5503) differ from 9404.
    result = evaluate_origin_rule(sample_json_djo, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "ORIGIN_RULE_C"
    assert result["applicable_origin_rule"] == "C"
    assert result["ncm_rule_info"]["mercosul_rule"]["raw_rule"] == "MP ou MaxMNO 45%"


def test_does_not_confer_origin_when_process_description_empty(sample_json_djo, fake_pass_validation_result):
    d = copy.deepcopy(sample_json_djo)
    d["Descrição do processo produtivo"] = ""
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "DOES_NOT_CONFER_ORIGIN"


def test_does_not_confer_origin_when_ncm_not_found(sample_json_djo, fake_pass_validation_result):
    d = copy.deepcopy(sample_json_djo)
    d["Código NCM"] = "9999.99.99"
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "DOES_NOT_CONFER_ORIGIN"


def test_correct_djo_when_base_validation_fails(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Razão social do produtor"]["Razão social"] = ""
    result = evaluate_origin_rule(d, "industrial")  # real validate_djo call -> FAIL
    assert result["final_status"] == "CORRECT_DJO"


def test_manual_verification_when_base_validation_needs_manual_review(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["Preço FOB"] = ""  # Rule 8 -> MANUAL_VERIFICATION, forces overall MANUAL_VERIFICATION
    result = evaluate_origin_rule(d, "industrial")
    assert result["final_status"] == "MANUAL_VERIFICATION"


def test_de_minimis_explicitly_excluded_gives_does_not_confer_origin(sample_json_djo, fake_pass_validation_result):
    # Real ACE18 rule for NCM 5102.11.00: "MP mais MaxMNO 45%. Não se
    # aplica de minimis..." - both conditions fail here -> AND fails ->
    # de_minimis_applies is False -> DOES_NOT_CONFER_ORIGIN (not MANUAL_VERIFICATION).
    d = copy.deepcopy(sample_json_djo)
    d["Código NCM"] = "5102.11.00"
    d["Materiales"]["nao_originarios"]["Items"] = [
        {"NCM/SH": "5102.20.00", "Descrição": "X", "País origem": "CHINA",
         "Valor (US$)": "1", "% s/Valor FOB": "70%", "Fornecedor/Fabricante": "Y"}
    ]
    d["Materiales"]["nao_originarios"]["Somatório"] = "70%"
    result = evaluate_origin_rule(d, "industrial", validation_result=fake_pass_validation_result)
    assert result["final_status"] == "DOES_NOT_CONFER_ORIGIN"
