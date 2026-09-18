# -*- coding: utf-8 -*-
"""Rules 6 & 7 — src/validation/material_validator.py. No network required."""
import copy

from src.validation.material_validator import validate_table_completeness, validate_table_percentages


def test_table_completeness_pass(sample_json_djo):
    result = validate_table_completeness(sample_json_djo["Materiales"])
    assert result["status"] == "PASS"
    assert result["details"]["originarios_outros_estados_partes"]["status"] == "NOT_APPLICABLE"


def test_table_completeness_allows_nao_informado_only_in_estado_parte_table(sample_json_djo):
    # NCM/SH and Valor (US$) are already "[Não Informado]" in the fixture's
    # originarios_estado_parte_produtor table - must still PASS (Rule 6.1).
    result = validate_table_completeness(sample_json_djo["Materiales"])
    assert result["details"]["originarios_estado_parte_produtor"]["status"] == "PASS"


def test_table_completeness_fails_when_nao_originarios_missing_column(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["nao_originarios"]["Items"][0]["País origem"] = "[Não Informado]"
    result = validate_table_completeness(materiales)
    assert result["status"] == "FAIL"
    assert result["details"]["nao_originarios"]["status"] == "FAIL"


def test_table_completeness_fails_when_estado_parte_missing_disallowed_column(sample_json_djo):
    # Descrição is NOT in the allowed-blank set even for
    # originarios_estado_parte_produtor.
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["originarios_estado_parte_produtor"]["Items"][0]["Descrição"] = "[Não Informado]"
    result = validate_table_completeness(materiales)
    assert result["details"]["originarios_estado_parte_produtor"]["status"] == "FAIL"


def test_table_percentage_pass(sample_json_djo):
    result = validate_table_percentages(sample_json_djo["Materiales"])
    assert result["status"] == "PASS"
    assert result["details"]["nao_originarios"]["calculated_sum"] == 32.75


def test_table_percentage_fails_on_mismatch(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["nao_originarios"]["Somatório"] = "50%"
    result = validate_table_percentages(materiales)
    assert result["status"] == "FAIL"


def test_table_percentage_not_applicable_when_no_somatorio_declared(sample_json_djo):
    result = validate_table_percentages(sample_json_djo["Materiales"])
    assert result["details"]["terceiros_paises_ptc"]["status"] == "NOT_APPLICABLE"
