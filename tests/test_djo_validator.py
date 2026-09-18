# -*- coding: utf-8 -*-
"""Rules 1, 2, 4, 8 — src/djo/validator.py. No network required."""
import copy

from src.djo.validator import (
    validate_djo_approval,
    validate_producer_information,
    validate_mandatory_fields,
    validate_preco_fob,
)


def test_djo_approval_pass_when_both_fields_present(sample_json_djo):
    result = validate_djo_approval(sample_json_djo)
    assert result["status"] == "PASS"


def test_djo_approval_not_applicable_when_not_yet_approved(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Código da aprovação DJO"] = ""
    result = validate_djo_approval(d)
    assert result["status"] == "NOT_APPLICABLE"


def test_producer_information_pass(sample_json_djo):
    result = validate_producer_information(sample_json_djo)
    assert result["status"] == "PASS"


def test_producer_information_fails_when_razao_social_missing(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Razão social do produtor"]["Razão social"] = ""
    result = validate_producer_information(d)
    assert result["status"] == "FAIL"
    assert "Razão social do produtor" in result["details"]["missing_fields"]


def test_producer_information_fails_when_endereco_missing(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Domicílio legal e parque industrial do produtor"]["Endereço"] = ""
    result = validate_producer_information(d)
    assert result["status"] == "FAIL"


def test_mandatory_fields_pass(sample_json_djo):
    result = validate_mandatory_fields(sample_json_djo)
    assert result["status"] == "PASS"


def test_mandatory_fields_fails_on_not_informado(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Unidade de medida"] = "[Não Informado]"
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Unidade de medida" in result["details"]["missing_fields"]


# --- Rule 4 FOB exception: "Valor FOB (USD)" is not required when
# 'originarios_estado_parte_produtor' is the ONLY material table with
# items -------------------------------------------------------------------

def test_mandatory_fields_passes_with_empty_fob_when_only_producer_table_present(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"]["Items"] = []  # leave only originarios_estado_parte_produtor
    d["Valor FOB (USD)"] = ""
    result = validate_mandatory_fields(d)
    assert result["status"] == "PASS"


def test_mandatory_fields_passes_with_populated_fob_when_only_producer_table_present(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"]["Items"] = []
    result = validate_mandatory_fields(d)
    assert result["status"] == "PASS"


def test_mandatory_fields_fails_on_empty_fob_when_nao_originarios_present(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Valor FOB (USD)"] = ""  # nao_originarios already has items in the fixture
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Valor FOB (USD)" in result["details"]["missing_fields"]


def test_mandatory_fields_fails_on_empty_fob_when_originarios_outros_estados_partes_present(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"]["Items"] = []
    d["Materiales"]["originarios_outros_estados_partes"] = {
        "Items": [{"NCM/SH": "1111.11.11", "Descrição": "X", "País origem": "ARGENTINA",
                   "Valor (US$)": "1", "% s/Valor FOB": "1%", "Fornecedor/Fabricante": "Y"}],
        "Somatório": "1%",
    }
    d["Valor FOB (USD)"] = ""
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Valor FOB (USD)" in result["details"]["missing_fields"]


def test_mandatory_fields_fails_on_empty_fob_when_terceiros_paises_ptc_present(sample_json_djo):
    d = copy.deepcopy(sample_json_djo)
    d["Materiales"]["nao_originarios"]["Items"] = []
    d["Materiales"]["terceiros_paises_ptc"] = {
        "Items": [{"NCM/SH": "2222.22.22", "Descrição": "X", "País origem": "CHINA",
                   "Valor (US$)": "1", "% s/Valor FOB": "1%", "Fornecedor/Fabricante": "Y"}],
        "Somatório": "1%",
    }
    d["Valor FOB (USD)"] = ""
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Valor FOB (USD)" in result["details"]["missing_fields"]


def test_mandatory_fields_fails_on_empty_fob_when_producer_table_plus_another_present(sample_json_djo):
    # originarios_estado_parte_produtor + nao_originarios both present (the
    # fixture's default configuration) -> the exception must NOT apply.
    d = copy.deepcopy(sample_json_djo)
    d["Valor FOB (USD)"] = ""
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Valor FOB (USD)" in result["details"]["missing_fields"]


def test_mandatory_fields_fails_on_empty_fob_when_no_material_table_present(sample_json_djo):
    # No material table at all -> existing (pre-exception) behavior is
    # preserved: "Valor FOB (USD)" is still required.
    d = copy.deepcopy(sample_json_djo)
    for table in ("originarios_estado_parte_produtor", "originarios_outros_estados_partes",
                  "nao_originarios", "terceiros_paises_ptc"):
        d["Materiales"][table]["Items"] = []
    d["Valor FOB (USD)"] = ""
    result = validate_mandatory_fields(d)
    assert result["status"] == "FAIL"
    assert "Valor FOB (USD)" in result["details"]["missing_fields"]


def test_preco_fob_pass_at_100(sample_json_djo):
    result = validate_preco_fob(sample_json_djo["Materiales"])
    assert result["status"] == "PASS"
    assert result["details"]["value"] == 100.0


def test_preco_fob_fails_over_100(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["Preço FOB"] = "150%"
    result = validate_preco_fob(materiales)
    assert result["status"] == "FAIL"


def test_preco_fob_missing_is_manual_verification_not_zero(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["Preço FOB"] = ""
    result = validate_preco_fob(materiales)
    assert result["status"] == "MANUAL_VERIFICATION"


# --- Rule 8 exception (same scenario as Rule 4's): "Preço FOB" is
# NOT_APPLICABLE, not MANUAL_VERIFICATION, when
# 'originarios_estado_parte_produtor' is the only material table present ---

def test_preco_fob_not_applicable_when_only_producer_table_present(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["nao_originarios"]["Items"] = []  # leave only originarios_estado_parte_produtor
    materiales["Preço FOB"] = ""
    result = validate_preco_fob(materiales)
    assert result["status"] == "NOT_APPLICABLE"


def test_preco_fob_still_manual_verification_when_producer_table_plus_another_present(sample_json_djo):
    # originarios_estado_parte_produtor + nao_originarios both present (the
    # fixture's default configuration) -> the exception must NOT apply.
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    materiales["Preço FOB"] = ""
    result = validate_preco_fob(materiales)
    assert result["status"] == "MANUAL_VERIFICATION"


def test_preco_fob_still_manual_verification_when_no_material_table_present(sample_json_djo):
    materiales = copy.deepcopy(sample_json_djo["Materiales"])
    for table in ("originarios_estado_parte_produtor", "originarios_outros_estados_partes",
                  "nao_originarios", "terceiros_paises_ptc"):
        materiales[table]["Items"] = []
    materiales["Preço FOB"] = ""
    result = validate_preco_fob(materiales)
    assert result["status"] == "MANUAL_VERIFICATION"
