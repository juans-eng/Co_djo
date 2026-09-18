# -*- coding: utf-8 -*-
"""
models/djo.py
==============

Structural constants describing the shape of a normalized `json_djo` dict
(produced by `djo.normalizer.normalize`). `json_djo` itself stays a plain
dict throughout the pipeline (matching the notebook reference
implementation and keeping every field trivially JSON-serializable) —
this module documents that shape and centralizes the few constants that
are shared across multiple validators, rather than introducing a parallel
class hierarchy that could drift from the real structure.

json_djo shape (see djo/normalizer.py:normalize for the authoritative
construction):
    {
        "Código da aprovação DJO": str,
        "Número da DJO": str,
        "Data de apresentação": str,
        "Razão social do produtor": {"Razão social": str, "CNPJ/CPF": str, "Inscrição Estadual": str},
        "Domicílio legal e parque industrial do produtor": {"Endereço": str, "Tel": str, "E-mail": str},
        "Razão social do exportador": {...same shape...},
        "Domicílio legal e parque industrial do exportador": {...same shape...},
        "Código NCM": str,
        "NALADI": str,
        "Denominação comercial do produto a exportar": str,
        "Valor FOB (USD)": str,
        "Unidade de medida": str,
        "Descrição do processo produtivo": str,
        "Materiales": {
            "originarios_estado_parte_produtor": {"Items": [ITEM, ...], "Somatório": str},
            "originarios_outros_estados_partes": {...same shape...},
            "nao_originarios": {...same shape...},
            "terceiros_paises_ptc": {...same shape...},
            "Porcentagem Total de Matérias Primas, Componentes ou Partes": str,
            "Valor agregado no processo Industrial (Deduzidos os tributos restituídos ou a restituir em caso de exportação)": str,
            "Preço FOB": str,
        },
        "Observações": str,
    }

    ITEM shape (one row of a material table):
    {
        "NCM/SH": str, "Descrição": str, "País origem": str,
        "Valor (US$)": str, "% s/Valor FOB": str, "Fornecedor/Fabricante": str,
    }
"""

# The 4 material tables that can appear under json_djo["Materiales"].
MATERIAL_TABLE_NAMES = [
    "originarios_estado_parte_produtor",
    "originarios_outros_estados_partes",
    "nao_originarios",
    "terceiros_paises_ptc",
]

# Columns every material-table item is expected to carry.
ITEM_COLUMNS = ["NCM/SH", "Descrição", "País origem", "Valor (US$)", "% s/Valor FOB", "Fornecedor/Fabricante"]
