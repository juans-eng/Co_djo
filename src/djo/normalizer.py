# -*- coding: utf-8 -*-
"""
djo/normalizer.py
===================

DJO normalization: turns the raw OCR text (from `djo/extractor.py`) into
the structured `json_djo` dict every downstream module consumes. Ported
verbatim from `codigo/extraction_djo.ipynb`'s parser cell — same
sequential, key-anchored extraction engine, same table-column heuristics,
same OCR-quirk tolerances (accent-tolerant "Somatório", country-name
fallback for glued Descrição/País origem, etc.). See that notebook cell's
own comments for the full history of why each of these exists; they are
preserved here rather than re-explained.

No business logic was changed in this port — only moved out of the
notebook and given a public entry point (`normalize`).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Known field labels of the DJO form — used to recognize where a "blank"
# block ends (i.e. the next non-empty line is already the next label),
# instead of swallowing that label as if it were the current field's value.
# ---------------------------------------------------------------------------
_LABEL_PATTERNS = [
    r"Código da aprovação DJO:",
    r"Número da DJO:",
    r"Data de apresentação:",
    r"Razão social do produtor:",
    r"CNPJ/CPF:",
    r"Inscrição Estadual:",
    r"Domicílio legal e parque industrial:",
    r"Endereço:",
    r"Telefone:",
    r"E-mail:",
    r"Razão social do exportador:",
    r"Código NCM\s*-\s*NALADI:",
    r"Denominação comercial do produto a exportar",
    r"Valor FOB \(USD\):",
    r"Unidade de medida:",
    r"Descrição do processo produtivo:",
    r"Materiais utilizados:",
    r"Materiais originários do Estado Parte produtor:",
    r"Materiais originários de outros Estados Partes:",
    r"Materiais não originários:",
    r"Materiais de terceiros países que tenham cumprido com a PTC:",
    r"Porcentagem Total de Mat[eé]rias Primas",
    r"Valor agregado no processo Industrial",
    r"Preço FOB:",
    r"Observações:",
    r"Somat.rio",
    r"Página\s*\d+\s*de\s*\d+",
    r"DECLARAÇÃO JURAMENTADA DE ORIGEM",
    r"NCM/SH",
    r"DECLARO PARA OS DEVIDOS FINS",
]
_LABEL_RE = re.compile(r"^\s*(?:" + "|".join(_LABEL_PATTERNS) + r")", re.IGNORECASE)


def _is_label_line(line: str) -> bool:
    return bool(_LABEL_RE.match(line))


# ---------------------------------------------------------------------------
# 0) Pre-processing
# ---------------------------------------------------------------------------
def preprocess_ocr_text(extracted_text: str) -> Tuple[str, str]:
    """Returns (codigo_aprovacao_djo, texto_limpo_para_extracao_sequencial)."""
    texto = extracted_text

    codigo_aprovacao = ""
    for m in re.finditer(r"Código da aprovação DJO:", texto):
        for linha in texto[m.end():].split("\n"):
            s = linha.strip()
            if not s:
                continue
            if s == "[Não Informado]" or _is_label_line(s):
                break
            codigo_aprovacao = s
            break
        if codigo_aprovacao:
            break

    texto = re.sub(
        r"Código da aprovação DJO:[^\n]*\n(?:\s*\n)?(?:[ \t]*\[Não Informado\][ \t]*\n)?",
        "\n",
        texto,
    )
    texto = re.sub(r"DECLARAÇÃO JURAMENTADA DE ORIGEM", "\n", texto)
    texto = re.sub(
        r"DECLARO PARA OS DEVIDOS FINS.*?(?=Página\s*\d+\s*de\s*\d+|$)",
        "\n",
        texto,
        flags=re.DOTALL,
    )
    texto = re.sub(r"Página\s*\d+\s*de\s*\d+", "\n", texto)
    texto = texto.replace("<<<", "\n")

    return codigo_aprovacao, texto


# ---------------------------------------------------------------------------
# 1) Sequential extraction engine
# ---------------------------------------------------------------------------
def sequential_extract(texto: str, keys: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    resultado: Dict[str, Any] = {}
    blocos_debug: List[Dict[str, Any]] = []
    cursor = 0

    for i, key in enumerate(keys):
        m = re.search(key["pattern"], texto[cursor:], re.IGNORECASE)
        if not m:
            blocos_debug.append({"key": key["name"], "found": False, "matched_label": None, "raw_block": ""})
            continue

        value_start = cursor + m.end()
        matched_label = m.group(0)

        next_start = len(texto)
        for key_seguinte in keys[i + 1:]:
            m2 = re.search(key_seguinte["pattern"], texto[value_start:], re.IGNORECASE)
            if m2:
                next_start = value_start + m2.start()
                break

        raw_block = texto[value_start:next_start]
        blocos_debug.append({
            "key": key["name"], "found": True,
            "matched_label": matched_label.strip(), "raw_block": raw_block,
        })

        handler = key.get("handler")
        if handler is not None:
            valor = handler(raw_block)
            if key.get("merge"):
                resultado.update(valor)
            else:
                resultado[key["name"]] = valor

        cursor = value_start

    return resultado, blocos_debug


# ---------------------------------------------------------------------------
# 2) Field handlers
# ---------------------------------------------------------------------------
def _texto_limpo(bloco: str) -> str:
    valor = " ".join(bloco.split())
    return "" if valor == "[Não Informado]" else valor


def handler_texto(bloco: str) -> str:
    return _texto_limpo(bloco)


def _extrai_label_no_bloco(bloco: str, label_regex: str) -> str:
    m = re.search(label_regex, bloco)
    if not m:
        return ""
    resto = bloco[m.end():]
    linhas = resto.split("\n")
    primeira = linhas[0].strip()
    if primeira and not _is_label_line(primeira):
        return primeira
    for linha in linhas[1:]:
        s = linha.strip()
        if not s:
            continue
        return "" if _is_label_line(s) else s
    return ""


def handler_empresa(bloco: str) -> Dict[str, str]:
    cnpj = _extrai_label_no_bloco(bloco, r"CNPJ/CPF:")
    insc = _extrai_label_no_bloco(bloco, r"Inscrição Estadual:")

    sem_labels = re.sub(r"CNPJ/CPF:.*", "", bloco)
    sem_labels = re.sub(r"Inscrição Estadual:.*", "", sem_labels)
    nome = " ".join(l.strip() for l in sem_labels.split("\n") if l.strip())

    return {"Razão social": nome, "CNPJ/CPF": cnpj, "Inscrição Estadual": insc}


def handler_domicilio(bloco: str) -> Dict[str, str]:
    return {
        "Endereço": _extrai_label_no_bloco(bloco, r"Endereço:"),
        "Tel": _extrai_label_no_bloco(bloco, r"Telefone:"),
        "E-mail": _extrai_label_no_bloco(bloco, r"E-mail:"),
    }


def handler_ncm_naladi_denominacao(bloco: str) -> Dict[str, str]:
    codigos = re.findall(r"\d{4}\.\d{2}\.\d{2}", bloco)
    ncm = codigos[0] if len(codigos) > 0 else ""
    naladi = codigos[1] if len(codigos) > 1 else ""

    sem_header = re.sub(r"Denominação comercial do produto a exportar", " ", bloco)
    for c in codigos[:2]:
        sem_header = sem_header.replace(c, " ", 1)

    return {
        "Código NCM": ncm,
        "NALADI": naladi,
        "Denominação comercial do produto a exportar": " ".join(sem_header.split()),
    }


# --- material tables ---------------------------------------------------
def _tokenize_columns(line: str):
    return list(re.finditer(r"\S+(?:[ \t]\S+)*", line))


# Country names (always uppercase in these documents' material tables),
# used ONLY as a fallback to split Descrição from País origem when the OCR
# doesn't leave the usual double-space gap between the two columns (e.g.
# "...Outros. CHILE" with a single space). Not an exhaustive list of every
# country — covers Mercosul and common trade partners; a country outside
# the list simply doesn't trigger the fallback (never worse than before).
KNOWN_COUNTRIES = {
    "BRASIL", "ARGENTINA", "PARAGUAI", "URUGUAI", "VENEZUELA", "BOLÍVIA",
    "CHILE", "COLÔMBIA", "PERU", "EQUADOR", "CHINA", "ITÁLIA", "ALEMANHA",
    "FRANÇA", "ESPANHA", "PORTUGAL", "JAPÃO", "ÍNDIA", "MÉXICO", "CANADÁ",
    "HOLANDA", "BÉLGICA", "SUÍÇA", "ÁUSTRIA", "RÚSSIA", "TAIWAN", "VIETNÃ",
    "TAILÂNDIA", "INDONÉSIA", "MALÁSIA", "TURQUIA", "ISRAEL", "AUSTRÁLIA",
    "CUBA", "PANAMÁ", "POLÔNIA", "HUNGRIA", "ROMÊNIA", "UCRÂNIA",
    "DINAMARCA", "SUÉCIA", "NORUEGA", "FINLÂNDIA", "IRLANDA", "GRÉCIA",
    "EGITO", "MARROCOS", "NIGÉRIA", "CINGAPURA", "FILIPINAS",
    "ESTADOS UNIDOS", "REINO UNIDO", "COREIA DO SUL", "COREIA DO NORTE",
    "PAÍSES BAIXOS", "ÁFRICA DO SUL", "NOVA ZELÂNDIA", "COSTA RICA",
    "EL SALVADOR", "REPÚBLICA DOMINICANA", "REPÚBLICA TCHECA",
    "ARÁBIA SAUDITA", "EMIRADOS ÁRABES UNIDOS", "HONG KONG",
}
_COUNTRY_WORD_COUNTS = sorted({len(c.split()) for c in KNOWN_COUNTRIES}, reverse=True)


def _split_descricao_pais(merged_text: str):
    words = merged_text.split(" ")
    for word_count in _COUNTRY_WORD_COUNTS:
        if len(words) <= word_count:
            continue
        tail = " ".join(words[-word_count:])
        if tail.upper() in KNOWN_COUNTRIES:
            return " ".join(words[:-word_count]).strip(), tail
    return merged_text.strip(), ""


def _parse_item_row(line: str, ncm_token_match):
    ncm = ncm_token_match.group().strip()
    base = ncm_token_match.end()
    resto_bruto = line[base:]

    pct_matches = list(re.finditer(r"[\d.,]+\s*%", resto_bruto))
    if pct_matches:
        pct_m = pct_matches[-1]
        pct_val = re.sub(r"\s+", "", pct_m.group())
        antes_pct = resto_bruto[:pct_m.start()]
        pos_pct_abs = base + pct_m.start()
        pos_fornecedor_abs = base + pct_m.end()
        fornecedor = resto_bruto[pct_m.end():].strip()
    else:
        pct_val = ""
        antes_pct = resto_bruto
        pos_pct_abs = base + len(resto_bruto)
        pos_fornecedor_abs = pos_pct_abs
        fornecedor = ""

    valor_matches = list(re.finditer(r"\[Não Informado\]|[\d.,]+", antes_pct))
    if valor_matches:
        valor_m = valor_matches[-1]
        valor_val = valor_m.group()
        antes_valor = antes_pct[:valor_m.start()]
        pos_valor_abs = base + valor_m.start()
    else:
        valor_val = ""
        antes_valor = antes_pct
        pos_valor_abs = base + len(antes_pct)

    subtokens = _tokenize_columns(antes_valor)
    if len(subtokens) >= 2:
        descricao = subtokens[0].group().strip()
        pais = " ".join(t.group().strip() for t in subtokens[1:])
        pos_descricao_abs = base + subtokens[0].start()
        pos_pais_abs = base + subtokens[1].start()
    elif len(subtokens) == 1:
        merged = subtokens[0].group().strip()
        descricao, pais = _split_descricao_pais(merged)
        pos_descricao_abs = base + subtokens[0].start()
        if pais:
            pos_pais_abs = pos_descricao_abs + (len(merged) - len(pais))
        else:
            pos_pais_abs = pos_valor_abs
    else:
        descricao, pais = "", ""
        pos_descricao_abs = pos_pais_abs = base

    item = {
        "NCM/SH": ncm,
        "Descrição": descricao,
        "País origem": pais,
        "Valor (US$)": valor_val,
        "% s/Valor FOB": pct_val,
        "Fornecedor/Fabricante": fornecedor,
    }
    ancoras = [
        ("Descrição", pos_descricao_abs),
        ("País origem", pos_pais_abs),
        ("Valor (US$)", pos_valor_abs),
        ("% s/Valor FOB", pos_pct_abs),
        ("Fornecedor/Fabricante", pos_fornecedor_abs),
    ]
    return item, ancoras


def handler_tabela(texto_secao: str) -> Dict[str, Any]:
    linhas = texto_secao.split("\n")
    nao_vazias = [l for l in linhas if l.strip()]

    resultado: Dict[str, Any] = {"Items": [], "Somatório": ""}
    if not nao_vazias:
        return resultado
    if len(nao_vazias) == 1 and "Não Informado" in nao_vazias[0]:
        return resultado

    m_som = re.search(r"Somat.rio:\s*([\d.,]+%)", texto_secao)
    if m_som:
        resultado["Somatório"] = m_som.group(1)

    header_idx = next((i for i, l in enumerate(linhas) if "NCM/SH" in l), None)
    if header_idx is None:
        return resultado

    header_line = linhas[header_idx]
    ncm_col_start = len(header_line) - len(header_line.lstrip())

    item_atual = None
    ancoras_atuais = None
    itens = []

    for linha in linhas[header_idx + 1:]:
        if not linha.strip():
            continue
        if re.search(r"Somat.rio:\s*[\d.,]+%", linha):
            break
        if "NCM/SH" in linha:
            continue

        indent = len(linha) - len(linha.lstrip())
        ncm_m = re.match(r"^\s*(\d{4}\.\d{2}\.\d{2}|\[Não Informado\])", linha)
        nova_linha = ncm_m is not None and abs(indent - ncm_col_start) <= 3

        if nova_linha:
            if item_atual:
                itens.append(item_atual)
            item_atual, ancoras_atuais = _parse_item_row(linha, ncm_m)
        else:
            if item_atual is None or not ancoras_atuais:
                continue
            for tok in _tokenize_columns(linha):
                texto_tok = tok.group().strip()
                melhor_campo = min(ancoras_atuais, key=lambda a: abs(a[1] - tok.start()))[0]
                if item_atual[melhor_campo]:
                    item_atual[melhor_campo] += " " + texto_tok
                else:
                    item_atual[melhor_campo] = texto_tok

    if item_atual:
        itens.append(item_atual)

    for item in itens:
        m_pct = re.match(r"\s*([\d.,]+%)", item["% s/Valor FOB"])
        if m_pct:
            item["% s/Valor FOB"] = m_pct.group(1)

    resultado["Items"] = itens
    return resultado


# ---------------------------------------------------------------------------
# 3) Ordered key list — order matches the real document flow
# ---------------------------------------------------------------------------
def build_keys() -> List[Dict[str, Any]]:
    return [
        {"name": "numero_djo", "pattern": r"Número da DJO:", "handler": handler_texto},
        {"name": "data_apresentacao", "pattern": r"Data de apresentação:", "handler": handler_texto},
        {"name": "razao_social_produtor", "pattern": r"Razão social do produtor:", "handler": handler_empresa},
        {"name": "domicilio_produtor", "pattern": r"Domicílio legal e parque industrial:", "handler": handler_domicilio},
        {"name": "razao_social_exportador", "pattern": r"Razão social do exportador:", "handler": handler_empresa},
        {"name": "domicilio_exportador", "pattern": r"Domicílio legal e parque industrial:", "handler": handler_domicilio},
        {"name": "ncm_naladi_denominacao", "pattern": r"Código NCM\s*-\s*NALADI:", "handler": handler_ncm_naladi_denominacao, "merge": True},
        {"name": "valor_fob_usd", "pattern": r"Valor FOB \(USD\):", "handler": handler_texto},
        {"name": "unidade_medida", "pattern": r"Unidade de medida:", "handler": handler_texto},
        {"name": "descricao_processo_produtivo", "pattern": r"Descrição do processo produtivo:", "handler": handler_texto},
        {"name": "materiais_utilizados_header", "pattern": r"Materiais utilizados:", "handler": None},
        {"name": "originarios_estado_parte_produtor", "pattern": r"Materiais originários do Estado Parte produtor:", "handler": handler_tabela},
        {"name": "originarios_outros_estados_partes", "pattern": r"Materiais originários de outros Estados Partes:", "handler": handler_tabela},
        {"name": "nao_originarios", "pattern": r"Materiais não originários:", "handler": handler_tabela},
        {"name": "terceiros_paises_ptc", "pattern": r"Materiais de terceiros países que tenham cumprido com a PTC:", "handler": handler_tabela},
        {"name": "porcentagem_total_materias_primas", "pattern": r"Porcentagem Total de Mat[eé]rias Primas, Componentes ou Partes:", "handler": handler_texto},
        {"name": "valor_agregado_processo_industrial", "pattern": r"Valor agregado no processo Industrial \(Deduzidos os tributos restituídos ou a restituir em caso de exportação\):", "handler": handler_texto},
        {"name": "preco_fob", "pattern": r"Preço FOB:", "handler": handler_texto},
        {"name": "observacoes", "pattern": r"Observações:", "handler": handler_texto},
    ]


# ---------------------------------------------------------------------------
# 4) Public entry point
# ---------------------------------------------------------------------------
def normalize(extracted_text: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Turns raw OCR text into the structured json_djo dict. Returns
    (json_djo, debug_blocks) — debug_blocks lets callers inspect exactly
    what raw text was captured for each field, in document order, useful
    for diagnosing a document that doesn't match the expected DJO layout.
    """
    codigo_aprovacao, texto_limpo = preprocess_ocr_text(extracted_text)
    campos, blocos_debug = sequential_extract(texto_limpo, build_keys())

    vazio_tabela = {"Items": [], "Somatório": ""}

    json_final = {
        "Código da aprovação DJO": codigo_aprovacao,
        "Número da DJO": campos.get("numero_djo", ""),
        "Data de apresentação": campos.get("data_apresentacao", ""),
        "Razão social do produtor": campos.get("razao_social_produtor", {}),
        "Domicílio legal e parque industrial do produtor": campos.get("domicilio_produtor", {}),
        "Razão social do exportador": campos.get("razao_social_exportador", {}),
        "Domicílio legal e parque industrial do exportador": campos.get("domicilio_exportador", {}),
        "Código NCM": campos.get("Código NCM", ""),
        "NALADI": campos.get("NALADI", ""),
        "Denominação comercial do produto a exportar": campos.get("Denominação comercial do produto a exportar", ""),
        "Valor FOB (USD)": campos.get("valor_fob_usd", ""),
        "Unidade de medida": campos.get("unidade_medida", ""),
        "Descrição do processo produtivo": campos.get("descricao_processo_produtivo", ""),
        "Materiales": {
            "originarios_estado_parte_produtor": campos.get("originarios_estado_parte_produtor", vazio_tabela),
            "originarios_outros_estados_partes": campos.get("originarios_outros_estados_partes", vazio_tabela),
            "nao_originarios": campos.get("nao_originarios", vazio_tabela),
            "terceiros_paises_ptc": campos.get("terceiros_paises_ptc", vazio_tabela),
            "Porcentagem Total de Matérias Primas, Componentes ou Partes": campos.get("porcentagem_total_materias_primas", ""),
            "Valor agregado no processo Industrial (Deduzidos os tributos restituídos ou a restituir em caso de exportação)": campos.get("valor_agregado_processo_industrial", ""),
            "Preço FOB": campos.get("preco_fob", ""),
        },
        "Observações": campos.get("observacoes", ""),
    }
    return json_final, blocos_debug


# Backwards-compatible alias matching the notebook's function name.
parsear_djo = normalize
