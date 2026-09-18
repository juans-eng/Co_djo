# -*- coding: utf-8 -*-
"""
gemini_service.py
==================

Thin wrapper around the Gemini API, used by the DJO validation core for the
three tasks in this system that genuinely need language understanding
rather than deterministic logic:

  1. `match_process_materials(...)` — Rule 5: decide whether the raw
     materials/inputs named in "Descrição do processo produtivo" are
     represented in the material tables' "Descrição" columns. This requires
     semantic judgement (e.g. "tecido" vs "TECIDO PERCAL 233 FIOS ..."),
     which is why it is delegated to an LLM instead of a keyword search.

  2. `summarize_validation_result(...)` / `summarize_origin_rule_result(...)`
     — human-readable summaries of the already-computed, deterministic
     `validation_result` / origin-rule result produced by the validation
     core. Both are presentation only: they never decide PASS/FAIL or any
     status, they just narrate a result that was already decided in
     Python, in the same tone/style so the two read as one consistent
     summary when concatenated (see djo_validation_core.ipynb's
     `generate_full_report`).

  3. `evaluate_rule_condition(...)` — origin-rule engine (ACE 18
     flowchart): a `mercosul_rule.raw_rule` clause that doesn't match any
     of the deterministic patterns (MP / MSP / MaxMNO / excluded
     positions) — e.g. "Reação química", "Processo biotecnológico", or a
     free-text requirement like "Devem ser elaborados a partir de leite
     produzido nos Estados Partes" — is a semantic judgement call against
     the process description, so it's delegated here as a fallback. It is
     only ever invoked when the deterministic conditions in the same rule
     could not already decide the outcome on their own (see
     djo_validation_core.ipynb's `evaluate_rule_conditions`).

All other business rules in this project are deterministic and MUST stay in
Python (see djo_validation_core.ipynb) — Gemini is not used to "decide" any
of them.

Configuration is read from environment variables (see `.env` at the project
root, loaded via `python-dotenv` by the caller):

    GEMINI_API_KEY   - required to actually call the API
    GEMINI_MODEL     - optional, defaults to "gemini-3.6-flash"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiServiceError(Exception):
    """Raised whenever the Gemini call cannot be trusted as a result: no API
    key configured, a request/network failure, or a response that doesn't
    parse into the expected structured shape. The validation core is
    expected to catch this and report MANUAL_VERIFICATION rather than
    crash or silently guess a PASS/FAIL."""


@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        )


def _get_client(config: GeminiConfig) -> genai.Client:
    if not config.api_key:
        raise GeminiServiceError(
            "GEMINI_API_KEY is not set. Create a .env file at the project "
            "root (see .env.example) with your Gemini API key."
        )
    return genai.Client(api_key=config.api_key)


# ---------------------------------------------------------------------------
# Rule 5: process-description materials vs. material-table descriptions
# ---------------------------------------------------------------------------
MATERIAL_MATCH_PROMPT_TEMPLATE = """Você é um assistente especializado em análise de documentos aduaneiros do Mercosul (Declaração Juramentada de Origem - DJO).

TAREFA (faça APENAS isto, nada além disso):
1. Leia a descrição do processo produtivo abaixo e identifique todos os
   insumos/matérias-primas EXPLICITAMENTE mencionados como entrada do
   processo.
2. NÃO inclua produtos intermediários ou finais que resultem da combinação
   ou reação de outros insumos. Por exemplo, em "reagir o insumo A com o
   insumo B para obter C", apenas A e B são insumos; C não deve ser
   listado.
3. NÃO invente insumos que não estejam explicitamente escritos no texto.
4. Para cada insumo identificado, verifique se ele está representado em
   alguma das descrições de materiais listadas abaixo (coluna "Descrição"
   das tabelas de materiais do DJO). Prefira correspondência textual exata,
   mas aceite equivalência semântica razoável (ex.: "tecido" corresponde a
   "TECIDO PERCAL 233 FIOS 134 G/M2 100% ALGODÃO").
5. NÃO afirme que um insumo foi encontrado se a correspondência não for
   razoável — nesse caso marque found=false.

DESCRIÇÃO DO PROCESSO PRODUTIVO:
\"\"\"
{descricao_processo}
\"\"\"

DESCRIÇÕES DE MATERIAIS DISPONÍVEIS NAS TABELAS DO DJO:
{materiais_listados}

Responda APENAS com um JSON no seguinte formato exato (sem texto adicional,
sem markdown, sem comentários):

{{
  "process_inputs": [
    {{
      "input": "<insumo identificado no texto do processo, em minúsculas>",
      "found": true | false,
      "matched_description": "<texto exato da coluna Descrição que corresponde, ou null se found=false>",
      "table": "<nome da tabela onde matched_description foi encontrado, ou null se found=false>"
    }}
  ],
  "all_inputs_found": true | false
}}

"all_inputs_found" deve ser true somente se TODOS os itens de
"process_inputs" tiverem found=true. Se nenhum insumo for identificado no
texto do processo, retorne "process_inputs": [] e "all_inputs_found": true.
"""


def _format_materials_for_prompt(material_items: List[Dict[str, str]]) -> str:
    if not material_items:
        return "(nenhuma descrição de material disponível nas tabelas do DJO)"
    lines = []
    for item in material_items:
        lines.append(f'- [{item["table"]}] "{item["descricao"]}"')
    return "\n".join(lines)


def match_process_materials(
    descricao_processo: str,
    material_items: List[Dict[str, str]],
    config: Optional[GeminiConfig] = None,
) -> Dict[str, Any]:
    """Calls Gemini to decide whether the inputs named in
    `descricao_processo` are represented in `material_items`.

    `material_items` is a flat list of {"table": <table_name>, "descricao":
    <Descrição value>} dicts, gathered across the 4 material tables (see
    djo_validation_core.ipynb's `collect_material_descriptions`).

    Returns a dict shaped exactly like the JSON schema in the prompt.
    Raises GeminiServiceError if the call fails or the response cannot be
    parsed into that shape — callers must treat that as
    MANUAL_VERIFICATION, never as an implicit PASS or FAIL.
    """
    config = config or GeminiConfig.from_env()
    client = _get_client(config)

    prompt = MATERIAL_MATCH_PROMPT_TEMPLATE.format(
        descricao_processo=descricao_processo.strip(),
        materiais_listados=_format_materials_for_prompt(material_items),
    )

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
    except Exception as exc:  # network/auth/quota/etc.
        raise GeminiServiceError(f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiServiceError(
            f"Gemini response was not valid JSON: {exc}. Raw response: {text[:500]}"
        ) from exc

    if "process_inputs" not in data or "all_inputs_found" not in data:
        raise GeminiServiceError(
            f"Gemini response is missing required keys. Got: {list(data.keys())}"
        )

    return data


# ---------------------------------------------------------------------------
# Final human-readable summary of the whole validation_result
# ---------------------------------------------------------------------------
SUMMARY_PROMPT_TEMPLATE = """Você é um assistente que explica, de forma clara e objetiva, o resultado de
uma validação automática de uma Declaração Juramentada de Origem (DJO) do
Mercosul para um analista de comércio exterior.

IMPORTANTE: os resultados abaixo já foram decididos por regras
determinísticas em Python. NÃO reavalie, não mude e não contradiga nenhum
status (PASS, FAIL, MANUAL_VERIFICATION, NOT_APPLICABLE) — apenas explique o
que eles significam, em português, de forma resumida e acionável.

RESULTADO DA VALIDAÇÃO (JSON):
{validation_result_json}

Escreva um resumo em português com:
1. Uma frase inicial com o status geral.
2. Uma lista curta (bullet points) destacando as regras que falharam
   (FAIL) ou que precisam de verificação manual (MANUAL_VERIFICATION), com
   uma explicação objetiva de cada uma.
3. Se todas as regras passaram, diga isso claramente em vez de gerar uma
   lista vazia.

Não inclua as regras "PASS" ou "NOT_APPLICABLE" na lista de destaques,
apenas mencione que elas passaram/não se aplicam na frase inicial, se
relevante. Responda em texto simples (sem markdown pesado, sem JSON).
"""


def summarize_validation_result(
    validation_result: Dict[str, Any],
    config: Optional[GeminiConfig] = None,
) -> str:
    """Generates a human-readable (Portuguese) summary of an already-computed
    validation_result, for eventual display in the frontend. This is purely
    presentational: it must never be used to alter or re-decide any rule's
    status. Raises GeminiServiceError on failure — the caller should fall
    back to showing the raw structured result if this is unavailable."""
    config = config or GeminiConfig.from_env()
    client = _get_client(config)

    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        validation_result_json=json.dumps(validation_result, ensure_ascii=False, indent=2)
    )

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except Exception as exc:
        raise GeminiServiceError(f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise GeminiServiceError("Gemini returned an empty summary.")
    return text


# ---------------------------------------------------------------------------
# Human-readable summary of the origin-rule (ACE 18 flowchart) result —
# same style/tone as summarize_validation_result above, so the two read as
# ONE consistent "type" of response when concatenated, instead of a polished
# paragraph followed by a raw numbered decision-trace dump.
# ---------------------------------------------------------------------------
ORIGIN_RULE_SUMMARY_PROMPT_TEMPLATE = """Você é um assistente que explica, de forma clara e objetiva, o resultado da
determinação da norma de origem do ACE 18 (Mercosul) para uma Declaração
Juramentada de Origem (DJO), para um analista de comércio exterior.

IMPORTANTE: o resultado abaixo já foi decidido por uma lógica
determinística em Python, seguindo o fluxograma oficial do ACE 18. NÃO
reavalie, não mude e não contradiga o status final (final_status) nem a
norma de origem aplicável (applicable_origin_rule) — apenas explique o que
eles significam e narre, em português, o caminho lógico percorrido no
fluxograma (decision_trace) até chegar a esse resultado.

RESULTADO DA DETERMINAÇÃO DA NORMA DE ORIGEM (JSON):
{origin_result_json}

Escreva um resumo em português com:
1. Uma frase inicial dizendo claramente qual foi o resultado: a norma de
   origem aplicável (A, B ou C), ou, se for o caso, que o produto não
   confere origem, que a DJO precisa ser corrigida, que é necessária
   verificação manual, ou que o fluxograma não se aplica a este produto.
2. Um parágrafo curto narrando o caminho percorrido no fluxograma (as
   principais perguntas/decisões, na ordem em que ocorreram, com base em
   decision_trace) que levou a esse resultado — por exemplo, se o produto
   tinha materiais de terceiros países, qual a NCM e a regra do Mercosul
   aplicada, quais condições (MP, MSP, MaxMNO, etc.) foram avaliadas e o
   que decidiram.

Responda em texto simples (sem markdown pesado, sem JSON, sem lista
numerada) — um texto corrido, no mesmo estilo de um resumo de validação
para o mesmo leitor.
"""


def summarize_origin_rule_result(
    origin_result: Dict[str, Any],
    config: Optional[GeminiConfig] = None,
) -> str:
    """Generates a human-readable (Portuguese) narration of an
    already-computed origin-rule result (evaluate_origin_rule's return
    value: final_status, applicable_origin_rule, decision_trace, ...),
    in the same presentational style as `summarize_validation_result` —
    intended to be concatenated with it into one consistent-sounding
    summary rather than mixed with the raw decision_trace. Never alters or
    re-decides final_status/applicable_origin_rule. Raises
    GeminiServiceError on failure — the caller should fall back to the
    deterministic `simple_explanation` field if this is unavailable."""
    config = config or GeminiConfig.from_env()
    client = _get_client(config)

    # Only the fields relevant to narrating the flowchart outcome — avoids
    # re-sending the (already summarized elsewhere) full validation_result.
    payload = {
        "final_status": origin_result.get("final_status"),
        "applicable_origin_rule": origin_result.get("applicable_origin_rule"),
        "ncm_rule_info": origin_result.get("ncm_rule_info"),
        "decision_trace": origin_result.get("decision_trace"),
    }

    prompt = ORIGIN_RULE_SUMMARY_PROMPT_TEMPLATE.format(
        origin_result_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except Exception as exc:
        raise GeminiServiceError(f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    if not text:
        raise GeminiServiceError("Gemini returned an empty origin-rule summary.")
    return text


# ---------------------------------------------------------------------------
# Origin-rule engine fallback: free-text mercosul_rule conditions that don't
# match any deterministic pattern (MP / MSP / MaxMNO / excluded positions)
# ---------------------------------------------------------------------------
RULE_CONDITION_PROMPT_TEMPLATE = """Você é um especialista em regras de origem do Mercosul (ACE 18).

Uma regra de origem aplicável à NCM do produto contém a(s) seguinte(s)
condição(ões), em texto livre, que não puderam ser avaliadas por lógica
determinística:

{condition_texts}

Sua ÚNICA tarefa é decidir se o produto descrito abaixo SATISFAZ pelo menos
uma dessas condições (se mais de uma for listada, trate-as como alternativas
equivalentes — basta uma ser satisfeita para o resultado ser true).

Baseie sua decisão SOMENTE na descrição do processo produtivo e nos
materiais listados abaixo. NÃO invente informações que não estejam
presentes nesses dados. Se a informação disponível não for suficiente para
decidir com segurança, responda "result": null (não adivinhe).

DESCRIÇÃO DO PROCESSO PRODUTIVO:
\"\"\"
{descricao_processo}
\"\"\"

MATERIAIS NÃO ORIGINÁRIOS / DE TERCEIROS PAÍSES DECLARADOS NA DJO:
{materiais_listados}

Responda APENAS com um JSON no formato exato (sem texto adicional, sem
markdown):

{{
  "result": true | false | null,
  "reasoning": "<explicação objetiva, em português, da sua decisão, citando o que no processo ou nos materiais levou a ela>"
}}
"""


def _format_rule_condition_materials(material_items: List[Dict[str, str]]) -> str:
    if not material_items:
        return "(nenhum material não originário/de terceiros países declarado)"
    lines = []
    for item in material_items:
        ncm = item.get("ncm") or "?"
        pais = item.get("pais_origem") or "?"
        descricao = item.get("descricao") or ""
        tabela = item.get("table") or "?"
        lines.append(f'- [{tabela}] NCM {ncm} ({pais}): "{descricao}"')
    return "\n".join(lines)


def evaluate_rule_condition(
    condition_texts: List[str],
    descricao_processo: str,
    material_items: List[Dict[str, str]],
    config: Optional[GeminiConfig] = None,
) -> Dict[str, Any]:
    """Asks Gemini whether the product satisfies at least one of
    `condition_texts` (free-text mercosul_rule clauses the deterministic
    parser couldn't classify), given the process description and the
    non-originating/third-country material items.

    `material_items`: list of {"table", "ncm", "pais_origem", "descricao"}.

    Returns {"result": true|false|null, "reasoning": str}. Raises
    GeminiServiceError on failure — callers must treat that (and a `null`
    result) as inconclusive, i.e. MANUAL_VERIFICATION, never as an assumed
    pass or fail.
    """
    config = config or GeminiConfig.from_env()
    client = _get_client(config)

    prompt = RULE_CONDITION_PROMPT_TEMPLATE.format(
        condition_texts="\n".join(f"- {c}" for c in condition_texts),
        descricao_processo=descricao_processo.strip(),
        materiais_listados=_format_rule_condition_materials(material_items),
    )

    try:
        response = client.models.generate_content(
            model=config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )
    except Exception as exc:
        raise GeminiServiceError(f"Gemini request failed: {exc}") from exc

    text = (response.text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiServiceError(
            f"Gemini response was not valid JSON: {exc}. Raw response: {text[:500]}"
        ) from exc

    if "result" not in data:
        raise GeminiServiceError(f"Gemini response is missing 'result' key. Got: {list(data.keys())}")

    return data

