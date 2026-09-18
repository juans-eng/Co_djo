# -*- coding: utf-8 -*-
"""
gemini/service.py
===================

Thin wrapper around the Gemini API — the ONLY module in this backend that
imports the `google.genai` SDK directly. Ported from `codigo/gemini_service.py`,
with one addition: simple request pacing (see `_rate_limit` /
`GEMINI_MIN_INTERVAL_SECONDS`) plus a short retry with backoff on transient
errors (HTTP 429/503 - "rate limited" / "high demand"), so a burst of calls
(Rule 5 + both summaries + any origin-rule Gemini fallback, all in one
`djo_service.validate_djo(...)` run) doesn't trip the API's requests-per-
minute limit. Every other caller in this backend goes through this module
— nothing else touches the Gemini SDK.

Four tasks genuinely need language understanding rather than deterministic
logic, and all four live here:

  1. `match_process_materials(...)` — Rule 5: decide whether the raw
     materials/inputs named in "Descrição do processo produtivo" are
     represented in the material tables' "Descrição" columns.

  2. `summarize_validation_result(...)` / `summarize_origin_rule_result(...)`
     — human-readable (Portuguese) narration of the already-computed,
     deterministic validation_result / origin-rule result. Presentation
     only: neither ever decides a status, they narrate one already decided
     in Python, in the same tone/style so they read as one summary when
     concatenated (see `services/djo_service.py`).

  3. `evaluate_rule_condition(...)` — origin-rule engine (ACE 18
     flowchart): a `mercosul_rule.raw_rule` clause that doesn't match any
     deterministic pattern (MP / MSP / MaxMNO / excluded positions) — e.g.
     "Reação química" — is a semantic judgement call, delegated here as a
     fallback. Only invoked when the deterministic conditions in the same
     rule could not already decide the outcome on their own (see
     `ace18/rule_engine.py`).

All other business rules in this backend are deterministic and MUST stay
in Python — Gemini is not used to "decide" any of them.

Configuration is read from environment variables (see `.env.example`):

    GEMINI_API_KEY               - required to actually call the API
    GEMINI_MODEL                 - optional, defaults to "gemini-3.6-flash"
    GEMINI_MIN_INTERVAL_SECONDS  - optional, defaults to 4.0 (rate pacing)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_MIN_INTERVAL_SECONDS = 4.0  # conservative pacing for typical free-tier RPM limits
MAX_RETRIES_ON_TRANSIENT_ERROR = 2


class GeminiServiceError(Exception):
    """Raised whenever the Gemini call cannot be trusted as a result: no API
    key configured, a request/network failure, or a response that doesn't
    parse into the expected structured shape. Callers are expected to catch
    this and report MANUAL_VERIFICATION (or an equivalent fallback) rather
    than crash or silently guess a result."""


@dataclass
class GeminiConfig:
    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(
            api_key=os.environ.get("GEMINI_API_KEY") or None,
            model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
            min_interval_seconds=float(os.environ.get("GEMINI_MIN_INTERVAL_SECONDS", DEFAULT_MIN_INTERVAL_SECONDS)),
        )


def _get_client(config: GeminiConfig) -> genai.Client:
    if not config.api_key:
        raise GeminiServiceError(
            "GEMINI_API_KEY is not set. Add it to your .env file "
            "(see .env.example) before requesting a Gemini-backed step."
        )
    return genai.Client(api_key=config.api_key)


# ---------------------------------------------------------------------------
# Request pacing + transient-error retry (section 15: avoid RPM problems)
# ---------------------------------------------------------------------------
_last_call_at: float = 0.0


def _rate_limit(min_interval_seconds: float) -> None:
    """Sleeps just enough to keep consecutive Gemini calls from this
    process at least `min_interval_seconds` apart. Process-wide (module
    level), so it paces calls across Rule 5, both summaries, and any
    origin-rule fallback evaluation happening in the same request."""
    global _last_call_at
    if min_interval_seconds <= 0:
        return
    elapsed = time.monotonic() - _last_call_at
    remaining = min_interval_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _last_call_at = time.monotonic()


def _is_transient_error(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "503" in text or "RESOURCE_EXHAUSTED" in text or "UNAVAILABLE" in text


def _generate_content(config: GeminiConfig, prompt: str, *, json_mode: bool, temperature: float):
    """Shared call path for every Gemini request in this module: paces
    requests, retries a bounded number of times on transient (429/503)
    errors with a short backoff, and always raises GeminiServiceError
    (never the raw SDK exception) on final failure."""
    client = _get_client(config)
    gen_config = types.GenerateContentConfig(
        temperature=temperature,
        **({"response_mime_type": "application/json"} if json_mode else {}),
    )

    last_exc: Optional[Exception] = None
    for attempt in range(MAX_RETRIES_ON_TRANSIENT_ERROR + 1):
        _rate_limit(config.min_interval_seconds)
        try:
            return client.models.generate_content(model=config.model, contents=prompt, config=gen_config)
        except Exception as exc:  # network/auth/quota/model errors from the SDK
            last_exc = exc
            if attempt < MAX_RETRIES_ON_TRANSIENT_ERROR and _is_transient_error(exc):
                time.sleep(2 ** attempt)  # 1s, 2s backoff before the next attempt
                continue
            raise GeminiServiceError(f"Gemini request failed: {exc}") from exc

    # Unreachable in practice (the loop above always returns or raises),
    # kept only to satisfy static analysis.
    raise GeminiServiceError(f"Gemini request failed: {last_exc}")


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
    return "\n".join(f'- [{item["table"]}] "{item["descricao"]}"' for item in material_items)


def match_process_materials(
    descricao_processo: str,
    material_items: List[Dict[str, str]],
    config: Optional[GeminiConfig] = None,
) -> Dict[str, Any]:
    """Calls Gemini to decide whether the inputs named in
    `descricao_processo` are represented in `material_items`.

    `material_items`: [{"table": <table_name>, "descricao": <Descrição value>}, ...].

    Returns a dict shaped exactly like the JSON schema in the prompt.
    Raises GeminiServiceError if the call fails or the response cannot be
    parsed into that shape — callers must treat that as inconclusive, never
    as an implicit PASS or FAIL.
    """
    config = config or GeminiConfig.from_env()
    prompt = MATERIAL_MATCH_PROMPT_TEMPLATE.format(
        descricao_processo=descricao_processo.strip(),
        materiais_listados=_format_materials_for_prompt(material_items),
    )

    response = _generate_content(config, prompt, json_mode=True, temperature=0)

    text = (response.text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GeminiServiceError(
            f"Gemini response was not valid JSON: {exc}. Raw response: {text[:500]}"
        ) from exc

    if "process_inputs" not in data or "all_inputs_found" not in data:
        raise GeminiServiceError(f"Gemini response is missing required keys. Got: {list(data.keys())}")

    return data


# ---------------------------------------------------------------------------
# Human-readable summary of the whole validation_result
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
    validation_result. Purely presentational — never alters or re-decides
    any rule's status. Raises GeminiServiceError on failure — the caller
    should fall back to showing the raw structured result if unavailable."""
    config = config or GeminiConfig.from_env()
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        validation_result_json=json.dumps(validation_result, ensure_ascii=False, indent=2)
    )

    response = _generate_content(config, prompt, json_mode=False, temperature=0.2)

    text = (response.text or "").strip()
    if not text:
        raise GeminiServiceError("Gemini returned an empty summary.")
    return text


# ---------------------------------------------------------------------------
# Human-readable summary of the origin-rule (ACE 18 flowchart) result — same
# style/tone as summarize_validation_result, so the two read as ONE
# consistent summary when concatenated.
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
    already-computed origin-rule result. Never alters or re-decides
    final_status/applicable_origin_rule. Raises GeminiServiceError on
    failure — the caller should fall back to the deterministic
    `simple_explanation` field if unavailable."""
    config = config or GeminiConfig.from_env()
    payload = {
        "final_status": origin_result.get("final_status"),
        "applicable_origin_rule": origin_result.get("applicable_origin_rule"),
        "ncm_rule_info": origin_result.get("ncm_rule_info"),
        "decision_trace": origin_result.get("decision_trace"),
    }
    prompt = ORIGIN_RULE_SUMMARY_PROMPT_TEMPLATE.format(
        origin_result_json=json.dumps(payload, ensure_ascii=False, indent=2)
    )

    response = _generate_content(config, prompt, json_mode=False, temperature=0.2)

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

    `material_items`: [{"table", "ncm", "pais_origem", "descricao"}, ...].

    Returns {"result": true|false|null, "reasoning": str}. Raises
    GeminiServiceError on failure — callers must treat that (and a `null`
    result) as inconclusive, never as an assumed pass or fail.
    """
    config = config or GeminiConfig.from_env()
    prompt = RULE_CONDITION_PROMPT_TEMPLATE.format(
        condition_texts="\n".join(f"- {c}" for c in condition_texts),
        descricao_processo=descricao_processo.strip(),
        materiais_listados=_format_rule_condition_materials(material_items),
    )

    response = _generate_content(config, prompt, json_mode=True, temperature=0)

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
