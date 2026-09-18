# -*- coding: utf-8 -*-
"""
app.py
=======

Streamlit MVP for the DJO / ACE 18 validation backend. This file is a
thin presentation/orchestration layer ONLY: it collects the three inputs
(PDF, product category, LLM provider/model) from the sidebar, makes
exactly one call to `services.djo_service.validate_djo_pdf(...)`, and
renders the already-structured, already-decided response. No validation
rule, no ACE 18 lookup, and no LLM call is implemented here — see
ARCHITECTURE.md for the full pipeline this sits on top of.

All user-facing text is Brazilian Portuguese; internal identifiers
(rule slugs, JSON keys such as `raw_rule`/`matched_ncm_expression`,
material-table names) are left untouched — only their on-screen labels
are translated.

Run with:

    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import streamlit as st

# Make `import src...` work regardless of Streamlit's working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()  # local dev: GEMINI_API_KEY / GROQ_API_KEY / LLMWHISPERER_API_KEY from .env — never sent to the browser

# Streamlit Community Cloud provides configured secrets via `st.secrets`
# (the dashboard's "Secrets" panel), not automatically as OS environment
# variables in every Streamlit version. Every module below still reads
# credentials via `os.environ.get(...)` unchanged (see
# src/gemini/service.py, src/llm/groq.py, src/djo/extractor.py) — this
# bridges `st.secrets` into `os.environ` once at startup so that same code
# keeps working locally (.env) and on Streamlit Cloud (st.secrets) without
# any business-logic change. Never overwrites a value already set (e.g.
# from a real local .env), and never raises when no secrets.toml exists
# (plain local dev) since `st.secrets` behaves as empty in that case.
try:
    for _env_key in (
        "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_MIN_INTERVAL_SECONDS",
        "GROQ_API_KEY", "GROQ_MODEL", "GROQ_MIN_INTERVAL_SECONDS",
        "LLMWHISPERER_API_KEY", "LLMWHISPERER_WAIT_TIMEOUT",
        "ACE18_PDF_PATH", "ACE18_CACHE_PATH",
    ):
        if _env_key in st.secrets and not os.environ.get(_env_key):
            os.environ[_env_key] = str(st.secrets[_env_key])
except Exception:
    pass  # no st.secrets configured (e.g. local dev with only .env) — fine

from src.services.djo_service import validate_djo_pdf
from src.llm.service import SUPPORTED_PROVIDERS, list_available_models
from src.models.origin_rule import STATUS_LABELS

LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imgs", "marca-FIESC-reduzida_cor.png")

st.set_page_config(page_title="Validação de DJO", page_icon="📋", layout="wide")

# --- Labels: internal value <-> on-screen Portuguese text ------------------
PRODUCT_CATEGORY_LABELS = {
    "industrial": "Industrial",
    "game": "Jogos e Sortidos",
    "automotive": "Automotivo",
}
PROVIDER_LABELS = {"gemini": "Gemini", "groq": "Groq"}

# (streamlit call, banner text, short label for the metric widget)
STATUS_DISPLAY_PT = {
    "VALID": ("success", "✅ VÁLIDO", "Válido"),
    "CORRECT_DJO": ("warning", "⚠️ DJO PRECISA SER CORRIGIDA", "Requer correção"),
    "DOES_NOT_CONFER_ORIGIN": ("error", "❌ NÃO CONFERE ORIGEM", "Não confere origem"),
    "MANUAL_VERIFICATION": ("warning", "🕵️ VERIFICAÇÃO MANUAL NECESSÁRIA", "Verificação manual"),
    "NOT_APPLICABLE": ("info", "ℹ️ NÃO APLICÁVEL", "Não aplicável"),
    "INVALID_DJO": ("error", "❌ DJO INVÁLIDA", "DJO inválida"),
    "PROCESSING_ERROR": ("error", "❌ ERRO DE PROCESSAMENTO", "Erro de processamento"),
}

ORIGIN_STATUS_LABELS_PT = {
    "ORIGIN_RULE_A": "Norma de Origem A",
    "ORIGIN_RULE_B": "Norma de Origem B",
    "ORIGIN_RULE_C": "Norma de Origem C",
    "DOES_NOT_CONFER_ORIGIN": "Não confere origem",
    "CORRECT_DJO": "Requer correção",
    "MANUAL_VERIFICATION": "Verificação manual",
    "NOT_APPLICABLE": "Não aplicável",
}

RULE_LABELS_PT = {
    "djo_approval": "Aprovação da DJO",
    "producer_information": "Informações do produtor",
    "ncm_ace18": "NCM / ACE 18",
    "mandatory_fields": "Campos obrigatórios",
    "process_materials": "Materiais do processo produtivo",
    "table_completeness": "Completude das tabelas",
    "table_percentage": "Percentuais das tabelas",
    "preco_fob": "Preço FOB",
}

RULE_STATUS_LABELS_PT = {
    "PASS": "✅ Aprovado",
    "FAIL": "❌ Reprovado",
    "MANUAL_VERIFICATION": "🕵️ Verificação manual",
    "NOT_APPLICABLE": "➖ Não aplicável",
}

# --- Navy-blue sidebar styling ----------------------------------------------
SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1F44 0%, #0E2A5C 100%);
}
[data-testid="stSidebar"] * {
    color: #F2F5FA !important;
}
[data-testid="stSidebar"] [data-testid="stImage"] {
    background-color: #FFFFFF;
    padding: 18px 12px;
    border-radius: 10px;
    display: flex;
    justify-content: center;
    margin-bottom: 8px;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.25);
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] input {
    background-color: #FFFFFF !important;
    border-radius: 6px;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #0A1F44 !important;
}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #F2F5FA !important;
    font-weight: 600;
}
</style>
"""


def _icon_for_result(result) -> str:
    if result is True:
        return "✅"
    if result is False:
        return "❌"
    return "❓"  # inconclusivo


def _save_upload_to_temp(uploaded_file) -> str:
    """Grava o PDF enviado em um arquivo temporário privado para que o
    pipeline de extração (que lê a partir de um caminho de arquivo) possa
    utilizá-lo. O chamador é responsável por apagá-lo (ver bloco `finally`
    abaixo) — o arquivo nunca é mantido em disco, e o usuário nunca
    precisa colocar arquivos dentro do repositório."""
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="djo_upload_")
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def _render_decision_trace(decision_trace):
    if not decision_trace:
        st.caption("Nenhuma sequência de decisão disponível para este resultado.")
        return
    for step in decision_trace:
        icon = _icon_for_result(step.get("result"))
        with st.expander(f"{icon} {step['step']}. {step['question']}", expanded=False):
            st.write(step.get("reason", ""))
            if step.get("input_used"):
                st.caption("Entrada utilizada:")
                st.json(step["input_used"], expanded=False)


def _render_validations_table(validations):
    rows = [
        {
            "": RULE_STATUS_LABELS_PT.get(v["status"], v["status"]),
            "Regra": RULE_LABELS_PT.get(v["rule"], v["rule"]),
            "Mensagem": v["message"],
        }
        for v in validations
    ]
    st.dataframe(rows, hide_index=True, width="stretch")


def _needs_article14_answer(result) -> bool:
    origin_rule = result.get("origin_rule")
    if not origin_rule or origin_rule.get("status") != "MANUAL_VERIFICATION":
        return False
    trace = origin_rule.get("decision_trace") or []
    return bool(trace) and "Artigo 14" in trace[-1].get("question", "")


def _run_pipeline(pdf_path, product_category, llm_provider, llm_model, article14_compliant=None):
    with st.spinner(
        "Processando a DJO — extraindo o documento, validando as regras e avaliando a norma de origem do "
        "ACE 18... (isso aciona o OCR e o provedor de IA selecionado; pode levar até um minuto)"
    ):
        return validate_djo_pdf(
            pdf_path=pdf_path,
            product_category=product_category,
            article14_compliant=article14_compliant,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )


def _render_result(result):
    status = result["status"]
    kind, banner_text, short_label = STATUS_DISPLAY_PT.get(status, ("info", status, status))
    getattr(st, kind)(banner_text)

    if status == "PROCESSING_ERROR":
        st.error(f"**{result['error']['type']}**: {result['error']['message']}")
        return

    st.markdown("### Validação da DJO")
    col1, col2, col3 = st.columns(3)
    col1.metric("Status da DJO", short_label)
    col2.metric("NCM", (result.get("ncm") or {}).get("code") or "—")
    col3.metric("Categoria do produto", PRODUCT_CATEGORY_LABELS.get(result["product_category"], result["product_category"]))

    if result.get("djo_validation"):
        st.markdown("#### Validações das regras de negócio (Regras 1–8)")
        _render_validations_table(result["djo_validation"]["validations"])

    if result.get("origin_rule"):
        st.markdown("---")
        st.markdown("### Regra de Origem ACE 18")
        orule = result["origin_rule"]
        col1, col2 = st.columns(2)
        col1.metric("Regra aplicável", orule.get("applicable_rule") or "—")
        col2.metric("Status", ORIGIN_STATUS_LABELS_PT.get(orule.get("status"), orule.get("status") or "—"))
        if orule.get("status") in STATUS_LABELS:
            st.caption(STATUS_LABELS[orule["status"]])

        if result.get("mercosul_rule"):
            with st.expander("Regra do Mercosul aplicada a esta NCM"):
                st.write(f"**NCM correspondente:** {result['mercosul_rule'].get('matched_ncm_expression')}")
                st.write(f"**Regra aplicada:** {result['mercosul_rule'].get('raw_rule')}")
                if result["mercosul_rule"].get("parsed_details"):
                    st.json(result["mercosul_rule"]["parsed_details"], expanded=False)

        st.markdown("#### Sequência de decisão")
        _render_decision_trace(orule.get("decision_trace") or result.get("decision_trace") or [])

    if result.get("summary"):
        st.markdown("---")
        st.markdown("### Explicação")
        st.caption(
            f"Provedor de IA: {PROVIDER_LABELS.get(result.get('llm_provider'), result.get('llm_provider'))}  |  "
            f"Modelo: {result.get('llm_model')}"
        )
        st.info(result["summary"])
        st.caption(
            "Esta explicação é gerada pelo provedor de IA selecionado apenas para narrar, em linguagem natural, "
            "o resultado acima. O status, a regra aplicável e a sequência de decisão são sempre decididos por "
            "lógica determinística em Python — a IA nunca os substitui."
        )

    with st.expander("Resposta estruturada completa (JSON)"):
        st.json(result)


def _render_sidebar():
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)
    with st.sidebar:
        if os.path.isfile(LOGO_PATH):
            st.image(LOGO_PATH, width="stretch")

        st.markdown("### Configurações")

        product_category_label = st.selectbox("Categoria do produto", list(PRODUCT_CATEGORY_LABELS.values()))
        product_category = next(k for k, v in PRODUCT_CATEGORY_LABELS.items() if v == product_category_label)

        provider_label = st.selectbox("Provedor de LLM", [PROVIDER_LABELS[p] for p in SUPPORTED_PROVIDERS])
        llm_provider = next(k for k, v in PROVIDER_LABELS.items() if v == provider_label)

        available_models = list_available_models(llm_provider)
        llm_model = st.selectbox("Modelo", available_models) if available_models else None

    return product_category, llm_provider, llm_model


def main():
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "last_inputs" not in st.session_state:
        st.session_state.last_inputs = None

    product_category, llm_provider, llm_model = _render_sidebar()

    st.title("📋 Validação de Origem — ACE 18")
    st.caption(
        "Validação de Declaração Juramentada de Origem (DJO) conforme o regime de origem do ACE 18 (MERCOSUL)."
    )

    st.markdown("#### Envie a Declaração Juramentada de Origem")
    uploaded_file = st.file_uploader("Arraste o arquivo PDF aqui ou clique para selecionar", type=["pdf"])
    submitted = st.button("Validar DJO", type="primary")

    if submitted:
        if uploaded_file is None:
            st.warning("Por favor, envie um PDF da DJO antes de validar.")
            return

        pdf_path = _save_upload_to_temp(uploaded_file)
        try:
            result = _run_pipeline(pdf_path, product_category, llm_provider, llm_model)
        finally:
            os.remove(pdf_path)  # nunca deixa o PDF enviado em disco após o processamento

        st.session_state.last_result = result
        st.session_state.last_inputs = {
            "uploaded_file": uploaded_file,
            "product_category": product_category,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        }

    result = st.session_state.last_result
    if result is None:
        return

    _render_result(result)

    # Pergunta de acompanhamento sobre Jogos e Sortidos (Artigo 14) — ver
    # ARCHITECTURE.md. O pipeline é reexecutado sobre o mesmo arquivo
    # enviado assim que a pergunta é respondida (refaz o OCR; ver a nota
    # sobre extensões futuras em ARCHITECTURE.md).
    if _needs_article14_answer(result):
        st.markdown("---")
        st.markdown("### Informação adicional necessária")
        st.warning((result["origin_rule"]["decision_trace"] or [{}])[-1].get("question", "O produto cumpre o Artigo 14?"))
        answer_label = st.radio(
            "O produto cumpre com a regra de Jogos e Sortidos do Artigo 14°?",
            ["Sim", "Não"], horizontal=True, key="article14_answer",
        )
        if st.button("Enviar resposta e reavaliar"):
            inputs = st.session_state.last_inputs
            uploaded_file = inputs["uploaded_file"]
            pdf_path = _save_upload_to_temp(uploaded_file)
            try:
                new_result = _run_pipeline(
                    pdf_path, inputs["product_category"], inputs["llm_provider"], inputs["llm_model"],
                    article14_compliant=(answer_label == "Sim"),
                )
            finally:
                os.remove(pdf_path)
            st.session_state.last_result = new_result
            st.rerun()


if __name__ == "__main__":
    main()
