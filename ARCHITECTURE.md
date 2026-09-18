# DJO Validation System — Architecture Guide

This document explains how this project is put together: the modular
Python backend that implements ACE 18 (MERCOSUL) rules-of-origin
validation for **Declaração Juramentada de Origem (DJO)** documents, and
the **Streamlit MVP** on top of it. It's written for whoever needs to run,
extend, or eventually replace either layer — without having to read
through the validation engine's internals first.

> **Reference implementation note.** The business logic was prototyped
> and validated in three Jupyter notebooks under `codigo/`
> (`extraction_djo.ipynb`, `estraction_ac18.ipynb`,
> `djo_validation_core.ipynb`). Those notebooks are kept as-is, as the
> functional reference — `src/` is a direct, behavior-preserving port of
> that same logic, not a rewrite, and `app.py` is a thin UI on top of
> `src/`, not a second implementation of anything.

---

## 1. System architecture

```
User (browser)
    ↓
Streamlit (app.py)                 <- presentation/orchestration only
    ↓
services.djo_service.validate_djo_pdf(...)   <- the one entry point
    ↓
djo.extractor        — PDF -> raw OCR text (LLM Whisperer)
    ↓
djo.normalizer        — raw OCR text -> structured json_djo
    ↓
validation.validation_service   — Rules 1-8
    ↓
origin.decision_engine          — ACE 18 origin-rule flowchart
    ↓
llm.service → LLMProvider ────┬── GeminiProvider (Gemini API)
                               └── GroqProvider   (Groq API)
    ↓
structured, JSON-serializable response
    ↓
Streamlit presentation (app.py renders the response — it never computes any of it)
```

**Frontend/backend boundary.** Everything below `services.djo_service` is
plain Python with no UI dependency at all — `app.py` is one possible
caller among others (a future API, a script, a test). Nothing in `src/`
imports `streamlit`.

---

## 2. Project structure

```
project/
├── app.py                       # Streamlit entry point (streamlit run app.py)
│
├── src/
│   ├── config.py                 # env-overridable file paths (ACE 18 PDF/cache)
│   │
│   ├── djo/
│   │   ├── extractor.py          # PDF -> raw OCR text (LLM Whisperer)
│   │   ├── normalizer.py         # raw OCR text -> structured json_djo
│   │   └── validator.py          # Rules 1, 2, 4, 8 (DJO-level checks)
│   │
│   ├── ace18/
│   │   ├── ncm_service.py        # ACE 18 NCM lookup + cache + Rule 3
│   │   ├── rule_parser.py        # parses mercosul_rule.raw_rule text
│   │   └── rule_engine.py        # evaluates parsed conditions (MP/MSP/MaxMNO/...)
│   │
│   ├── validation/
│   │   ├── material_validator.py   # Rules 6, 7 (material tables)
│   │   ├── process_validator.py    # Rule 5 (LLM semantic matching)
│   │   └── validation_service.py   # orchestrates Rules 1-8 -> validation_result
│   │
│   ├── origin/
│   │   ├── decision_engine.py    # the ACE 18 origin-rule flowchart itself
│   │   └── decision_trace.py     # trace-building + explanation helpers
│   │
│   ├── llm/                       # provider-agnostic LLM abstraction
│   │   ├── base.py                # LLMProvider interface, LLMProviderError, RateLimiter
│   │   ├── gemini.py              # GeminiProvider (adapts gemini/service.py)
│   │   ├── groq.py                # GroqProvider (talks to the Groq HTTP API directly)
│   │   └── service.py             # create_provider(name, model), list_available_models(name)
│   │
│   ├── gemini/
│   │   └── service.py            # the ONLY module that imports the google.genai SDK
│   │
│   ├── models/
│   │   ├── djo.py                # json_djo shape + shared constants
│   │   ├── validation.py         # ValidationStatus, make_result, shared helpers
│   │   └── origin_rule.py        # PRODUCT_CATEGORIES, status labels
│   │
│   └── services/
│       └── djo_service.py        # <-- the single entry point (see section 4)
│
├── dados/                         # sample PDFs + the ACE 18 cache (existing project data)
├── codigo/                        # reference notebooks (kept as-is — see note above)
├── tests/                         # pytest suite (see section 11)
├── examples/run_local.py          # CLI example
├── .env.example
├── .gitignore
├── requirements.txt
└── ARCHITECTURE.md                # this file
```

**Separation of responsibilities**, module by module:

| Module | Responsibility | Talks to an LLM? |
|---|---|---|
| `djo/extractor.py` | PDF → raw text (OCR only, no interpretation) | No |
| `djo/normalizer.py` | raw text → structured `json_djo` dict | No |
| `djo/validator.py` | Rules 1, 2, 4, 8 — simple `json_djo` field checks | No |
| `ace18/ncm_service.py` | NCM lookup against the cached ACE 18 table; Rule 3 | No |
| `ace18/rule_parser.py` | Parses `raw_rule` text into criteria/conditions | No |
| `ace18/rule_engine.py` | Evaluates parsed conditions against the product | Only via the configured `LLMProvider`, for free-text conditions |
| `validation/material_validator.py` | Rules 6, 7 — material-table checks | No |
| `validation/process_validator.py` | Rule 5 — process description vs. materials | Yes (the only rule requiring semantic judgement) |
| `validation/validation_service.py` | Orchestrates Rules 1-8 into one result | No (delegates) |
| `origin/decision_engine.py` | The ACE 18 origin-rule flowchart | No (delegates) |
| `llm/base.py` | Provider-agnostic contract (`LLMProvider`) | — |
| `llm/gemini.py` / `llm/groq.py` | Concrete providers | Yes — these are the only two modules that talk to an LLM API |
| `llm/service.py` | Provider factory (`create_provider`, `list_available_models`) | No |
| `services/djo_service.py` | Top-level orchestration + response shaping | No (delegates) |
| `app.py` | Streamlit UI: collects inputs, calls `djo_service`, renders the response | No |

**Principle carried over from the notebooks:** deterministic business rules
stay in plain Python. An LLM is used **only** where genuine semantic
language understanding is required — matching free-text process inputs
against material descriptions (Rule 5), evaluating a handful of free-text
ACE 18 rule conditions the parser can't classify (e.g. "Reação química"),
and writing the two human-readable summaries. **An LLM never decides
PASS/FAIL/A/B/C for anything** — it only narrates or judges semantics that
were never expressed as structured data to begin with. This holds
regardless of which provider is selected.

---

## 3. The LLM provider abstraction

`src/llm/base.py` defines `LLMProvider`, an interface with exactly the
four methods the business logic needs:

```python
class LLMProvider(ABC):
    name: str
    model: str

    def match_process_materials(self, descricao_processo, material_items) -> dict: ...
    def summarize_validation_result(self, validation_result) -> str: ...
    def summarize_origin_rule_result(self, origin_result) -> str: ...
    def evaluate_rule_condition(self, condition_texts, descricao_processo, material_items) -> dict: ...
```

`validation/process_validator.py`, `ace18/rule_engine.py`, and
`services/djo_service.py` depend **only** on this interface — none of
them import `google.genai` or know Groq exists. Two implementations exist
today:

- **`GeminiProvider`** (`llm/gemini.py`) — a thin adapter over
  `gemini/service.py` (unchanged from before this abstraction existed;
  still the only module touching the Gemini SDK).
- **`GroqProvider`** (`llm/groq.py`) — talks to Groq's OpenAI-compatible
  `chat/completions` endpoint directly over HTTP, reusing the **exact
  same prompt templates** as `gemini/service.py` (imported, not
  duplicated) for all four tasks, so switching providers changes which
  model answers the prompt, never the prompt or the expected response
  shape.

Both implement the same pacing + bounded-retry-on-transient-error pattern
(429/503 handling), each with its own rate limiter, so selecting Groq
never throttles a concurrently-running Gemini call or vice versa.

`src/llm/service.py` is the only place that maps a provider **name**
("gemini"/"groq", case-insensitive) to a concrete class:

```python
from src.llm.service import create_provider
provider = create_provider("groq", model="openai/gpt-oss-120b")
```

`create_provider(...)` never makes a network call — it only reads
`GROQ_API_KEY`/`GEMINI_API_KEY` from the environment and constructs the
object; a missing/invalid key or model only ever surfaces as an
`LLMProviderError` on the first actual call, exactly like the
Gemini-only behavior before this abstraction existed. An unrecognized
provider name is the one thing rejected eagerly, in `validate_djo_pdf`,
before any OCR runs (see section 4).

### Adding a third provider later

1. Create `src/llm/<provider>.py` with a class implementing `LLMProvider`'s
   four methods (reuse the prompt templates from `gemini/service.py`
   unless the new provider genuinely needs different ones).
2. Add one `if name == "<provider>":` branch to
   `llm/service.py:create_provider` and, if it should appear in the
   Streamlit dropdown, to `list_available_models` and
   `SUPPORTED_PROVIDERS`.
3. Nothing else changes — `process_validator.py`, `rule_engine.py`, and
   `djo_service.py` already work with any `LLMProvider`.

---

## 4. The one function you need to call

```python
from src.services.djo_service import validate_djo_pdf

result = validate_djo_pdf(
    pdf_path="path/to/the/uploaded/djo.pdf",
    product_category="industrial",   # "industrial" | "game" | "automotive"
    article14_compliant=None,        # only relevant for "game" — see section 6
    llm_provider="gemini",           # "gemini" | "groq"
    llm_model=None,                  # optional override; defaults to the provider's own default
)
```

This is the **only** function `app.py` (or a future API layer) needs to
call. It never raises — every failure mode (bad input, an unrecognized
`llm_provider`, OCR failure, an LLM API failure, an unexpected error) is
captured into the response itself (`status: "PROCESSING_ERROR"`, see
section 7). Everything it returns is plain `dict`/`list`/`str`/`float`/
`bool`/`None` — safe to `json.dumps(...)` or return directly from a web
framework or a Streamlit page, with **no** Docling objects, DataFrames,
LLM SDK objects, or exceptions anywhere in the tree, and no API key
anywhere in it either.

---

## 5. Execution flow

```
PDF upload
    ↓
validate_djo_pdf(pdf_path, product_category, article14_compliant?, llm_provider, llm_model)
    ↓
llm.service.create_provider(llm_provider, llm_model)   — fails fast on an unrecognized provider name
    ↓
djo.extractor          — OCR (LLM Whisperer)
    ↓
djo.normalizer         — structured json_djo
    ↓
validation.validation_service.validate_djo   — Rules 1-8 (Rule 5 uses the provider above)
    ↓
origin.decision_engine.evaluate_origin_rule  — ACE 18 flowchart
    (internally calls ace18.ncm_service for the NCM lookup; uses the
     provider above for any free-text rule condition)
    ↓
provider.summarize_validation_result(...) + provider.summarize_origin_rule_result(...)
    ↓
JSON response
    ↓
frontend (app.py, or any other caller)
```

If `product_category == "automotive"`, the pipeline still runs OCR +
normalization (so you can display the extracted DJO fields), but **skips
validation, the origin-rule flowchart, and both LLM summaries entirely**
— the response's `status` is `NOT_APPLICABLE` and
`djo_validation`/`origin_rule` are `None`.

---

## 6. Product category & the Games/Article 14 decision

`product_category` is **required on every call** and is **never inferred**
— always send exactly one of `"industrial"`, `"game"`, `"automotive"`.

- **`"automotive"`**: the ACE 18 origin-rule flowchart does not apply.
  `status` will be `"NOT_APPLICABLE"`.
- **`"industrial"`**: runs the full pipeline; `article14_compliant` is
  ignored.
- **`"game"`**: the flowchart needs one extra yes/no input — whether the
  product complies with the Games & Assortments rule (Art. 14). Call
  `validate_djo_pdf(...)` **without** `article14_compliant` first; if the
  response's `origin_rule.status` comes back as `"MANUAL_VERIFICATION"`
  and the last `decision_trace` entry's `question` mentions "Artigo 14",
  **ask the user that yes/no question**, then call `validate_djo_pdf(...)`
  again with `article14_compliant=True` or `False`.

`app.py` implements exactly this: `_needs_article14_answer(result)`
checks those two conditions and renders the follow-up question only when
they hold — it never reconstructs the flowchart itself, it just reads the
already-computed `decision_trace`. The follow-up call currently re-runs
OCR + normalization on the same uploaded file (see section 15 for why,
and the caching improvement that would avoid it).

---

## 7. Response schema

```jsonc
{
  "status": "VALID",                 // see section 8 for all possible values
  "product_category": "industrial",
  "agreement": "ACE_18",
  "llm_provider": "gemini",          // "gemini" | "groq" — whichever generated the summaries below
  "llm_model": "gemini-3.6-flash",

  "djo": { /* the full normalized json_djo — see below */ },

  "djo_validation": {
    "overall_status": "PASS",
    "agreement": "ACE_18",
    "validations": [
      {"rule": "djo_approval", "status": "PASS", "message": "..."},
      {"rule": "producer_information", "status": "PASS", "message": "..."},
      {"rule": "ncm_ace18", "status": "PASS", "message": "...",
       "mercosul_rule": {"matched_ncm_expression": "...", "raw_rule": "..."}},
      {"rule": "mandatory_fields", "status": "PASS", "message": "..."},
      {"rule": "process_materials", "status": "PASS", "message": "...", "details": {}},
      {"rule": "table_completeness", "status": "PASS", "message": "...", "details": {}},
      {"rule": "table_percentage", "status": "PASS", "message": "...", "details": {}},
      {"rule": "preco_fob", "status": "PASS", "message": "...", "details": {}}
    ]
  },

  "ncm": {
    "code": "9404.90.00",
    "found_in_ace18": true,
    "match_info": {"match_type": "CHAPTER", "matched_ncm_expression": "ex Capítulo 94(*)", "table_index": 22}
  },

  "mercosul_rule": {
    "matched_ncm_expression": "ex Capítulo 94(*)",
    "raw_rule": "MP ou MaxMNO 45%",
    "parsed_details": {"criteria": ["MP"], "max_mno_percentage": 45, "operator": "OR", "de_minimis_applies": true, "chemical_processes": []}
  },

  "origin_rule": {
    "applicable_rule": "C",              // "A" | "B" | "C" | null
    "status": "ORIGIN_RULE_C",           // see section 8
    "decision_trace": [
      {"step": 1, "question": "...", "input_used": {}, "result": true, "reason": "..."},
      "..."
    ],
    "explanation": "Sequência de decisões do fluxograma ACE 18:\n1. ...\n\nConclusão: ..."
  },

  "decision_trace": [ /* same array as origin_rule.decision_trace, exposed at top level too for convenience */ ],

  "summary": "<LLM-authored validation summary>\n\n<LLM-authored origin-rule summary, or the deterministic explanation if the LLM was unavailable>",

  "error": null   // only non-null when status == "PROCESSING_ERROR": {"type": "...", "message": "..."}
}
```

`djo` is the full normalized document — see `src/models/djo.py`'s
docstring for the exact field-by-field shape. `matched_ncm_expression`
and `raw_rule` are always present when the NCM was found, exactly as
produced by the ACE 18 lookup — nothing here is dropped or renamed by
the presentation layer.

---

## 8. Status handling

**Top-level `status`:**

| Status | Meaning | What to show |
|---|---|---|
| `VALID` | Origin rule A, B, or C was determined | Green/success state; show `origin_rule.applicable_rule` |
| `CORRECT_DJO` | Base validation failed — the DJO itself has errors | Show `djo_validation.validations` filtered to `FAIL`/`MANUAL_VERIFICATION`; ask the user to fix and resubmit |
| `DOES_NOT_CONFER_ORIGIN` | Validation passed, but the product doesn't qualify for origin under ACE 18 | Red/negative state; show `origin_rule.explanation` |
| `MANUAL_VERIFICATION` | Something couldn't be decided automatically (missing game answer, an LLM condition came back inconclusive, a `de minimis` tolerance with no available threshold, ...) | Yellow/pending state; show `origin_rule.explanation` and/or the relevant `decision_trace` entry |
| `NOT_APPLICABLE` | `product_category == "automotive"` | Informational state only |
| `INVALID_DJO` | The PDF doesn't look like a DJO at all | Ask the user to check the uploaded file |
| `PROCESSING_ERROR` | An actual failure (bad `product_category`, unrecognized `llm_provider`, OCR error, unexpected exception) | Show `error.message`; **never** show a stack trace |

`origin_rule.status` is the origin engine's own, more granular status —
one of `ORIGIN_RULE_A` / `ORIGIN_RULE_B` / `ORIGIN_RULE_C` /
`DOES_NOT_CONFER_ORIGIN` / `CORRECT_DJO` / `MANUAL_VERIFICATION` /
`NOT_APPLICABLE`. Per-rule statuses inside `djo_validation.validations[*]`
are always one of `PASS`, `FAIL`, `MANUAL_VERIFICATION`,
`NOT_APPLICABLE`.

---

## 9. Deterministic logic vs. LLM responsibility

```
Deterministic Python logic
        ↓
Authoritative validation result (status, applicable rule, decision_trace)

LLM (Gemini or Groq)
        ↓
Natural-language explanation of an already-decided result
```

Both `summarize_validation_result` and `summarize_origin_rule_result` are
given the already-computed structured result and are explicitly
instructed (in the prompt) not to reinterpret or contradict any status —
only to narrate it in Portuguese. If the call fails for any reason
(missing key, network error, malformed response), `djo_service` falls
back to a short placeholder (validation summary) or the deterministic
`simple_explanation` (origin summary) — it never blocks the response or
silently invents a different status.

---

## 10. Streamlit application (`app.py`)

`app.py` is a single-file MVP: a navy-blue sidebar (logo + configuration)
and a main area (upload + validate) that on submit calls
`validate_djo_pdf(...)` once and renders the response. It holds no
business logic — every helper function in it (`_render_validations_table`,
`_render_decision_trace`, `_needs_article14_answer`, ...) either formats
already-structured data or reads a field to decide what to show; none of
them compute a status. **All user-facing text is Brazilian Portuguese**
(labels, buttons, status text, messages) — internal identifiers (rule
slugs like `process_materials`, JSON keys like `raw_rule`/
`matched_ncm_expression`, material-table names like
`originarios_estado_parte_produtor`) are left untouched; only their
on-screen presentation is translated, via small label-mapping dicts near
the top of the file (`PRODUCT_CATEGORY_LABELS`, `STATUS_DISPLAY_PT`,
`ORIGIN_STATUS_LABELS_PT`, `RULE_LABELS_PT`, `RULE_STATUS_LABELS_PT`).

Key behaviors:

- **Sidebar (`_render_sidebar`)**: holds all configuration — the company
  logo (`imgs/marca-FIESC-reduzida_cor.png`, the FIESC brand mark; loaded
  from the existing `imgs/` folder, not recreated), then "Categoria do
  produto" (Industrial/Jogos e Sortidos/Automotivo →
  industrial/game/automotive), "Provedor de LLM" (Gemini/Groq), and
  "Modelo" (populated from `llm.service.list_available_models(...)` for
  whichever provider is selected — reusing the existing provider
  abstraction, not a second model list). These are plain (non-form)
  widgets, so the model dropdown updates immediately when the provider
  changes. Styled navy-blue via one `st.markdown(<style>...)` block
  (`SIDEBAR_CSS`) scoped to `[data-testid="stSidebar"]` — a dark navy
  gradient background (`#0A1F44` → `#0E2A5C`, chosen to contrast against
  the logo's own medium blue, `#0E4194`, sampled directly from the PNG),
  white sidebar text, and a white rounded card behind the logo and behind
  each select box (so both stay fully readable regardless of the navy
  background). No other part of the app is restyled.
- **Main area**: only the upload widget and the "Validar DJO" button —
  the category/provider/model controls live in the sidebar and are not
  repeated here.
- **Upload handling**: the uploaded PDF is written to a private temp file
  (`tempfile.mkstemp`), passed to `validate_djo_pdf`, and deleted in a
  `finally` block — never left on disk, and the user never needs to place
  a file inside the repository.
- **Loading state**: a single `st.spinner(...)` wraps the one call to
  `validate_djo_pdf` — there's no attempt to fabricate a multi-step
  progress bar the backend doesn't actually expose (`validate_djo_pdf` is
  one opaque call from the UI's point of view).
- **Result rendering**: status banner, a metrics row (status/NCM/category),
  the Rules 1-8 table, the origin-rule card (applicable rule + status),
  the decision sequence (one expander per `decision_trace` entry, ✅/❌/❓
  for True/False/None), the LLM explanation (labeled with
  `llm_provider`/`llm_model`), and a raw-JSON expander for anything not
  otherwise surfaced. All of it in Portuguese.
- **Games/Art. 14 follow-up**: rendered only when
  `_needs_article14_answer(result)` is true (see section 6); answering it
  re-calls `validate_djo_pdf` with `article14_compliant` set.
- **Errors**: a `PROCESSING_ERROR` status renders `error.type`/`message`
  in an `st.error(...)` box — never a Python traceback.
- **Secrets**: `app.py` calls `load_dotenv()` once at import time so
  `GEMINI_API_KEY`/`GROQ_API_KEY`/`LLMWHISPERER_API_KEY` are available to
  the backend modules that read them directly from `os.environ` — `app.py`
  itself never reads, stores, displays, or forwards any key. Nothing in
  the response payload it renders contains a key either (see section 7).

---

## 11. Local execution

```powershell
# from the project root, with the venv already set up (see requirements.txt)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# fill in .env from the template
copy .env.example .env
# then edit .env: GEMINI_API_KEY=..., and/or GROQ_API_KEY=..., LLMWHISPERER_API_KEY=...

# run the Streamlit app
.venv\Scripts\streamlit.exe run app.py
# or, if the venv's Scripts folder is on PATH:
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`), upload a
sample PDF from `dados/djos/`, pick a product category and LLM provider,
and click **Validate DJO**.

You only need the API key for whichever LLM provider you select in the
dropdown — Gemini and Groq keys are independent; the app works with
either one configured (or both).

For a non-UI run (e.g. scripting, or verifying the pipeline without a
browser):

```powershell
.venv\Scripts\python.exe examples\run_local.py "dados\djos\DJO 3 (1).pdf" industrial
.venv\Scripts\python.exe examples\run_local.py "dados\djos\DJO 1 (1).pdf" game --article14 true
.venv\Scripts\python.exe examples\run_local.py "dados\djos\DJO 3 (1).pdf" industrial --llm-provider groq --llm-model openai/gpt-oss-120b
```

The ACE 18 NCM cache (`dados/reos_ace18_cache.json`) already exists in
this project, so no Docling/OCR run happens for ACE 18 lookups — only the
DJO PDF itself gets OCR'd (via LLM Whisperer) on each call.

---

## 12. Tests

```powershell
.venv\Scripts\python.exe -m pytest              # offline suite (no API keys needed) — 81 tests
.venv\Scripts\python.exe -m pytest -m integration   # + the one real end-to-end test (needs LLMWHISPERER_API_KEY + GEMINI_API_KEY)
```

Coverage: DJO validation (Rules 1/2/4/8), ACE 18 NCM lookup (against the
real cache), material-table validation (Rules 6/7), the origin-rule
decision engine (A/B/C, de minimis, NCM-not-found, empty process
description), product-category handling (industrial/game/automotive,
invalid category, the full Games/Article-14 flow), the LLM provider
abstraction (`test_llm_provider.py` — factory behavior, both providers
implementing the full interface, Groq's HTTP layer mocked including
retry-on-transient-error, missing-key handling for both providers,
unsupported provider/model), Gemini service error handling directly
(missing key, malformed response, transient-error retry), and full
response serialization (including the new `llm_provider`/`llm_model`
fields and a dedicated "unsupported provider fails before OCR" test).
Gemini, Groq, and LLM Whisperer are mocked everywhere except the one
`@pytest.mark.integration` test, which is skipped automatically unless
both required API keys are present in the environment.

`app.py` itself is verified two ways (not part of the automated `pytest`
suite, since it's a thin UI layer over already-tested logic):
`streamlit.testing.v1.AppTest` runs the full script in a simulated
session and asserts no exception is raised and the expected widgets
render; a small offline script exercises `app.py`'s own helper functions
(`_needs_article14_answer`, `_run_pipeline`, `_render_result`) against
constructed and mocked results, including the unsupported-provider and
automotive-category paths.

---

## 13. Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | To use the Gemini provider | Get one at https://aistudio.google.com/apikey |
| `GEMINI_MODEL` | No (defaults to `gemini-3.6-flash`) | Override the Gemini model |
| `GEMINI_MIN_INTERVAL_SECONDS` | No (defaults to `4.0`) | Minimum pacing between consecutive Gemini calls |
| `GROQ_API_KEY` | To use the Groq provider | Get one at https://console.groq.com/keys |
| `GROQ_MODEL` | No (defaults to `openai/gpt-oss-120b`) | Override the Groq model |
| `GROQ_MIN_INTERVAL_SECONDS` | No (defaults to `2.0`) | Minimum pacing between consecutive Groq calls |
| `LLMWHISPERER_API_KEY` | For any PDF extraction | Get one at https://unstract.com |
| `LLMWHISPERER_WAIT_TIMEOUT` | No (defaults to `200`) | Seconds to wait for one OCR job |
| `ACE18_PDF_PATH` / `ACE18_CACHE_PATH` | No | Only needed to point at a different ACE 18 source/cache than `dados/` |

None of these are ever hard-coded, logged, or included in any response —
see `src/llm/gemini.py`, `src/llm/groq.py`, and `src/djo/extractor.py`.
**Never commit a real `.env` file** — see `.gitignore` (also covers
`.streamlit/secrets.toml`, in case that's used later for a Streamlit
Community Cloud deployment).

---

## 14. Future API (FastAPI behind or instead of Streamlit)

`services/djo_service.validate_djo_pdf(...)` is already shaped to sit
directly behind an HTTP endpoint — it takes plain arguments and returns a
plain, JSON-serializable dict, with no Streamlit dependency anywhere
below it. Wiring it into FastAPI is a thin adapter, not a rewrite, and
can run **alongside** `app.py` (same backend, two frontends) or **replace**
it entirely:

```python
from fastapi import FastAPI, UploadFile, Form
from src.services.djo_service import validate_djo_pdf

app = FastAPI()

@app.post("/djo/validate")
async def validate(
    file: UploadFile,
    product_category: str = Form(...),
    llm_provider: str = Form("gemini"),
    llm_model: str | None = Form(None),
    article14_compliant: bool | None = Form(None),
):
    tmp_path = save_upload_to_tmp(file)   # your own upload-handling helper
    result = validate_djo_pdf(tmp_path, product_category, article14_compliant, llm_provider=llm_provider, llm_model=llm_model)
    return result   # already a plain dict — FastAPI serializes it directly
```

Nothing in `src/` imports a web framework or Streamlit — `djo_service`
stays usable from a CLI, a notebook, a queue worker, a Streamlit app, or
an API endpoint without change.

## 15. Business rule note — the "producer-table-only" FOB exceptions

Two different rules check two different FOB-related fields, and both
gained the same, narrow exception:

- **Rule 4** (`validate_mandatory_fields`) normally requires
  `json_djo["Valor FOB (USD)"]` (a currency amount) to be present,
  alongside "Unidade de medida" and "Descrição do processo produtivo".
- **Rule 8** (`validate_preco_fob`) normally requires
  `json_djo["Materiales"]["Preço FOB"]` (a *different* field — a
  percentage, not a currency amount) to be present, in order to check it
  against a 100% ceiling; when it's missing, the rule reports
  `MANUAL_VERIFICATION` rather than assuming 0%.

**The exception**: neither field is required — Rule 4 becomes `PASS` and
Rule 8 becomes `NOT_APPLICABLE` (instead of `MANUAL_VERIFICATION`) —
**when `originarios_estado_parte_produtor` is the ONLY material table
with items**, i.e. every other table (`originarios_outros_estados_partes`,
`nao_originarios`, `terceiros_paises_ptc`) is empty. That configuration
represents a DJO with no cross-border materials to price at all, so both
fields being blank there is expected, not a defect — this was confirmed
against a real DJO PDF with exactly this table configuration, which
surfaced the same gap in Rule 8 shortly after the Rule 4 fix shipped.

Both rules share one check, `_only_producer_originating_table_present`
(defined once in `djo/validator.py`, used by both), which itself reuses
the existing `MATERIAL_TABLE_NAMES` list from `models/djo.py` — the same
table-name list every other material-table check in the codebase already
uses — rather than hard-coding the four names again or duplicating the
presence check. If `originarios_estado_parte_produtor` is combined with
any other table, or no table has items at all, both fields are still
required/checked exactly as before; only the single-table case is
exempt. No other field in either rule, and no other validation rule, was
touched. See `tests/test_djo_validator.py` for the full matrix of cases
(empty/filled FOB × each table configuration, for both rules).

## 16. Deployment: GitHub → Streamlit Community Cloud

- **Repository**: https://github.com/juans-eng/Co_djo.git
- **Streamlit entry point**: `app.py` (`streamlit run app.py`) — the only
  entry point; nothing else in the repo needs to be run to serve the app.
- **Python version**: Streamlit Community Cloud selects the Python
  version manually, per-app, in the "Advanced settings" dialog at deploy
  time (it is not auto-detected from a file) — pick **3.12** there. Local
  development was done on 3.14, but 3.12 is Cloud's own default and the
  most broadly compatible with `docling`'s heavier dependencies (fewer
  risks of a missing prebuilt wheel during the Cloud build); nothing in
  this codebase requires a specific minor version.
- **Runtime assets already in the repo** (no manual upload step needed):
  `imgs/marca-FIESC-reduzida_cor.png` (sidebar logo),
  `dados/reos_ace18_cache.json` (the ACE 18 rules cache Rule 3/the origin
  engine query — committed so the app never needs to re-run Docling
  against the source PDF at runtime), `dados/ACE_018_221_pt (1).pdf` (that
  cache's source PDF, kept as the fallback rebuild path if the cache is
  ever deleted), and the sample DJO/certificate PDFs under `dados/` (for
  local testing/demos — the deployed app itself only ever processes
  whatever PDF the user uploads).

### Local development

```powershell
pip install -r requirements.txt
copy .env.example .env
# edit .env: GEMINI_API_KEY=..., GROQ_API_KEY=..., LLMWHISPERER_API_KEY=...
streamlit run app.py
```

### Streamlit Community Cloud

1. Push this repository to GitHub (already configured — see below).
2. In Streamlit Community Cloud, create a new app pointed at this repo,
   branch `main`, entry point `app.py`.
3. In the app's **Settings → Secrets**, paste (see
   `.streamlit/secrets.toml.example` for the full template):
   ```toml
   GEMINI_API_KEY = "..."
   GROQ_API_KEY = "..."
   LLMWHISPERER_API_KEY = "..."
   ```
   All three are required for full functionality — `LLMWHISPERER_API_KEY`
   isn't an LLM provider choice, it's what performs OCR on the uploaded
   PDF, so the app can't process any document without it regardless of
   which of Gemini/Groq is selected.
4. Deploy. No other configuration is required — `app.py` bridges
   `st.secrets` into `os.environ` at startup (see the code comment right
   after `load_dotenv()` in `app.py`), so every module below it keeps
   reading credentials exactly the way it already did
   (`os.environ.get(...)`) with zero business-logic change between local
   and Cloud.

**Never** commit `.env` or `.streamlit/secrets.toml` — both are
git-ignored; only their `.example` templates are tracked. Secrets are
entered directly into the Cloud dashboard, never into the repository.

### Adding a future LLM provider's credentials

Follow `src/llm/service.py`'s "Adding a third provider later" note (§3)
for the code side. For credentials specifically: add the new key to
`.env.example` (empty placeholder) and to
`.streamlit/secrets.toml.example` (empty placeholder + comment), then add
the real value to your local `.env` and to the Cloud app's Secrets panel
— never to a tracked file.

## 17. Future extension points (not implemented, by design)

- **Decision flow**: a business-decision step consuming this response
  (auto-approve / route to manual review / reject) belongs *after* this
  backend's output, as its own stage.
- **More trade agreements**: register a new entry in
  `ace18.ncm_service.build_agreement_services(...)`-style factory with a
  service exposing the same `.query_ncm(ncm)` shape as `ACE18Service` —
  no other module needs to change.
- **A third LLM provider**: see section 3, "Adding a third provider
  later" — one new module + one factory branch.
- **Skip re-OCR on the Article-14 follow-up call**: both `validate_djo_pdf`
  and `app.py`'s follow-up currently re-extract + re-validate on the
  second ("now answer Art. 14") call for a game product. Caching
  `json_djo` (keyed by a request/session id) between the two calls would
  avoid the repeat OCR + LLM cost — not implemented in this pass.
