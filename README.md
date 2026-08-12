# EarningsIQ

A multi-agent AI system that analyzes SEC filings and delivers investment-grade insights for retail investors. Ask a natural language question about any public company — EarningsIQ routes it through a pipeline of specialized agents, retrieves and chunks the relevant 10-Q or 10-K filings, and returns a structured recommendation with cited evidence.

---

## Architecture

Seven agents, each with a distinct role and loop style:

| Agent | Model (via OpenRouter) | Pattern |
|---|---|---|
| Orchestrator | gemini-3-flash-preview | Single LLM call → strict-schema intent JSON |
| Scraper | No LLM | Deterministic SEC EDGAR client |
| Extraction | gemini-3-flash-preview | Tool-collection loop — all 5 tools must fire |
| Risk Scoring | gpt-5.2 (pinned to OpenAI first-party) | Strict ordered loop — named forced tool per step |
| Comparison | claude-sonnet-4.6 | Open-ended loop, exits on terminal tool |
| Web Search | gemini-3-flash-preview | Standard tool loop (Tavily) |
| Advice | claude-sonnet-4.6 | No tools — pure synthesis, Pydantic output |

Models are routed per-agent through OpenRouter: budget models where the capability requirements are easy, frontier models where correctness is hardest. Three eval tiers guard the system:

1. **Capability harness** (`evals/capability_harness.py`) — can the models do the mechanics? Probes each model+provider pair for named forced tool choice, parallel tool calls, and strict structured outputs. Provider capability drifts, so it re-runs before demos and deploys.
2. **Promptfoo evals** (`evals/*_eval.yaml`) — do the agents meet their contracts? Per-agent assertions on routing, schemas, and output shape.
3. **Analyst benchmark** (`evals/analyst_benchmark.py`) — is the analysis actually good? Runs the real pipeline over 12 diverse queries, gathers fresh professional research per query (Tavily), then scores the output with deterministic numeric cross-checks plus a cross-family LLM judge (gpt-5.2 grades the sonnet-written analysis) on factual accuracy, coverage, grounding, and directional agreement. `uv run python -m evals.analyst_benchmark`; use `--frozen-research <snapshot>` for agent-regression comparisons where reference drift is pinned out.

**Orchestrator** classifies intent (`single_analysis`, `comparison`, `advice`, `web_only`), selects filing type (`10-Q` vs `10-K` vs both), and determines how many periods are needed — one LLM call with strict `json_schema` structured outputs, validated by Pydantic.

**Scraper** pulls filings from the SEC EDGAR public API. A custom `parse_sections()` parser extracts structured sections (MD&A, Risk Factors, Financial Statements, Business Overview) with TOC-skip logic and backward-looking TOC detection to handle the inconsistencies across company filings.

**Extraction Agent** runs a tool-collection loop where the LLM must call all 5 tools exactly once: financial metrics, risk factors, segment performance, management outlook, and notable changes. A `MAX_TURNS` guard raises `ExtractionTimeoutError` if the loop stalls.

**Risk Scoring Agent** enforces strict tool ordering — `fetch_historical_signals` (real Postgres query) → `score_risk_components` → `finalize_overall_risk`. Python recomputes the weighted average independently rather than trusting the LLM's arithmetic.

**Comparison Agent** uses regex mode detection before any LLM call to avoid wasted tokens. Runs an open-ended loop combining vector search (pgvector cosine similarity) and structured signal retrieval, exiting only when the terminal tool `cite_and_answer` fires.

**Advice Agent** takes the extracted signals, risk result, and optional web summary, and returns a validated `AdviceOutput` Pydantic model: recommendation, confidence, reasoning, key risks, key positives, and a disclaimer.

---

## Key Design Decisions

- **Status enum pipeline** — `pending → chunked → embedded → extracted → scored`. Each agent checks status before running; idempotent by design.
- **In-memory handoff** — `extracted_signals` dict passed directly from Extraction → Risk Scoring, avoiding a redundant DB read.
- **DB-backed tool** — Risk Scoring's `fetch_historical_signals` executes a real Postgres query and returns the results to the model as context, grounding the scoring in historical data.
- **Terminal tool pattern** — Comparison Agent loop exits only when `cite_and_answer` fires, never on `stop_reason` alone.
- **Section-aware chunking** — chunks carry section and filing type metadata so citations in the final answer are attributable to a specific section of a specific filing.
- **Fast path / slow path** — if data is already in the DB, the API returns advice immediately. If not, it enqueues an RQ job and the frontend polls until the pipeline completes.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.12 + FastAPI |
| LLM access | OpenRouter (`/chat/completions` via OpenAI SDK) — per-agent model routing across Google/OpenAI/Anthropic |
| Observability | LangSmith tracing (optional — enabled via `LANGSMITH_TRACING`) |
| Data validation | Pydantic v2 |
| Database | PostgreSQL 16 + pgvector (HNSW index) |
| Embeddings | Voyage AI voyage-finance-2 (finance-domain, 1024 dims) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Web search | Tavily API |
| Queue | Redis + Redis Queue (RQ) |
| Frontend | React 18 + TypeScript + Vite + Tailwind |
| Deploy | Railway (API + DB + Redis + worker) + Vercel (frontend) |

---

## Data Source

SEC EDGAR public API — free, legally clean, no scraping. 10-Q (quarterly) and 10-K (annual) filings only. 8-K was ruled out because earnings call transcript exhibits are inconsistently filed across companies.
