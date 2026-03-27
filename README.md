# EarningsIQ

A multi-agent AI system that analyzes SEC filings and delivers investment-grade insights for retail investors. Ask a natural language question about any public company — EarningsIQ routes it through a pipeline of specialized agents, retrieves and chunks the relevant 10-Q or 10-K filings, and returns a structured recommendation with cited evidence.

---

## Architecture

Seven agents, each with a distinct role and loop style:

| Agent | Model | Pattern |
|---|---|---|
| Orchestrator | claude-haiku-4-5 | Single LLM call → intent JSON |
| Scraper | No LLM | Deterministic SEC EDGAR client |
| Extraction | claude-sonnet-4-5 | Tool-collection loop — all 5 tools must fire |
| Risk Scoring | claude-sonnet-4-5 | Strict ordered loop — 3 tools in sequence |
| Comparison | claude-sonnet-4-5 | Open-ended loop, exits on terminal tool |
| Web Search | claude-haiku-4-5 | Standard tool loop (Tavily) |
| Advice | claude-sonnet-4-5 | No tools — pure synthesis, Pydantic output |

**Orchestrator** classifies intent (`single_analysis`, `comparison`, `advice`, `web_only`), selects filing type (`10-Q` vs `10-K` vs both), and determines how many periods are needed — all in a single LLM call with assistant prefill to force raw JSON.

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
| LLM SDK | Anthropic SDK |
| Data validation | Pydantic v2 |
| Database | PostgreSQL 16 + pgvector (HNSW index) |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Web search | Tavily API |
| Queue | Redis + Redis Queue (RQ) |
| Frontend | React 18 + TypeScript + Vite + Tailwind |
| Deploy | Railway (API + DB + Redis + worker) + Vercel (frontend) |

---

## Data Source

SEC EDGAR public API — free, legally clean, no scraping. 10-Q (quarterly) and 10-K (annual) filings only. 8-K was ruled out because earnings call transcript exhibits are inconsistently filed across companies.
