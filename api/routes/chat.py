# POST /chat — the main query endpoint.
# Fast path: if ticker data is already in DB, runs advice and returns the answer immediately.
# Slow path: if data is missing, enqueues the full pipeline and returns a job_id to poll.

import logging
from datetime import date

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agents.advice import run_advice
from agents.comparison import is_comparison_query, run_comparison
from agents.orchestrator import run_orchestrator
from agents.web_search import run_web_search
from api.tasks import FRESHNESS_WINDOW_DAYS, ticker_is_ready
from db.connection import get_connection
from db.queries import get_latest_filing, get_signals_for_filings
from llm import LLMError
from scraper.edgar_client import get_10k_filings, get_10q_filings, get_cik

router = APIRouter()
logger = logging.getLogger(__name__)

# Bank holding companies and other entities with non-standard SEC filing structures not supported in v1.
UNSUPPORTED_TICKERS = {"JPM", "BAC", "WFC", "GS", "C", "MS", "BRK.A", "BRK.B", "USB", "PNC"}


class ChatRequest(BaseModel):
    query: str = Field(max_length=500)
    ticker: str | None = None
    filing_type: str | None = None
    intent: str | None = None


# Loads the most recent signals, risk score, and filing metadata for a ticker from the DB.
def _load_context(ticker: str, filing_type: str) -> tuple[dict, dict, dict | None]:
    with get_connection() as conn:
        latest = get_latest_filing(conn, ticker, filing_type)
        signals_list = get_signals_for_filings(conn, [latest["id"]]) if latest else []

        risk_result = {}
        if latest:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM risk_scores WHERE filing_id = %s",
                    (latest["id"],),
                )
                row = cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    risk_result = dict(zip(cols, row))

    return (signals_list[0] if signals_list else {}), risk_result, latest


# "Last quarter" questions route to the 10-Q, but a fiscal Q4 has no 10-Q — those numbers only
# appear in the 10-K. When the latest 10-K is more recent than the latest 10-Q, analyze the 10-K
# instead. If the company doesn't file one of the two forms, fall back to whichever it does file.
def _resolve_filing_type(ticker: str, filing_type: str) -> str:
    if filing_type != "10-Q":
        return filing_type

    with get_connection() as conn:
        local_q = get_latest_filing(conn, ticker, "10-Q")
        local_k = get_latest_filing(conn, ticker, "10-K")

    # If our newest local filing is inside the quarterly cadence, the DB is current enough to decide.
    local_dates = [f["filed_at"] for f in (local_q, local_k) if f and f["filed_at"]]
    if local_dates and (date.today() - max(local_dates)).days < FRESHNESS_WINDOW_DAYS:
        if local_k and (not local_q or local_k["filed_at"] > local_q["filed_at"]):
            return "10-K"
        return "10-Q"

    # DB is stale or empty — ask EDGAR which form carries the most recent period.
    try:
        cik = get_cik(ticker)
        edgar_q = get_10q_filings(cik, limit=1)
        edgar_k = get_10k_filings(cik, limit=1)
    except Exception:
        return "10-Q"  # EDGAR unreachable or unknown ticker — let the main flow handle it

    if not edgar_q:
        return "10-K" if edgar_k else "10-Q"
    if edgar_k and edgar_k[0]["filed_at"] > edgar_q[0]["filed_at"]:
        return "10-K"
    return "10-Q"


@router.post("/chat")
def chat(body: ChatRequest, request: Request):
    # Maps typed LLM failures and loop timeouts to error responses. Pipeline failures surface
    # via /job/{id}.
    try:
        return _chat(body, request)
    except (LLMError, TimeoutError) as e:
        logger.warning(f"Chat request failed: {type(e).__name__}: {e}")
        return {"type": "error", "message": "The analysis service hit a temporary problem — please try again."}


def _chat(body: ChatRequest, request: Request):
    # Skip orchestrator if ticker and filing_type are already known (e.g. retry after pipeline job).
    # The retry carries the first pass's orchestrator intent; the regex is only a fallback.
    if body.ticker and body.filing_type:
        ticker = body.ticker.upper()
        filing_type = body.filing_type
        intent = {"intent": "single_analysis", "ticker": ticker, "filing_types": [filing_type], "periods_needed": 1, "web_search_needed": False}
        intent_type = body.intent or ("comparison" if is_comparison_query(body.query) else "single_analysis")
    else:
        intent = run_orchestrator(body.query)
        ticker = intent.get("ticker", "").upper()
        filing_type = intent["filing_types"][0] if intent.get("filing_types") else "10-Q"
        intent_type = intent.get("intent")

    # Web-only
    if intent_type == "web_only":
        summary = run_web_search(body.query)
        return {"type": "web", "answer": summary}

    # Block unsupported tickers before any pipeline work starts.
    if ticker and ticker in UNSUPPORTED_TICKERS:
        return {"type": "error", "message": f"{ticker} is a financial institution whose SEC filings use a non-standard structure not supported in v1. Try a company like AAPL, MSFT, NVDA, or TSLA."}

    # Comparison query — uses vector search + structured signals
    if intent_type == "comparison" and is_comparison_query(body.query):
        if not ticker:
            return {"type": "error", "message": "Please mention a specific publicly traded company (e.g. Apple, MSFT, Tesla)."}

        if not ticker_is_ready(ticker, filing_type):
            try:
                cik = get_cik(ticker)
                filings = get_10q_filings(cik, limit=1) if filing_type == "10-Q" else get_10k_filings(cik, limit=1)
            except ValueError:
                return {"type": "error", "message": f"'{ticker}' was not found in SEC EDGAR. Please check the ticker and try again."}

            if not filings:
                summary = run_web_search(body.query)
                return {"type": "web", "answer": f"Note: {ticker} does not file {filing_type} reports (it may be an ETF or index fund). Here's what I found via web search:\n\n{summary}"}

            from api.tasks import run_full_pipeline
            job = request.app.state.queue.enqueue(
                run_full_pipeline,
                ticker,
                filing_type,
                intent.get("periods_needed", 1),
                job_timeout=600,
            )
            return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status.", "ticker": ticker, "filing_type": filing_type, "intent": intent_type}

        result = run_comparison(body.query)
        return {"type": "comparison", **result}

    # Guard: no ticker detected — ask user to specify a company.
    if not ticker:
        return {"type": "error", "message": "Please mention a specific publicly traded company (e.g. Apple, MSFT, Tesla)."}

    # Fiscal-Q4 routing: skip when the caller already pinned a filing type (retry after pipeline job).
    if not body.filing_type:
        filing_type = _resolve_filing_type(ticker, filing_type)

    # Check if ticker data is ready in DB
    if not ticker_is_ready(ticker, filing_type):
        # Pre-check EDGAR before enqueuing a long pipeline job — avoids 2-minute wait for ETFs/invalid tickers.
        try:
            cik = get_cik(ticker)
            filings = get_10q_filings(cik, limit=1) if filing_type == "10-Q" else get_10k_filings(cik, limit=1)
        except ValueError:
            return {"type": "error", "message": f"'{ticker}' was not found in SEC EDGAR. Please check the ticker and try again."}

        if not filings:
            # ETF, index fund, or non-filing entity — fall back to web search.
            summary = run_web_search(body.query)
            return {"type": "web", "answer": f"Note: {ticker} does not file {filing_type} reports (it may be an ETF or index fund). Here's what I found via web search:\n\n{summary}"}

        from api.tasks import run_full_pipeline
        job = request.app.state.queue.enqueue(
            run_full_pipeline,
            ticker,
            filing_type,
            intent.get("periods_needed", 1),
            job_timeout=600,
        )
        return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status.", "ticker": ticker, "filing_type": filing_type, "intent": intent_type}

    # Fast path — data exists, run advice
    extracted_signals, risk_result, latest_filing = _load_context(ticker, filing_type)
    web_summary = run_web_search(body.query) if intent.get("web_search_needed") else None
    advice = run_advice(ticker, extracted_signals, risk_result, web_summary)

    return {
        "type": "advice",
        "ticker": ticker,
        "filing_type": filing_type,
        "period": latest_filing["period"] if latest_filing else None,
        "filed_at": latest_filing["filed_at"].isoformat() if latest_filing and latest_filing["filed_at"] else None,
        **advice.model_dump(),
    }
