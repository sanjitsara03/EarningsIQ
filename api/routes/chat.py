# POST /chat — fast path answers from data already in the DB; slow path enqueues the pipeline and returns a job_id.

import logging
from datetime import date

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agents.advice import run_advice, run_analysis
from agents.comparison import is_comparison_query, run_comparison
from agents.orchestrator import run_orchestrator
from agents.web_search import run_web_search, run_web_search_detailed
from api.rate_limit import check_rate_limit
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


# Fiscal Q4 has no 10-Q — route to the 10-K when it's newer; fall back to whichever form the company files.
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
    # Every /chat request can trigger LLM spend — rate-limit before any model call.
    check_rate_limit(request, "chat")
    # Typed LLM failures and loop timeouts map to error responses; pipeline failures surface via /job/{id}.
    try:
        return _chat(body, request)
    except (LLMError, TimeoutError) as e:
        logger.warning(f"Chat request failed: {type(e).__name__}: {e}")
        return {"type": "error", "message": "The analysis service hit a temporary problem — please try again."}


def _chat(body: ChatRequest, request: Request):
    # Ticker + filing_type known (retry after a pipeline job) — skip the orchestrator; regex is only a fallback.
    if body.ticker and body.filing_type:
        ticker = body.ticker.upper()
        filing_type = body.filing_type
        intent = {"intent": "single_analysis", "ticker": ticker, "filing_types": [filing_type], "periods_needed": 1, "web_search_needed": False}
        intent_type = body.intent or ("comparison" if is_comparison_query(body.query) else "single_analysis")
    else:
        intent = run_orchestrator(body.query)
        # The orchestrator can emit "ticker": null (e.g. "how are markets doing?") — guard before .upper().
        ticker = (intent.get("ticker") or "").upper()
        filing_type = intent["filing_types"][0] if intent.get("filing_types") else "10-Q"
        intent_type = intent.get("intent")

    if intent_type == "web_only":
        summary, sources = run_web_search_detailed(body.query)
        return {
            "type": "web",
            "answer": summary,
            "sources": [{"title": s["title"], "url": s["url"]} for s in sources],
        }

    # Block unsupported tickers before any pipeline work starts.
    if ticker and ticker in UNSUPPORTED_TICKERS:
        return {"type": "error", "message": f"{ticker} is a financial institution whose SEC filings use a non-standard structure not supported in v1. Try a company like AAPL, MSFT, NVDA, or TSLA."}

    if intent_type == "comparison" and is_comparison_query(body.query):
        if not ticker:
            return {"type": "error", "message": "Please mention a specific publicly traded company (e.g. Apple, MSFT, Tesla)."}

        # Backfill missing quarters; the post-pipeline retry hint pins periods_needed=1, so no infinite retry.
        periods_needed = intent.get("periods_needed") or 1
        if not ticker_is_ready(ticker, filing_type, min_filings=periods_needed):
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
                periods_needed,
                job_timeout=600,
            )
            return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status.", "ticker": ticker, "filing_type": filing_type, "intent": intent_type}

        result = run_comparison(body.query)
        # tool_trace is for the benchmark judge, not the API — serve only answer + citations.
        return {"type": "comparison", "answer": result["answer"], "citations": result.get("citations", [])}

    if not ticker:
        return {"type": "error", "message": "Please mention a specific publicly traded company (e.g. Apple, MSFT, Tesla)."}

    # Fiscal-Q4 routing: skip when the caller already pinned a filing type (retry after pipeline job).
    if not body.filing_type:
        filing_type = _resolve_filing_type(ticker, filing_type)

    if not ticker_is_ready(ticker, filing_type):
        # Pre-check EDGAR before enqueuing a long pipeline job — avoids 2-minute wait for ETFs/invalid tickers.
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

    # Fast path: single_analysis gets a factual answer with no stance, advice gets the buy/hold/sell card.
    extracted_signals, risk_result, latest_filing = _load_context(ticker, filing_type)
    web_summary = run_web_search(body.query) if intent.get("web_search_needed") else None
    period = latest_filing["period"] if latest_filing else None
    provenance = {
        "ticker": ticker,
        "filing_type": filing_type,
        "period": period,
        "filed_at": latest_filing["filed_at"].isoformat() if latest_filing and latest_filing["filed_at"] else None,
    }

    if intent_type == "single_analysis":
        analysis = run_analysis(
            body.query, ticker, extracted_signals, risk_result, web_summary,
            filing_type, str(period) if period else None,
        )
        return {"type": "analysis", **provenance, **analysis.model_dump()}

    advice = run_advice(
        ticker, extracted_signals, risk_result, web_summary,
        filing_type, str(period) if period else None,
    )
    return {"type": "advice", **provenance, **advice.model_dump()}
