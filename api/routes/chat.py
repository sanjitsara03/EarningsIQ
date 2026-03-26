# POST /chat — the main query endpoint.
# Fast path: if ticker data is already in DB, runs advice and returns the answer immediately.
# Slow path: if data is missing, enqueues the full pipeline and returns a job_id to poll.

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from agents.advice import run_advice
from agents.comparison import is_comparison_query, run_comparison
from agents.orchestrator import run_orchestrator
from agents.web_search import run_web_search
from api.tasks import ticker_is_ready
from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker, get_signals_for_filings
from scraper.edgar_client import get_10k_filings, get_10q_filings, get_cik

router = APIRouter()


class ChatRequest(BaseModel):
    query: str = Field(max_length=500)
    ticker: str | None = None
    filing_type: str | None = None


# Loads the most recent signals and risk score for a ticker from the DB.
def _load_context(ticker: str, filing_type: str) -> tuple[dict, dict]:
    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=1)
        signals_list = get_signals_for_filings(conn, filing_ids) if filing_ids else []

        risk_result = {}
        if filing_ids:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM risk_scores WHERE filing_id = %s",
                    (filing_ids[0],),
                )
                row = cur.fetchone()
                if row:
                    cols = [d[0] for d in cur.description]
                    risk_result = dict(zip(cols, row))

    return (signals_list[0] if signals_list else {}), risk_result


@router.post("/chat")
def chat(body: ChatRequest, request: Request):
    # Skip orchestrator if ticker and filing_type are already known (e.g. retry after pipeline job).
    if body.ticker and body.filing_type:
        ticker = body.ticker.upper()
        filing_type = body.filing_type
        intent = {"intent": "single_analysis", "ticker": ticker, "filing_types": [filing_type], "periods_needed": 1, "web_search_needed": False}
        intent_type = "comparison" if is_comparison_query(body.query) else "single_analysis"
    else:
        intent = run_orchestrator(body.query)
        ticker = intent.get("ticker", "").upper()
        filing_type = intent["filing_types"][0] if intent.get("filing_types") else "10-Q"
        intent_type = intent.get("intent")

    # Web-only
    if intent_type == "web_only":
        summary = run_web_search(body.query)
        return {"type": "web", "answer": summary}

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
            return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status.", "ticker": ticker, "filing_type": filing_type}

        result = run_comparison(body.query)
        return {"type": "comparison", **result}

    # Guard: no ticker detected — ask user to specify a company.
    if not ticker:
        return {"type": "error", "message": "Please mention a specific publicly traded company (e.g. Apple, MSFT, Tesla)."}

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
        return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status.", "ticker": ticker, "filing_type": filing_type}

    # Fast path — data exists, run advice
    extracted_signals, risk_result = _load_context(ticker, filing_type)
    web_summary = run_web_search(body.query) if intent.get("web_search_needed") else None
    advice = run_advice(ticker, extracted_signals, risk_result, web_summary)

    return {"type": "advice", "ticker": ticker, "filing_type": filing_type, **advice.model_dump()}
