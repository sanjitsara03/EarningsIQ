# POST /chat — the main query endpoint.
# Fast path: if ticker data is already in DB, runs advice and returns the answer immediately.
# Slow path: if data is missing, enqueues the full pipeline and returns a job_id to poll.

from fastapi import APIRouter, Request
from pydantic import BaseModel

from agents.advice import run_advice
from agents.comparison import is_comparison_query, run_comparison
from agents.orchestrator import run_orchestrator
from agents.web_search import run_web_search
from api.tasks import ticker_is_ready
from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker, get_signals_for_filings

router = APIRouter()


class ChatRequest(BaseModel):
    query: str


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
        result = run_comparison(body.query)
        return {"type": "comparison", **result}

    # Check if ticker data is ready in DB
    if not ticker_is_ready(ticker, filing_type):
        from api.tasks import run_full_pipeline
        job = request.app.state.queue.enqueue(
            run_full_pipeline,
            ticker,
            filing_type,
            intent.get("periods_needed", 1),
            job_timeout=600,
        )
        return {"type": "queued", "job_id": job.id, "status": "queued", "message": f"Ingesting {ticker} — poll /job/{job.id} for status."}

    # Fast path — data exists, run advice
    extracted_signals, risk_result = _load_context(ticker, filing_type)
    web_summary = run_web_search(body.query) if intent.get("web_search_needed") else None
    advice = run_advice(ticker, extracted_signals, risk_result, web_summary)

    return {"type": "advice", "ticker": ticker, **advice.model_dump()}
