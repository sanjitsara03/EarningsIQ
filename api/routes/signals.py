# GET /signals/{ticker} — returns extracted signals for the most recent filing.

from fastapi import APIRouter, HTTPException

from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker, get_signals_for_filings

router = APIRouter()


@router.get("/signals/{ticker}")
def get_signals(ticker: str, filing_type: str = "10-Q", limit: int = 1):
    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker.upper(), filing_type, limit=limit)
        if not filing_ids:
            raise HTTPException(status_code=404, detail=f"No filings found for {ticker}")
        signals = get_signals_for_filings(conn, filing_ids)
        if not signals:
            raise HTTPException(status_code=404, detail=f"No signals found for {ticker} — run /ingest first")
    return {"ticker": ticker.upper(), "signals": signals}
