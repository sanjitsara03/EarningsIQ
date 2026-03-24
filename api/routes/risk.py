# GET /risk/{ticker} — returns the most recent risk score for a ticker.

from fastapi import APIRouter, HTTPException

from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker

router = APIRouter()


@router.get("/risk/{ticker}")
def get_risk(ticker: str, filing_type: str = "10-Q"):
    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker.upper(), filing_type, limit=1)
        if not filing_ids:
            raise HTTPException(status_code=404, detail=f"No filings found for {ticker}")
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM risk_scores WHERE filing_id = %s",
                (filing_ids[0],),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"No risk score found for {ticker} — run /ingest first")
            cols = [d[0] for d in cur.description]
            result = dict(zip(cols, row))
    return {"ticker": ticker.upper(), "risk": result}
