# RQ job functions executed by workers.
# Each function is a self-contained pipeline step that can be enqueued independently.
# run_full_pipeline chains ingest → extraction → risk scoring for a ticker that isn't in the DB yet.

import logging
import time

from agents.extraction import run_extraction
from agents.risk_scoring import run_risk_scoring
from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker, get_signals_for_filings
from pipeline.ingest import ingest

logger = logging.getLogger(__name__)


# Ingests a ticker, then runs extraction and risk scoring on each new filing.
def run_full_pipeline(ticker: str, filing_type: str = "10-Q", limit: int = 1) -> dict:
    logger.info(f"Starting full pipeline for {ticker} {filing_type}")

    ingest(ticker, filing_type=filing_type, limit=limit)

    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=limit)

    results = []
    for filing_id in filing_ids:
        logger.info(f"Running extraction for filing_id={filing_id}")
        extracted = run_extraction(filing_id)

        if filing_type == "10-K":
            time.sleep(60)  # 10-K prompts are large enough to hit the 30k TPM limit
        logger.info(f"Running risk scoring for filing_id={filing_id}")
        risk = run_risk_scoring(filing_id, ticker, extracted)

        results.append({"filing_id": filing_id, "risk_tier": risk["risk_tier"], "overall_score": risk["overall_score"]})

    logger.info(f"Pipeline complete for {ticker}: {results}")
    return {"ticker": ticker, "filings_processed": results}


# Checks whether a ticker already has fully scored data in the DB.
def ticker_is_ready(ticker: str, filing_type: str = "10-Q") -> bool:
    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=1)
        if not filing_ids:
            return False
        signals = get_signals_for_filings(conn, filing_ids)
        return len(signals) > 0
