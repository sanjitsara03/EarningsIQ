# RQ job functions executed by workers.
# Each function is a self-contained pipeline step that can be enqueued independently.
# run_full_pipeline chains ingest → extraction → risk scoring for a ticker that isn't in the DB yet.

import logging
import time
from datetime import date

from agents.extraction import run_extraction
from agents.risk_scoring import run_risk_scoring
from db.connection import get_connection
from db.queries import filing_exists, get_filing_ids_for_ticker, get_latest_filing, get_signals_for_filings
from pipeline.ingest import ingest
from scraper.edgar_client import get_10k_filings, get_10q_filings, get_cik

logger = logging.getLogger(__name__)

# Companies file quarterly, so a filing younger than this can't have a successor on EDGAR yet.
# Below this age we trust the DB and skip the network check entirely.
FRESHNESS_WINDOW_DAYS = 85


# Ingests a ticker, then runs extraction and risk scoring on each new filing.
def run_full_pipeline(ticker: str, filing_type: str = "10-Q", limit: int = 1) -> dict:
    logger.info(f"Starting full pipeline for {ticker} {filing_type}")

    ingest(ticker, filing_type=filing_type, limit=limit)

    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=limit)

    results = []
    for i, filing_id in enumerate(filing_ids):
        logger.info(f"Running extraction for filing_id={filing_id}")
        extracted = run_extraction(filing_id)

        if filing_type == "10-K":
            time.sleep(60)  # wait for TPM window to reset before risk scoring
        logger.info(f"Running risk scoring for filing_id={filing_id}")
        risk = run_risk_scoring(filing_id, ticker, extracted)

        results.append({"filing_id": filing_id, "risk_tier": risk["risk_tier"], "overall_score": risk["overall_score"]})

        if filing_type == "10-K" and i < len(filing_ids) - 1:
            time.sleep(60)  # wait before next filing's extraction to avoid TPM stacking

    logger.info(f"Pipeline complete for {ticker}: {results}")
    return {"ticker": ticker, "filings_processed": results}


# Checks whether a ticker has fully scored data in the DB that is still current.
# "Current" means: either the latest local filing is young enough that no successor can exist yet,
# or EDGAR confirms we already hold the newest filing. A newer filing on EDGAR returns False,
# which sends the chat route down the existing slow path (enqueue pipeline → poll → retry).
def ticker_is_ready(ticker: str, filing_type: str = "10-Q") -> bool:
    with get_connection() as conn:
        latest = get_latest_filing(conn, ticker, filing_type)
        if latest is None:
            return False
        signals = get_signals_for_filings(conn, [latest["id"]])
        if not signals:
            return False

    # Inside the quarterly filing cadence — nothing newer can exist, skip the EDGAR round trip.
    if latest["filed_at"] and (date.today() - latest["filed_at"]).days < FRESHNESS_WINDOW_DAYS:
        return True

    # Stale window: ask EDGAR whether a newer filing has been published since we ingested this one.
    try:
        cik = get_cik(ticker)
        filings = get_10q_filings(cik, limit=1) if filing_type == "10-Q" else get_10k_filings(cik, limit=1)
    except Exception:
        logger.warning(f"EDGAR freshness check failed for {ticker} — serving existing data")
        return True  # EDGAR unreachable: serve what we have rather than failing the request

    if not filings:
        return True

    with get_connection() as conn:
        return filing_exists(conn, filings[0]["accession"])
