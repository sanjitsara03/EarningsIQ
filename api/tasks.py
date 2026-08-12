# RQ job functions executed by workers.
# Each function is a self-contained pipeline step that can be enqueued independently.
# run_full_pipeline chains ingest → extraction → risk scoring for a ticker that isn't in the DB yet.

import logging
from datetime import date

from agents.extraction import run_extraction
from agents.risk_scoring import run_risk_scoring
from db.connection import get_connection
from db.queries import (
    filing_exists,
    get_filing_ids_for_ticker,
    get_filing_statuses,
    get_latest_filing,
    get_signals_for_filings,
)
from llm import flush_traces, traced
from pipeline.ingest import ingest
from scraper.edgar_client import get_10k_filings, get_10q_filings, get_cik

logger = logging.getLogger(__name__)

# RQ workers import this module directly — configure logging here so pipeline/agent warnings
# (e.g. dropped unverifiable quotes) reach the worker logs. No-op when already configured.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Filings younger than this cannot have a successor on EDGAR (quarterly cadence); below this age
# the freshness check skips the network round trip.
FRESHNESS_WINDOW_DAYS = 85


# Ingests a ticker, then runs extraction and risk scoring on each filing that still needs it.
# Status gating: 'scored' filings are skipped; 'embedded'/'extracted' filings run the remaining
# stages (inserts are upserts). If ingestion produced nothing new and no filing needs work, the
# job raises before any LLM call. flush_traces() runs after the @traced root span on the inner
# function has closed; RQ work-horses exit immediately after the job.
def run_full_pipeline(ticker: str, filing_type: str = "10-Q", limit: int = 1) -> dict:
    try:
        return _run_full_pipeline(ticker, filing_type, limit)
    finally:
        flush_traces()


@traced("full_pipeline")
def _run_full_pipeline(ticker: str, filing_type: str, limit: int) -> dict:
    logger.info(f"Starting full pipeline for {ticker} {filing_type}")

    new_ids = ingest(ticker, filing_type=filing_type, limit=limit)

    with get_connection() as conn:
        filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=limit)
        statuses = get_filing_statuses(conn, filing_ids)

    todo = [fid for fid in filing_ids if statuses.get(fid) in ("embedded", "extracted")]
    if not todo:
        if new_ids:
            # Ingested but not embedded — shouldn't happen, but don't silently succeed.
            raise RuntimeError(f"Filings {new_ids} were ingested but none reached 'embedded' status.")
        raise RuntimeError(
            f"No new filings ingested for {ticker} {filing_type} and none need processing. "
            "Likely cause: ingestion failed upstream (e.g. embedding API credits) — check worker logs."
        )

    results = []
    for filing_id in todo:
        logger.info(f"Running extraction for filing_id={filing_id} (status={statuses.get(filing_id)})")
        extracted = run_extraction(filing_id)

        logger.info(f"Running risk scoring for filing_id={filing_id}")
        risk = run_risk_scoring(filing_id, ticker, extracted)

        results.append({"filing_id": filing_id, "risk_tier": risk["risk_tier"], "overall_score": risk["overall_score"]})

    logger.info(f"Pipeline complete for {ticker}: {results}")
    return {"ticker": ticker, "filings_processed": results}


# Checks whether a ticker has fully scored data in the DB that is still current.
# "Current" means: either the latest local filing is young enough that no successor can exist yet,
# or EDGAR confirms we already hold the newest filing. A newer filing on EDGAR returns False,
# which sends the chat route down the existing slow path (enqueue pipeline → poll → retry).
# min_filings > 1 (multi-quarter comparisons) additionally requires that many filings with signals,
# so a 4-quarter question backfills history instead of running against a single quarter.
def ticker_is_ready(ticker: str, filing_type: str = "10-Q", min_filings: int = 1) -> bool:
    with get_connection() as conn:
        latest = get_latest_filing(conn, ticker, filing_type)
        if latest is None:
            return False
        check_ids = (
            get_filing_ids_for_ticker(conn, ticker, filing_type, limit=min_filings)
            if min_filings > 1
            else [latest["id"]]
        )
        if len(check_ids) < min_filings:
            return False
        signals = get_signals_for_filings(conn, check_ids)
        if len(signals) < min_filings:
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
        return True  # EDGAR unreachable — serve existing data

    if not filings:
        return True

    with get_connection() as conn:
        return filing_exists(conn, filings[0]["accession"])
