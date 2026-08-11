# Deletes a ticker's filings (CASCADE wipes chunks, signals, and risk scores) and re-runs the
# full pipeline so they are re-fetched, re-parsed, re-embedded, re-extracted, and re-scored.
# Use after a parser or extraction fix that invalidates stored chunks.
#
# Usage: uv run python scripts/reingest_filings.py AAPL --type 10-Q
#        uv run python scripts/reingest_filings.py AAPL --type 10-Q --delete-only
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete and re-ingest a ticker's filings")
    parser.add_argument("ticker")
    parser.add_argument("--type", default="10-Q", choices=["10-Q", "10-K"], dest="filing_type")
    parser.add_argument("--delete-only", action="store_true", help="delete rows; let the app's freshness check re-ingest on next query")
    args = parser.parse_args()
    ticker = args.ticker.upper()

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, accession, period, status FROM filings WHERE ticker = %s AND filing_type = %s ORDER BY period",
                (ticker, args.filing_type),
            )
            rows = cur.fetchall()
            for r in rows:
                logger.info(f"Deleting filing id={r[0]} accession={r[1]} period={r[2]} status={r[3]}")
            cur.execute("DELETE FROM filings WHERE ticker = %s AND filing_type = %s", (ticker, args.filing_type))
    logger.info(f"Deleted {len(rows)} {args.filing_type} filing(s) for {ticker} (chunks/signals/risk cascade).")

    if args.delete_only:
        logger.info("--delete-only: the app's freshness check will re-ingest on the next query.")
        return

    from api.tasks import run_full_pipeline

    # Note: this fetches the N NEWEST filings from EDGAR, which can differ from the deleted set
    # when newer filings exist — the window shifts forward, which is what a demo wants.
    logger.info(f"Re-ingesting the {max(len(rows), 1)} newest {args.filing_type} filing(s) from EDGAR.")
    result = run_full_pipeline(ticker, args.filing_type, limit=max(len(rows), 1))
    logger.info(f"Re-ingest complete: {result}")

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT f.period, s.eps, s.revenue_usd FROM filings f JOIN signals s ON s.filing_id = f.id
               WHERE f.ticker = %s AND f.filing_type = %s ORDER BY f.period DESC""",
            (ticker, args.filing_type),
        )
        for period, eps, revenue in cur.fetchall():
            logger.info(f"{ticker} {period}: eps={eps} revenue_usd={float(revenue):,.0f}" if revenue else f"{ticker} {period}: eps={eps}")


if __name__ == "__main__":
    main()
