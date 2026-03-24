# Data access layer for the filings table.
# All functions take a psycopg2 connection
import json

from psycopg2.extensions import connection


# Inserts a new row into the filings table and returns the generated id.
def insert_filing(
    conn: connection,
    ticker: str,
    cik: str,
    filing_type: str,
    period: str,
    accession: str,
    filed_at: str | None,
) -> int:
    """Insert a new filing row and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO filings (ticker, cik, filing_type, period, accession, filed_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (ticker, cik, filing_type, period, accession, filed_at),
        )
        return cur.fetchone()[0]


# Checks whether a filing with this accession number already exists. Used as a dedup guard before fetching.
def filing_exists(conn: connection, accession: str) -> bool:
    """Return True if this accession number is already in the filings table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM filings WHERE accession = %s",
            (accession,),
        )
        return cur.fetchone() is not None


# Returns the current pipeline status for a filing, or None if the accession isn't found.
def get_filing_status(conn: connection, accession: str) -> str | None:
    """Return the pipeline status for this accession, or None if not found."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT status FROM filings WHERE accession = %s",
            (accession,),
        )
        row = cur.fetchone()
        return row[0] if row else None


# Fetches all chunks for a filing ordered by chunk_index. Returns list of dicts with section and content.
def get_chunks_for_filing(conn: connection, filing_id: int) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT section, content FROM chunks WHERE filing_id = %s ORDER BY chunk_index",
            (filing_id,),
        )
        return [{"section": row[0], "content": row[1]} for row in cur.fetchall()]


# Inserts extracted signals into the signals table. JSONB fields are serialized from Python objects.
def insert_signals(conn: connection, filing_id: int, results: dict) -> None:
    fm = results.get("extract_financial_metrics", {})
    outlook = results.get("extract_management_outlook", {})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signals (
                filing_id,
                revenue, eps, gross_margin, operating_margin, revenue_yoy_delta,
                guidance_revenue, guidance_period, guidance_withdrawn,
                segments, notable_changes, risk_factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                filing_id,
                fm.get("revenue"),
                fm.get("eps"),
                fm.get("gross_margin"),
                fm.get("operating_margin"),
                fm.get("revenue_yoy_delta"),
                outlook.get("guidance_revenue"),
                outlook.get("guidance_period"),
                outlook.get("withdrawn"),
                json.dumps(results.get("extract_segment_performance")),
                json.dumps(results.get("extract_notable_changes")),
                json.dumps(results.get("extract_risk_factors")),
            ),
        )


# Advances a filing to the next pipeline status (e.g. pending → chunked → embedded → extracted → scored).
def update_filing_status(conn: connection, filing_id: int, status: str) -> None:
    """Advance a filing to the next pipeline status."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE filings SET status = %s WHERE id = %s",
            (status, filing_id),
        )
