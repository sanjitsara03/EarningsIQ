# Data access layer for the filings table.
# All functions take a psycopg2 connection
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


# Advances a filing to the next pipeline status (e.g. pending → chunked → embedded → extracted → scored).
def update_filing_status(conn: connection, filing_id: int, status: str) -> None:
    """Advance a filing to the next pipeline status."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE filings SET status = %s WHERE id = %s",
            (status, filing_id),
        )
