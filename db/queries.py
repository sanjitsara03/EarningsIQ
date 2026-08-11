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
# Upserts on filing_id; re-runs are idempotent.
def insert_signals(conn: connection, filing_id: int, results: dict) -> None:
    fm = results.get("extract_financial_metrics", {})
    outlook = results.get("extract_management_outlook", {})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO signals (
                filing_id,
                revenue_usd, eps, gross_margin, operating_margin, revenue_yoy_delta,
                guidance_revenue_usd, guidance_period, guidance_withdrawn,
                segments, notable_changes, risk_factors
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id) DO UPDATE SET
                revenue_usd = EXCLUDED.revenue_usd,
                eps = EXCLUDED.eps,
                gross_margin = EXCLUDED.gross_margin,
                operating_margin = EXCLUDED.operating_margin,
                revenue_yoy_delta = EXCLUDED.revenue_yoy_delta,
                guidance_revenue_usd = EXCLUDED.guidance_revenue_usd,
                guidance_period = EXCLUDED.guidance_period,
                guidance_withdrawn = EXCLUDED.guidance_withdrawn,
                segments = EXCLUDED.segments,
                notable_changes = EXCLUDED.notable_changes,
                risk_factors = EXCLUDED.risk_factors
            """,
            (
                filing_id,
                fm.get("revenue_usd"),
                fm.get("eps"),
                fm.get("gross_margin"),
                fm.get("operating_margin"),
                fm.get("revenue_yoy_delta"),
                outlook.get("guidance_revenue_usd"),
                outlook.get("guidance_period"),
                outlook.get("withdrawn"),
                json.dumps(results.get("extract_segment_performance")),
                json.dumps(results.get("extract_notable_changes")),
                json.dumps(results.get("extract_risk_factors")),
            ),
        )


# Fetches past signals for a ticker ordered by period descending, optionally excluding one filing
# (the one currently being processed). Used as a historical baseline by risk scoring and the
# extraction plausibility guard.
def get_historical_signals(conn: connection, ticker: str, exclude_filing_id: int | None = None) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.period, s.revenue_usd, s.eps, s.gross_margin, s.operating_margin,
                   s.revenue_yoy_delta, s.guidance_revenue_usd, s.guidance_withdrawn
            FROM signals s
            JOIN filings f ON s.filing_id = f.id
            WHERE f.ticker = %s AND (%s::int IS NULL OR s.filing_id != %s)
            ORDER BY f.period DESC
            LIMIT 8
            """,
            (ticker, exclude_filing_id, exclude_filing_id),
        )
        cols = ["period", "revenue_usd", "eps", "gross_margin", "operating_margin",
                "revenue_yoy_delta", "guidance_revenue_usd", "guidance_withdrawn"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# Returns {filing_id: status} for a set of filing IDs. Used by the pipeline to skip completed stages.
def get_filing_statuses(conn: connection, filing_ids: list[int]) -> dict[int, str]:
    if not filing_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, status FROM filings WHERE id = ANY(%s)",
            (filing_ids,),
        )
        return dict(cur.fetchall())


# Inserts a completed risk score row into the risk_scores table.
# Upserts on filing_id; re-runs are idempotent.
def insert_risk_score(
    conn: connection,
    filing_id: int,
    scores: dict,
    overall_score: float,
    risk_tier: str,
    executive_summary: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO risk_scores (
                filing_id, guidance_cut_risk, demand_uncertainty, margin_pressure,
                competitive_threat, macro_exposure, overall_score, risk_tier, executive_summary
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (filing_id) DO UPDATE SET
                guidance_cut_risk = EXCLUDED.guidance_cut_risk,
                demand_uncertainty = EXCLUDED.demand_uncertainty,
                margin_pressure = EXCLUDED.margin_pressure,
                competitive_threat = EXCLUDED.competitive_threat,
                macro_exposure = EXCLUDED.macro_exposure,
                overall_score = EXCLUDED.overall_score,
                risk_tier = EXCLUDED.risk_tier,
                executive_summary = EXCLUDED.executive_summary
            """,
            (
                filing_id,
                scores["guidance_cut_risk"],
                scores["demand_uncertainty"],
                scores["margin_pressure"],
                scores["competitive_threat"],
                scores["macro_exposure"],
                overall_score,
                risk_tier,
                executive_summary,
            ),
        )


# Returns the ticker for a filing id, or None if the filing doesn't exist.
def get_filing_ticker(conn: connection, filing_id: int) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT ticker FROM filings WHERE id = %s", (filing_id,))
        row = cur.fetchone()
        return row[0] if row else None


# Returns the most recent filing row for a ticker and filing type, or None if we've never ingested one.
# Used for freshness checks (compare filed_at/accession against EDGAR) and for showing the filing date in results.
def get_latest_filing(conn: connection, ticker: str, filing_type: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, period, filed_at, accession FROM filings
            WHERE ticker = %s AND filing_type = %s
            ORDER BY period DESC
            LIMIT 1
            """,
            (ticker, filing_type),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip(["id", "period", "filed_at", "accession"], row))


# Returns filing IDs for a ticker and filing type, ordered by period descending. Used by the Comparison Agent.
def get_filing_ids_for_ticker(conn: connection, ticker: str, filing_type: str, limit: int = 4) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id FROM filings
            WHERE ticker = %s AND filing_type = %s
            ORDER BY period DESC
            LIMIT %s
            """,
            (ticker, filing_type, limit),
        )
        return [row[0] for row in cur.fetchall()]


# Returns signals rows for a list of filing IDs, joined with period and ticker for context.
def get_signals_for_filings(conn: connection, filing_ids: list[int]) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT f.ticker, f.period, f.filing_type, f.filed_at,
                   s.revenue_usd, s.eps, s.gross_margin, s.operating_margin,
                   s.revenue_yoy_delta, s.guidance_revenue_usd, s.guidance_withdrawn,
                   s.segments, s.notable_changes
            FROM signals s
            JOIN filings f ON s.filing_id = f.id
            WHERE s.filing_id = ANY(%s)
            ORDER BY f.period DESC
            """,
            (filing_ids,),
        )
        cols = ["ticker", "period", "filing_type", "filed_at", "revenue_usd", "eps", "gross_margin",
                "operating_margin", "revenue_yoy_delta", "guidance_revenue_usd",
                "guidance_withdrawn", "segments", "notable_changes"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# Advances a filing to the next pipeline status (e.g. pending → chunked → embedded → extracted → scored).
def update_filing_status(conn: connection, filing_id: int, status: str) -> None:
    """Advance a filing to the next pipeline status."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE filings SET status = %s WHERE id = %s",
            (status, filing_id),
        )
