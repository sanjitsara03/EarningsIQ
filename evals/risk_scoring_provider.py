# Promptfoo provider for the Risk Scoring agent.
# Fetches existing risk score from DB if present, otherwise runs the agent.

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker
from agents.risk_scoring import run_risk_scoring


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        vars = context.get("vars", {})
        ticker = vars["ticker"]
        filing_type = vars.get("filing_type", "10-Q")

        with get_connection() as conn:
            filing_ids = get_filing_ids_for_ticker(conn, ticker, filing_type, limit=1)

            if not filing_ids:
                return {"error": f"No {filing_type} filing found for {ticker} in DB"}

            filing_id = filing_ids[0]

            # Return existing score if already computed
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT overall_score, risk_tier, executive_summary FROM risk_scores WHERE filing_id = %s",
                    (filing_id,),
                )
                row = cur.fetchone()

            if row:
                result = {"overall_score": float(row[0]), "risk_tier": row[1], "executive_summary": row[2]}
                return {"output": json.dumps(result)}

            # Otherwise fetch signals and run agent
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.revenue_usd, s.eps, s.gross_margin, s.operating_margin,
                           s.revenue_yoy_delta, s.guidance_revenue_usd, s.guidance_withdrawn,
                           s.segments, s.notable_changes, s.risk_factors
                    FROM signals s WHERE s.filing_id = %s
                    """,
                    (filing_id,),
                )
                row = cur.fetchone()

            if not row:
                return {"error": f"No signals found for filing_id={filing_id}. Run extraction first."}

            cols = ["revenue_usd", "eps", "gross_margin", "operating_margin", "revenue_yoy_delta",
                    "guidance_revenue_usd", "guidance_withdrawn", "segments", "notable_changes", "risk_factors"]
            extracted_signals = dict(zip(cols, row))

        result = run_risk_scoring(filing_id, ticker, extracted_signals)
        return {"output": json.dumps(result, default=str)}
    except Exception as e:
        return {"error": str(e)}
