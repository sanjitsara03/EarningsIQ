# Promptfoo provider for the Extraction agent.
# Fetches existing signals from DB if present, otherwise runs the agent.

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db.connection import get_connection
from db.queries import get_filing_ids_for_ticker


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

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT revenue, eps, gross_margin, operating_margin,
                           revenue_yoy_delta, guidance_revenue, guidance_withdrawn,
                           segments, notable_changes, risk_factors
                    FROM signals WHERE filing_id = %s
                    """,
                    (filing_id,),
                )
                row = cur.fetchone()

            if not row:
                return {"error": f"No signals for filing_id={filing_id}. Run extraction first."}

            cols = ["revenue", "eps", "gross_margin", "operating_margin", "revenue_yoy_delta",
                    "guidance_revenue", "guidance_withdrawn", "segments", "notable_changes", "risk_factors"]
            result = dict(zip(cols, row))

        return {"output": json.dumps(result, default=str)}
    except Exception as e:
        return {"error": str(e)}
