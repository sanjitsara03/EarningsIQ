import logging
import os

import anthropic
from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import get_historical_signals, insert_risk_score, update_filing_status
from tools.risk_tools import (
    TOOLS_BY_STEP,
    compute_overall_score,
    handle_finalize_overall_risk,
    handle_score_risk_components,
)

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"

SYSTEM_PROMPT = """You are a financial risk analyst scoring a company's SEC filing against its historical performance.

You will be given the current filing's extracted signals and must call three tools in strict order:
1. fetch_historical_signals — retrieve this company's historical data from the database
2. score_risk_components — score each of the five risk components 1–10 using the current filing and history
3. finalize_overall_risk — write a concise executive summary of your risk assessment

Base your scores on concrete evidence: margin trends, guidance changes, risk factor language, and segment performance.
Do not call the next tool until the previous one has returned."""


# Formats extracted signals as a readable prompt section for the LLM.
def _build_prompt(ticker: str, extracted_signals: dict) -> str:
    return f"""Company: {ticker}

Current filing extracted signals:
{extracted_signals}

Call fetch_historical_signals first to retrieve this company's historical baseline, then score the risk components."""


# Formats historical signals rows as a readable string to return as a tool result to the LLM.
def _format_historical(rows: list[dict]) -> str:
    if not rows:
        return "No historical data found for this ticker."
    lines = ["Historical signals (most recent first):"]
    for row in rows:
        lines.append(
            f"  {row['period']}: revenue={row['revenue']}, gross_margin={row['gross_margin']}%, "
            f"operating_margin={row['operating_margin']}%, revenue_yoy_delta={row['revenue_yoy_delta']}%, "
            f"guidance_withdrawn={row['guidance_withdrawn']}"
        )
    return "\n".join(lines)


# Runs the strict 3-step tool loop. Accepts extracted_signals from the Extraction Agent directly (no second DB read).
# Inserts into risk_scores table and updates filing status to 'scored'.
def run_risk_scoring(filing_id: int, ticker: str, extracted_signals: dict) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": _build_prompt(ticker, extracted_signals)}]
    scores = {}
    executive_summary = ""

    for step in range(3):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOLS_BY_STEP[step],
            tool_choice={"type": "any"},
            messages=messages,
        )

        tool_use = next(b for b in response.content if b.type == "tool_use")
        logger.info(f"Step {step + 1}/3 — tool called: {tool_use.name}")

        # Tool 1: execute DB query and return real data to the LLM
        if tool_use.name == "fetch_historical_signals":
            with get_connection() as conn:
                rows = get_historical_signals(conn, tool_use.input["ticker"])
            tool_result_content = _format_historical(rows)

        # Tool 2: collect component scores, compute weighted average
        elif tool_use.name == "score_risk_components":
            scores = handle_score_risk_components(tool_use.input)
            overall_score, risk_tier = compute_overall_score(scores)
            tool_result_content = f"Scores recorded. Overall score: {overall_score} ({risk_tier}). Now write the executive summary."

        # Tool 3: collect executive summary
        elif tool_use.name == "finalize_overall_risk":
            executive_summary = handle_finalize_overall_risk(tool_use.input)
            tool_result_content = "Done."

        # Append assistant turn and tool result, then continue to next step
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tool_use.id, "content": tool_result_content}
            ],
        })

    with get_connection() as conn:
        insert_risk_score(conn, filing_id, scores, overall_score, risk_tier, executive_summary)
        update_filing_status(conn, filing_id, "scored")

    logger.info(f"Risk scoring complete for filing_id={filing_id} — tier={risk_tier}, score={overall_score}")

    return {
        "scores": scores,
        "overall_score": overall_score,
        "risk_tier": risk_tier,
        "executive_summary": executive_summary,
    }


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("filing_id", type=int)
    parser.add_argument("ticker", type=str)
    args = parser.parse_args()

    # Load extracted signals from DB for standalone testing
    from db.queries import get_chunks_for_filing
    import json as _json

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM signals WHERE filing_id = %s", (args.filing_id,))
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            extracted_signals = dict(zip(cols, row)) if row else {}

    result = run_risk_scoring(args.filing_id, args.ticker, extracted_signals)
    print(json.dumps(result, indent=2))
