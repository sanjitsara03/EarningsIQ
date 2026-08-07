import logging

from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import get_historical_signals, insert_risk_score, update_filing_status
from llm import (
    WrongToolError,
    assistant_msg_from_response,
    chat,
    named_tool_choice,
    to_openai_tools,
    tool_result_msgs,
    traced,
    user_msg,
)
from tools.risk_tools import (
    STEP_ORDER,
    TOOLS,
    compute_overall_score,
    handle_finalize_overall_risk,
    handle_score_risk_components,
)

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_TOOLS = to_openai_tools(TOOLS)

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


# Forces one step's named tool and verifies the model called it. One corrective retry, then
# WrongToolError.
def _forced_step(tool_name: str, messages: list[dict]):
    for attempt in range(2):
        # gpt-5.2 endpoints do not declare parallel_tool_calls; sending it with
        # require_parameters=true returns a routing 404.
        resp = chat(
            "risk_scoring", messages, system=SYSTEM_PROMPT,
            tools=OPENAI_TOOLS, tool_choice=named_tool_choice(tool_name),
        )
        tc = resp.tool_calls[0]
        if tc.name == tool_name:
            return resp, tc
        logger.warning(f"Expected tool {tool_name!r} but got {tc.name!r} (attempt {attempt + 1})")
        if attempt == 1:
            raise WrongToolError("risk_scoring", tool_name, tc.name)
        messages.append(assistant_msg_from_response(resp))
        messages.extend(
            tool_result_msgs([(c.id, f"Error: you must call {tool_name} for this step.") for c in resp.tool_calls])
        )


# Runs the strict 3-step tool loop. Accepts extracted_signals from the Extraction Agent directly (no second DB read).
# Inserts into risk_scores table and updates filing status to 'scored'.
@traced("risk_scoring")
def run_risk_scoring(filing_id: int, ticker: str, extracted_signals: dict) -> dict:
    messages = [user_msg(_build_prompt(ticker, extracted_signals))]
    scores = {}
    executive_summary = ""
    overall_score: float | None = None
    risk_tier: str | None = None

    for step, tool_name in enumerate(STEP_ORDER):
        resp, tool_call = _forced_step(tool_name, messages)
        logger.info(f"Step {step + 1}/3 — tool called: {tool_call.name}")

        # Tool 1: execute DB query and return real data to the LLM
        if tool_name == "fetch_historical_signals":
            with get_connection() as conn:
                rows = get_historical_signals(conn, tool_call.args.get("ticker", ticker))
            tool_result_content = _format_historical(rows)

        # Tool 2: collect component scores, compute weighted average
        elif tool_name == "score_risk_components":
            scores = handle_score_risk_components(tool_call.args)
            overall_score, risk_tier = compute_overall_score(scores)
            tool_result_content = f"Scores recorded. Overall score: {overall_score} ({risk_tier}). Now write the executive summary."

        # Tool 3: collect executive summary
        else:
            executive_summary = handle_finalize_overall_risk(tool_call.args)
            tool_result_content = "Done."

        # Append assistant turn and tool results, then continue to next step. Every call id must
        # be answered, including duplicates.
        pairs = [(tool_call.id, tool_result_content)]
        pairs += [(tc.id, "Ignored duplicate call.") for tc in resp.tool_calls if tc.id != tool_call.id]
        messages.append(assistant_msg_from_response(resp))
        messages.extend(tool_result_msgs(pairs))

    # Never persist a scoreless row.
    if overall_score is None or risk_tier is None:
        raise RuntimeError("score_risk_components step did not produce a score")

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
