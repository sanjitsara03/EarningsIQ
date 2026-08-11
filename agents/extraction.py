import logging

from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import get_chunks_for_filing, get_filing_ticker, get_historical_signals, insert_signals, update_filing_status
from llm import (
    LLMOutputError,
    LoopStallError,
    WrongToolError,
    assistant_msg_from_response,
    chat,
    named_tool_choice,
    to_openai_tools,
    tool_result_msgs,
    traced,
    user_msg,
)
from tools.extraction_tools import TOOL_HANDLERS, TOOLS

# Extreme tripwire for unit-CLASS errors only (a millions figure stored as dollars shifts values
# by x1e6). Company-specific revenue scale is checked adaptively in _revenue_implausibility —
# these bounds intentionally say nothing about what "normal" revenue looks like.
REVENUE_MIN_USD = 1e4   # $10K
REVENUE_MAX_USD = 5e12  # $5T

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TURNS = 10
OPENAI_TOOLS = to_openai_tools(TOOLS)

SYSTEM_PROMPT = """You are a financial analyst extracting structured data from SEC 10-Q and 10-K filings.

You will be given filing text organized by section. You must call ALL FIVE extraction tools exactly once each:
- extract_financial_metrics
- extract_risk_factors
- extract_segment_performance
- extract_management_outlook
- extract_notable_changes

Rules:
- Use only information present in the filing. Do not infer or guess missing values — omit them.
- Always include verbatim quotes exactly as they appear in the text.
- Report revenue exactly as printed in the financial tables, plus the table's stated unit
  (from captions like "In millions, except per share amounts") and that caption verbatim.
  Never convert the revenue figure yourself.
- Call all five tools even if some sections have limited data."""


class ExtractionTimeoutError(Exception):
    pass


# Returns a description of why the normalized revenue looks wrong, or None if it's plausible.
# The checks are company-relative, so no assumption about a "normal" revenue range is made:
#   1. Segment cross-check — segments carry their own independently-cited unit and must roughly
#      sum to total revenue. Works on a first-ever extraction; a unit error on either side
#      shifts the ratio by ~x1000.
#   2. Prior-period ratio — this ticker's own history (current filing's row excluded).
#   3. Extreme absolute tripwire — catches unit-class errors when neither adaptive check has data.
def _revenue_implausibility(filing_id: int, results: dict) -> str | None:
    revenue_usd = results["extract_financial_metrics"].get("revenue_usd")
    if revenue_usd is None:
        return None

    if not (REVENUE_MIN_USD <= revenue_usd <= REVENUE_MAX_USD):
        return f"revenue_usd={revenue_usd:,.0f} is outside the unit-sanity tripwire (${REVENUE_MIN_USD:,.0f}–${REVENUE_MAX_USD:,.0f})"

    seg_sum = sum(s.get("revenue_usd") or 0 for s in results.get("extract_segment_performance") or [])
    if seg_sum and not (1 / 3 <= revenue_usd / seg_sum <= 3):
        return (
            f"revenue_usd={revenue_usd:,.0f} disagrees with the segment revenue sum "
            f"{seg_sum:,.0f} (ratio {revenue_usd / seg_sum:.2f}) — one side likely has a unit error"
        )

    with get_connection() as conn:
        ticker = get_filing_ticker(conn, filing_id)
        rows = get_historical_signals(conn, ticker, exclude_filing_id=filing_id) if ticker else []
    prior = next((r["revenue_usd"] for r in rows if r.get("revenue_usd")), None)
    if prior and not (0.1 <= revenue_usd / float(prior) <= 10):
        return f"revenue_usd={revenue_usd:,.0f} is more than 10x away from the prior period's {float(prior):,.0f}"
    return None


# Runs a tool handler, converting schema-violating arguments (missing keys, unknown units) into
# a typed error instead of an untyped crash.
def _run_handler(tool_name: str, args: dict) -> dict:
    try:
        return TOOL_HANDLERS[tool_name](args)
    except (KeyError, ValueError, TypeError) as e:
        raise LLMOutputError("extraction", f"tool {tool_name} returned invalid arguments: {type(e).__name__}: {e}")


# Groups chunks by section and builds a single prompt string with section headers.
def _build_prompt(chunks: list[dict]) -> str:
    sections: dict[str, list[str]] = {}
    for chunk in chunks:
        sections.setdefault(chunk["section"], []).append(chunk["content"])

    parts = ["Here is the SEC filing content organized by section:\n"]
    for section, contents in sections.items():
        parts.append(f"\n=== {section.upper()} ===\n")
        parts.extend(contents)

    parts.append("\n\nExtract all required information by calling all five tools.")
    return "\n".join(parts)


# Runs the tool-collection loop for a single filing. Exits when all 5 tools have fired or MAX_TURNS is reached.
# Returns the collected results dict and persists signals to the DB.
@traced("extraction")
def run_extraction(filing_id: int) -> dict:
    with get_connection() as conn:
        chunks = get_chunks_for_filing(conn, filing_id)

    if not chunks:
        raise ValueError(f"No chunks found for filing_id={filing_id}")

    messages = [user_msg(_build_prompt(chunks))]
    results = {}
    turns = 0
    stalled_turns = 0

    while len(results) < 5 and turns < MAX_TURNS:
        resp = chat("extraction", messages, system=SYSTEM_PROMPT, tools=OPENAI_TOOLS, tool_choice="required")
        turns += 1

        before = len(results)
        for tc in resp.tool_calls:
            # Unknown tool names are a hard failure.
            if tc.name not in TOOL_HANDLERS:
                raise WrongToolError("extraction", "one of the five extraction tools", tc.name)
            if tc.name not in results:
                results[tc.name] = _run_handler(tc.name, tc.args)
                logger.info(f"Tool called: {tc.name} (turn {turns})")

        if len(results) == 5:
            break

        # Every call id must be acknowledged, including duplicate calls the dedup above skipped.
        messages.append(assistant_msg_from_response(resp))
        messages.extend(tool_result_msgs([(tc.id, "OK") for tc in resp.tool_calls]))

        # Stall guard: one nudge after a no-progress turn; two consecutive no-progress turns abort.
        missing = set(TOOL_HANDLERS.keys()) - set(results.keys())
        if len(results) == before:
            stalled_turns += 1
            if stalled_turns >= 2:
                raise LoopStallError("extraction", f"no new tools after {turns} turns; missing {missing}")
            messages.append(user_msg(f"You still need to call these tools: {sorted(missing)}"))
        else:
            stalled_turns = 0

    if len(results) < 5:
        missing = set(TOOL_HANDLERS.keys()) - set(results.keys())
        raise ExtractionTimeoutError(
            f"Extraction incomplete after {MAX_TURNS} turns. Missing: {missing}"
        )

    # Plausibility guard: one corrective re-extraction of the metrics on an implausible revenue,
    # then a hard error. An off-by-a-million figure must never be persisted.
    problem = _revenue_implausibility(filing_id, results)
    if problem:
        logger.warning(f"Implausible revenue for filing_id={filing_id} — retrying metrics: {problem}")
        retry_messages = [
            messages[0],
            user_msg(
                f"Your earlier extract_financial_metrics call produced an implausible figure: {problem}. "
                "Re-read the financial statements and their unit caption, then call extract_financial_metrics again."
            ),
        ]
        resp = chat(
            "extraction", retry_messages, system=SYSTEM_PROMPT,
            tools=OPENAI_TOOLS, tool_choice=named_tool_choice("extract_financial_metrics"),
        )
        tc = resp.tool_calls[0]
        if tc.name != "extract_financial_metrics":
            raise WrongToolError("extraction", "extract_financial_metrics", tc.name)
        # Merge over the first pass so optional fields it captured (eps, margins) survive a
        # retry that only re-reports the revenue figure.
        retry_metrics = _run_handler(tc.name, tc.args)
        results["extract_financial_metrics"] = {
            **results["extract_financial_metrics"],
            **{k: v for k, v in retry_metrics.items() if v is not None},
        }
        problem = _revenue_implausibility(filing_id, results)
        if problem:
            raise LLMOutputError("extraction", f"revenue failed plausibility after retry: {problem}")

    with get_connection() as conn:
        insert_signals(conn, filing_id, results)
        update_filing_status(conn, filing_id, "extracted")

    logger.info(f"Extraction complete for filing_id={filing_id}")
    return results


if __name__ == "__main__":
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("filing_id", type=int, help="filing_id to extract signals from")
    args = parser.parse_args()

    results = run_extraction(args.filing_id)
    print(json.dumps(results, indent=2))
