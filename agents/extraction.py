import logging

from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import get_chunks_for_filing, insert_signals, update_filing_status
from llm import (
    LoopStallError,
    WrongToolError,
    assistant_msg_from_response,
    chat,
    to_openai_tools,
    tool_result_msgs,
    traced,
    user_msg,
)
from tools.extraction_tools import TOOL_HANDLERS, TOOLS

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
- Revenue and financial figures should be in millions USD unless clearly stated otherwise.
- Call all five tools even if some sections have limited data."""


class ExtractionTimeoutError(Exception):
    pass


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
                results[tc.name] = TOOL_HANDLERS[tc.name](tc.args)
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
