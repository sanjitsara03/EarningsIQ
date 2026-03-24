import logging
import os

import anthropic
from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import get_chunks_for_filing, insert_signals, update_filing_status
from tools.extraction_tools import TOOL_HANDLERS, TOOLS

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TURNS = 10
MODEL = "claude-sonnet-4-5"

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
def run_extraction(filing_id: int) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    with get_connection() as conn:
        chunks = get_chunks_for_filing(conn, filing_id)

    if not chunks:
        raise ValueError(f"No chunks found for filing_id={filing_id}")

    messages = [{"role": "user", "content": _build_prompt(chunks)}]
    results = {}
    turns = 0

    while len(results) < 5 and turns < MAX_TURNS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            tool_choice={"type": "any"},
            messages=messages,
        )
        turns += 1

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        for tool_use in tool_uses:
            if tool_use.name not in results:
                results[tool_use.name] = TOOL_HANDLERS[tool_use.name](tool_use.input)
                logger.info(f"Tool called: {tool_use.name} (turn {turns})")

        if len(results) == 5:
            break

        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": t.id, "content": "OK"}
                for t in tool_uses
            ],
        })

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
