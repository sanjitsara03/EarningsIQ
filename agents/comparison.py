# The Comparison Agent answers questions that require looking across multiple filings —
# trends, QoQ/YoY comparisons, "how has X changed over time."
# It combines vector search (semantic) with structured signals (numeric) and exits only
# when the terminal tool cite_and_answer is called.

import logging
import os
import re

import anthropic
from dotenv import load_dotenv

from db.connection import get_connection
from tools.rag_tools import (
    TOOLS,
    handle_cite_and_answer,
    handle_fetch_structured_signals,
    handle_resolve_filing_ids,
    handle_vector_search_chunks,
)

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5"
MAX_TURNS = 10

SYSTEM_PROMPT = """You are a financial analyst answering questions that require comparing data across multiple SEC filing periods.

You have four tools available:
1. resolve_filing_ids — look up filing IDs for a ticker (call this first)
2. vector_search_chunks — semantic search over filing text for relevant passages
3. fetch_structured_signals — get numeric signals (revenue, margins, etc.) across periods
4. cite_and_answer — return your final answer with citations (call this when done)

Always resolve filing IDs first. Use both vector search and structured signals to build evidence.
When you have enough to answer the question thoroughly, call cite_and_answer."""

# Patterns that indicate a comparison question. Checked before any LLM call.
COMPARISON_TRIGGERS = [
    r"\b(over time|historically|trend|quarter[s]?\s+(ago|over|vs|versus))\b",
    r"\b(last\s+(year|quarter|q[1-4]))\b",
    r"\b(compare|comparison|q[1-4]\s+vs\.?\s+q[1-4])\b",
    r"\b(year[- ]over[- ]year|yoy|qoq)\b",
]


# Returns True if the query contains comparison language.
def is_comparison_query(query: str) -> bool:
    query_lower = query.lower()
    return any(re.search(p, query_lower) for p in COMPARISON_TRIGGERS)


# Formats tool handler results as a readable string to pass back to the LLM as a tool result.
def _format_tool_result(name: str, result) -> str:
    if isinstance(result, list) and len(result) == 0:
        return "No data found."
    if name == "vector_search_chunks":
        lines = []
        for chunk in result:
            lines.append(f"[filing_id={chunk['filing_id']} section={chunk['section']}]\n{chunk['content']}")
        return "\n\n".join(lines)
    if name == "fetch_structured_signals":
        lines = []
        for s in result:
            lines.append(
                f"{s['period']} ({s['filing_type']}): revenue={s['revenue']}, "
                f"gross_margin={s['gross_margin']}%, operating_margin={s['operating_margin']}%, "
                f"revenue_yoy_delta={s['revenue_yoy_delta']}%, guidance_withdrawn={s['guidance_withdrawn']}"
            )
        return "\n".join(lines)
    return str(result)


# Open-ended tool loop that exits when the LLM calls cite_and_answer. Returns the answer and citations.
def run_comparison(query: str) -> dict:
    if not is_comparison_query(query):
        raise ValueError(f"Query does not appear to be a comparison question: {query!r}")

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    messages = [{"role": "user", "content": query}]
    turns = 0

    while turns < MAX_TURNS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        turns += 1

        tool_uses = [b for b in response.content if b.type == "tool_use"]

        # No tool calls — LLM finished without calling cite_and_answer, which is unexpected
        if not tool_uses:
            logger.warning("Comparison agent stopped without calling cite_and_answer.")
            text = next((b.text for b in response.content if hasattr(b, "text")), "")
            return {"answer": text, "citations": []}

        tool_results = []
        final_result = None

        with get_connection() as conn:
            for tool_use in tool_uses:
                logger.info(f"Tool called: {tool_use.name} (turn {turns})")

                if tool_use.name == "resolve_filing_ids":
                    result = handle_resolve_filing_ids(tool_use.input, conn)
                elif tool_use.name == "vector_search_chunks":
                    result = handle_vector_search_chunks(tool_use.input, conn)
                elif tool_use.name == "fetch_structured_signals":
                    result = handle_fetch_structured_signals(tool_use.input, conn)
                elif tool_use.name == "cite_and_answer":
                    final_result = handle_cite_and_answer(tool_use.input)
                    result = "Done."

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": _format_tool_result(tool_use.name, result) if tool_use.name != "cite_and_answer" else "Done.",
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        if final_result is not None:
            logger.info(f"Comparison complete after {turns} turns.")
            return final_result

    raise TimeoutError(f"Comparison agent did not call cite_and_answer within {MAX_TURNS} turns.")


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "How has MSFT revenue trended over the last quarter?"
    result = run_comparison(query)
    print(json.dumps(result, indent=2, default=str))
