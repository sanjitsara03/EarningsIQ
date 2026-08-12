# Comparison Agent — answers multi-period trend questions with vector search + structured signals.

import logging
import re

from dotenv import load_dotenv

from db.connection import get_connection
from llm import (
    assistant_msg_from_response,
    chat,
    to_openai_tools,
    tool_result_msgs,
    traced,
    user_msg,
)
from tools.rag_tools import (
    TOOLS,
    handle_cite_and_answer,
    handle_fetch_structured_signals,
    handle_resolve_filing_ids,
    handle_vector_search_chunks,
)

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TURNS = 10
OPENAI_TOOLS = to_openai_tools(TOOLS)

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
                f"{s['period']} ({s['filing_type']}): revenue_usd={s['revenue_usd']}, "
                f"gross_margin={s['gross_margin']}%, operating_margin={s['operating_margin']}%, "
                f"revenue_yoy_delta={s['revenue_yoy_delta']}%, guidance_withdrawn={s['guidance_withdrawn']}"
            )
        return "\n".join(lines)
    return str(result)


# Tool loop exiting on cite_and_answer; tool_trace lets the benchmark judge count DB-derived numbers as grounded.
@traced("comparison")
def run_comparison(query: str) -> dict:
    if not is_comparison_query(query):
        raise ValueError(f"Query does not appear to be a comparison question: {query!r}")

    messages = [user_msg(query)]
    turns = 0
    nudged = False
    tool_trace: list[dict] = []

    while turns < MAX_TURNS:
        resp = chat("comparison", messages, system=SYSTEM_PROMPT, tools=OPENAI_TOOLS)
        turns += 1

        # No tool calls: nudge once toward cite_and_answer, then degrade to an uncited text answer.
        if not resp.tool_calls:
            if not nudged:
                nudged = True
                logger.warning("Comparison agent answered in text — nudging toward cite_and_answer.")
                messages.append(assistant_msg_from_response(resp))
                messages.append(user_msg("Deliver your final answer by calling the cite_and_answer tool."))
                continue
            logger.warning("Comparison agent stopped without calling cite_and_answer.")
            return {"answer": resp.text or "", "citations": [], "tool_trace": tool_trace}

        pairs = []
        final_result = None

        with get_connection() as conn:
            for tc in resp.tool_calls:
                logger.info(f"Tool called: {tc.name} (turn {turns})")

                if tc.name == "resolve_filing_ids":
                    content = _format_tool_result(tc.name, handle_resolve_filing_ids(tc.args, conn))
                elif tc.name == "vector_search_chunks":
                    content = _format_tool_result(tc.name, handle_vector_search_chunks(tc.args, conn))
                elif tc.name == "fetch_structured_signals":
                    content = _format_tool_result(tc.name, handle_fetch_structured_signals(tc.args, conn))
                elif tc.name == "cite_and_answer":
                    final_result = handle_cite_and_answer(tc.args)
                    content = "Done."
                else:
                    logger.warning(f"Unknown tool called: {tc.name}")
                    content = (
                        f"Error: unknown tool '{tc.name}'. Available tools: resolve_filing_ids, "
                        "vector_search_chunks, fetch_structured_signals, cite_and_answer."
                    )

                if tc.name != "cite_and_answer":
                    tool_trace.append({"tool": tc.name, "args": tc.args, "result": content[:4000]})
                pairs.append((tc.id, content))

        messages.append(assistant_msg_from_response(resp))
        messages.extend(tool_result_msgs(pairs))

        if final_result is not None:
            logger.info(f"Comparison complete after {turns} turns.")
            return {**final_result, "tool_trace": tool_trace}

    raise TimeoutError(f"Comparison agent did not call cite_and_answer within {MAX_TURNS} turns.")


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "How has MSFT revenue trended over the last quarter?"
    result = run_comparison(query)
    print(json.dumps(result, indent=2, default=str))
