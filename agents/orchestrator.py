# The orchestrator is the first thing that runs when a user asks a question.
# Its only job is to read the question and figure out
# what kind of work needs to happen: which company, which filing type, how many periods,
# and whether we need a web search on top of the filing data.
#
# It makes a single LLM call using the cheapest/fastest model and returns a small
# JSON blob that the rest of the system uses to decide what to do next.

import json
import logging
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a query router for a financial analysis system that reads SEC filings.

Given a user question, return a single JSON object with no extra text, markdown, or explanation.

JSON shape:
{
  "intent": one of "single_analysis" | "comparison" | "advice" | "web_only",
  "ticker": uppercase stock ticker (e.g. "AAPL"),
  "filing_types": array containing "10-Q" and/or "10-K",
  "periods_needed": integer, how many filing periods to retrieve (default 1),
  "web_search_needed": boolean
}

Routing rules:

intent:
- "single_analysis" — standard question about one company's recent performance or risks
- "comparison" — question explicitly comparing metrics across time periods or quarters
- "advice" — buy/hold/sell or investment decision questions
- "web_only" — question requiring current news or data not found in filings

filing_types:
- ["10-Q"] — questions about recent quarterly performance, last quarter, margins, revenue
- ["10-K"] — questions about long-term risks, annual performance, business overview
- ["10-Q", "10-K"] — investment/advice questions needing both recent and annual context

periods_needed:
- 1 for most questions
- Match the number explicitly requested ("last 4 quarters" → 4, "past year" → 4)

web_search_needed:
- true if the question requires current stock price, recent news, or analyst sentiment
- false for all filing-based questions

Return only the JSON object. No other text."""


# Single LLM call — classifies the user query and returns a structured intent dict.
def run_orchestrator(query: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": query},
            {"role": "assistant", "content": "{"},
        ],
    )

    raw = "{" + response.content[0].text.strip()
    intent = json.loads(raw)
    logger.info(f"Orchestrator: {query!r} → {intent}")
    return intent


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "How did Apple do last quarter?"
    result = run_orchestrator(query)
    print(json.dumps(result, indent=2))
