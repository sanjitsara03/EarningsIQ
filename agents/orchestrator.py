# Orchestrator — one cheap LLM call that turns the user query into the intent JSON routing all downstream work.

import json
import logging
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

from llm import LLMOutputError, chat_json, traced, user_msg

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query router for a financial analysis system that reads SEC filings.

Given a user question, return a single JSON object with no extra text, markdown, or explanation.

JSON shape:
{
  "intent": one of "single_analysis" | "comparison" | "advice" | "web_only",
  "ticker": uppercase stock ticker (e.g. "AAPL"), or "" (empty string) if no specific publicly traded company is mentioned,
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


# extra="forbid" emits additionalProperties: false; ticker == "" means no specific company mentioned.
class IntentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["single_analysis", "comparison", "advice", "web_only"]
    ticker: str
    filing_types: list[Literal["10-Q", "10-K"]]
    periods_needed: int
    web_search_needed: bool


ORCHESTRATOR_SCHEMA = IntentOutput.model_json_schema()


# Classifies the query into a validated intent dict; one corrective retry, then LLMOutputError.
@traced("orchestrator")
def run_orchestrator(query: str) -> dict:
    messages = [user_msg(query)]
    data: dict = {}

    for attempt in range(2):
        data = chat_json(
            "orchestrator", messages, system=SYSTEM_PROMPT,
            schema_name="orchestrator_intent", schema=ORCHESTRATOR_SCHEMA,
        )
        try:
            intent = IntentOutput.model_validate(data)
        except ValidationError as e:
            logger.warning(f"Orchestrator intent failed validation (attempt {attempt + 1}): {e}")
            messages = [
                user_msg(query),
                {"role": "assistant", "content": json.dumps(data)},
                user_msg(f"That response failed validation:\n{e}\nReturn a corrected JSON object."),
            ]
            continue
        logger.info(f"Orchestrator: {query!r} → {intent.model_dump()}")
        return intent.model_dump()

    raise LLMOutputError("orchestrator", "intent failed validation after retry", json.dumps(data))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "How did Apple do last quarter?"
    result = run_orchestrator(query)
    print(json.dumps(result, indent=2))
