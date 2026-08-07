# The Advice Agent is the final step in the pipeline. It takes everything gathered 
# extracted signals, risk score, and optional web search results and synthesizes
# a buy/hold/sell recommendation.

import json
import logging
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

from llm import LLMOutputError, chat_json, traced, user_msg

load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior financial analyst synthesizing research into an investment recommendation.

You will be given:
- Extracted financial signals from SEC filings (revenue, margins, guidance, risks, segments)
- A risk score with component breakdown and executive summary
- Optionally, recent web search results with current news and analyst sentiment

Your job is to weigh all of this evidence and return a JSON object with exactly this shape:
{
  "recommendation": "buy" | "hold" | "sell",
  "confidence": "high" | "medium" | "low",
  "reasoning": "3-5 sentences citing specific evidence from the data provided",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "key_positives": ["positive 1", "positive 2", "positive 3"],
  "disclaimer": "This is not financial advice."
}

Rules:
- reasoning must cite specific numbers or quotes from the provided data
- key_risks and key_positives must each have exactly 3 items
- disclaimer must always be exactly "This is not financial advice."
- Return only the JSON object, no other text."""


# extra="forbid" emits additionalProperties: false in the schema.
class AdviceOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation: Literal["buy", "hold", "sell"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str
    key_risks: list[str]
    key_positives: list[str]
    disclaimer: str


ADVICE_SCHEMA = AdviceOutput.model_json_schema()


# Builds the user prompt from all available inputs.
def _build_prompt(ticker: str, extracted_signals: dict, risk_result: dict, web_summary: str | None) -> str:
    parts = [f"Company: {ticker}\n"]

    parts.append("=== EXTRACTED SIGNALS ===")
    parts.append(json.dumps(extracted_signals, indent=2, default=str))

    parts.append("\n=== RISK SCORE ===")
    parts.append(json.dumps(risk_result, indent=2, default=str))

    if web_summary:
        parts.append("\n=== WEB SEARCH RESULTS ===")
        parts.append(web_summary)

    parts.append("\nReturn your recommendation as a JSON object.")
    return "\n".join(parts)


# Single LLM call — synthesizes all inputs into a validated AdviceOutput.
# One corrective retry on validation failure, then LLMOutputError.
@traced("advice")
def run_advice(
    ticker: str,
    extracted_signals: dict,
    risk_result: dict,
    web_summary: str | None = None,
) -> AdviceOutput:
    prompt = _build_prompt(ticker, extracted_signals, risk_result, web_summary)
    messages = [user_msg(prompt)]
    data: dict = {}

    for attempt in range(2):
        data = chat_json(
            "advice", messages, system=SYSTEM_PROMPT,
            schema_name="advice_output", schema=ADVICE_SCHEMA,
        )
        try:
            output = AdviceOutput.model_validate(data)
        except ValidationError as e:
            logger.warning(f"Advice output failed validation (attempt {attempt + 1}): {e}")
            messages = [
                user_msg(prompt),
                {"role": "assistant", "content": json.dumps(data)},
                user_msg(f"That response failed validation:\n{e}\nReturn a corrected JSON object."),
            ]
            continue
        logger.info(f"Advice for {ticker}: {output.recommendation} ({output.confidence} confidence)")
        return output

    raise LLMOutputError("advice", "advice failed validation after retry", json.dumps(data))


if __name__ == "__main__":
    import sys
    from db.connection import get_connection

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"

    # Load signals and risk score from DB for standalone testing
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT s.*, f.ticker, f.period FROM signals s
                JOIN filings f ON s.filing_id = f.id
                WHERE f.ticker = %s ORDER BY f.period DESC LIMIT 1
                """,
                (ticker,),
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            extracted_signals = dict(zip(cols, row)) if row else {}

            cur.execute(
                """
                SELECT r.* FROM risk_scores r
                JOIN filings f ON r.filing_id = f.id
                WHERE f.ticker = %s ORDER BY f.period DESC LIMIT 1
                """,
                (ticker,),
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
            risk_result = dict(zip(cols, row)) if row else {}

    result = run_advice(ticker, extracted_signals, risk_result)
    print(json.dumps(result.model_dump(), indent=2))
