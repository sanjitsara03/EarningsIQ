# Advice Agent — final pipeline step: synthesizes signals, risk score, and web results into buy/hold/sell.

import json
import logging
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

from llm import LLMOutputError, chat_json, traced, user_msg

load_dotenv()

logger = logging.getLogger(__name__)

# Shared data-interpretation rules for both the advice and analysis prompts.
DATA_RULES = """Data rules:
- All monetary signal fields ending in _usd are raw US dollars (e.g. 109400000000 = $109.4 billion).
- 10-Q and 10-K filings rarely contain forward guidance — companies give guidance on earnings
  calls, not in filings. A missing or null guidance_revenue_usd means "not stated in this
  filing", NEVER that guidance was withdrawn, pulled, or suspended. Only guidance_withdrawn:
  true (a quote-verified extracted flag) justifies the word "withdrawn".
- When the source filing is a 10-K, every figure is a FULL FISCAL YEAR total. Frame each number
  as annual (e.g. "FY2026 revenue of $X"), never as quarterly. If the user asked about the last
  quarter, state explicitly that the fiscal Q4 is reported inside the annual 10-K and these are
  full-year figures."""

SYSTEM_PROMPT = f"""You are a senior financial analyst synthesizing research into an investment recommendation.

You will be given:
- Extracted financial signals from SEC filings (revenue, margins, guidance, risks, segments)
- A risk score with component breakdown and executive summary
- Optionally, recent web search results with current news and analyst sentiment

{DATA_RULES}

Your job is to weigh all of this evidence and return a JSON object with exactly this shape:
{{
  "recommendation": "buy" | "hold" | "sell",
  "confidence": "high" | "medium" | "low",
  "reasoning": "3-5 sentences citing specific evidence from the data provided",
  "key_risks": ["risk 1", "risk 2", "risk 3"],
  "key_positives": ["positive 1", "positive 2", "positive 3"],
  "disclaimer": "This is not financial advice."
}}

Rules:
- reasoning must cite specific numbers or quotes from the provided data
- key_risks and key_positives must each have exactly 3 items
- disclaimer must always be exactly "This is not financial advice."
- Return only the JSON object, no other text."""

ANALYSIS_SYSTEM_PROMPT = f"""You are a senior financial analyst answering a factual question about a company's SEC filing.

You will be given the user's question, extracted financial signals from the filing, a risk
score, and optionally recent web search results.

{DATA_RULES}

Return a JSON object with exactly this shape:
{{
  "answer": "markdown answering the user's question directly, citing specific figures",
  "highlights": ["notable point 1", "notable point 2"]
}}

Rules:
- Answer the question that was actually asked. Every metric the user explicitly asked for must
  appear in answer with its value — or an explicit statement that the filing does not contain it.
- Do NOT give a buy/hold/sell opinion or investment stance — this is a factual question.
- highlights: 2-4 short supporting points worth knowing beyond the direct answer.
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


# Factual answer to a single_analysis question — no investment stance.
class AnalysisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    highlights: list[str]


ADVICE_SCHEMA = AdviceOutput.model_json_schema()
ANALYSIS_SCHEMA = AnalysisOutput.model_json_schema()


# Builds the user prompt; filing_type/period ground the DATA_RULES framing, query only on the analysis route.
def _build_prompt(
    ticker: str,
    extracted_signals: dict,
    risk_result: dict,
    web_summary: str | None,
    filing_type: str | None = None,
    period: str | None = None,
    query: str | None = None,
) -> str:
    parts = [f"Company: {ticker}"]
    if filing_type:
        source = f"Source filing: {filing_type}"
        if period:
            source += f", period ended {period}"
        parts.append(source)
    if query:
        parts.append(f'\nUser question: "{query}"')
    parts.append("")

    parts.append("=== EXTRACTED SIGNALS ===")
    parts.append(json.dumps(extracted_signals, indent=2, default=str))

    parts.append("\n=== RISK SCORE ===")
    parts.append(json.dumps(risk_result, indent=2, default=str))

    if web_summary:
        parts.append("\n=== WEB SEARCH RESULTS ===")
        parts.append(web_summary)

    parts.append("\nReturn your answer as a JSON object.")
    return "\n".join(parts)


# Shared call-validate-retry loop: one corrective retry on validation failure, then LLMOutputError.
def _chat_validated(prompt: str, system: str, schema_name: str, schema: dict, model_cls):
    messages = [user_msg(prompt)]
    data: dict = {}
    for attempt in range(2):
        data = chat_json("advice", messages, system=system, schema_name=schema_name, schema=schema)
        try:
            return model_cls.model_validate(data)
        except ValidationError as e:
            logger.warning(f"{schema_name} failed validation (attempt {attempt + 1}): {e}")
            messages = [
                user_msg(prompt),
                {"role": "assistant", "content": json.dumps(data)},
                user_msg(f"That response failed validation:\n{e}\nReturn a corrected JSON object."),
            ]
    raise LLMOutputError("advice", f"{schema_name} failed validation after retry", json.dumps(data))


# Single LLM call — synthesizes all inputs into a validated AdviceOutput.
@traced("advice")
def run_advice(
    ticker: str,
    extracted_signals: dict,
    risk_result: dict,
    web_summary: str | None = None,
    filing_type: str | None = None,
    period: str | None = None,
) -> AdviceOutput:
    prompt = _build_prompt(ticker, extracted_signals, risk_result, web_summary, filing_type, period)
    output = _chat_validated(prompt, SYSTEM_PROMPT, "advice_output", ADVICE_SCHEMA, AdviceOutput)
    logger.info(f"Advice for {ticker}: {output.recommendation} ({output.confidence} confidence)")
    return output


# Single LLM call — answers a factual single_analysis question with a validated AnalysisOutput.
@traced("analysis")
def run_analysis(
    query: str,
    ticker: str,
    extracted_signals: dict,
    risk_result: dict,
    web_summary: str | None = None,
    filing_type: str | None = None,
    period: str | None = None,
) -> AnalysisOutput:
    prompt = _build_prompt(ticker, extracted_signals, risk_result, web_summary, filing_type, period, query)
    output = _chat_validated(prompt, ANALYSIS_SYSTEM_PROMPT, "analysis_output", ANALYSIS_SCHEMA, AnalysisOutput)
    logger.info(f"Analysis for {ticker}: {len(output.answer)} chars, {len(output.highlights)} highlights")
    return output


if __name__ == "__main__":
    import sys

    from db.connection import get_connection

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
                SELECT s.*, f.ticker, f.period, f.filing_type FROM signals s
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

    result = run_advice(
        ticker, extracted_signals, risk_result,
        filing_type=extracted_signals.get("filing_type"),
        period=str(extracted_signals.get("period") or "") or None,
    )
    print(json.dumps(result.model_dump(), indent=2))
