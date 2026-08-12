# Judging layer for the analyst benchmark: a cross-family LLM judge (gpt-5.2 grades the
# sonnet-written outputs), a cheap numeric-claims extractor, and deterministic cross-checks.
import json
import logging
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from llm import LLMOutputError, chat_json, user_msg

logger = logging.getLogger(__name__)

# Comparison tolerances for the deterministic numeric checks.
REVENUE_REL_TOL = 0.02      # ±2%
EPS_ABS_TOL = 0.03          # ±$0.03
EPS_REL_TOL = 0.02          # or ±2%
YOY_PP_TOL = 1.5            # ±1.5 percentage points


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int
    evidence: list[str]

    # Range enforced via validator, not Field(ge/le) — those emit minimum/maximum keywords
    # that strict json_schema mode rejects.
    @field_validator("score")
    @classmethod
    def _score_in_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("score must be 1-5")
        return v


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    factual_accuracy: DimensionScore
    coverage: DimensionScore
    grounding: DimensionScore
    directional_agreement: DimensionScore
    reference_caveats: list[str]
    overall_comment: str


class NumericClaims(BaseModel):
    model_config = ConfigDict(extra="forbid")

    period_label: str | None
    period_end_year: int | None
    period_end_month: int | None
    period_scope: Literal["quarter", "year"] | None
    revenue_usd: float | None
    eps_diluted: float | None
    revenue_yoy_pct: float | None
    analyst_consensus: Literal["buy", "hold", "sell"] | None
    consensus_quote: str | None


JUDGE_SCHEMA = JudgeVerdict.model_json_schema()
CLAIMS_SCHEMA = NumericClaims.model_json_schema()

JUDGE_SYSTEM_PROMPT = """You are grading an AI financial analyst's answer against professional research sources.

Judge ONLY against the numbered sources provided — never against your own knowledge; your
training data is outdated. Score each dimension 1-5 and give concrete evidence strings that
cite source tags [S1]..[Sn]:

- factual_accuracy: 5 = no number or claim contradicted by any source; 3 = minor discrepancies
  (rounding, imprecise period); 1 = material contradiction (wrong revenue magnitude, wrong
  direction of change).
- coverage: 5 = addresses every major theme the sources emphasize; 3 = covers headline results
  but misses one recurring professional theme; 1 = misses most of what professionals lead with.
  List the missed themes in evidence.
- grounding: 5 = every claim traceable to the DATA PROVIDED TO THE AGENT section (its actual
  filing data and search results, when provided) or to the sources; 3 = one or two unsupported
  embellishments; 1 = substantial claims supported by nothing provided. Claims traceable to the
  agent's provided data are GROUNDED even when no numbered source mentions them — that is the
  product working as designed. List genuinely ungrounded claims verbatim in evidence.
- directional_agreement: 5 = same bullish/bearish stance as professional consensus, OR a
  divergent stance argued from specific evidence; 3 = adjacent stance weakly argued; 1 =
  opposite stance with no engagement with the consensus view.

Period rule: the agent analyzed the filing period stated in the prompt. If sources discuss a
NEWER quarter, do NOT count newer-quarter facts as agent errors — record the gap in
reference_caveats. If sources conflict with each other, note it in reference_caveats and do not
penalize the agent for siding with either.

Guidance rule: companies issue guidance on earnings calls and press releases, not in 10-Q/10-K
filings. An agent statement that guidance is absent FROM THE FILING is accurate scoping, not an
error, even when sources report call-issued guidance — record that gap in reference_caveats."""

EXTRACTOR_SYSTEM_PROMPT = """You extract numeric financial claims from text. Report ONLY values
explicitly stated in the text — never estimate or fill gaps from knowledge. Use null for anything
not stated. revenue_usd must be in raw US dollars (convert stated millions/billions).
period_end_year and period_end_month: the calendar year and month (1-12) the reported period
ENDS in, when determinable from the text (e.g. "quarter ended June 27, 2026" -> year 2026,
month 6); null when not determinable.
period_scope: "quarter" when the reported figures cover a single quarter, "year" when they cover
a full fiscal year (10-K annual totals); null when unclear.
analyst_consensus: only when the text states an analyst rating consensus (map Strong Buy/
Moderate Buy/Overweight/Outperform -> buy; Hold/Neutral/Market Perform -> hold; Sell/
Underweight/Underperform -> sell); include the verbatim phrase as consensus_quote."""


# Formats sources as numbered [S1]..[Sn] blocks for the judge prompt.
def format_sources(sources: list[dict]) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        blocks.append(
            f"[S{i}] {s.get('title', 'untitled')} — {s.get('url', '')} (retrieved {s.get('retrieved_at', '?')})\n"
            f"{(s.get('content') or '')[:2000]}"
        )
    return "\n\n".join(blocks)


# Runs the LLM judge for one query. One corrective Pydantic retry, then LLMOutputError.
def run_judge(bq, run_record: dict, sources: list[dict]) -> dict:
    prompt = (
        f"User query: {bq.query}\n"
        f"Route taken: {run_record.get('route')}\n"
        f"Filing period analyzed: {run_record.get('filing_period') or 'unknown'} "
        f"(filed {run_record.get('filed_at') or 'unknown'})\n"
        f"Judge focus: {bq.judge_focus}\n\n"
        f"=== AGENT OUTPUT ===\n{json.dumps(run_record.get('agent_output'), indent=1, default=str)[:8000]}\n\n"
        + (
            f"=== DATA PROVIDED TO THE AGENT ===\n"
            f"{json.dumps(run_record.get('agent_inputs'), indent=1, default=str)[:6000]}\n\n"
            if run_record.get("agent_inputs") else ""
        )
        + f"=== PROFESSIONAL SOURCES ===\n{format_sources(sources)}"
    )
    messages = [user_msg(prompt)]
    data: dict = {}
    for attempt in range(2):
        data = chat_json(
            "benchmark_judge", messages, system=JUDGE_SYSTEM_PROMPT,
            schema_name="judge_verdict", schema=JUDGE_SCHEMA,
        )
        try:
            return JudgeVerdict.model_validate(data).model_dump()
        except ValidationError as e:
            logger.warning(f"Judge verdict failed validation (attempt {attempt + 1}): {e}")
            messages = [
                user_msg(prompt),
                {"role": "assistant", "content": json.dumps(data)},
                user_msg(f"That response failed validation:\n{e}\nReturn a corrected JSON object."),
            ]
    raise LLMOutputError("benchmark_judge", "verdict failed validation after retry", json.dumps(data))


# All-null claims dict, used whenever extraction fails for any reason.
def empty_claims() -> dict:
    return {f: None for f in NumericClaims.model_fields}


# Extracts numeric claims from agent output or concatenated reference content. Never raises —
# a failed extraction degrades to empty claims (the judge still runs; deterministic numeric
# checks come back not_comparable).
def extract_numeric_claims(text: str, *, is_reference: bool) -> dict:
    role = "professional research sources" if is_reference else "an AI analyst's answer"
    try:
        data = chat_json(
            "benchmark_extractor",
            [user_msg(f"Extract the numeric claims from this text ({role}):\n\n{text[:12000]}")],
            system=EXTRACTOR_SYSTEM_PROMPT,
            schema_name="numeric_claims", schema=CLAIMS_SCHEMA,
        )
        return NumericClaims.model_validate(data).model_dump()
    except Exception as e:
        logger.warning(f"Numeric-claims extraction failed — treating as empty: {type(e).__name__}: {e}")
        return empty_claims()


def _compare(agent_val, ref_val, *, rel_tol=None, abs_tol=None) -> str:
    if agent_val is None or ref_val is None:
        return "not_comparable"
    if abs_tol is not None and abs(agent_val - ref_val) <= abs_tol:
        return "match"
    if rel_tol is not None and ref_val != 0 and abs(agent_val - ref_val) / abs(ref_val) <= rel_tol:
        return "match"
    return "mismatch"


# YoY growth appears as either percent (16.0) or a fraction (0.16) depending on the source —
# extracted signals store fractions, analyst sources state percent. A 100x-off pair that matches
# after rescaling is a representation difference, not a numeric error.
def _compare_yoy(agent_val, ref_val) -> str:
    result = _compare(agent_val, ref_val, abs_tol=YOY_PP_TOL)
    if result != "mismatch":
        return result
    for a, r in ((agent_val * 100, ref_val), (agent_val, ref_val * 100)):
        if _compare(a, r, abs_tol=YOY_PP_TOL) == "match":
            return "match"
    return "mismatch"


# Same-period test on structured (year, month, scope) fields — free-text labels from two
# independent extractions almost never string-match, which would misclassify real mismatches.
# Periods differ only when both sides state a field AND they disagree. Scope matters because a
# fiscal year and its final quarter share an end month: 10-K annual totals vs a source's Q4
# figures is a period difference, not a numeric error.
def _periods_differ(agent_claims: dict, ref_claims: dict) -> bool:
    for field in ("period_end_year", "period_end_month", "period_scope"):
        a, r = agent_claims.get(field), ref_claims.get(field)
        if a is not None and r is not None and a != r:
            return True
    return False


# Deterministic checks: intent routing, numeric cross-check with tolerances, consensus alignment.
def deterministic_checks(bq, run_record: dict, agent_claims: dict, ref_claims: dict) -> dict:
    intent_actual = (run_record.get("intent_actual") or {}).get("intent")
    checks: dict = {"intent_match": intent_actual == bq.intent_expectation, "intent_actual": intent_actual}

    numeric = {
        "agent_period": agent_claims.get("period_label"),
        "reference_period": ref_claims.get("period_label"),
    }
    differ = _periods_differ(agent_claims, ref_claims)
    for metric, compare in [
        ("revenue_usd", lambda a, r: _compare(a, r, rel_tol=REVENUE_REL_TOL)),
        ("eps_diluted", lambda a, r: _compare(a, r, abs_tol=EPS_ABS_TOL, rel_tol=EPS_REL_TOL)),
        ("revenue_yoy_pct", _compare_yoy),
    ]:
        result = compare(agent_claims.get(metric), ref_claims.get(metric))
        if result == "mismatch" and differ:
            result = "different_period"
        numeric[metric] = {"agent": agent_claims.get(metric), "reference": ref_claims.get(metric), "result": result}
    checks["numeric"] = numeric

    # Consensus alignment (advice queries). "opposite" is a flag, not a failure — well-argued
    # divergence is scored by the judge's directional_agreement dimension.
    agent_rec = None
    if isinstance(run_record.get("agent_output"), dict):
        agent_rec = run_record["agent_output"].get("recommendation")
    ref_rec = ref_claims.get("analyst_consensus")
    if agent_rec and ref_rec:
        order = {"buy": 0, "hold": 1, "sell": 2}
        gap = abs(order[agent_rec] - order[ref_rec])
        result = {0: "aligned", 1: "adjacent", 2: "opposite"}[gap]
    else:
        result = "n/a"
    checks["consensus"] = {
        "agent_recommendation": agent_rec, "reference_consensus": ref_rec,
        "consensus_quote": ref_claims.get("consensus_quote"), "result": result,
    }
    return checks
