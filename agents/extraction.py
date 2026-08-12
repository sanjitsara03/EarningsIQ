import logging
import re

from dotenv import load_dotenv

from db.connection import get_connection
from db.queries import (
    get_chunks_for_filing,
    get_filing_ticker,
    get_historical_signals,
    insert_signals,
    update_filing_status,
)
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
- Report ONLY values present in the provided filing text. Never supply a number from memory,
  estimation, or general knowledge — if a figure (for example diluted EPS) does not appear in
  the text, omit that field entirely.
- Always include verbatim quotes exactly as they appear in the text. Every quote field
  (unit_quote, eps_quote, verbatim_quote) is verified verbatim against the filing; a value
  whose quote cannot be found is discarded.
- When you report eps, you must also report eps_quote: the exact line containing the diluted
  EPS figure.
- Report revenue exactly as printed in the financial tables, plus the table's stated unit
  (from captions like "In millions, except per share amounts") and that caption verbatim.
  Never convert the revenue figure yourself.
- 10-Q income statements show both a "Three Months Ended" column and a year-to-date column
  ("Six Months Ended" or "Nine Months Ended"). ALL metrics — revenue, EPS, margins, YoY change —
  must come from the three-month (current quarter) column. For 10-Ks, use the current fiscal
  year column. Never mix columns: when a statement row shows several values, pick the one under
  the three-month (or current-year) current-period header.
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


_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'", "‛": "'",   # single curly quotes
    "“": '"', "”": '"', "„": '"',   # double curly quotes
    "–": "-", "—": "-", "−": "-",   # en/em dash, minus
    "\xa0": " ",
})


# Canonical form for verbatim-quote comparison: typographic characters flattened, case folded,
# whitespace runs collapsed.
def _normalize_quote(s: str) -> str:
    s = s.translate(_QUOTE_TRANSLATION).casefold()
    return re.sub(r"\s+", " ", s).strip()


# Second-stage form: alphanumerics only. Tolerates dropped punctuation ("$ 1.84" vs "$1.84")
# while still requiring every word in sequence.
def _alnum_only(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# True when the quote appears in the source the model was shown, anchored at token boundaries —
# an unanchored substring test would let "1,84" verify against "21,845". Deterministic two-stage
# match; no fuzzy matching, since near-miss acceptance would defeat the purpose.
def _quote_in_source(quote, norm_source: str, alnum_source: str) -> bool:
    if not quote or not str(quote).strip():
        return False
    norm = _normalize_quote(str(quote))
    if norm and re.search(rf"(?<![a-z0-9]){re.escape(norm)}(?![a-z0-9])", norm_source):
        return True
    alnum = _alnum_only(norm)
    # A quote that normalizes to nothing has no verifiable content, and boundary padding keeps
    # token sequences intact ("1 84" must not match inside "21 845").
    return bool(alnum) and f" {alnum} " in f" {alnum_source} "


# True when the numeric value appears as a token sequence inside the quote text (e.g. eps 2.02
# inside "Diluted earnings per share $ 2.02"). Binds the reported figure to its citation — a real
# quote paired with a different number must not verify.
def _number_in_text(value, text: str) -> bool:
    if value is None or not text:
        return False
    alnum_text = f" {_alnum_only(_normalize_quote(str(text)))} "
    candidates = {str(value), f"{float(value):.2f}", f"{float(value):g}"}
    return any(f" {_alnum_only(_normalize_quote(c))} " in alnum_text for c in candidates)


# Verifies the metrics tool's quotes. Both failures route through the corrective-retry path as
# problem strings. After the retry (drop_unverified_eps=True) an eps failure degrades to dropping
# the field — eps is legitimately optional — while unit_quote stays hard: it justifies
# revenue_usd, which must never be persisted uncited.
def _verify_metrics_quotes(
    results: dict, norm_source: str, alnum_source: str, *, drop_unverified_eps: bool = False
) -> str | None:
    fm = results.get("extract_financial_metrics", {})

    problems = []
    if fm.get("eps") is not None:
        quote_ok = _quote_in_source(fm.get("eps_quote"), norm_source, alnum_source)
        value_ok = quote_ok and _number_in_text(fm.get("eps"), fm.get("eps_quote"))
        if not value_ok:
            reason = "was not found verbatim in the filing text" if not quote_ok else "does not contain the reported eps"
            if drop_unverified_eps:
                logger.warning(f"eps_quote {reason} after retry — dropping eps. quote={fm.get('eps_quote')!r} eps={fm.get('eps')}")
                fm["eps"] = None
                fm["eps_quote"] = None
            else:
                problems.append(f"eps_quote {fm.get('eps_quote')!r} {reason}")

    if fm.get("revenue_usd") is not None and not _quote_in_source(fm.get("unit_quote"), norm_source, alnum_source):
        problems.append(f"unit_quote {fm.get('unit_quote')!r} was not found verbatim in the filing text")
    return "; ".join(problems) or None


# Verifies every quote-bearing field against the text the model was shown, applying the soft
# policies in place (drop unverifiable items/fields with a warning). Returns a retryable problem
# string for unit_quote/eps_quote failures, or None.
def _verify_quotes(results: dict, source_text: str) -> str | None:
    norm_source = _normalize_quote(source_text)
    alnum_source = _alnum_only(norm_source)

    risks = results.get("extract_risk_factors") or []
    kept = [r for r in risks if _quote_in_source(r.get("verbatim_quote"), norm_source, alnum_source)]
    for r in risks:
        if r not in kept:
            logger.warning(f"Dropping risk factor with unverifiable quote: {r.get('label')!r}")
    results["extract_risk_factors"] = kept

    # The outlook's substance (guidance figure, withdrawn flag, period) is only as good as its
    # quote. A missing, empty, or unverifiable quote voids all of it — an empty quote must not
    # slip guidance through, and a fabricated "we withdrew guidance" must not survive its
    # rejected citation.
    outlook = results.get("extract_management_outlook") or {}
    has_substance = outlook.get("guidance_revenue_usd") is not None or outlook.get("withdrawn")
    if has_substance and not _quote_in_source(outlook.get("verbatim_quote"), norm_source, alnum_source):
        logger.warning(f"Outlook quote unverifiable — dropping guidance fields. quote={outlook.get('verbatim_quote')!r}")
        outlook["verbatim_quote"] = None
        outlook["guidance_revenue_usd"] = None
        outlook["guidance_period"] = None
        outlook["withdrawn"] = None

    changes = results.get("extract_notable_changes") or []
    kept_changes = [c for c in changes if _quote_in_source(c.get("verbatim_quote"), norm_source, alnum_source)]
    for c in changes:
        if c not in kept_changes:
            logger.warning(f"Dropping notable change with unverifiable quote: {c.get('metric')!r}")
    results["extract_notable_changes"] = kept_changes

    return _verify_metrics_quotes(results, norm_source, alnum_source)


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

    prompt_text = _build_prompt(chunks)
    messages = [user_msg(prompt_text)]
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

    # Quote verification (soft drops applied in place; unit_quote failure is hard) runs before the
    # plausibility guard so the retry message names the more fundamental defect. Either hard
    # problem gets one corrective re-extraction of the metrics, then a hard error — an
    # unverifiable or off-by-a-million figure must never be persisted.
    hard_problem = _verify_quotes(results, prompt_text)
    problem = hard_problem or _revenue_implausibility(filing_id, results)
    if problem:
        logger.warning(f"Metrics failed verification for filing_id={filing_id} — retrying: {problem}")
        retry_messages = [
            messages[0],
            user_msg(
                f"Your earlier extract_financial_metrics call produced an implausible figure or an "
                f"unverifiable citation: {problem}. Re-read the financial statements; copy unit_quote "
                "(and eps_quote if you report eps) exactly as printed — the entire row including any "
                "parenthetical note references in its label — then call extract_financial_metrics again."
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
        norm_source = _normalize_quote(prompt_text)
        hard_problem = _verify_metrics_quotes(
            results, norm_source, _alnum_only(norm_source), drop_unverified_eps=True
        )
        problem = hard_problem or _revenue_implausibility(filing_id, results)
        if problem:
            raise LLMOutputError("extraction", f"metrics failed verification after retry: {problem}")

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
