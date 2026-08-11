# Network-free smoke tests: every module imports, schemas are strict-mode-safe, and the adapter's
# pure helpers behave. Runs on bare `pytest` (the capability harness is excluded via addopts).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_all_modules_import():
    import agents.advice
    import agents.comparison
    import agents.extraction
    import agents.orchestrator
    import agents.risk_scoring
    import agents.web_search
    import api.main
    import api.tasks
    import evals.capability_harness
    import llm  # noqa: F401


def test_schemas_are_strict_safe():
    from agents.advice import ADVICE_SCHEMA
    from agents.orchestrator import ORCHESTRATOR_SCHEMA

    for schema in (ORCHESTRATOR_SCHEMA, ADVICE_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_tool_conversion_shape():
    from llm import to_openai_tools
    from tools.extraction_tools import TOOLS

    converted = to_openai_tools(TOOLS)
    assert len(converted) == 5
    for tool in converted:
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}


def test_risk_step_order_matches_tools():
    from tools.risk_tools import STEP_ORDER, TOOLS

    assert [t["name"] for t in TOOLS] == STEP_ORDER


def test_tool_result_msgs_shape():
    from llm import tool_result_msgs

    msgs = tool_result_msgs([("id1", "OK"), ("id2", "data")])
    assert msgs == [
        {"role": "tool", "tool_call_id": "id1", "content": "OK"},
        {"role": "tool", "tool_call_id": "id2", "content": "data"},
    ]


def test_parse_args_rejects_non_objects():
    import json

    import pytest

    from llm.adapter import _parse_args

    assert _parse_args("") == {}
    assert _parse_args('{"a": 1}') == {"a": 1}
    with pytest.raises(json.JSONDecodeError):
        _parse_args("[1, 2]")
    with pytest.raises(json.JSONDecodeError):
        _parse_args('"just a string"')


def test_unit_normalization():
    from tools.extraction_tools import handle_extract_financial_metrics, handle_extract_management_outlook, to_usd

    assert to_usd(94930, "millions") == 94_930_000_000
    assert to_usd(1.5, "billions") == 1_500_000_000
    assert to_usd(250000, "thousands") == 250_000_000
    assert to_usd(None, "millions") is None

    fm = handle_extract_financial_metrics({"period": "Q3", "revenue": 109400, "unit": "millions", "unit_quote": "In millions"})
    assert fm["revenue_usd"] == 109_400_000_000
    assert "revenue" not in fm and "unit" not in fm
    assert fm["unit_quote"] == "In millions"

    outlook = handle_extract_management_outlook({"guidance_revenue": 10.2, "guidance_revenue_unit": "billions", "guidance_period": "Q4", "withdrawn": False, "verbatim_quote": "x"})
    assert outlook["guidance_revenue_usd"] == 10_200_000_000
    # A figure without a unit is dropped, never guessed.
    no_unit = handle_extract_management_outlook({"guidance_revenue": 96000, "guidance_period": "Q4", "withdrawn": False, "verbatim_quote": "x"})
    assert no_unit["guidance_revenue_usd"] is None

    from tools.extraction_tools import handle_extract_segment_performance

    segs = handle_extract_segment_performance({"unit": "millions", "segments": [{"name": "Cloud", "revenue": 1100, "growth": 24.0}]})
    assert segs == [{"name": "Cloud", "revenue_usd": 1_100_000_000, "growth": 24.0}]

    import pytest as _pytest

    with _pytest.raises(ValueError):
        to_usd(100, "million")  # unknown unit must raise, never guess


def test_revenue_plausibility_checks():
    from agents.extraction import REVENUE_MAX_USD, REVENUE_MIN_USD, _revenue_implausibility

    assert REVENUE_MIN_USD <= 94_930_000_000 <= REVENUE_MAX_USD

    # Segment cross-check fires before any DB access, so these run without a database.
    def results(revenue_usd, seg_revenues):
        return {
            "extract_financial_metrics": {"revenue_usd": revenue_usd},
            "extract_segment_performance": [{"name": f"s{i}", "revenue_usd": r} for i, r in enumerate(seg_revenues)],
        }

    # Total carries a millions-as-dollars error; segments are correct -> caught, company-independent.
    problem = _revenue_implausibility(999, results(94_930_000, [60_000_000_000, 34_000_000_000]))
    assert problem and "segment" in problem

    # Unit-class extremes are caught by the tripwire even with no segments and no history.
    problem = _revenue_implausibility(999, results(5_000, []))
    assert problem and "tripwire" in problem

    # None revenue is not an error (10-K stub financials).
    assert _revenue_implausibility(999, results(None, [])) is None


# Synthetic 10-Q reproducing Apple's geometry: the real Item 1 heading sits right after the TOC
# (so _is_toc_entry rejects it) and the statement title is the only recoverable anchor.
def _apple_style_10q() -> str:
    toc = "\n".join(f"Item {n}. {t} {i}" for i, (n, t) in enumerate([
        ("1", "Financial Statements"), ("2", "Management's Discussion and Analysis"),
        ("3", "Quantitative and Qualitative Disclosures"), ("4", "Controls and Procedures"),
        ("1A", "Risk Factors"), ("5", "Other Information"), ("6", "Exhibits"),
    ]))
    return f"""<html><body>
<p>COVER PAGE</p>
<p>{toc}</p>
<p>Item 1. Financial Statements</p>
<p>Apple Inc. CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)</p>
<p>(In millions, except number of shares, which are reflected in thousands, and per share amounts)</p>
<p>{'Total net sales 109,417 Diluted earnings per share 2.02 ' * 60}</p>
<p>Item 2. Management's Discussion and Analysis of Financial Condition</p>
<p>{'The Company posted strong quarterly results driven by iPhone. ' * 60}</p>
<p>Item 1A. Risk Factors</p>
<p>{'Macroeconomic conditions could materially affect results. ' * 60}</p>
</body></html>"""


def test_10q_financials_fallback():
    from scraper.parser import parse_sections

    sections = parse_sections(_apple_style_10q(), "10-Q")
    assert "CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS" in sections["financials"]
    assert "In millions, except number of shares" in sections["financials"]
    assert "Diluted earnings per share" in sections["financials"]
    # MD&A must not be truncated or mislabeled by the fallback.
    assert "posted strong quarterly results" in sections["mda"]
    assert "posted strong quarterly results" not in sections["financials"]


def test_10q_fallback_noop_when_financials_ok():
    from scraper.parser import parse_sections

    # Well-formed filing: Item 1 heading far from any TOC, with a large body.
    html = f"""<html><body>
<p>Item 1. Financial Statements</p>
<p>{'Revenue 1,000 Diluted earnings per share 1.00 ' * 80}</p>
<p>CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS</p>
<p>{'More statement content here. ' * 40}</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>{'Discussion text. ' * 80}</p>
</body></html>"""
    sections = parse_sections(html, "10-Q")
    # The slice must still begin at the Item 1 heading, not the later statement title.
    assert sections["financials"].startswith("Item 1. Financial Statements")


def test_10q_fallback_rejects_eof_anchor_when_mda_missing():
    from scraper.parser import parse_sections

    # No MD&A heading at all; the only statement-title match sits after risk factors (e.g. an
    # exhibit-index line near EOF). The anchor must be rejected — it doesn't precede the other
    # found sections.
    html = f"""<html><body>
<p>Item 1A. Risk Factors</p>
<p>{'Business risks described here. ' * 80}</p>
<p>Exhibit index: Consolidated Statements of Operations (filed herewith)</p>
</body></html>"""
    sections = parse_sections(html, "10-Q")
    assert sections["financials"] == ""


def test_10q_fallback_rejects_anchor_after_mda():
    from scraper.parser import parse_sections

    toc = "\n".join(f"Item {n}. {t}" for n, t in [
        ("1", "Financial Statements"), ("2", "Management's Discussion"), ("3", "Quantitative"),
        ("4", "Controls"), ("1A", "Risk Factors"), ("6", "Exhibits"),
    ])
    # Statement title appears ONLY inside MD&A prose — the fallback must not fire.
    html = f"""<html><body>
<p>{toc}</p>
<p>Item 2. Management's Discussion and Analysis</p>
<p>See our condensed consolidated statements of operations for details. {'Discussion. ' * 100}</p>
</body></html>"""
    sections = parse_sections(html, "10-Q")
    assert sections["financials"] == ""
    assert "Discussion." in sections["mda"]


def test_eps_requires_quote():
    from tools.extraction_tools import handle_extract_financial_metrics

    base = {"period": "Q3", "revenue": 109417, "unit": "millions", "unit_quote": "In millions"}
    dropped = handle_extract_financial_metrics({**base, "eps": 2.02})
    assert dropped["eps"] is None
    kept = handle_extract_financial_metrics({**base, "eps": 2.02, "eps_quote": "Diluted earnings per share 2.02"})
    assert kept["eps"] == 2.02


def test_quote_normalization_and_match():
    from agents.extraction import _alnum_only, _normalize_quote, _quote_in_source

    source = "The Company’s revenue was $\xa0109,417 million — up 16%.\nDiluted   earnings per share $ 2.02"
    norm = _normalize_quote(source)
    alnum = _alnum_only(norm)

    assert _quote_in_source("The Company's revenue was $ 109,417 million - up 16%.", norm, alnum)
    assert _quote_in_source("Diluted earnings per share $2.02", norm, alnum)  # alnum stage
    assert not _quote_in_source("Apple paid a $38 billion repatriation tax in 2018", norm, alnum)
    assert not _quote_in_source("", norm, alnum)
    assert not _quote_in_source(None, norm, alnum)
    assert not _quote_in_source("---", norm, alnum)  # punctuation-only must not pass vacuously

    # Token-boundary anchoring: short numeric fragments must not match inside larger numbers.
    src2 = _normalize_quote("Total shares 21,845 thousand and deferred revenue was 21,840")
    alnum2 = _alnum_only(src2)
    assert not _quote_in_source("1,84", src2, alnum2)
    assert not _quote_in_source("$1.84", src2, alnum2)


def test_eps_value_must_appear_in_quote():
    from agents.extraction import _verify_quotes

    source = "(In millions, except per share amounts) Diluted earnings per share 2.02"
    results = {
        "extract_financial_metrics": {
            "revenue_usd": 1e11, "unit_quote": "(In millions, except per share amounts)",
            "eps": 2.50, "eps_quote": "Diluted earnings per share 2.02",  # real quote, wrong figure
        },
    }
    assert _verify_quotes(results, source) is None
    assert results["extract_financial_metrics"]["eps"] is None

    results["extract_financial_metrics"].update({"eps": 2.02, "eps_quote": "Diluted earnings per share 2.02"})
    _verify_quotes(results, source)
    assert results["extract_financial_metrics"]["eps"] == 2.02


def test_outlook_substance_requires_verified_quote():
    from agents.extraction import _verify_quotes

    source = "Management expects continued growth in Services."
    # Empty quote must not slip guidance through.
    results = {"extract_management_outlook": {"guidance_revenue_usd": 5e9, "verbatim_quote": "", "withdrawn": False, "guidance_period": "Q4"}}
    _verify_quotes(results, source)
    assert results["extract_management_outlook"]["guidance_revenue_usd"] is None

    # A fabricated withdrawal must not survive its rejected citation.
    results = {"extract_management_outlook": {"withdrawn": True, "guidance_period": "Q4", "verbatim_quote": "we withdrew all guidance"}}
    _verify_quotes(results, source)
    assert results["extract_management_outlook"]["withdrawn"] is None
    assert results["extract_management_outlook"]["guidance_period"] is None


def test_verify_quotes_policies():
    from agents.extraction import _verify_quotes

    source = (
        "Revenue was strong. (In millions, except per share amounts) Diluted earnings per share 2.02. "
        "Demand may fluctuate with enterprise spending. Management expects growth to continue."
    )
    results = {
        "extract_financial_metrics": {
            "revenue_usd": 109_417_000_000, "unit_quote": "(In millions, except per share amounts)",
            "eps": 2.02, "eps_quote": "Diluted earnings per share 2.02",
        },
        "extract_risk_factors": [
            {"label": "demand", "verbatim_quote": "Demand may fluctuate with enterprise spending."},
            {"label": "fabricated", "verbatim_quote": "Aliens may disrupt the supply chain."},
        ],
        "extract_management_outlook": {"guidance_revenue_usd": 1e9, "verbatim_quote": "Management expects growth to continue."},
        "extract_notable_changes": [{"metric": "margin", "verbatim_quote": "Nothing like this appears."}],
    }
    hard = _verify_quotes(results, source)
    assert hard is None
    assert results["extract_financial_metrics"]["eps"] == 2.02
    assert [r["label"] for r in results["extract_risk_factors"]] == ["demand"]
    assert results["extract_management_outlook"]["guidance_revenue_usd"] == 1e9
    assert results["extract_notable_changes"] == []

    # Unverifiable unit_quote is a HARD failure.
    results["extract_financial_metrics"]["unit_quote"] = "(In thousands, totally invented)"
    hard = _verify_quotes(results, source)
    assert hard and "unit_quote" in hard


def test_env_override_errors_are_clear(monkeypatch):
    import pytest

    from llm.config import get_agent_config

    monkeypatch.setenv("LLM_PROVIDER_ADVICE", "not-json{")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER_ADVICE"):
        get_agent_config("advice")
