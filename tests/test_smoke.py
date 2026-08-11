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


def test_env_override_errors_are_clear(monkeypatch):
    import pytest

    from llm.config import get_agent_config

    monkeypatch.setenv("LLM_PROVIDER_ADVICE", "not-json{")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER_ADVICE"):
        get_agent_config("advice")
