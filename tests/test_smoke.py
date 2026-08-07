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


def test_env_override_errors_are_clear(monkeypatch):
    import pytest

    from llm.config import get_agent_config

    monkeypatch.setenv("LLM_PROVIDER_ADVICE", "not-json{")
    with pytest.raises(RuntimeError, match="LLM_PROVIDER_ADVICE"):
        get_agent_config("advice")
