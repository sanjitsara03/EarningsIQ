# Capability harness: probes every (model, provider-pin) pair in llm/config.py for the exact
# capabilities each agent depends on. Provider capability drifts weekly on OpenRouter, so this is
# a permanent, re-runnable gate — run it before migrating an agent, before demos, and after any
# provider incident. A full run costs a few cents.
#
# Usage:
#   uv run python -m evals.capability_harness                    # full matrix
#   uv run python -m evals.capability_harness --agent risk_scoring
#   uv run python -m evals.capability_harness --include-cache-probe
#
# Report: evals/output/capability_report.json (+ timestamped history copy).
import argparse
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai

from llm import (
    NoToolCallError,
    assistant_msg_from_response,
    chat,
    chat_json,
    get_agent_config,
    named_tool_choice,
    to_openai_tools,
    tool_result_msgs,
    user_msg,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

# Tiny fake tools for the forced-choice probes — three steps mirroring the risk loop's shape.
FAKE_STEP_TOOLS = [
    {
        "name": f"step_{letter}",
        "description": f"Record the value for step {letter}. Call when instructed to perform step {letter}.",
        "input_schema": {
            "type": "object",
            "properties": {"value": {"type": "string", "description": "A short value to record."}},
            "required": ["value"],
        },
    }
    for letter in "abc"
]

# Fake tools for the terminal-loop probe: a lookup and a terminal answer tool.
FAKE_LOOP_TOOLS = [
    {
        "name": "lookup_data",
        "description": "Look up a revenue figure for a company and period. Call before answering.",
        "input_schema": {
            "type": "object",
            "properties": {"company": {"type": "string"}, "period": {"type": "string"}},
            "required": ["company", "period"],
        },
    },
    {
        "name": "cite_and_answer",
        "description": "Deliver the final answer with citations. This ends the task — call it exactly once, after looking up the data you need.",
        "input_schema": {
            "type": "object",
            "properties": {"answer": {"type": "string"}, "citations": {"type": "array", "items": {"type": "string"}}},
            "required": ["answer", "citations"],
        },
    },
]

FAKE_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for current information.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

# A small fake filing for the extraction probe — enough substance that all five tools have data.
FAKE_FILING = """=== FINANCIALS ===
(In millions, except per share amounts)
=== MDA ===
Revenue for the quarter was $2,450 million, up 12% year over year, driven by strength in the
Cloud segment. Gross margin was 61.2%, and operating margin was 28.4%. Diluted EPS was $1.84.
The Cloud segment grew 24% to $1,100 million; Devices declined 3% to $650 million; Services
grew 9% to $700 million. Management expects full-year revenue of approximately $10,200 million
and noted that prior guidance is unchanged. Operating expenses increased 8% due to continued
investment in AI infrastructure, which management flagged as the primary driver of the
quarter-over-quarter margin decline of 120 basis points.
=== RISK_FACTORS ===
Demand for our products may fluctuate with enterprise IT spending cycles. We face intense
competition in cloud infrastructure from larger rivals with greater resources. Supply chain
constraints for AI accelerators could limit our capacity expansion. Foreign exchange volatility
affects approximately 40% of our revenue. New AI regulations in the EU may increase compliance
costs materially in fiscal 2027."""


# One probe result row for the report.
def _result(probe, agent, status, *, failure_mode=None, detail=None, usage=(0, 0), latency_ms=0, provider_served=""):
    cfg = get_agent_config(agent)
    return {
        "probe": probe,
        "agent": agent,
        "model": cfg.model,
        "provider_requested": cfg.provider,
        "provider_served": provider_served,
        "status": status,
        "failure_mode": failure_mode,
        "detail": detail,
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        "latency_ms": latency_ms,
    }


# Classifies an exception from a forced-tool-choice call into the known failure modes.
def _classify_error(e: Exception) -> str:
    if isinstance(e, openai.NotFoundError):
        return "404_no_endpoints"
    if isinstance(e, openai.BadRequestError):
        return "400_rejected"
    if isinstance(e, NoToolCallError):
        return "silent_prose"
    return f"error:{type(e).__name__}"


# P1a/P1b — strict json_schema structured output validates against the real Pydantic model.
def probe_json_schema_strict(agent: str, probe_name: str) -> dict:
    if agent == "orchestrator":
        from agents.orchestrator import ORCHESTRATOR_SCHEMA, SYSTEM_PROMPT, IntentOutput

        schema_name, schema, model_cls = "orchestrator_intent", ORCHESTRATOR_SCHEMA, IntentOutput
        prompt = "How did Apple do last quarter?"
        system = SYSTEM_PROMPT
    else:
        from agents.advice import ADVICE_SCHEMA, SYSTEM_PROMPT, AdviceOutput

        schema_name, schema, model_cls = "advice_output", ADVICE_SCHEMA, AdviceOutput
        prompt = (
            "Ticker: TESTCO. Signals: revenue 2450 (millions), eps 1.84, gross margin 61.2%, "
            "revenue up 12% YoY. Risk score: 4.2 (medium) — competition and supply chain. "
            "Give your investment recommendation."
        )
        system = SYSTEM_PROMPT

    start = time.monotonic()
    try:
        data = chat_json(agent, [user_msg(prompt)], system=system, schema_name=schema_name, schema=schema)
        model_cls(**data)
        return _result(probe_name, agent, "pass", latency_ms=int((time.monotonic() - start) * 1000))
    except Exception as e:
        return _result(probe_name, agent, "fail", failure_mode=_classify_error(e), detail=str(e)[:300])


# P2a — named forced tool choice across three sequential steps on the pinned provider.
# P2b — the fallback mechanism: single-tool tools list + tool_choice "required".
def probe_forced_tools(agent: str, mechanism: str) -> dict:
    probe_name = "forced_named_tool" if mechanism == "named" else "single_tool_required"
    messages = [
        user_msg(
            "We will proceed through three steps in order: step a, then step b, then step c. "
            "For each step, call the corresponding tool with any short value. Begin with step a."
        )
    ]
    steps = []
    prompt_toks = completion_toks = 0
    provider_served = ""
    start = time.monotonic()

    try:
        for letter in "abc":
            tool_name = f"step_{letter}"
            if mechanism == "named":
                tools = to_openai_tools(FAKE_STEP_TOOLS)
                tool_choice = named_tool_choice(tool_name)
            else:
                tools = to_openai_tools([t for t in FAKE_STEP_TOOLS if t["name"] == tool_name])
                tool_choice = "required"

            # gpt-5.2 endpoints do not declare parallel_tool_calls; sending it with
            # require_parameters=true returns a routing 404.
            resp = chat(agent, messages, tools=tools, tool_choice=tool_choice)
            provider_served = resp.provider or provider_served
            prompt_toks += resp.prompt_tokens
            completion_toks += resp.completion_tokens

            names = [tc.name for tc in resp.tool_calls]
            steps.append({"forced": tool_name, "got": names, "provider": resp.provider})
            if any(n != tool_name for n in names):
                return _result(
                    probe_name, agent, "fail", failure_mode="silent_wrong_tool",
                    detail={"steps": steps}, usage=(prompt_toks, completion_toks),
                    provider_served=provider_served,
                )

            messages.append(assistant_msg_from_response(resp))
            messages.extend(tool_result_msgs([(tc.id, "recorded") for tc in resp.tool_calls]))

        return _result(
            probe_name, agent, "pass", detail={"steps": steps},
            usage=(prompt_toks, completion_toks),
            latency_ms=int((time.monotonic() - start) * 1000), provider_served=provider_served,
        )
    except Exception as e:
        return _result(
            probe_name, agent, "fail", failure_mode=_classify_error(e),
            detail={"steps": steps, "error": str(e)[:300]}, provider_served=provider_served,
        )


# P3 — tool_choice "required" collects all five real extraction tools within 10 turns.
def probe_required_parallel(agent: str) -> dict:
    from tools.extraction_tools import TOOLS

    system = (
        "You are extracting structured data from a SEC filing. You must call ALL FIVE extraction "
        "tools exactly once each. Use only information present in the filing."
    )
    messages = [user_msg(FAKE_FILING + "\n\nExtract all required information by calling all five tools.")]
    openai_tools = to_openai_tools(TOOLS)
    collected: set[str] = set()
    prompt_toks = completion_toks = 0
    parallel_first_turn = False
    provider_served = ""
    start = time.monotonic()

    try:
        for turn in range(10):
            resp = chat(agent, messages, system=system, tools=openai_tools, tool_choice="required")
            provider_served = resp.provider or provider_served
            prompt_toks += resp.prompt_tokens
            completion_toks += resp.completion_tokens

            names = {tc.name for tc in resp.tool_calls}
            if turn == 0 and len(names) == 5:
                parallel_first_turn = True
            collected |= names
            if len(collected) >= 5:
                return _result(
                    "required_parallel_collection", agent, "pass",
                    detail={"turns_needed": turn + 1, "parallel_first_turn": parallel_first_turn},
                    usage=(prompt_toks, completion_toks),
                    latency_ms=int((time.monotonic() - start) * 1000), provider_served=provider_served,
                )
            messages.append(assistant_msg_from_response(resp))
            messages.extend(tool_result_msgs([(tc.id, "OK") for tc in resp.tool_calls]))

        return _result(
            "required_parallel_collection", agent, "fail", failure_mode="incomplete_collection",
            detail={"collected": sorted(collected)}, usage=(prompt_toks, completion_toks),
            provider_served=provider_served,
        )
    except Exception as e:
        return _result("required_parallel_collection", agent, "fail", failure_mode=_classify_error(e), detail=str(e)[:300])


# P4 — auto tool choice reaches the terminal tool within 10 turns, after at least one lookup.
def probe_terminal_loop(agent: str) -> dict:
    system = (
        "You answer questions about company revenue using your tools. Look up the data you need "
        "with lookup_data, then deliver your final answer by calling cite_and_answer. "
        "cite_and_answer ends the task."
    )
    messages = [user_msg("How did TestCo's revenue change from Q1 to Q2 2026?")]
    openai_tools = to_openai_tools(FAKE_LOOP_TOOLS)
    lookups = 0
    prompt_toks = completion_toks = 0
    provider_served = ""
    start = time.monotonic()

    try:
        for turn in range(10):
            resp = chat(agent, messages, system=system, tools=openai_tools)
            provider_served = resp.provider or provider_served
            prompt_toks += resp.prompt_tokens
            completion_toks += resp.completion_tokens

            if not resp.tool_calls:
                return _result(
                    "terminal_tool_loop", agent, "fail", failure_mode="stopped_without_terminal",
                    detail={"turn": turn + 1, "lookups": lookups}, usage=(prompt_toks, completion_toks),
                    provider_served=provider_served,
                )

            pairs = []
            for tc in resp.tool_calls:
                if tc.name == "cite_and_answer":
                    status = "pass" if lookups >= 1 else "fail"
                    return _result(
                        "terminal_tool_loop", agent, status,
                        failure_mode=None if status == "pass" else "terminal_before_lookup",
                        detail={"turns": turn + 1, "lookups": lookups},
                        usage=(prompt_toks, completion_toks),
                        latency_ms=int((time.monotonic() - start) * 1000), provider_served=provider_served,
                    )
                lookups += 1
                pairs.append((tc.id, "Q1 2026 revenue: $2,100M. Q2 2026 revenue: $2,450M."))

            messages.append(assistant_msg_from_response(resp))
            messages.extend(tool_result_msgs(pairs))

        return _result(
            "terminal_tool_loop", agent, "fail", failure_mode="burned_all_turns",
            detail={"lookups": lookups}, usage=(prompt_toks, completion_toks), provider_served=provider_served,
        )
    except Exception as e:
        return _result("terminal_tool_loop", agent, "fail", failure_mode=_classify_error(e), detail=str(e)[:300])


# P5 — auto tool loop terminates with text within 5 turns.
def probe_auto_termination(agent: str) -> dict:
    system = "You are a research assistant with a web search tool. Answer the question, searching as needed."
    messages = [user_msg("What is the latest news about TestCo's earnings?")]
    openai_tools = to_openai_tools([FAKE_SEARCH_TOOL])
    prompt_toks = completion_toks = 0
    provider_served = ""
    start = time.monotonic()

    try:
        for turn in range(5):
            resp = chat(agent, messages, system=system, tools=openai_tools)
            provider_served = resp.provider or provider_served
            prompt_toks += resp.prompt_tokens
            completion_toks += resp.completion_tokens

            if not resp.tool_calls:
                ok = bool(resp.text and resp.text.strip()) and resp.finish_reason == "stop"
                return _result(
                    "auto_loop_termination", agent, "pass" if ok else "fail",
                    failure_mode=None if ok else "no_text_at_stop",
                    detail={"turns": turn + 1, "finish_reason": resp.finish_reason},
                    usage=(prompt_toks, completion_toks),
                    latency_ms=int((time.monotonic() - start) * 1000), provider_served=provider_served,
                )

            pairs = [
                (tc.id, "TestCo reported Q2 revenue of $2,450M, beating estimates by 4%. Shares rose 6%.")
                for tc in resp.tool_calls
            ]
            messages.append(assistant_msg_from_response(resp))
            messages.extend(tool_result_msgs(pairs))

        return _result("auto_loop_termination", agent, "fail", failure_mode="never_terminated", usage=(prompt_toks, completion_toks), provider_served=provider_served)
    except Exception as e:
        return _result("auto_loop_termination", agent, "fail", failure_mode=_classify_error(e), detail=str(e)[:300])


# P7 (optional) — informational: does prompt caching pass through? Never blocks.
def probe_cache_passthrough(agent: str) -> dict:
    big_system = "You are a financial analyst. " + ("Context sentence for caching probe. " * 300)
    try:
        chat(agent, [user_msg("Say OK.")], system=big_system)
        resp2 = chat(agent, [user_msg("Say OK again.")], system=big_system)
        status = "warn" if resp2.cached_tokens == 0 else "pass"
        return _result(
            "cache_passthrough", agent, status,
            failure_mode=None if status == "pass" else "no_cached_tokens_reported",
            detail={"cached_tokens_second_call": resp2.cached_tokens},
            usage=(resp2.prompt_tokens, resp2.completion_tokens),
        )
    except Exception as e:
        return _result("cache_passthrough", agent, "warn", failure_mode=_classify_error(e), detail=str(e)[:300])


# Maps each agent gate to the probes that decide it.
def run_probes(only_agent: str | None = None, include_cache_probe: bool = False) -> dict:
    plan = [
        ("orchestrator", lambda: [probe_json_schema_strict("orchestrator", "json_schema_strict")]),
        ("advice", lambda: [probe_json_schema_strict("advice", "json_schema_strict")]),
        ("risk_scoring", lambda: [probe_forced_tools("risk_scoring", "named"), probe_forced_tools("risk_scoring", "single")]),
        ("extraction", lambda: [probe_required_parallel("extraction")]),
        ("comparison", lambda: [probe_terminal_loop("comparison")]),
        ("web_search", lambda: [probe_auto_termination("web_search")]),
    ]

    results = []
    for agent, runner in plan:
        if only_agent and agent != only_agent:
            continue
        logger.info(f"--- probing {agent} ({get_agent_config(agent).model}) ---")
        results.extend(runner())

    if include_cache_probe and (not only_agent or only_agent == "comparison"):
        results.append(probe_cache_passthrough("comparison"))

    # Gate verdicts. Risk scoring distinguishes which forcing mechanism passed.
    gates = {}
    by_probe = {(r["agent"], r["probe"]): r for r in results}
    for agent in ["orchestrator", "advice", "extraction", "comparison", "web_search"]:
        rows = [r for r in results if r["agent"] == agent and r["status"] != "warn"]
        if rows:
            gates[agent] = "pass" if all(r["status"] == "pass" for r in rows) else "fail"
    named = by_probe.get(("risk_scoring", "forced_named_tool"))
    single = by_probe.get(("risk_scoring", "single_tool_required"))
    if named or single:
        expected_provider = "openai"
        named_ok = named and named["status"] == "pass" and expected_provider in named["provider_served"].lower()
        single_ok = single and single["status"] == "pass" and expected_provider in single["provider_served"].lower()
        gates["risk_scoring"] = "pass_named" if named_ok else ("pass_fallback_only" if single_ok else "fail")

    report = {
        "run_at": datetime.now(UTC).isoformat(),
        "results": results,
        "summary": {
            "gates": gates,
            "total_usage": {
                "prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in results),
                "completion_tokens": sum(r["usage"]["completion_tokens"] for r in results),
            },
        },
    }
    return report


VALID_AGENTS = {"orchestrator", "advice", "risk_scoring", "extraction", "comparison", "web_search"}


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenRouter capability harness")
    parser.add_argument("--agent", choices=sorted(VALID_AGENTS), help="probe a single agent gate")
    parser.add_argument("--include-cache-probe", action="store_true")
    args = parser.parse_args()

    report = run_probes(args.agent, args.include_cache_probe)
    if not report["results"]:
        print("ERROR: no probes ran — the gate cannot vacuously pass.")
        return 1

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "capability_report.json").write_text(json.dumps(report, indent=2, default=str))
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    (OUTPUT_DIR / f"capability_report_{ts}.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n=== Capability gates ===")
    for agent, verdict in report["summary"]["gates"].items():
        print(f"  {agent:14s} {verdict}")
    usage = report["summary"]["total_usage"]
    print(f"  tokens: {usage['prompt_tokens']} in / {usage['completion_tokens']} out")

    failed = [v for v in report["summary"]["gates"].values() if v == "fail"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
