# Analyst benchmark: runs the real agent pipeline over the query set, gathers fresh professional
# research via Tavily, and scores agent output with deterministic checks plus a cross-family LLM judge.
# Routing helpers are imported from the chat route so benchmark routing cannot drift from prod.
#
# Usage:
#   uv run python -m evals.analyst_benchmark                       # full run (~$1, 5-8 min)
#   uv run python -m evals.analyst_benchmark --query aapl_advice   # filter by id (repeatable)
#   uv run python -m evals.analyst_benchmark --intent advice       # filter by intent
#   uv run python -m evals.analyst_benchmark --skip-research       # reuse latest research snapshot
#   uv run python -m evals.analyst_benchmark --frozen-research F   # pinned snapshot (regression mode)
#   uv run python -m evals.analyst_benchmark --ingest-missing      # inline-ingest unready tickers
#   uv run python -m evals.analyst_benchmark --rejudge REPORT      # re-score stored agent outputs
#   uv run python -m evals.analyst_benchmark --compare REPORT      # per-dimension deltas vs a report
#
# Preconditions: tickers in the query set must be ingested (checked; remediation printed).
# Known limitation: per-query token usage is not recorded; wall-clock latency per stage is.
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.benchmark_judge import (
    deterministic_checks,
    extract_numeric_claims,
    run_judge,
)
from evals.benchmark_queries import QUERIES, BenchmarkQuery, default_research_queries

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"

DIMENSIONS = ["factual_accuracy", "coverage", "grounding", "directional_agreement"]
# Hallucination-class failures matter most for this product.
WORST_QUERY_WEIGHTS = {"factual_accuracy": 2, "coverage": 1, "grounding": 2, "directional_agreement": 1}


# Replicates chat.py's _chat() routing minus HTTP/queue, reusing the route's internals so it can't drift.
def run_agent_for_query(bq: BenchmarkQuery, ingest_missing: bool) -> dict:
    from agents.advice import run_advice, run_analysis
    from agents.comparison import is_comparison_query, run_comparison
    from agents.orchestrator import run_orchestrator
    from agents.web_search import run_web_search
    from api.routes.chat import _load_context, _resolve_filing_type
    from api.tasks import run_full_pipeline, ticker_is_ready

    start = time.monotonic()
    record: dict = {"status": "ok", "route": None, "intent_actual": None, "agent_output": None,
                    "filing_period": None, "filed_at": None, "error": None}
    try:
        intent = run_orchestrator(bq.query)
        record["intent_actual"] = intent
        intent_type = intent.get("intent")
        ticker = (intent.get("ticker") or "").upper()
        filing_type = intent["filing_types"][0] if intent.get("filing_types") else bq.filing_type

        if intent_type == "web_only":
            from agents.web_search import run_web_search_detailed

            record["route"] = "web"
            summary, sources = run_web_search_detailed(bq.query)
            record["agent_output"] = summary
            # The judge needs the agent's own search results to trace grounding.
            record["agent_inputs"] = {"web_sources": sources}
        elif intent_type == "comparison" and is_comparison_query(bq.query):
            if not ticker_is_ready(ticker, filing_type):
                if not ingest_missing:
                    record["status"] = "skipped_not_ingested"
                    return record
                logger.info(f"Ingesting {ticker} {filing_type} inline (--ingest-missing)...")
                run_full_pipeline(ticker, filing_type, intent.get("periods_needed", 1))
            record["route"] = "comparison"
            result = run_comparison(bq.query)
            # The judge needs the tool results the agent saw, or DB-derived numbers read as ungrounded.
            record["agent_inputs"] = {"tool_trace": result.pop("tool_trace", [])}
            record["agent_output"] = result
        else:
            filing_type = _resolve_filing_type(ticker, filing_type)
            if not ticker_is_ready(ticker, filing_type):
                if not ingest_missing:
                    record["status"] = "skipped_not_ingested"
                    return record
                logger.info(f"Ingesting {ticker} {filing_type} inline (--ingest-missing)...")
                run_full_pipeline(ticker, filing_type, intent.get("periods_needed", 1))
            signals, risk, latest = _load_context(ticker, filing_type)
            record["filing_period"] = latest["period"] if latest else None
            record["filed_at"] = latest["filed_at"].isoformat() if latest and latest["filed_at"] else None
            web = run_web_search(bq.query) if intent.get("web_search_needed") else None
            # Filing-derived claims are grounded even when absent from web sources — the judge needs the inputs.
            record["agent_inputs"] = {"signals": signals, "risk": risk, "web_summary": web}
            period = str(record["filing_period"]) if record["filing_period"] else None
            if intent_type == "single_analysis":
                record["route"] = "analysis"
                record["agent_output"] = run_analysis(
                    bq.query, ticker, signals, risk, web, filing_type, period).model_dump()
            else:
                record["route"] = "advice"
                record["agent_output"] = run_advice(
                    ticker, signals, risk, web, filing_type, period).model_dump()
    except Exception as e:
        record["status"] = "agent_error"
        record["error"] = f"{type(e).__name__}: {e}"[:500]
    record["latency_ms"] = int((time.monotonic() - start) * 1000)
    return record


# Structured Tavily search (the web agent's helper only returns a formatted string); one retry, then empty.
def _tavily_raw(query: str, max_results: int = 3) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    for attempt in range(2):
        try:
            resp = client.search(query=query, max_results=max_results)
            return resp.get("results", [])
        except Exception as e:
            logger.warning(f"Tavily search failed (attempt {attempt + 1}): {e}")
            time.sleep(1)
    return []


# Gathers, dedupes, and stamps professional sources for one query.
def research_query(bq: BenchmarkQuery) -> dict:
    start = time.monotonic()
    searches = list(bq.research_queries) or default_research_queries(bq.ticker or "")
    seen: dict[str, dict] = {}
    for search in searches:
        for r in _tavily_raw(search):
            url = r.get("url")
            if url and url not in seen:
                seen[url] = {
                    "url": url, "title": r.get("title"), "content": r.get("content"),
                    "score": r.get("score"), "tavily_query": search,
                    "retrieved_at": datetime.now(UTC).isoformat(),
                }
        time.sleep(0.5)
    sources = sorted(seen.values(), key=lambda s: s.get("score") or 0, reverse=True)[:6]
    status = "ok" if len(sources) >= 3 else ("partial" if sources else "empty")
    return {"status": status, "sources": sources,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "latency_ms": int((time.monotonic() - start) * 1000)}


def judge_query(bq: BenchmarkQuery, run_record: dict, sources: list[dict]) -> tuple[dict, dict]:
    start = time.monotonic()
    agent_text = json.dumps(run_record.get("agent_output"), default=str)
    ref_text = "\n\n".join((s.get("content") or "")[:2000] for s in sources)

    # extract_numeric_claims never raises — one flaky extractor call must not abort the whole run.
    agent_claims = extract_numeric_claims(agent_text, is_reference=False)
    ref_claims = extract_numeric_claims(ref_text, is_reference=True)
    checks = deterministic_checks(bq, run_record, agent_claims, ref_claims)

    try:
        verdict = run_judge(bq, run_record, sources)
        judge = {"status": "ok", "verdict": verdict}
    except Exception as e:
        logger.warning(f"Judge failed for {bq.id}: {e}")
        judge = {"status": "judge_failed", "verdict": None, "error": f"{type(e).__name__}: {e}"[:300]}
    judge["latency_ms"] = int((time.monotonic() - start) * 1000)
    return checks, judge


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _weighted_mean(verdict: dict) -> float:
    total = sum(WORST_QUERY_WEIGHTS[d] * verdict[d]["score"] for d in DIMENSIONS)
    return round(total / sum(WORST_QUERY_WEIGHTS.values()), 2)


def build_aggregate(query_records: list[dict]) -> dict:
    scored = [q for q in query_records if q["judge"].get("status") == "ok"]
    dimension_means = {}
    for d in DIMENSIONS:
        vals = [q["judge"]["verdict"][d]["score"] for q in scored]
        dimension_means[d] = round(sum(vals) / len(vals), 2) if vals else None

    ran = [q for q in query_records if q["run"]["status"] == "ok"]
    intent_ok = sum(1 for q in ran if q.get("deterministic_checks", {}).get("intent_match"))
    numeric_results = [
        m["result"]
        for q in query_records
        for name, m in (q.get("deterministic_checks", {}).get("numeric") or {}).items()
        if isinstance(m, dict) and "result" in m
    ]
    comparable = [r for r in numeric_results if r in ("match", "mismatch")]
    consensus_counts: dict[str, int] = {}
    for q in query_records:
        r = (q.get("deterministic_checks", {}).get("consensus") or {}).get("result")
        if r:
            consensus_counts[r] = consensus_counts.get(r, 0) + 1

    worst = sorted(
        ({"id": q["id"], "weighted_mean": _weighted_mean(q["judge"]["verdict"]),
          "top_issue": next(iter(q["judge"]["verdict"]["factual_accuracy"]["evidence"]), "")}
         for q in scored),
        key=lambda w: w["weighted_mean"],
    )[:3]

    return {
        "queries_total": len(query_records), "queries_scored": len(scored),
        "dimension_means": dimension_means,
        "deterministic": {
            "intent_match": f"{intent_ok}/{len(ran)}",
            "numeric_match": f"{comparable.count('match')}/{len(comparable)} comparable",
            "consensus": consensus_counts,
        },
        "worst_queries": worst,
    }


def write_markdown(report: dict, path: Path) -> None:
    cfg = report["config"]
    lines = [
        "# Analyst Benchmark Summary",
        (
            f"\nRun: {report['run_at']} · git {report['git_sha']} · judge {cfg['judge_model']} · "
            f"research {cfg['research_mode']} ({cfg.get('research_snapshot', '-')})"
        ),
        "\n## Scores (1-5; means over judged queries)\n",
        "| dimension | mean |", "|---|---|",
    ]
    for d, v in report["aggregate"]["dimension_means"].items():
        lines.append(f"| {d} | {v if v is not None else 'n/a'} |")
    det = report["aggregate"]["deterministic"]
    lines += [
        f"\nDeterministic: intent {det['intent_match']} · numeric {det['numeric_match']} · consensus {det['consensus']}",
        "\n## Per-query\n",
        "| id | route | status | fact | cov | grnd | dir | numeric | consensus |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for q in report["queries"]:
        v = (q["judge"] or {}).get("verdict")
        scores = [str(v[d]["score"]) if v else "-" for d in DIMENSIONS]
        numeric = q.get("deterministic_checks", {}).get("numeric") or {}
        # Unambiguous one-char codes ('match' and 'mismatch' share a first letter).
        codes = {"match": "✓", "mismatch": "✗", "different_period": "p", "not_comparable": "·"}
        n_summary = ",".join(codes.get(m["result"], "?") for k, m in numeric.items() if isinstance(m, dict) and "result" in m) or "-"
        cons = (q.get("deterministic_checks", {}).get("consensus") or {}).get("result", "-")
        lines.append(f"| {q['id']} | {q['run'].get('route') or '-'} | {q['run']['status']} | "
                     f"{' | '.join(scores)} | {n_summary} | {cons} |")

    lines.append("\n## Worst queries\n")
    for w in report["aggregate"]["worst_queries"]:
        lines.append(f"- **{w['id']}** (weighted {w['weighted_mean']}): {w['top_issue']}")

    lines.append("\n## Reference set\n")
    for q in report["queries"]:
        srcs = (q.get("research") or {}).get("sources") or []
        if srcs:
            lines.append(f"**{q['id']}**")
            lines += [f"- {s['url']} (retrieved {s['retrieved_at'][:10]})" for s in srcs]
    lines.append(
        "\n*Drift note: fresh-research runs are only meaningful as current alignment; use "
        "--frozen-research with an unchanged judge/agent-model/filing-period set for "
        "agent-regression comparisons.*"
    )
    path.write_text("\n".join(lines))


# Rejudge keeps the original run's provenance — only the judge is stamped fresh, else --compare flags all rows n/c.
def build_report(query_records: list[dict], research_mode: str, research_snapshot: str | None,
                 base_report: dict | None = None) -> dict:
    from llm import AGENT_MODELS

    if base_report:
        config = dict(base_report["config"])
        config["research_mode"] = "rejudge"
        config["judge_model"] = AGENT_MODELS["benchmark_judge"].model
        config["rejudged_from"] = research_snapshot
        git_sha = base_report.get("git_sha", "unknown")
    else:
        config = {
            "judge_model": AGENT_MODELS["benchmark_judge"].model,
            "agent_models": {name: cfg.model for name, cfg in AGENT_MODELS.items()},
            "research_mode": research_mode,
            "research_snapshot": research_snapshot,
        }
        git_sha = _git_sha()
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "config": config,
        "queries": query_records,
        "aggregate": build_aggregate(query_records),
    }


def compare_reports(current: dict, previous: dict) -> None:
    prev_by_id = {q["id"]: q for q in previous["queries"]}
    same_setup = (
        current["config"]["judge_model"] == previous["config"]["judge_model"]
        and current["config"]["agent_models"] == previous["config"]["agent_models"]
        and current["config"].get("research_snapshot") == previous["config"].get("research_snapshot")
    )
    print("\n=== Comparison vs previous report ===")
    if not same_setup:
        print("(setup differs — judge/models/research snapshot changed; deltas shown as n/c where periods differ)")
    for q in current["queries"]:
        prev = prev_by_id.get(q["id"])
        if not prev or not (q["judge"].get("verdict") and prev["judge"].get("verdict")):
            continue
        comparable = same_setup and q["run"].get("filing_period") == prev["run"].get("filing_period")
        deltas = []
        for d in DIMENSIONS:
            delta = q["judge"]["verdict"][d]["score"] - prev["judge"]["verdict"][d]["score"]
            deltas.append(f"{d[:4]}:{'n/c' if not comparable else f'{delta:+d}'}")
        print(f"  {q['id']:18s} {' '.join(deltas)}")


def rejudge_report(previous: dict) -> list[dict]:
    by_id = {q.id: q for q in QUERIES}
    records = []
    for q in previous["queries"]:
        bq = by_id.get(q["id"])
        if bq is None or q["run"]["status"] != "ok":
            records.append(q)
            continue
        sources = (q.get("research") or {}).get("sources") or []
        if not sources:
            q["judge"] = {"status": "no_references", "verdict": None}
            records.append(q)
            continue
        checks, judge = judge_query(bq, q["run"], sources)
        q["deterministic_checks"], q["judge"] = checks, judge
        records.append(q)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyst benchmark")
    parser.add_argument("--query", action="append", help="filter by query id (repeatable)")
    parser.add_argument("--intent", choices=["single_analysis", "comparison", "advice", "web_only"])
    parser.add_argument("--skip-research", action="store_true", help="reuse benchmark_research_latest.json")
    parser.add_argument("--frozen-research", type=Path, help="pinned research snapshot path")
    parser.add_argument("--ingest-missing", action="store_true")
    parser.add_argument("--rejudge", type=Path, help="re-score an existing report's agent outputs")
    parser.add_argument("--compare", type=Path, help="print per-dimension deltas vs a previous report")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    try:
        base_report = None
        if args.rejudge:
            base_report = json.loads(args.rejudge.read_text())
            records = rejudge_report(base_report)
            research_mode, snapshot_name = "rejudge", str(args.rejudge.name)
        else:
            selected = [q for q in QUERIES
                        if (not args.query or q.id in args.query)
                        and (not args.intent or q.intent_expectation == args.intent)]
            if not selected:
                print("ERROR: no queries selected — the benchmark cannot vacuously pass.")
                return 1

            # RESEARCH (before RUN so a bad snapshot path fails fast)
            if args.frozen_research:
                research_map = json.loads(args.frozen_research.read_text())
                research_mode, snapshot_name = "frozen", args.frozen_research.name
            elif args.skip_research:
                latest = OUTPUT_DIR / "benchmark_research_latest.json"
                research_map = json.loads(latest.read_text())
                research_mode, snapshot_name = "reused_latest", "benchmark_research_latest.json"
            else:
                logger.info("--- RESEARCH stage ---")
                research_map = {q.id: research_query(q) for q in selected}
                snapshot_name = f"benchmark_research_{ts}.json"
                (OUTPUT_DIR / snapshot_name).write_text(json.dumps(research_map, indent=1))
                # Merge into (never clobber) latest — a filtered run must not shrink the map --skip-research uses.
                latest_path = OUTPUT_DIR / "benchmark_research_latest.json"
                merged = json.loads(latest_path.read_text()) if latest_path.exists() else {}
                merged.update(research_map)
                latest_path.write_text(json.dumps(merged, indent=1))
                research_mode = "fresh"

            records = []
            for bq in selected:
                logger.info(f"--- RUN {bq.id} ---")
                run_record = run_agent_for_query(bq, args.ingest_missing)
                research = research_map.get(bq.id) or {"status": "empty", "sources": []}
                # intent_match needs no references — an empty research set must not fail correct routing.
                base_checks = {}
                if run_record["status"] == "ok":
                    intent_actual = (run_record.get("intent_actual") or {}).get("intent")
                    base_checks = {"intent_match": intent_actual == bq.intent_expectation,
                                   "intent_actual": intent_actual}
                rec = {"id": bq.id, "query": bq.query, "ticker": bq.ticker,
                       "intent_expectation": bq.intent_expectation,
                       "run": run_record, "research": research,
                       "deterministic_checks": base_checks, "judge": {"status": "not_judged", "verdict": None}}
                if run_record["status"] == "skipped_not_ingested":
                    logger.warning(f"{bq.id}: ticker {bq.ticker} not ingested — run with --ingest-missing "
                                   f"or ingest via the app first")
                elif run_record["status"] == "ok":
                    if research["sources"]:
                        logger.info(f"--- JUDGE {bq.id} ---")
                        checks, judge = judge_query(bq, run_record, research["sources"])
                        rec["deterministic_checks"], rec["judge"] = checks, judge
                    else:
                        rec["judge"] = {"status": "no_references", "verdict": None}
                        logger.warning(f"{bq.id}: no references gathered — judge skipped")
                records.append(rec)

        report = build_report(records, research_mode, snapshot_name, base_report)
        summary_path = OUTPUT_DIR / "benchmark_summary.md"
        if args.rejudge:
            # A rejudge must not overwrite the canonical report — its agent outputs are historical.
            summary_path = OUTPUT_DIR / f"benchmark_summary_rejudge_{ts}.md"
            (OUTPUT_DIR / f"benchmark_report_rejudge_{ts}.json").write_text(json.dumps(report, indent=1, default=str))
            write_markdown(report, summary_path)
        else:
            (OUTPUT_DIR / "benchmark_report.json").write_text(json.dumps(report, indent=1, default=str))
            (OUTPUT_DIR / f"benchmark_report_{ts}.json").write_text(json.dumps(report, indent=1, default=str))
            write_markdown(report, OUTPUT_DIR / "benchmark_summary.md")
            write_markdown(report, OUTPUT_DIR / f"benchmark_summary_{ts}.md")

        if args.compare:
            compare_reports(report, json.loads(args.compare.read_text()))

        agg = report["aggregate"]
        print(f"\n=== Analyst benchmark: {agg['queries_scored']}/{agg['queries_total']} scored ===")
        for d, v in agg["dimension_means"].items():
            print(f"  {d:22s} {v if v is not None else 'n/a'}")
        det = agg["deterministic"]
        print(f"  intent {det['intent_match']} · numeric {det['numeric_match']} · consensus {det['consensus']}")
        print(f"  report: {summary_path}")

        # A rejudge can never re-run agents, so agent_error records are historical — only judge failures count.
        failed = [q for q in records
                  if (q["run"]["status"] == "agent_error" and not args.rejudge)
                  or q["judge"].get("status") == "judge_failed"]
        for q in failed:
            logger.error(f"{q['id']}: run={q['run']['status']} judge={q['judge'].get('status')} "
                         f"error={q['run'].get('error') or q['judge'].get('error')}")
        return 1 if failed else 0
    finally:
        from llm import flush_traces

        flush_traces()


if __name__ == "__main__":
    sys.exit(main())
