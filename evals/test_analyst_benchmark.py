# Pytest wrapper for the analyst benchmark. Excluded from bare `pytest` (see pyproject addopts)
# because it runs the full agent pipeline + Tavily research + LLM judge; run explicitly with:
# pytest -m benchmark
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.analyst_benchmark import judge_query, research_query, run_agent_for_query
from evals.benchmark_queries import QUERIES

pytestmark = pytest.mark.benchmark

# One cheap query per major route.
SMOKE_IDS = ["aapl_single", "aapl_advice", "nvda_web"]
DIMENSIONS = ["factual_accuracy", "coverage", "grounding", "directional_agreement"]


@pytest.fixture(scope="module")
def records():
    out = []
    for bq in [q for q in QUERIES if q.id in SMOKE_IDS]:
        run_record = run_agent_for_query(bq, ingest_missing=False)
        research = research_query(bq)
        checks, judge = ({}, {"status": "no_references", "verdict": None})
        if run_record["status"] == "ok" and research["sources"]:
            checks, judge = judge_query(bq, run_record, research["sources"])
        out.append({"id": bq.id, "run": run_record, "research": research,
                    "deterministic_checks": checks, "judge": judge})
    return out


def test_all_smoke_queries_ran(records):
    assert len(records) == len(SMOKE_IDS)
    errors = [r["id"] for r in records if r["run"]["status"] == "agent_error"]
    assert not errors, f"agent errors: {errors}"


def test_judged_records_have_full_verdicts(records):
    judged = [r for r in records if r["judge"]["status"] == "ok"]
    assert judged, "no records were judged — research or judge failed for every query"
    for r in judged:
        verdict = r["judge"]["verdict"]
        for d in DIMENSIONS:
            assert 1 <= verdict[d]["score"] <= 5
            assert verdict[d]["evidence"], f"{r['id']}: {d} has no evidence"
