# Network-free unit tests for pure logic that has bitten us before: unit conversion, risk
# weighting, the benchmark's numeric comparators, comparison-trigger routing, and the rate limiter.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- to_usd (tools/extraction_tools.py) --------------------------------------


def test_to_usd_converts_by_cited_unit():
    from tools.extraction_tools import to_usd

    assert to_usd(94.8, "billions") == 94_800_000_000
    assert to_usd(27_843, "millions") == 27_843_000_000
    assert to_usd(500, "thousands") == 500_000
    assert to_usd(42.5, "dollars") == 42.5
    assert to_usd(None, "millions") is None


def test_to_usd_rejects_unknown_unit():
    from tools.extraction_tools import to_usd

    with pytest.raises(ValueError):
        to_usd(1, "lakhs")


# --- risk weighting (tools/risk_tools.py) ------------------------------------


def test_risk_weights_sum_to_one():
    from tools.risk_tools import WEIGHTS

    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9


def test_overall_risk_weighted_average_and_tiers():
    from tools.risk_tools import WEIGHTS, compute_overall_score

    # Uniform component scores must produce that same score; tiers: <4 low, <7 medium, else high.
    assert compute_overall_score(dict.fromkeys(WEIGHTS, 1)) == (1.0, "low")
    assert compute_overall_score(dict.fromkeys(WEIGHTS, 4)) == (4.0, "medium")
    assert compute_overall_score(dict.fromkeys(WEIGHTS, 7)) == (7.0, "high")
    assert compute_overall_score(dict.fromkeys(WEIGHTS, 10)) == (10.0, "high")


# --- benchmark numeric comparators (evals/benchmark_judge.py) -----------------


def test_yoy_comparison_normalizes_fraction_vs_percent():
    from evals.benchmark_judge import _compare_yoy

    assert _compare_yoy(0.16, 16.0) == "match"  # stored fraction vs source percent
    assert _compare_yoy(16.0, 0.16) == "match"
    assert _compare_yoy(16.0, 16.5) == "match"  # within the 1.5pp tolerance
    assert _compare_yoy(16.0, 26.0) == "mismatch"
    assert _compare_yoy(None, 16.0) == "not_comparable"


def test_periods_differ_on_structured_fields():
    from evals.benchmark_judge import _periods_differ

    q = {"period_end_year": 2026, "period_end_month": 6, "period_scope": "quarter"}
    assert not _periods_differ(q, dict(q))
    # A fiscal year and its final quarter share an end month — scope must separate them.
    assert _periods_differ(q, {**q, "period_scope": "year"})
    assert _periods_differ(q, {**q, "period_end_year": 2025})
    # Unstated fields never count as a difference.
    assert not _periods_differ(q, {"period_end_year": None, "period_end_month": None, "period_scope": None})


# --- comparison-trigger routing (agents/comparison.py) ------------------------


def test_comparison_trigger_detection():
    from agents.comparison import is_comparison_query

    assert is_comparison_query("How has MSFT margin trended over time?")
    assert is_comparison_query("Compare Q1 vs Q2 revenue for Tesla")
    assert is_comparison_query("NVDA year-over-year growth")
    assert not is_comparison_query("Should I buy Apple?")
    assert not is_comparison_query("What are NVIDIA's biggest risks?")


# --- rate limiter (api/rate_limit.py) -----------------------------------------


class _FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops = []

    def incr(self, key):
        self.ops.append(("incr", key))
        return self

    def expire(self, key, ttl):
        self.ops.append(("expire", key))
        return self

    def execute(self):
        results = []
        for op, key in self.ops:
            if op == "incr":
                self.redis.counts[key] = self.redis.counts.get(key, 0) + 1
                results.append(self.redis.counts[key])
            else:
                results.append(True)
        return results


class _FakeRedis:
    def __init__(self):
        self.counts = {}

    def pipeline(self):
        return _FakePipeline(self)


class _BrokenRedis:
    def pipeline(self):
        raise ConnectionError("redis down")


def test_rate_limiter_per_ip_window():
    from api import rate_limit

    r = _FakeRedis()
    for _ in range(rate_limit.PER_IP_PER_MINUTE):
        assert rate_limit.evaluate(r, "1.2.3.4", "chat", now=1000) is None
    assert rate_limit.evaluate(r, "1.2.3.4", "chat", now=1000) == rate_limit.PER_IP_MESSAGE
    # A different IP in the same window is unaffected; a new minute resets the caller.
    assert rate_limit.evaluate(r, "5.6.7.8", "chat", now=1000) is None
    assert rate_limit.evaluate(r, "1.2.3.4", "chat", now=1060) is None


def test_rate_limiter_daily_budget(monkeypatch):
    from api import rate_limit

    r = _FakeRedis()
    monkeypatch.setattr(rate_limit, "DAILY_GLOBAL_BUDGET", 3)
    for i in range(3):
        assert rate_limit.evaluate(r, f"ip{i}", "chat", now=1000 + i * 60) is None
    assert rate_limit.evaluate(r, "ip9", "chat", now=9000) == rate_limit.DAILY_MESSAGE


def test_rate_limiter_fails_open_when_redis_down():
    from api import rate_limit

    assert rate_limit.evaluate(_BrokenRedis(), "1.2.3.4", "chat") is None
