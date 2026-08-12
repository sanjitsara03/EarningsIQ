# Analyst-benchmark query set — each query runs through the real agent pipeline.
# Comparison wording must match COMPARISON_TRIGGERS (agents/comparison.py) and single_analysis
# wording must avoid those trigger words, or the chat routing misroutes them.
from dataclasses import dataclass, field

INTENTS = {"single_analysis", "comparison", "advice", "web_only"}


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    query: str
    ticker: str | None
    filing_type: str
    intent_expectation: str
    judge_focus: str
    research_queries: tuple = field(default_factory=tuple)


# Three research templates per ticker: results coverage, consensus ratings, recommendation.
def default_research_queries(ticker: str) -> list[str]:
    return [
        f"{ticker} latest quarterly earnings results revenue EPS",
        f"{ticker} stock analyst ratings consensus price target",
        f"{ticker} stock buy hold sell analyst recommendation",
    ]


QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        id="aapl_advice", query="Should I buy Apple stock right now?",
        ticker="AAPL", filing_type="10-Q", intent_expectation="advice",
        judge_focus="Weight the recommendation vs street consensus, and coverage of the valuation and AI narrative.",
    ),
    BenchmarkQuery(
        id="msft_advice", query="Is Microsoft a good investment after its latest quarterly results?",
        ticker="MSFT", filing_type="10-Q", intent_expectation="advice",
        judge_focus="Weight Azure/cloud growth and capex commentary coverage.",
    ),
    BenchmarkQuery(
        id="googl_advice", query="Buy, hold, or sell Google?",
        ticker="GOOGL", filing_type="10-Q", intent_expectation="advice",
        judge_focus="Weight how search/AI competition is framed relative to professional coverage.",
    ),
    BenchmarkQuery(
        id="orcl_advice", query="Should I invest in Oracle based on its latest filing?",
        ticker="ORCL", filing_type="10-Q", intent_expectation="advice",
        judge_focus="Fresh-ingest ticker. Weight cloud backlog (RPO) coverage.",
    ),
    BenchmarkQuery(
        id="nvda_single", query="How did NVIDIA do in its most recent quarterly filing?",
        ticker="NVDA", filing_type="10-Q", intent_expectation="single_analysis",
        judge_focus="Weight data-center revenue and growth-rate accuracy — numeric-check friendly.",
    ),
    BenchmarkQuery(
        id="aapl_single", query="What did Apple report for revenue and EPS in its most recent quarter?",
        ticker="AAPL", filing_type="10-Q", intent_expectation="single_analysis",
        judge_focus="Pure numeric cross-check: revenue and diluted EPS must match professional coverage.",
    ),
    BenchmarkQuery(
        id="googl_single", query="What are the biggest risks in Alphabet's latest quarterly filing?",
        ticker="GOOGL", filing_type="10-Q", intent_expectation="single_analysis",
        judge_focus="Risk coverage vs professional risk commentary; filings-only risks are legitimate.",
    ),
    BenchmarkQuery(
        id="msft_single", query="Summarize Microsoft's most recent quarterly performance.",
        ticker="MSFT", filing_type="10-Q", intent_expectation="single_analysis",
        judge_focus="Balanced coverage: revenue, margins, segments, guidance.",
    ),
    BenchmarkQuery(
        id="msft_comparison", query="How has Microsoft's revenue changed year over year?",
        ticker="MSFT", filing_type="10-Q", intent_expectation="comparison",
        judge_focus="YoY revenue figure accuracy vs professional coverage.",
    ),
    BenchmarkQuery(
        id="nvda_comparison", query="Compare NVIDIA's gross margin last quarter vs the prior quarter.",
        ticker="NVDA", filing_type="10-Q", intent_expectation="comparison",
        judge_focus="Margin direction and magnitude accuracy.",
    ),
    BenchmarkQuery(
        id="nvda_web", query="What is the latest analyst sentiment on NVIDIA stock?",
        ticker="NVDA", filing_type="10-Q", intent_expectation="web_only",
        judge_focus="Freshness: items should be dated and current; penalize stale training-data facts.",
    ),
    BenchmarkQuery(
        id="aapl_web", query="What's the current news around Apple and AI?",
        ticker="AAPL", filing_type="10-Q", intent_expectation="web_only",
        judge_focus="Coverage of the professional news set; dated citations.",
    ),
]

assert len({q.id for q in QUERIES}) == len(QUERIES), "benchmark query ids must be unique"
assert all(q.intent_expectation in INTENTS for q in QUERIES), "unknown intent_expectation"
