# Pytest wrapper for the capability harness. Excluded from plain `pytest` runs (see pyproject
# addopts) because it makes real OpenRouter calls; run explicitly with: pytest -m capability
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.capability_harness import run_probes

pytestmark = pytest.mark.capability

GATE_AGENTS = ["orchestrator", "advice", "web_search", "extraction", "comparison", "risk_scoring"]


# One full harness run shared across all gate assertions.
@pytest.fixture(scope="module")
def report():
    return run_probes()


@pytest.mark.parametrize("agent", GATE_AGENTS)
def test_gate(report, agent):
    verdict = report["summary"]["gates"].get(agent)
    assert verdict is not None, f"no gate verdict recorded for {agent}"
    assert verdict in ("pass", "pass_named", "pass_fallback_only"), f"{agent} gate: {verdict}"
