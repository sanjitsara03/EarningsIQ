# Promptfoo provider for the Advice agent.
# Accepts pre-built signals and risk data as vars so no DB is needed.

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.advice import run_advice


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        vars = context.get("vars", {})
        ticker = vars["ticker"]
        extracted_signals = json.loads(vars["extracted_signals"])
        risk_result = json.loads(vars["risk_result"])
        web_summary = vars.get("web_summary")

        result = run_advice(
            ticker, extracted_signals, risk_result, web_summary,
            filing_type=vars.get("filing_type"), period=vars.get("period"),
        )
        return {"output": json.dumps(result.model_dump())}
    except Exception as e:
        return {"error": str(e)}
