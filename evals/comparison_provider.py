# Promptfoo provider for the Comparison agent.

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.comparison import run_comparison


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        result = run_comparison(prompt)
        return {"output": json.dumps(result, default=str)}
    except Exception as e:
        return {"error": str(e)}
