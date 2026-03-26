# Promptfoo Python provider for the orchestrator agent.
# Receives the user query and returns the raw JSON string output from run_orchestrator.

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.orchestrator import run_orchestrator


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        result = run_orchestrator(prompt)
        return {"output": json.dumps(result)}
    except Exception as e:
        return {"error": str(e)}
