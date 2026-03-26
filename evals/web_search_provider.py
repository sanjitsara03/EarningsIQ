# Promptfoo provider for the Web Search agent.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.web_search import run_web_search


def call_api(prompt: str, options: dict, context: dict) -> dict:
    try:
        result = run_web_search(prompt)
        return {"output": result}
    except Exception as e:
        return {"error": str(e)}
