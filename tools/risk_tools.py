# Risk Scoring Agent tools; the full list is declared each request, each step's tool forced by named tool_choice.

TOOLS = [
    {
        "name": "fetch_historical_signals",
        "description": (
            "Fetch historical signals for this company from the database. "
            "Call this first to establish a baseline before scoring any risk components."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "The company's stock ticker, e.g. 'AAPL'.",
                }
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "score_risk_components",
        "description": (
            "Score each of the five risk components on a scale of 1–10 based on the current filing "
            "and the historical baseline you just retrieved. 1 = minimal risk, 10 = critical risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "guidance_cut_risk": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Risk that management cut or withdrew guidance vs prior periods.",
                },
                "demand_uncertainty": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Uncertainty in forward demand based on management language and trends.",
                },
                "margin_pressure": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Degree of gross/operating margin compression vs historical baseline.",
                },
                "competitive_threat": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Intensity of competitive threats based on risk factors language.",
                },
                "macro_exposure": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Exposure to macroeconomic headwinds such as rates, FX, or geopolitical risk.",
                },
            },
            "required": [
                "guidance_cut_risk",
                "demand_uncertainty",
                "margin_pressure",
                "competitive_threat",
                "macro_exposure",
            ],
        },
    },
    # Python recomputes the weighted average — the LLM's own overall score is never trusted.
    {
        "name": "finalize_overall_risk",
        "description": (
            "Write a concise executive summary of the overall risk assessment. "
            "3–5 sentences citing specific evidence from the filing and historical comparison."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "Plain English risk assessment citing specific evidence.",
                }
            },
            "required": ["executive_summary"],
        },
    },
]

# The strict call order the agent enforces — one forced tool per step.
STEP_ORDER = ["fetch_historical_signals", "score_risk_components", "finalize_overall_risk"]

WEIGHTS = {
    "guidance_cut_risk": 0.30,
    "demand_uncertainty": 0.25,
    "margin_pressure": 0.20,
    "competitive_threat": 0.15,
    "macro_exposure": 0.10,
}


# Computes weighted average from component scores and returns (overall_score, risk_tier).
def compute_overall_score(scores: dict) -> tuple[float, str]:
    overall = sum(scores[k] * w for k, w in WEIGHTS.items())
    if overall < 4:
        tier = "low"
    elif overall < 7:
        tier = "medium"
    else:
        tier = "high"
    return round(overall, 2), tier


def handle_score_risk_components(input: dict) -> dict:
    return input


def handle_finalize_overall_risk(input: dict) -> str:
    return input["executive_summary"]
