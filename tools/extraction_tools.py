# Tool schemas and handlers for the Extraction Agent.
# TOOLS is the provider-neutral flat shape; agents convert it via llm.to_openai_tools().
# TOOL_HANDLERS maps tool name → function that processes the LLM's input and returns stored data.

TOOLS = [
    {
        "name": "extract_financial_metrics",
        "description": (
            "Extract core financial metrics from the filing. "
            "Call this once after reading the financial statements and MD&A."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "description": "Reporting period, e.g. 'Q1 FY2025' or 'FY2024'.",
                },
                "revenue": {
                    "type": "number",
                    "description": "Total revenue in millions USD.",
                },
                "eps": {
                    "type": "number",
                    "description": "Diluted earnings per share.",
                },
                "gross_margin": {
                    "type": "number",
                    "description": "Gross margin as a percentage, e.g. 45.2.",
                },
                "operating_margin": {
                    "type": "number",
                    "description": "Operating margin as a percentage.",
                },
                "revenue_yoy_delta": {
                    "type": "number",
                    "description": "Revenue year-over-year change as a percentage.",
                },
            },
            "required": ["period", "revenue"],
        },
    },
    {
        "name": "extract_risk_factors",
        "description": (
            "Extract risk signals from the Risk Factors section. "
            "Return between 1 and 10 risks, each with a verbatim quote."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "risks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Risk category, e.g. 'macro', 'competitive', 'regulatory'.",
                            },
                            "label": {
                                "type": "string",
                                "description": "Short label for the risk, e.g. 'Rising interest rates'.",
                            },
                            "severity": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 10,
                                "description": "Severity score from 1 (low) to 10 (critical).",
                            },
                            "verbatim_quote": {
                                "type": "string",
                                "description": "Exact sentence or phrase from the filing.",
                            },
                        },
                        "required": ["category", "label", "severity", "verbatim_quote"],
                    },
                }
            },
            "required": ["risks"],
        },
    },
    {
        "name": "extract_segment_performance",
        "description": (
            "Extract business segment revenue breakdown. "
            "Call this once after reading the segment or geographic breakdown in the filing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Segment name, e.g. 'Intelligent Cloud'.",
                            },
                            "revenue": {
                                "type": "number",
                                "description": "Segment revenue in millions USD.",
                            },
                            "growth": {
                                "type": "number",
                                "description": "YoY revenue growth as a percentage.",
                            },
                        },
                        "required": ["name", "revenue"],
                    },
                }
            },
            "required": ["segments"],
        },
    },
    {
        "name": "extract_management_outlook",
        "description": (
            "Extract forward-looking guidance from the MD&A section. "
            "If guidance was withdrawn, set withdrawn=true and omit guidance_revenue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "guidance_revenue": {
                    "type": "number",
                    "description": "Guided revenue for the next period in millions USD. Omit if withdrawn.",
                },
                "guidance_period": {
                    "type": "string",
                    "description": "The period the guidance applies to, e.g. 'Q2 FY2025'.",
                },
                "withdrawn": {
                    "type": "boolean",
                    "description": "True if management explicitly withdrew or declined to give guidance.",
                },
                "verbatim_quote": {
                    "type": "string",
                    "description": "Exact sentence or phrase from the filing supporting this outlook.",
                },
            },
            "required": ["guidance_period", "withdrawn", "verbatim_quote"],
        },
    },
    {
        "name": "extract_notable_changes",
        "description": (
            "Extract metrics that management explicitly called out as changing meaningfully QoQ or YoY. "
            "Each change must include a verbatim quote."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "description": "Name of the metric, e.g. 'Cloud revenue'.",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down"],
                            },
                            "magnitude": {
                                "type": "string",
                                "description": "How much it changed, e.g. '12%' or '$2.1B'.",
                            },
                            "verbatim_quote": {
                                "type": "string",
                                "description": "Exact sentence or phrase from the filing.",
                            },
                        },
                        "required": ["metric", "direction", "magnitude", "verbatim_quote"],
                    },
                }
            },
            "required": ["changes"],
        },
    },
]


# Each handler receives the raw input dict from the LLM tool call and returns it unchanged.
# The extraction agent accumulates these into a results dict keyed by tool name.

def handle_extract_financial_metrics(input: dict) -> dict:
    return input


def handle_extract_risk_factors(input: dict) -> dict:
    return input["risks"]


def handle_extract_segment_performance(input: dict) -> dict:
    return input["segments"]


def handle_extract_management_outlook(input: dict) -> dict:
    return input


def handle_extract_notable_changes(input: dict) -> dict:
    return input["changes"]


TOOL_HANDLERS = {
    "extract_financial_metrics": handle_extract_financial_metrics,
    "extract_risk_factors": handle_extract_risk_factors,
    "extract_segment_performance": handle_extract_segment_performance,
    "extract_management_outlook": handle_extract_management_outlook,
    "extract_notable_changes": handle_extract_notable_changes,
}
