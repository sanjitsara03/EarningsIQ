# Single source of truth for which model serves each agent; the capability harness probes these configs.
import json
import os
from dataclasses import dataclass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# `provider` is OpenRouter's routing object (sent via extra_body); `extra` holds additional extra_body keys.
@dataclass(frozen=True)
class AgentLLMConfig:
    model: str
    max_tokens: int
    provider: dict | None = None
    extra: dict | None = None


# Risk scoring pins the first-party OpenAI provider with fallbacks disabled.
AGENT_MODELS: dict[str, AgentLLMConfig] = {
    "orchestrator": AgentLLMConfig(
        model="google/gemini-3-flash-preview",
        max_tokens=1024,
        provider={"require_parameters": True},
    ),
    "web_search": AgentLLMConfig(
        model="google/gemini-3-flash-preview",
        max_tokens=1024,
    ),
    "extraction": AgentLLMConfig(
        model="google/gemini-3-flash-preview",
        max_tokens=8192,
        provider={"require_parameters": True},
    ),
    "risk_scoring": AgentLLMConfig(
        model="openai/gpt-5.2",
        max_tokens=8192,  # includes reasoning tokens
        provider={"order": ["openai"], "allow_fallbacks": False, "require_parameters": True},
        extra={"reasoning": {"effort": "low"}},
    ),
    "comparison": AgentLLMConfig(
        model="anthropic/claude-sonnet-4.6",
        max_tokens=4096,
        provider={"require_parameters": True},
    ),
    "advice": AgentLLMConfig(
        model="anthropic/claude-sonnet-4.6",
        max_tokens=1024,
        provider={"require_parameters": True},
    ),
    # Benchmark roles; the judge is a different model family than the sonnet-written outputs it grades.
    "benchmark_judge": AgentLLMConfig(
        model="openai/gpt-5.2",
        max_tokens=8192,  # includes reasoning tokens
        provider={"order": ["openai"], "allow_fallbacks": False, "require_parameters": True},
        extra={"reasoning": {"effort": "low"}},
    ),
    "benchmark_extractor": AgentLLMConfig(
        model="google/gemini-3-flash-preview",
        max_tokens=1024,
        provider={"require_parameters": True},
    ),
}


# Applies LLM_*_<AGENT> env overrides; a model override drops the stored provider pin and extra.
def get_agent_config(agent: str) -> AgentLLMConfig:
    cfg = AGENT_MODELS[agent]
    key = agent.upper()

    model_override = os.getenv(f"LLM_MODEL_{key}")
    provider_override = os.getenv(f"LLM_PROVIDER_{key}")
    max_tokens_override = os.getenv(f"LLM_MAX_TOKENS_{key}")

    if not (model_override or provider_override or max_tokens_override):
        return cfg

    try:
        provider = json.loads(provider_override) if provider_override else (None if model_override else cfg.provider)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM_PROVIDER_{key} is not valid JSON: {e}") from e
    try:
        max_tokens = int(max_tokens_override) if max_tokens_override else cfg.max_tokens
    except ValueError as e:
        raise RuntimeError(f"LLM_MAX_TOKENS_{key} is not an integer: {max_tokens_override!r}") from e

    return AgentLLMConfig(
        model=model_override or cfg.model,
        max_tokens=max_tokens,
        provider=provider,
        extra=None if model_override else cfg.extra,
    )
