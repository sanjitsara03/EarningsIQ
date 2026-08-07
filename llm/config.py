# Per-agent model configuration for OpenRouter. This is the single source of truth for which
# model serves each agent — the capability harness in evals/ probes exactly these configs.
import json
import os
from dataclasses import dataclass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Frozen per-agent LLM settings. `provider` is OpenRouter's routing object (sent via extra_body);
# `extra` holds additional extra_body keys (e.g. reasoning effort for OpenAI models).
@dataclass(frozen=True)
class AgentLLMConfig:
    model: str
    max_tokens: int
    provider: dict | None = None
    extra: dict | None = None


# Per-agent model registry. Risk scoring pins the first-party OpenAI provider with fallbacks disabled.
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
}


# Resolves the config for an agent, applying env overrides:
# LLM_MODEL_<AGENT>, LLM_MAX_TOKENS_<AGENT>, LLM_PROVIDER_<AGENT> (JSON string).
# A model override drops the stored provider pin and extra unless LLM_PROVIDER_<AGENT> is also set.
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
