# The single boundary between agents and the LLM provider — agents call OpenRouter via chat()/chat_json()
# and never touch the SDK directly.
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import openai
from dotenv import load_dotenv
from openai import OpenAI

from llm.config import OPENROUTER_BASE_URL, get_agent_config
from llm.errors import LLMError, LLMOutputError, NoToolCallError, ToolArgumentError
from llm.tracing import traced_openai

load_dotenv()
logger = logging.getLogger(__name__)

# Per-agent request timeouts (seconds) where the default 60s isn't enough.
_TIMEOUTS = {"extraction": 120.0}

# Lazy singleton: importing never requires the key, and forked RQ work-horses build their own client.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set — required for all agent LLM calls.")
        _client = traced_openai(
            OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=api_key,
                default_headers={"X-Title": "EarningsAgentIQ"},
                max_retries=2,
                timeout=60.0,
            )
        )
    return _client


# One tool call, normalized; args is the parsed arguments dict.
@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict


# Normalized response; raw_message keeps the ChatCompletionMessage for the assistant echo.
@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    model: str = ""
    provider: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    raw_message: Any = None


# Converts flat tool schemas ({name, description, input_schema}) to the OpenAI function wrapper.
def to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


# The OpenAI shape for forcing one specific named tool.
def named_tool_choice(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


# Assistant-turn echo for tool loops; provider fields stripped, must carry content or tool_calls.
def assistant_msg_from_response(resp: LLMResponse) -> dict:
    msg = resp.raw_message.model_dump(include={"role", "content", "tool_calls"}, exclude_none=True)
    if "tool_calls" not in msg and not msg.get("content"):
        msg["content"] = ""
    return msg


# One {"role": "tool"} message per call id — every id in the preceding assistant message must be answered.
def tool_result_msgs(results: list[tuple[str, str]]) -> list[dict]:
    return [{"role": "tool", "tool_call_id": call_id, "content": content} for call_id, content in results]


# Internal marker for a tool call whose arguments were not a valid JSON object.
class _MalformedArgs(Exception):
    def __init__(self, name: str, raw: str):
        self.name = name
        self.raw = raw


# Parses tool-call arguments; empty string means a no-arg call. Non-object JSON is invalid.
def _parse_args(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("tool arguments are not a JSON object", raw, 0)
    return parsed


# Builds the normalized response from a completion; validates OpenRouter's 200-with-error quirk.
def _normalize(agent: str, completion) -> LLMResponse:
    if not getattr(completion, "choices", None):
        err = getattr(completion, "error", None) or getattr(completion, "model_extra", {}).get("error")
        raise LLMError(f"[{agent}] provider returned no choices (error: {err})")

    choice = completion.choices[0]
    msg = choice.message
    tool_calls = []
    for tc in msg.tool_calls or []:
        try:
            args = _parse_args(tc.function.arguments)
        except json.JSONDecodeError:
            raise _MalformedArgs(tc.function.name, tc.function.arguments)
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
    usage = getattr(completion, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return LLMResponse(
        text=msg.content,
        tool_calls=tool_calls,
        finish_reason=choice.finish_reason or "",
        model=completion.model or "",
        provider=str((getattr(completion, "model_extra", None) or {}).get("provider", "")),
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_tokens=getattr(details, "cached_tokens", 0) or 0,
        raw_message=msg,
    )


# Core call; messages must NOT contain a system entry — pass system= instead. One retry on malformed tool args.
def chat(
    agent: str,
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
    response_format: dict | None = None,
    parallel_tool_calls: bool | None = None,
) -> LLMResponse:
    cfg = get_agent_config(agent)

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "messages": ([{"role": "system", "content": system}] if system else []) + messages,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    if parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = parallel_tool_calls

    extra_body: dict[str, Any] = {}
    if cfg.provider:
        extra_body["provider"] = cfg.provider
    if cfg.extra:
        extra_body.update(cfg.extra)
    if extra_body:
        kwargs["extra_body"] = extra_body

    # Timeouts are passed per request; with_options() clones the client without the LangSmith wrapper.
    if agent in _TIMEOUTS:
        kwargs["timeout"] = _TIMEOUTS[agent]

    client = _get_client()
    last_malformed: _MalformedArgs | None = None
    for attempt in range(2):
        start = time.monotonic()
        try:
            completion = client.chat.completions.create(**kwargs)
        except openai.OpenAIError as e:
            raise LLMError(f"[{agent}] provider call failed: {type(e).__name__}: {e}") from e
        latency_ms = int((time.monotonic() - start) * 1000)

        finish = completion.choices[0].finish_reason if getattr(completion, "choices", None) else "?"
        provider = (getattr(completion, "model_extra", None) or {}).get("provider", "?")
        usage = getattr(completion, "usage", None)
        logger.info(
            f"llm agent={agent} model={cfg.model} served_by={provider} finish={finish} "
            f"tokens={getattr(usage, 'prompt_tokens', '?')}/{getattr(usage, 'completion_tokens', '?')} "
            f"latency_ms={latency_ms}"
        )

        try:
            resp = _normalize(agent, completion)
        except _MalformedArgs as e:
            # Truncated arguments raise immediately; a retry cannot repair them.
            if finish == "length":
                raise LLMOutputError(
                    agent, f"tool call '{e.name}' truncated (finish_reason=length) — raise max_tokens", e.raw
                )
            last_malformed = e
            logger.warning(f"[{agent}] malformed arguments for tool '{e.name}' (attempt {attempt + 1}) — retrying")
            continue

        forced = tool_choice == "required" or isinstance(tool_choice, dict)
        if forced and not resp.tool_calls:
            raise NoToolCallError(agent, tool_choice)
        return resp

    raise ToolArgumentError(agent, last_malformed.name, last_malformed.raw)


# Structured-output call (strict json_schema → dict); one retry on unparseable output, truncation raises.
def chat_json(
    agent: str,
    messages: list[dict],
    *,
    system: str | None = None,
    schema_name: str,
    schema: dict,
) -> dict:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
    }

    last_text: str | None = None
    for attempt in range(2):
        resp = chat(agent, messages, system=system, response_format=response_format)
        if resp.finish_reason == "length":
            raise LLMOutputError(agent, "output truncated (finish_reason=length) — raise max_tokens", resp.text)
        try:
            parsed = json.loads(resp.text or "")
            # A top-level array/string/number is valid JSON but not a valid structured output.
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        last_text = resp.text
        logger.warning(f"[{agent}] structured output not a valid JSON object (attempt {attempt + 1}) — retrying")

    raise LLMOutputError(agent, "output was not a valid JSON object after retry", last_text)
