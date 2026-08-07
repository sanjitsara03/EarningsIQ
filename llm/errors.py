# Typed error taxonomy for all agent LLM calls. One import point for api/routes/chat.py and RQ:
# adapter-level errors (transport/format) and agent-level errors (semantic guards) both live here.


# Base class for all LLM-layer failures.
class LLMError(Exception):
    pass


# Tool-call arguments were not valid JSON, even after one retry of the identical request.
class ToolArgumentError(LLMError):
    def __init__(self, agent: str, tool_name: str, raw_arguments: str):
        self.agent = agent
        self.tool_name = tool_name
        self.raw_arguments = raw_arguments
        super().__init__(f"[{agent}] tool '{tool_name}' returned unparseable arguments: {raw_arguments[:200]!r}")


# A forced tool_choice ("required" or a named function) came back with zero tool calls.
class NoToolCallError(LLMError):
    def __init__(self, agent: str, tool_choice):
        self.agent = agent
        self.tool_choice = tool_choice
        super().__init__(f"[{agent}] forced tool_choice={tool_choice!r} returned no tool calls")


# Structured output failed: truncated (finish_reason=length), unparseable JSON after one retry,
# or Pydantic validation failure after one corrective retry.
class LLMOutputError(LLMError):
    def __init__(self, agent: str, reason: str, raw_text: str | None = None):
        self.agent = agent
        self.reason = reason
        self.raw_text = raw_text
        super().__init__(f"[{agent}] structured output failed: {reason}")


# The model called a different tool than the one this step forced.
class WrongToolError(LLMError):
    def __init__(self, agent: str, expected: str, got: str):
        self.agent = agent
        self.expected = expected
        self.got = got
        super().__init__(f"[{agent}] expected tool '{expected}' but model called '{got}'")


# A tool loop stopped making progress (consecutive turns without usable tool calls or text).
class LoopStallError(LLMError):
    def __init__(self, agent: str, detail: str):
        self.agent = agent
        super().__init__(f"[{agent}] loop stalled: {detail}")
