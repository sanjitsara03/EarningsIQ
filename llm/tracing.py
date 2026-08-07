# Optional LangSmith tracing. No-op unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.
# One shared Client backs the OpenAI wrapper and every @traced span; flush_traces() drains its queue.
import logging
import os

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_ls_client = None


# True when the env opts into tracing. Checked once per process at first use.
def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


# Lazily constructs the shared LangSmith client.
def get_ls_client():
    global _ls_client
    if not tracing_enabled():
        return None
    if _ls_client is None:
        from langsmith import Client

        _ls_client = Client()
    return _ls_client


# Wraps an OpenAI client so every chat call is traced; pass-through when tracing is off.
def traced_openai(client):
    ls = get_ls_client()
    if ls is None:
        return client
    from langsmith.wrappers import wrap_openai

    return wrap_openai(client, tracing_extra={"client": ls})


# Decorator for agent entry points: @traced("orchestrator"). Identity when tracing is off.
def traced(name: str):
    def decorator(fn):
        ls = get_ls_client()
        if ls is None:
            return fn
        from langsmith import traceable

        return traceable(name=name, client=ls)(fn)

    return decorator


# Best-effort drain of the trace queue. Call at the end of RQ jobs; safe to call when tracing is off.
def flush_traces() -> None:
    if _ls_client is None:
        return
    try:
        _ls_client.flush()
    except Exception as e:
        logger.warning(f"LangSmith flush failed: {e}")
