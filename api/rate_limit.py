# Two-layer Redis limiter (per-IP/minute + global daily) for the public LLM-spend endpoints; fails open
# when Redis is unreachable — a broken limiter must never take the demo down.

import logging
import os
import time
from datetime import date

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

PER_IP_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_IP_PER_MIN", "8"))
DAILY_GLOBAL_BUDGET = int(os.environ.get("RATE_LIMIT_DAILY_GLOBAL", "400"))

DAILY_MESSAGE = "The demo's daily analysis budget is exhausted — please come back tomorrow."
PER_IP_MESSAGE = "Too many requests — please wait a minute and try again."


# Best-effort client IP: Railway (like most proxies) puts the real client first in X-Forwarded-For.
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Core check, split from FastAPI for testability; None = allowed (including on Redis failure).
def evaluate(redis, ip: str, scope: str, now: float | None = None) -> str | None:
    now = now if now is not None else time.time()
    minute = int(now // 60)
    try:
        pipe = redis.pipeline()
        pipe.incr(f"rl:{scope}:{ip}:{minute}")
        pipe.expire(f"rl:{scope}:{ip}:{minute}", 120)
        pipe.incr(f"rl:daily:{date.today().isoformat()}")
        pipe.expire(f"rl:daily:{date.today().isoformat()}", 172_800)
        ip_count, _, day_count, _ = pipe.execute()
    except Exception as e:
        logger.warning(f"Rate limiter unavailable ({type(e).__name__}: {e}) — allowing request")
        return None
    if day_count > DAILY_GLOBAL_BUDGET:
        return DAILY_MESSAGE
    if ip_count > PER_IP_PER_MINUTE:
        return PER_IP_MESSAGE
    return None


def check_rate_limit(request: Request, scope: str) -> None:
    message = evaluate(request.app.state.redis, _client_ip(request), scope)
    if message is not None:
        logger.warning(f"Rate limit hit: scope={scope} ip={_client_ip(request)}")
        raise HTTPException(status_code=429, detail=message)
