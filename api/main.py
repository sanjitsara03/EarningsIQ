# FastAPI app entry point. Mounts all route modules and sets up the Redis connection.

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from redis import Redis
from rq import Queue

from api.routes import chat, ingest, risk, signals

load_dotenv()

app = FastAPI(title="EarningsAgentIQ", version="1.0.0")

# Redis connection and RQ queue — shared across routes via app.state.
redis_conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
task_queue = Queue(connection=redis_conn)

app.state.redis = redis_conn
app.state.queue = task_queue

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(signals.router)
app.include_router(risk.router)


@app.get("/health")
def health():
    return {"status": "ok"}
