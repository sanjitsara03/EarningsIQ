# FastAPI app entry point. Mounts all route modules and sets up the Redis connection.

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from rq import Queue

from api.routes import chat, ingest, jobs, risk, signals

load_dotenv()

app = FastAPI(title="EarningsAgentIQ", version="1.0.0")

# Allow requests from the Vite dev server and future Vercel deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis connection and RQ queue — shared across routes via app.state.
redis_conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"))
task_queue = Queue(connection=redis_conn)

app.state.redis = redis_conn
app.state.queue = task_queue

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(signals.router)
app.include_router(risk.router)
app.include_router(jobs.router)


@app.get("/health")
def health():
    return {"status": "ok"}
