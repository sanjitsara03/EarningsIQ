# Starting EarningsAgentIQ (Local Dev)

> `SimpleWorker` (no-fork mode) is required on macOS due to fork-safety issues with OpenSSL/libpq.

## Full Startup Order

| Step | Command | Port |
|------|---------|------|
| 1 | `docker compose up -d` | 5432, 6379 |
| 2 | `PYTHONPATH=. uv run uvicorn api.main:app --reload --port 8000` | 8000 |
| 3 | `PYTHONPATH=. uv run rq worker --worker-class rq.SimpleWorker` | — |
| 4 | `cd frontend && npm run dev` | 5173 |
