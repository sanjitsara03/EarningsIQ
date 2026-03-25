# POST /ingest — enqueues a full pipeline job for a ticker.
# POST /job/{job_id} — returns the status and result of a queued job.

from fastapi import APIRouter, Request
from pydantic import BaseModel
from rq.job import Job
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

router = APIRouter()


class IngestRequest(BaseModel):
    ticker: str
    filing_type: str = "10-Q"
    limit: int = 1


class JobResponse(BaseModel):
    job_id: str
    status: str


# Enqueues a full pipeline job (ingest → extraction → risk scoring) for the given ticker.
@router.post("/ingest", response_model=JobResponse)
@limiter.limit("5/minute")
def enqueue_ingest(body: IngestRequest, request: Request):
    from api.tasks import run_full_pipeline
    job = request.app.state.queue.enqueue(
        run_full_pipeline,
        body.ticker.upper(),
        body.filing_type,
        body.limit,
        job_timeout=600,
    )
    return {"job_id": job.id, "status": "queued"}


# Returns the current status and result of a job by ID.
@router.get("/job/{job_id}")
def get_job_status(job_id: str, request: Request):
    job = Job.fetch(job_id, connection=request.app.state.redis)
    status = job.get_status().value
    result = job.result if status == "finished" else None
    return {"job_id": job_id, "status": status, "result": result}
