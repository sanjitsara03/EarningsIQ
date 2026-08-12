# POST /ingest — enqueues a full pipeline job for a ticker.
# Job status polling lives in api/routes/jobs.py (GET /job/{job_id}).

from fastapi import APIRouter, Request
from pydantic import BaseModel

from api.rate_limit import check_rate_limit

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
def enqueue_ingest(body: IngestRequest, request: Request):
    # Each pipeline job is the most expensive thing this API can do — rate-limit before enqueueing.
    check_rate_limit(request, "ingest")
    from api.tasks import run_full_pipeline
    job = request.app.state.queue.enqueue(
        run_full_pipeline,
        body.ticker.upper(),
        body.filing_type,
        body.limit,
        job_timeout=600,
    )
    return {"job_id": job.id, "status": "queued"}
