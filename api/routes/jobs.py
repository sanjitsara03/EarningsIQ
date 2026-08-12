# GET /job/{job_id} — polls an RQ job; returns status and the job result when finished.

from fastapi import APIRouter, HTTPException, Request

router = APIRouter()


@router.get("/job/{job_id}")
def get_job(job_id: str, request: Request):
    from rq.exceptions import NoSuchJobError
    from rq.job import Job

    try:
        job = Job.fetch(job_id, connection=request.app.state.redis)
        status = job.get_status()
        status_str = status.value if hasattr(status, "value") else str(status)
        return {
            "job_id": job_id,
            "status": status_str,
            "result": job.result if status_str == "finished" else None,
        }
    except NoSuchJobError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
