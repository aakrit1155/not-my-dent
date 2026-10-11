"""Short-lived in-process jobs. Run exactly one Uvicorn worker: jobs live in memory."""

import asyncio
import logging
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.pipeline import runner

log = logging.getLogger("notmydent")
router = APIRouter(prefix="/api", tags=["claims"])
JOBS: dict[str, dict] = {}
JOB_TTL_SECONDS = 60 * 60
MAX_ACTIVE_JOBS = 3
VIDEO_LIMIT = 220 * 1024 * 1024
IMAGE_LIMIT = 15 * 1024 * 1024
CHUNK = 1024 * 1024
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "video/x-msvideo", "video/avi", "application/octet-stream"}
PHOTO_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _expire_jobs():
    now = time.monotonic()
    for key in list(JOBS):
        job = JOBS[key]
        if job["state"] not in {"queued", "processing"} and now - job["updated"] > JOB_TTL_SECONDS:
            shutil.rmtree(job["dir"], ignore_errors=True)
            del JOBS[key]


async def _copy_limited(upload: UploadFile, dest: Path, limit: int, types: set[str]):
    if (upload.content_type or "").lower() not in types:
        raise HTTPException(415, f"Unsupported file type: {upload.content_type or 'unknown'}")
    size = 0
    try:
        with dest.open("wb") as target:
            while block := await upload.read(CHUNK):
                size += len(block)
                if size > limit:
                    raise HTTPException(413, f"File exceeds the {limit // 1024 // 1024} MiB limit")
                target.write(block)
        if not size:
            raise HTTPException(400, "Empty uploads are not accepted")
    finally:
        await upload.close()


def _to_urls(value, root: Path, prefix: str):
    """Replace result-file paths in the engine's report with download URLs."""
    if isinstance(value, dict):
        return {key: _to_urls(item, root, prefix) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_urls(item, root, prefix) for item in value]
    if isinstance(value, str) and value.startswith(f"{root}/"):
        return prefix + Path(value).relative_to(root).as_posix()
    return value


async def _run_job(job_id: str):
    job = JOBS[job_id]
    job["state"] = "processing"
    inputs = Path(job["dir"]) / "inputs"
    results = (Path(job["dir"]) / "results").resolve()
    try:
        report = await asyncio.to_thread(
            runner.run,
            inputs / "claim.image",
            inputs / "before.video",
            inputs / "after.video",
            job["claim_text"],
            results,
        )
        job["report"] = _to_urls(report, results, f"/api/jobs/{job_id}/files/")
        job["state"] = "complete"
    except Exception:
        # Log details server-side; don't expose file paths or tracebacks publicly.
        log.exception("Job %s failed", job_id)
        job["state"] = "failed"
        job["error"] = "Analysis failed. Check that both videos play and the photo is a valid image, then try again."
    finally:
        shutil.rmtree(inputs, ignore_errors=True)
        job["updated"] = time.monotonic()


@router.post("/verify-claim", status_code=202)
async def verify_claim(
    background_tasks: BackgroundTasks,
    video_before: UploadFile | None = File(default=None),
    video_after: UploadFile | None = File(default=None),
    claim_image: UploadFile | None = File(default=None),
    claim_text: str = Form(default=""),
):
    _expire_jobs()
    if sum(j["state"] in {"queued", "processing"} for j in JOBS.values()) >= MAX_ACTIVE_JOBS:
        raise HTTPException(503, "The analyzer is busy. Please retry shortly.")
    claim_text = claim_text.strip()
    if not claim_text or len(claim_text) > 2000:
        raise HTTPException(422, "Enter a claim description of 1–2000 characters")
    if not all((video_before, video_after, claim_image)):
        raise HTTPException(422, "Two videos and a vendor photo are required")
    assert video_before and video_after and claim_image
    directory = tempfile.mkdtemp(prefix="notmydent-")
    inputs = Path(directory) / "inputs"
    inputs.mkdir()
    try:
        await _copy_limited(video_before, inputs / "before.video", VIDEO_LIMIT, VIDEO_TYPES)
        await _copy_limited(video_after, inputs / "after.video", VIDEO_LIMIT, VIDEO_TYPES)
        await _copy_limited(claim_image, inputs / "claim.image", IMAGE_LIMIT, PHOTO_TYPES)
        job_id = secrets.token_urlsafe(18)
        JOBS[job_id] = {"state": "queued", "updated": time.monotonic(), "dir": directory, "claim_text": claim_text}
        background_tasks.add_task(_run_job, job_id)
        return {"job_id": job_id, "state": "queued", "status_url": f"/api/jobs/{job_id}"}
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    _expire_jobs()
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    if job["state"] == "complete":
        return {"state": "complete", "report": job["report"]}
    if job["state"] == "failed":
        return {"state": "failed", "error": job["error"]}
    return {"state": job["state"], "models": runner.STATUS["models"]}


@router.get("/jobs/{job_id}/files/{name:path}")
async def job_file(job_id: str, name: str):
    _expire_jobs()
    job = JOBS.get(job_id)
    if job is None or job["state"] != "complete":
        raise HTTPException(404, "File not found or expired")
    root = (Path(job["dir"]) / "results").resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "File not found or expired")
    return FileResponse(path, headers={"Cache-Control": "no-store"})
