"""Short-lived in-process jobs. Run exactly one Uvicorn worker in mock mode."""

import asyncio
import io
import secrets
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.pipeline.mock import process_claim_pipeline

router = APIRouter(prefix="/api", tags=["claims"])
JOBS: dict[str, dict] = {}
JOB_TTL_SECONDS = 20 * 60
MAX_ACTIVE_JOBS = 3
VIDEO_LIMIT = 220 * 1024 * 1024
IMAGE_LIMIT = 15 * 1024 * 1024
CHUNK = 1024 * 1024
VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-matroska", "application/octet-stream"}
PHOTO_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _expire_jobs():
    now = time.monotonic()
    for key in list(JOBS):
        job = JOBS[key]
        if job["state"] not in {"queued", "processing"} and now - job["updated"] > JOB_TTL_SECONDS:
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


async def _run_job(job_id: str, directory: str, claim_text: str, demo: bool):
    job = JOBS[job_id]
    job["state"] = "processing"
    try:
        base = Path(directory)
        report, png = await asyncio.to_thread(
            process_claim_pipeline,
            None if demo else base / "before.video",
            None if demo else base / "after.video",
            claim_text,
            None if demo else base / "vendor.image",
        )
        job["report"] = report
        job["image"] = png
        job["state"] = "complete"
    except Exception:
        # Avoid exposing internal file paths and tracebacks to a public endpoint.
        job["state"] = "failed"
        job["error"] = "The demo could not generate this report. Please try again."
    finally:
        shutil.rmtree(directory, ignore_errors=True)
        job["updated"] = time.monotonic()


@router.post("/verify-claim", status_code=202)
async def verify_claim(
    background_tasks: BackgroundTasks,
    video_before: UploadFile | None = File(default=None),
    video_after: UploadFile | None = File(default=None),
    claim_image: UploadFile | None = File(default=None),
    claim_text: str = Form(default=""),
    demo: bool = Form(default=False),
):
    _expire_jobs()
    if sum(j["state"] in {"queued", "processing"} for j in JOBS.values()) >= MAX_ACTIVE_JOBS:
        raise HTTPException(503, "Demo is busy. Please retry shortly.")
    claim_text = claim_text.strip()
    if not claim_text or len(claim_text) > 2000:
        raise HTTPException(422, "Enter a claim description of 1–2000 characters")
    if not demo and not all((video_before, video_after, claim_image)):
        raise HTTPException(422, "Two videos and a vendor photo are required, or choose sample inputs")
    directory = tempfile.mkdtemp(prefix="notmydent-")
    try:
        if not demo:
            assert video_before and video_after and claim_image
            await _copy_limited(video_before, Path(directory) / "before.video", VIDEO_LIMIT, VIDEO_TYPES)
            await _copy_limited(video_after, Path(directory) / "after.video", VIDEO_LIMIT, VIDEO_TYPES)
            await _copy_limited(claim_image, Path(directory) / "vendor.image", IMAGE_LIMIT, PHOTO_TYPES)
        # Demo mode discards client files, if supplied; never echo their contents.
        job_id = secrets.token_urlsafe(18)
        JOBS[job_id] = {"state": "queued", "updated": time.monotonic()}
        background_tasks.add_task(_run_job, job_id, directory, claim_text, demo)
        return {"job_id": job_id, "state": "queued", "status_url": f"/api/jobs/{job_id}"}
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        for upload in (video_before, video_after, claim_image):
            if upload is not None:
                await upload.close()


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    _expire_jobs()
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired")
    if job["state"] == "complete":
        return {"state": "complete", "report": {**job["report"], "proof_image_url": f"/api/jobs/{job_id}/proof"}}
    if job["state"] == "failed":
        return {"state": "failed", "error": job["error"]}
    return {"state": job["state"]}


@router.get("/jobs/{job_id}/proof")
async def proof(job_id: str):
    _expire_jobs()
    job = JOBS.get(job_id)
    if job is None or job["state"] != "complete":
        raise HTTPException(404, "Proof image not found or expired")
    return StreamingResponse(io.BytesIO(job["image"]), media_type="image/png", headers={"Cache-Control": "no-store"})
