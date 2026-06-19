"""FastAPI backend.

Endpoints:
  POST /upload    -> store an image or video, returns a file id
  POST /generate  -> queue a job (requires consent affirmation)
  GET  /status/{job_id}
  GET  /download/{job_id}
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

import config
from app import model_runner
from app.queue import JOBS

app = FastAPI(title="Local Motion Transfer")

# id -> stored file path
_FILES: dict[str, Path] = {}


def _save_upload(upload: UploadFile, allowed: set[str]) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(allowed)}")
    file_id = uuid.uuid4().hex[:12]
    dest = config.UPLOAD_DIR / f"{file_id}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    _FILES[file_id] = dest
    return file_id


@app.post("/upload")
async def upload(kind: str = Form(...), file: UploadFile = File(...)):
    """kind = 'image' or 'video'."""
    if kind == "image":
        file_id = _save_upload(file, config.ALLOWED_IMAGE_EXT)
    elif kind == "video":
        file_id = _save_upload(file, config.ALLOWED_VIDEO_EXT)
    else:
        raise HTTPException(400, "kind must be 'image' or 'video'")
    return {"file_id": file_id}


@app.post("/generate")
async def generate(
    image_id: str = Form(...),
    video_id: str = Form(...),
    consent_confirmed: bool = Form(False),
    steps: int = Form(config.DEFAULT_STEPS),
    seed: int = Form(config.DEFAULT_SEED),
):
    # Consent gate — see README "Responsible use".
    if not consent_confirmed:
        raise HTTPException(
            403,
            "Generation requires confirming you have the right to use this "
            "likeness (your own image, or a consenting subject).",
        )

    image_path = _FILES.get(image_id)
    video_path = _FILES.get(video_id)
    if image_path is None or video_path is None:
        raise HTTPException(404, "Unknown image_id or video_id. Upload first.")

    def job_fn(job):
        return model_runner.run_pipeline(
            job, image_path, video_path, steps=steps, seed=seed
        )

    job_id = JOBS.submit(job_fn, meta={"image_id": image_id, "video_id": video_id})
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "queue_position": JOBS.position(job_id) if job.status == "queued" else 0,
        "error": job.error,
        "ready": job.status == "done",
    }


@app.get("/download/{job_id}")
async def download(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job_id")
    if job.status != "done":
        raise HTTPException(409, f"Job is '{job.status}', not ready.")
    out = job.meta.get("output")
    if not out or not Path(out).exists():
        raise HTTPException(500, "Output file missing.")
    return FileResponse(out, media_type="video/mp4", filename=f"motion_transfer_{job_id}.mp4")


@app.get("/")
async def root():
    ok, detail = model_runner.model_configured()
    return {"ok": True, "model": "mimicmotion", "model_configured": ok, "detail": detail}
