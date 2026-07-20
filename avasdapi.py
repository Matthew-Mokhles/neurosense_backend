"""
app.py — AV-ASD Inference API
================================

Single endpoint: upload a video, get back per-label behavior probabilities
and binary predictions using the locked best-known recipe
(v19 + v18 cross-ensemble, w_v19=0.4, fixed thresholds; test F1=0.5240).

RUNNING
-------
  pip install fastapi uvicorn python-multipart
  uvicorn app:app --host 0.0.0.0 --port 8000

  Then open http://localhost:8000/docs for interactive Swagger UI.

CALLING (example)
------------------
  curl -X POST "http://localhost:8000/analyze" \\
       -F "file=@/path/to/clip.mp4"

NOTES
-----
- Inference is SEQUENTIAL on a single GPU: this matches the project's
  6 GB VRAM training constraint. One request fully occupies the GPU
  for the duration of its run (CLIP -> Whisper enc -> Whisper ASR ->
  E5 -> v19 x3 seeds -> v18 x3 seeds, each unloaded before the next
  loads). Concurrent requests are serialized via an asyncio lock so
  two videos never try to load models onto the GPU at once.
- Uploaded files are saved to a temp directory and deleted after
  processing (success or failure).
- See inference_pipeline.py for the exact recipe being reproduced and
  inference_pipeline.PROJECT_ROOT / V19_CKPT_DIR / V18_CKPT_DIR for the
  checkpoint paths — edit those (or set AVASD_PROJECT_ROOT env var) if
  your checkpoints live somewhere else.
"""

import os
import shutil
import tempfile
import time
import uuid
import asyncio
import threading
import logging
import traceback
from typing import Dict, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import avasd_inference as pipeline

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('avasd_api')

# Only one inference run on the GPU at a time — required for a 6 GB card.
# asyncio.Lock for the synchronous /analyze endpoint's own coroutine;
# threading.Lock for _run_job_sync, which runs in a worker thread (asyncio
# locks aren't safe to acquire across threads).
_gpu_lock = asyncio.Lock()
_gpu_thread_lock = threading.Lock()
_models_ready = False

ALLOWED_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB safety cap

router = APIRouter(prefix='/avasd', tags=['avasd'])

# ─────────────────────────────────────────────────────────────────────────
# Async job store
# ─────────────────────────────────────────────────────────────────────────
# AV-ASD inference (model load + CLIP/Whisper/E5/v19/v18 on a single clip)
# can take minutes, especially on CPU — far longer than a mobile client
# should hold one HTTP request open for. /analyze/start returns a job_id
# immediately after the upload finishes; the client polls
# /analyze/status/{job_id} (cheap, instant) until status != "pending",
# and can navigate away / go to background in between polls.
#
# In-memory only — jobs don't survive a server restart, which is fine for
# a single-process dev/small-deployment server. _JOB_TTL_SECONDS bounds
# memory growth from clients that start a job and never poll it again.
_JOB_TTL_SECONDS = 60 * 60  # 1 hour
_jobs: Dict[str, dict] = {}


def _cleanup_stale_jobs():
    now = time.time()
    stale = [jid for jid, j in _jobs.items() if now - j['created_at'] > _JOB_TTL_SECONDS]
    for jid in stale:
        _jobs.pop(jid, None)


def _run_job_sync(job_id: str, tmp_path: str, tmp_dir: str):
    """Runs in the threadpool: the actual long-running inference, fully
    detached from any client connection. Updates the shared job dict on
    completion (success or failure) so polling clients see the result."""
    try:
        with _gpu_thread_lock:
            _ensure_models_loaded_sync()
            result = pipeline.run_inference(tmp_path)
        _jobs[job_id] = {**_jobs[job_id], 'status': 'done', 'result': result}
        logger.info(f'Job {job_id} complete.')
    except Exception as e:
        logger.error(f'Job {job_id} failed: {e}\n{traceback.format_exc()}')
        _jobs[job_id] = {**_jobs[job_id], 'status': 'error', 'error': str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _ensure_models_loaded_sync():
    """Loads CLIP/Whisper/E5/v19/v18 once (GBs of weights). Called either
    eagerly at startup via preload(), or lazily on first /analyze hit if
    preload was skipped/failed."""
    global _models_ready
    if _models_ready:
        return
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline.preload_models(device)
    _models_ready = True


def preload():
    """Eagerly load the full CLIP/Whisper/E5/v19/v18 ensemble. This is by
    far the heaviest model in the unified API (GBs of weights across 8
    sub-models) — called from main.py's startup."""
    _ensure_models_loaded_sync()


class AnalyzeResponse(BaseModel):
    transcript: str
    probabilities: dict
    predictions: dict
    thresholds: dict


class ErrorResponse(BaseModel):
    error: str
    detail: str


@router.get('/health')
async def health():
    import torch
    return {
        'status': 'ok',
        'models_loaded': _models_ready,
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@router.get('/labels')
async def labels():
    """Returns the 10 behavior labels and their locked decision thresholds."""
    return {
        'labels': pipeline.LABELS,
    }


@router.post(
    '/analyze',
    response_model=AnalyzeResponse,
    responses={400: {'model': ErrorResponse}, 500: {'model': ErrorResponse}},
)
async def analyze(file: UploadFile = File(...)):
    """
    Upload a video clip and receive behavior predictions.

    The clip should ideally be ~30 seconds (the model was trained on
    30-second windows); shorter clips are padded with silence/black
    frames, longer clips are truncated.
    """
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. "
                   f"Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_dir = tempfile.mkdtemp(prefix='avasd_upload_')
    tmp_path = os.path.join(tmp_dir, f'input{ext}')

    try:
        size = 0
        with open(tmp_path, 'wb') as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=400,
                        detail=f'File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB.')
                out_f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail='Uploaded file is empty.')

        logger.info(f'Received {file.filename} ({size / 1e6:.1f} MB) -> {tmp_path}')

        async with _gpu_lock:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _ensure_models_loaded_sync)
            result = await loop.run_in_executor(None, pipeline.run_inference, tmp_path)

        logger.info(f'Inference complete for {file.filename}')
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Inference failed: {e}\n{traceback.format_exc()}')
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {str(e)}')
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Validates and streams an upload to a temp file. Returns
    (tmp_dir, tmp_path); caller owns cleanup of tmp_dir."""
    ext = os.path.splitext(file.filename or '')[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{ext}'. "
                   f"Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    tmp_dir = tempfile.mkdtemp(prefix='avasd_upload_')
    tmp_path = os.path.join(tmp_dir, f'input{ext}')

    size = 0
    with open(tmp_path, 'wb') as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=400,
                    detail=f'File too large. Max {MAX_UPLOAD_BYTES // (1024*1024)} MB.')
            out_f.write(chunk)

    if size == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail='Uploaded file is empty.')

    logger.info(f'Received {file.filename} ({size / 1e6:.1f} MB) -> {tmp_path}')
    return tmp_dir, tmp_path


class JobStartResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    status: str  # "pending" | "done" | "error"
    result: Optional[dict] = None
    error: Optional[str] = None


@router.post('/analyze/start', response_model=JobStartResponse)
async def analyze_start(file: UploadFile = File(...)):
    """
    Upload a video clip and get back a job_id immediately — the upload
    itself is the only thing this request waits on. Inference (which can
    take minutes, especially on CPU) runs afterward in the background;
    poll GET /avasd/analyze/status/{job_id} until status != "pending".

    This is the endpoint mobile clients should use instead of the
    synchronous /analyze, so a slow/CPU-bound inference run never holds
    an HTTP connection open long enough to time out or get killed by the
    OS/network when the app is backgrounded.
    """
    _cleanup_stale_jobs()
    tmp_dir, tmp_path = await _save_upload(file)

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {'status': 'pending', 'created_at': time.time(), 'result': None, 'error': None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_job_sync, job_id, tmp_path, tmp_dir)

    logger.info(f'Started job {job_id} for {file.filename}')
    return JobStartResponse(job_id=job_id)


@router.get('/analyze/status/{job_id}', response_model=JobStatusResponse)
async def analyze_status(job_id: str):
    """Cheap poll — instant response, no GPU/CPU work. Call this every
    few seconds until status is "done" or "error"."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail='Unknown or expired job_id.')
    return JobStatusResponse(status=job['status'], result=job.get('result'), error=job.get('error'))
