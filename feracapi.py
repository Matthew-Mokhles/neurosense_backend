"""
server.py
=========
FastAPI server exposing the FERAC inference pipeline over HTTP for the
Flutter mobile app, per the "server-based deployment" decision.

Endpoints:
    POST /predict           - single-frame emotion prediction (multipart/form-data, field "frame")
    POST /session/start     - start a new viewing session (resets temporal smoothing)
    POST /session/{id}/predict - predict within a specific session's smoothing context
    GET  /health            - liveness check

Why per-session smoothing matters: app_inference.predict_emotion() uses
ONE shared module-level smoother (see app_inference._pipeline). That's
fine for a single Python process driving one camera. Over HTTP, multiple
concurrent app instances (or even a single child closing and reopening
the app) would otherwise corrupt each other's temporal_smoothing history
with frames from a different face/session. This server keeps one
FERACPipeline (model + detector + aligner) SHARED across requests
— they're expensive to construct and stateless w.r.t. classification —
but gives each session its OWN TemporalSmoother instance, which IS
correct to share within a session and WRONG to share across sessions.

Run:
    pip install fastapi uvicorn python-multipart
    cd inference
    uvicorn server:app --host 0.0.0.0 --port 8000

Then from the same LAN, the Flutter app hits:
    http://<your-machine-LAN-IP>:8000/predict
"""

import time
import uuid
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from ferac_face_detector import FaceDetector
from ferac_face_alignment import FaceAligner
from ferac_inference import FERACInferenceModel
from ferac_temporal_smoothing import TemporalSmoother
from ferac_config import CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW


router = APIRouter(prefix="/ferac", tags=["ferac"])


# ─────────────────────────────────────────────────────────────────────────
# Shared, expensive-to-construct components — lazily built on first request
# and reused after that. These are stateless with respect to any single
# session's classification (the model and detector don't carry per-call
# history), so sharing them is safe and necessary for performance.
# ─────────────────────────────────────────────────────────────────────────

_model: Optional[FERACInferenceModel] = None
_detector: Optional[FaceDetector] = None
_aligner: Optional[FaceAligner] = None

# Per-session temporal smoothing state. Sessions are created via
# /session/start and identified by an opaque UUID the Flutter app stores
# for the lifetime of one viewing session (e.g. one video in the protocol,
# or one full app visit — your choice on the client side).
_sessions: Dict[str, TemporalSmoother] = {}
_session_last_seen: Dict[str, float] = {}

SESSION_TTL_SECONDS = 60 * 30  # auto-expire idle sessions after 30 minutes


def _ensure_loaded():
    global _model, _detector, _aligner
    if _model is None:
        _model = FERACInferenceModel()
        _detector = FaceDetector()
        _aligner = FaceAligner()


def preload():
    """Eagerly load the ConvNeXt model + MediaPipe detector/aligner.
    Called from main.py's startup."""
    _ensure_loaded()


def _decode_frame(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image bytes as JPEG/PNG.")
    return frame


def _run_single_frame(frame_bgr: np.ndarray):
    """Detect -> align -> classify ONE frame. Returns (label_or_None, confidence, probs_or_None)."""
    face = _detector.detect_largest(frame_bgr)
    if face is None:
        return None, 0.0, None

    crop = _detector.crop(frame_bgr, face)
    if crop.size == 0:
        return None, 0.0, None

    crop = _aligner.align(crop)
    label, confidence, probs = _model.predict(crop)
    return label, confidence, probs


def _cleanup_stale_sessions():
    now = time.time()
    stale = [sid for sid, last in _session_last_seen.items()
             if now - last > SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_last_seen.pop(sid, None)


# ─────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    emotion: Optional[str]
    confidence: float


class SessionStartResponse(BaseModel):
    session_id: str


# ─────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@router.post("/session/start", response_model=SessionStartResponse)
def start_session():
    """
    Call this once when the child starts watching a video / opens the
    emotion-tracking screen. Returns a session_id the app must include
    in subsequent /session/{id}/predict calls so temporal smoothing
    builds up correctly for THIS viewing session only.
    """
    _cleanup_stale_sessions()
    session_id = str(uuid.uuid4())
    _sessions[session_id] = TemporalSmoother(
        window=SMOOTHING_WINDOW, confidence_threshold=CONFIDENCE_THRESHOLD
    )
    _session_last_seen[session_id] = time.time()
    return SessionStartResponse(session_id=session_id)


@router.post("/session/{session_id}/predict", response_model=PredictResponse)
async def predict_in_session(session_id: str, frame: UploadFile = File(...)):
    """
    Smoothed prediction within an existing session. Use this for the
    real app flow: call /session/start once, then POST a frame here
    every N milliseconds while that session is active.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id. Call /session/start again.")

    frame_bytes = await frame.read()
    frame_bgr = _decode_frame(frame_bytes)

    await run_in_threadpool(_ensure_loaded)
    label, confidence, probs = await run_in_threadpool(_run_single_frame, frame_bgr)
    smoother = _sessions[session_id]
    _session_last_seen[session_id] = time.time()

    if probs is None:
        # No face this frame — return the smoother's current state
        # unchanged rather than forcing a None (a brief detection
        # dropout shouldn't flicker the UI to "no emotion").
        return PredictResponse(emotion=None, confidence=0.0)

    smoothed_label, smoothed_conf, _ = smoother.update(probs)
    return PredictResponse(emotion=smoothed_label, confidence=round(float(smoothed_conf), 4))


@router.post("/session/{session_id}/end")
def end_session(session_id: str):
    """Call when the child finishes watching / leaves the screen, to free memory promptly."""
    _sessions.pop(session_id, None)
    _session_last_seen.pop(session_id, None)
    return {"status": "ended"}


@router.post("/predict", response_model=PredictResponse)
async def predict_stateless(frame: UploadFile = File(...)):
    """
    Stateless single-frame prediction, NO temporal smoothing. Useful for
    quick testing/curl from a terminal, or a one-off snapshot feature.
    For the real app's continuous video-watching flow, prefer
    /session/start + /session/{id}/predict so predictions stabilize
    frame-to-frame instead of flickering.
    """
    frame_bytes = await frame.read()
    frame_bgr = _decode_frame(frame_bytes)
    await run_in_threadpool(_ensure_loaded)
    label, confidence, _ = await run_in_threadpool(_run_single_frame, frame_bgr)
    return PredictResponse(emotion=label, confidence=round(float(confidence), 4))
