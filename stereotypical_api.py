"""
NeuroSense Backend API
======================
FastAPI backend for ASD Detection

Endpoints:
    POST /predict/video      → بيبعت فيديو ويرجع report
    POST /predict/skeleton   → بيبعت skeleton array ويرجع report
    GET  /health             → بيتأكد إن الـ API شغال
    GET  /activities         → بيرجع الـ 11 activities

Run:
    uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
import numpy as np
import json
import tempfile
import os
import sys
from pathlib import Path
from typing import Optional

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from inference import ASDInference, ACTIVITY_MAP
from skeleton_extractor import SkeletonExtractor

# ============================================================
# ROUTER SETUP
# ============================================================
router = APIRouter(prefix="/asd-motion", tags=["asd-motion"])

# ============================================================
# Lazy-loaded, shared once across requests
# ============================================================
MODEL_PATH = PROJECT_DIR / 'checkpoints' / 'best_model.pth'

_inference_engine: Optional[ASDInference] = None
_skeleton_extractor: Optional[SkeletonExtractor] = None


def _get_inference_engine() -> ASDInference:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = ASDInference(model_path=MODEL_PATH)
    return _inference_engine


def _get_skeleton_extractor() -> SkeletonExtractor:
    global _skeleton_extractor
    if _skeleton_extractor is None:
        _skeleton_extractor = SkeletonExtractor()
    return _skeleton_extractor


def preload():
    """Eagerly load the ST-GCN model and the MediaPipe skeleton extractor.
    Called from main.py's startup."""
    _get_inference_engine()
    _get_skeleton_extractor()


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/health")
async def health_check():
    """تأكد إن الـ API شغال"""
    return {
        "status": "healthy",
        "model": "ASD_Detection_ST-GCN",
        "version": "1.0.0"
    }


@router.get("/activities")
async def get_activities():
    """بيرجع الـ 11 activities وتفاصيلها"""
    activities = []
    for key, info in ACTIVITY_MAP.items():
        activities.append({
            "key":     key,
            "display": info['display'],
            "theme":   info['theme'],
            "id":      info['id'],
        })
    return {"activities": activities, "total": len(activities)}


@router.post("/predict/video")
async def predict_from_video(
    video: UploadFile = File(..., description="فيديو الطفل (.mp4, .avi, .mov)"),
    activity: str     = Form(..., description="اسم الـ activity"),
    child_name: Optional[str] = Form(None),
    child_age:  Optional[int] = Form(None),
):
    """
    يستقبل فيديو ويرجع تقرير ASD

    - **video**: فيديو الطفل وهو بيعمل الـ activity
    - **activity**: اسم الـ activity (مثال: arm_swing, frog_pose)
    - **child_name**: اسم الطفل (اختياري)
    - **child_age**: عمر الطفل (اختياري)
    """

    print(f"🔎 DEBUG /predict/video: activity={activity!r} filename={video.filename!r} content_type={video.content_type!r}")

    allowed_types = [
    'video/mp4', 'video/avi', 'video/quicktime',
    'video/x-msvideo', 'video/webm', 'video/mkv',
    'video/x-matroska', 'video/3gpp', 'video/x-ms-wmv',
    'application/octet-stream',  # بعض الموبايلات بيبعتوا كده
    ]
    if video.content_type not in allowed_types:
        # log بس ومتوقفش
        print(f"⚠️ Content type: {video.content_type} — proceeding anyway")

    # تحقق من اسم الـ activity
    if activity not in ACTIVITY_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Activity '{activity}' غير موجودة. الـ activities المتاحة: {list(ACTIVITY_MAP.keys())}"
        )

    # حفظ الفيديو مؤقتاً
    suffix = Path(video.filename).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    print(f"🔎 DEBUG /predict/video: saved {len(content)} bytes to {tmp_path}")
    if len(content) == 0:
        os.remove(tmp_path)
        raise HTTPException(
            status_code=400,
            detail="❌ الفيديو وصل فاضي (0 bytes) — جرّب تسجّل/ترفع الفيديو تاني."
        )

    try:
        # استخراج الـ skeleton من الفيديو (blocking CPU work — لازم يتشغل في
        # threadpool عشان مايقفش الـ event loop ويوقف الـ API كله عن الرد)
        skeleton = await run_in_threadpool(_get_skeleton_extractor().extract, tmp_path)

        if skeleton is None:
            raise HTTPException(
                status_code=422,
                detail="❌ مش قادر يستخرج skeleton من الفيديو. تأكد إن الطفل واضح في الفيديو."
            )

        # معلومات الطفل
        child_info = None
        if child_name or child_age:
            child_info = {"name": child_name, "age": child_age}

        # Inference
        report = await run_in_threadpool(
            _get_inference_engine().predict_from_skeleton,
            skeleton=skeleton,
            activity=activity,
            child_info=child_info
        )

        return JSONResponse(content=report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        # امسح الفيديو المؤقت
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/predict/skeleton")
async def predict_from_skeleton(
    skeleton_file: UploadFile = File(..., description="skeleton file (.npz)"),
    activity: str  = Form(...),
    child_name: Optional[str] = Form(None),
    child_age:  Optional[int] = Form(None),
):
    """
    يستقبل skeleton جاهز ويرجع تقرير ASD
    (للتجربة والـ testing بدون فيديو)
    """

    if activity not in ACTIVITY_MAP:
        raise HTTPException(status_code=400, detail=f"Activity '{activity}' غير موجودة.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
        content = await skeleton_file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        child_info = None
        if child_name or child_age:
            child_info = {"name": child_name, "age": child_age}

        report = await run_in_threadpool(
            _get_inference_engine().predict_from_file,
            npz_path=tmp_path,
            activity=activity,
            child_info=child_info
        )

        return JSONResponse(content=report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/predict/session")
async def predict_full_session(
    videos: list[UploadFile] = File(..., description="فيديوهات الـ 11 activities"),
    activities: str = Form(..., description='JSON list: ["arm_swing","frog_pose",...]'),
    child_name: Optional[str] = Form(None),
    child_age:  Optional[int] = Form(None),
):
    """
    يستقبل فيديوهات كل الـ activities ويرجع تقرير شامل للـ session كلها
    """

    activity_list = json.loads(activities)

    if len(videos) != len(activity_list):
        raise HTTPException(
            status_code=400,
            detail=f"عدد الفيديوهات ({len(videos)}) مش متطابق مع عدد الـ activities ({len(activity_list)})"
        )

    child_info = None
    if child_name or child_age:
        child_info = {"name": child_name, "age": child_age}

    activities_skeletons = {}
    tmp_paths = []

    try:
        for video, activity in zip(videos, activity_list):
            if activity not in ACTIVITY_MAP:
                continue

            suffix = Path(video.filename).suffix or '.mp4'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await video.read()
                tmp.write(content)
                tmp_paths.append(tmp.name)

            skeleton = await run_in_threadpool(_get_skeleton_extractor().extract, tmp.name)
            if skeleton is not None:
                activities_skeletons[activity] = skeleton

        if not activities_skeletons:
            raise HTTPException(status_code=422, detail="❌ مش قادر يستخرج skeleton من أي فيديو.")

        report = await run_in_threadpool(
            _get_inference_engine().predict_full_session, activities_skeletons, child_info
        )
        return JSONResponse(content=report)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in tmp_paths:
            if os.path.exists(path):
                os.remove(path)
