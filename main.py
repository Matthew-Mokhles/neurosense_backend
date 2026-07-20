"""
main.py — NeuroSense Unified API
=================================
Single FastAPI server exposing every model behind one process/port.

Sub-APIs (all mounted as routers, no per-model server/process needed):
    /graphomotor/...   sentence-level dysgraphia (Keras CNN, image upload)
    /qchat/...         Q-CHAT-10 toddler autism screening (XGBoost + RF)
    /asd-motion/...    ASD stereotypical-movement detection (ST-GCN, video/skeleton)
    /ferac/...         facial expression recognition (ConvNeXt, webcam frame)
    /avasd/...         audio-visual autism behavior detection (CLIP+Whisper+E5 ensemble, video)

Speed design
------------
- All models are PRELOADED at server startup (see lifespan below) — every
  model is loaded once, in a background thread, before uvicorn starts
  accepting traffic, so the very first request to any route is already
  fast. Each model is then cached as a module-level singleton — never
  reloaded per request.
- Every blocking call (TF/Torch inference, OpenCV, MediaPipe) runs in
  Starlette's threadpool via run_in_threadpool / run_in_executor, so one
  slow request never blocks the asyncio event loop or other endpoints.
- GZipMiddleware compresses JSON/image responses.
- A single uvicorn process serves everything — one port, one set of
  /docs, no inter-service network hops.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

(Use 1 worker if these run on GPU — CUDA contexts don't share well across
process-based workers. For CPU-only horizontal scaling, raise --workers.
Startup will take longer now that everything preloads — AV-ASD's
CLIP/Whisper/E5/v19/v18 ensemble alone is several GB.)
"""

import os
import sys

# Several sub-modules (e.g. inference.py's ASDInference, stereotypical_api.py)
# print emoji status messages. Windows' default console/redirect encoding is
# cp1252, which can't encode them and crashes with UnicodeEncodeError the
# instant that print() runs — force UTF-8 on stdout/stderr before any of
# those modules get imported.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# This server has no GPU available (confirmed: every run logs "Cannot
# dlopen some GPU libraries" and falls back to CPU). Telling TensorFlow
# up front to not even look for a GPU skips several failed dlopen
# attempts at import time (CUDA/cuBLAS/cuDNN .dlls that don't exist on
# this box) — a little faster startup, a lot less log noise, zero
# behavior change since it was always going to end up on CPU anyway.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Prevent OpenCV's MSMF backend from using DXVA hardware-accelerated
# video decode — that path loads nvdxgdmal64.dll which access-violates
# on this host's GTX 1060 driver.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# Tell ANGLE (used internally by MediaPipe for OpenGL ES → DirectX
# translation) to not create a real GPU device. This prevents the
# legacy mp.solutions.pose graph from touching the NVIDIA driver at
# all, closing the last path to nvdxgdmal64.dll.
os.environ.setdefault("ANGLE_DEFAULT_PLATFORM", "swiftshader")
# OpenCV's OpenCL backend (T-API) auto-detects and creates a context on
# any available OpenCL platform the first time it's touched — on this
# host that includes NVIDIA's OpenCL ICD, which goes through the same
# driver media layer (nvdxgdmal64.dll) implicated in every crash so
# far. cv2 is imported by nearly every router in this process, so any
# of them could trigger this. Disabling OpenCL outright removes that
# path; none of this codebase's cv2 usage (resize/cvtColor/decode on
# small frames) needs GPU acceleration to be fast enough.
os.environ.setdefault("OPENCV_OPENCL_RUNTIME", "")
os.environ.setdefault("OPENCV_OPENCL_DEVICE", "disabled")

# This process loads torch (bundles Intel MKL/OpenMP), TensorFlow Lite via
# MediaPipe, XGBoost, and scikit-learn all into the same interpreter. Each
# of these ships its own private copy of the Intel OpenMP runtime
# (libiomp5md.dll on Windows). When a second copy gets loaded into a
# process that already has one, the OpenMP runtime aborts the process
# outright — no Python exception, no traceback, the process just dies.
# This is almost certainly the cause of the silent crash right after
# "All models preloaded" (that's the point where every library above has
# finally been imported and their thread pools start interacting).
# Setting this tells the runtime to tolerate duplicate copies instead of
# aborting. It must be set before any of torch/xgboost/sklearn/mediapipe
# are imported, hence its placement here at the very top of the file.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Also cap OpenMP's own thread pool (separate from torch.set_num_threads
# below) to reduce contention between the several thread pools now living
# in one process on a memory/CPU-constrained host.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import logging
import time
from contextlib import asynccontextmanager

import cv2
cv2.ocl.setUseOpenCL(False)

# IMPORTANT: torch must be imported before sklearn/xgboost/scipy in this
# process. On Windows, torch's bundled OpenMP/MKL DLLs (torch/lib/c10.dll
# etc.) collide with the ones scikit-learn's scipy dependency loads first,
# causing "OSError: [WinError 1114] DLL initialization routine failed" if
# sklearn (pulled in transitively by qchat_api -> xgboost_random_forest_model
# -> imblearn -> sklearn) loads first. Importing torch here, before any
# router import, forces the correct DLL load order for the whole process.
import torch

# By default torch spins up one intra-op thread per CPU core. With
# TensorFlow, MediaPipe/XNNPACK, and sklearn each also running their own
# thread pools in this same process, that's significant thread/stack
# memory and scheduler contention on top of an already memory-constrained
# host — a likely contributor to the native access-violation crashes
# we've seen in c10.dll under load. Capping it trades a little CPU
# parallelism for materially less overhead and contention.
torch.set_num_threads(min(4, torch.get_num_threads()))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("neurosense_api")

from graphomotorapi import router as graphomotor_router, preload as preload_graphomotor
from qchat_api import router as qchat_router, preload as preload_qchat
from stereotypical_api import router as asd_motion_router, preload as preload_asd_motion
from feracapi import router as ferac_router, preload as preload_ferac
from avasdapi import router as avasd_router

_PRELOADERS = [
    ("graphomotor", preload_graphomotor),
    ("qchat", preload_qchat),
    ("asd-motion", preload_asd_motion),
    ("ferac", preload_ferac),
]


def _preload_all_sync():
    """Runs in a worker thread so the asyncio event loop stays free during
    startup. Sequential, not parallel: several of these load onto the
    same GPU/CPU resources and concurrent loading would just contend for
    the same disk/CPU/VRAM bandwidth without finishing any faster."""
    for name, fn in _PRELOADERS:
        t0 = time.perf_counter()
        logger.info(f"Preloading {name} ...")
        try:
            fn()
            logger.info(f"{name} ready in {time.perf_counter() - t0:.1f}s")
        except Exception:
            logger.exception(f"Failed to preload {name} — its routes will error until fixed.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()
    logger.info("Preloading all models ...")
    await loop.run_in_executor(None, _preload_all_sync)
    logger.info(f"All models preloaded in {time.perf_counter() - t0:.1f}s. Server ready.")
    yield


app = FastAPI(
    title="NeuroSense Unified API",
    description=(
        "One server, every NeuroSense model: graphomotor dysgraphia, "
        "Q-CHAT-10 screening, ASD stereotypical-movement detection, "
        "facial expression recognition (FERAC), and audio-visual ASD "
        "behavior detection (AV-ASD). See /docs for all endpoints."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(graphomotor_router)
app.include_router(qchat_router)
app.include_router(asd_motion_router)
app.include_router(ferac_router)
app.include_router(avasd_router)


@app.get("/", tags=["meta"])
def root():
    return {
        "status": "ok",
        "service": "NeuroSense Unified API",
        "models": {
            "graphomotor": "/graphomotor (sentence dysgraphia)",
            "qchat": "/qchat (Q-CHAT-10 screening)",
            "asd_motion": "/asd-motion (ST-GCN stereotypical movement)",
            "ferac": "/ferac (facial expression recognition)",
            "avasd": "/avasd (audio-visual ASD behavior detection)",
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe. By the time the server accepts connections, the
    startup lifespan has already finished preloading every model — so a
    200 here means all models are loaded, not just that the process is up.
    Per-model detail is still available at each sub-API's /<name>/health."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, workers=1)
