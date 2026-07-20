"""
app_ferac.py — standalone server for facial expression recognition
(ConvNeXt + MediaPipe face landmarks). See app_graphomotor.py for why
this was split out of the unified main.py.

Run:
    uvicorn app_ferac:app --host 0.0.0.0 --port 8004
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("ANGLE_DEFAULT_PLATFORM", "swiftshader")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

import torch
torch.set_num_threads(min(4, torch.get_num_threads()))

from feracapi import router as ferac_router, preload as preload_ferac


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_ferac()
    yield


app = FastAPI(title="NeuroSense — FERAC", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(ferac_router)


@app.get("/health")
def health():
    return {"status": "ok"}
