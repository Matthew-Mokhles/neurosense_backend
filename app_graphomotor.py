"""
app_graphomotor.py — standalone server for the sentence-dysgraphia model.

Split out of main.py: running graphomotor (TensorFlow), qchat (XGBoost/
sklearn), asd-motion + ferac + avasd (torch + MediaPipe) all in one
process caused intermittent native-level crashes (no Python traceback,
process just dies) from colliding OpenMP/native runtimes across
TF/torch/MediaPipe/sklearn. Running each model in its own process
removes that collision entirely.

Run:
    uvicorn app_graphomotor:app --host 0.0.0.0 --port 8001
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from graphomotorapi import router as graphomotor_router, preload as preload_graphomotor


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_graphomotor()
    yield


app = FastAPI(title="NeuroSense — Graphomotor", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(graphomotor_router)


@app.get("/health")
def health():
    return {"status": "ok"}
