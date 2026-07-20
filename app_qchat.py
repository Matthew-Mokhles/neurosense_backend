"""
app_qchat.py — standalone server for the Q-CHAT-10 screening model
(XGBoost + Random Forest). See app_graphomotor.py for why this was
split out of the unified main.py.

Run:
    uvicorn app_qchat:app --host 0.0.0.0 --port 8002
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from qchat_api import router as qchat_router, preload as preload_qchat


@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_qchat()
    yield


app = FastAPI(title="NeuroSense — Q-CHAT-10", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(qchat_router)


@app.get("/health")
def health():
    return {"status": "ok"}
