"""
graphomotorapi.py
FastAPI router for sentence-level dysgraphia detection (lazy-loaded model).
"""

import os
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from starlette.concurrency import run_in_threadpool

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentence_dysgraphia_model_final.h5")

router = APIRouter(prefix="/graphomotor", tags=["graphomotor"])

_model = None


def _get_model():
    global _model
    if _model is None:
        from tensorflow.keras.models import load_model
        _model = load_model(MODEL_PATH)
    return _model


def preload():
    """Eagerly load the model. Called from main.py's startup so the first
    real request doesn't pay the load cost."""
    _get_model()


# =========================
# PREPROCESSING
# =========================
def preprocess(img):

    img = cv2.resize(img, (224, 224))

    # Reduce camera noise slightly
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)

    # Same preprocessing used during training
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    img = preprocess_input(img)

    return np.expand_dims(img, axis=0)


# =========================
# TEST-TIME AUGMENTATION
# =========================
def tta_predict(img):

    predictions = []

    for angle in [0, -5, 5]:

        if angle == 0:
            aug = img.copy()

        else:
            h, w = img.shape[:2]

            M = cv2.getRotationMatrix2D(
                (w // 2, h // 2),
                angle,
                1.0
            )

            aug = cv2.warpAffine(
                img,
                M,
                (w, h)
            )

        x = preprocess(aug)

        pred = _get_model().predict(
            x,
            verbose=0
        )[0]

        predictions.append(pred)

    # Average predictions
    return np.mean(predictions, axis=0)


# =========================
# HEALTH CHECK
# =========================
@router.get("/")
def health():
    return {"status": "running", "model_loaded": _model is not None}


# =========================
# PREDICTION
# =========================
@router.post("/predict")
async def predict(file: UploadFile = File(...)):

    content = await file.read()

    nparr = np.frombuffer(
        content,
        np.uint8
    )

    img = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )

    if img is None:
        return {"error": "Invalid image"}

    pred = await run_in_threadpool(tta_predict, img)

    low_score = float(pred[0])
    dys_score = float(pred[1])

    confidence = float(np.max(pred))

    label = (
        "Potential Dysgraphia"
        if dys_score > low_score
        else "Low Potential Dysgraphia"
    )

    print("--------------------------------")
    print("Prediction:", pred)
    print("Low Score:", low_score)
    print("Potential Score:", dys_score)
    print("Confidence:", confidence)
    print("Predicted Class:", label)
    print("--------------------------------")

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "dysgraphia_score": round(dys_score, 4),
        "low_potential_score": round(low_score, 4)
    }
