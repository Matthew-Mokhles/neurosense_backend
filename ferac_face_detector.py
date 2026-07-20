"""
face_detector.py
================
Face detection using MediaPipe Tasks API (mp.tasks.vision.FaceDetector).

IMPORTANT — API version note: mediapipe >= 0.10 removed the legacy
`mp.solutions.face_detection` ("Solutions" API) entirely. This module
targets the current Tasks API (`mp.tasks.vision.FaceDetector`), which is
also the API surface MediaPipe's official Flutter/Android/iOS plugins
bind to — so this Python reference implementation and your eventual
Flutter on-device implementation use the same underlying model format
(.task bundles), not two different MediaPipe generations.

Why MediaPipe over YOLOv8-face/RetinaFace (per task brief's priority
order): the target deployment is a Flutter mobile app. MediaPipe Tasks
ships official, maintained on-device bundles for both face detection AND
face landmarking for Android/iOS/Flutter, running efficiently on CPU
without a separate ONNX/TFLite conversion step. YOLOv8-face and
RetinaFace are heavier PyTorch models with no first-party Flutter
runtime — shipping them on mobile would require a separate export and a
custom native bridge.

Handles multiple faces by selecting the LARGEST detected face, per spec.

Model bundle: downloaded automatically on first use to
~/.cache/ferac_mediapipe/blaze_face_short_range.tflite (a small, official
Google-hosted MediaPipe model bundle). If you're deploying somewhere
without internet access, pre-download this file once and pass its local
path to FaceDetector(model_path=...).
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from ferac_config import FACE_DETECTION_MIN_CONFIDENCE, FACE_MARGIN_RATIO

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ferac_mediapipe")
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)
_DEFAULT_MODEL_PATH = os.path.join(_CACHE_DIR, "blaze_face_short_range.tflite")


def _ensure_model_downloaded(model_path: str, url: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading MediaPipe face detector model bundle to {model_path} ...")
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")
    return model_path


@dataclass
class FaceBox:
    """Pixel-space bounding box. keypoints maps name -> (x, y) in pixel space."""
    x: int
    y: int
    w: int
    h: int
    confidence: float
    keypoints: dict

    @property
    def area(self) -> int:
        return self.w * self.h

    def expanded(self, margin_ratio: float, frame_shape) -> "FaceBox":
        """Return a copy expanded by margin_ratio on each side, clamped to frame bounds."""
        H, W = frame_shape[:2]
        mx = int(self.w * margin_ratio)
        my = int(self.h * margin_ratio)
        x0 = max(0, self.x - mx)
        y0 = max(0, self.y - my)
        x1 = min(W, self.x + self.w + mx)
        y1 = min(H, self.y + self.h + my)
        return FaceBox(x0, y0, x1 - x0, y1 - y0, self.confidence, self.keypoints)


class FaceDetector:
    """
    Usage:
        detector = FaceDetector()
        face = detector.detect_largest(frame_bgr)   # FaceBox or None
        crop = detector.crop(frame_bgr, face)         # np.ndarray (face-only image)
    """

    # BlazeFace short-range keypoint order (6 keypoints)
    _KEYPOINT_NAMES = [
        "right_eye", "left_eye", "nose_tip",
        "mouth_center", "right_ear_tragion", "left_ear_tragion",
    ]

    def __init__(self, min_confidence: float = FACE_DETECTION_MIN_CONFIDENCE,
                 model_path: Optional[str] = None):
        """
        model_path: local path to a .task/.tflite face-detector bundle. If
        None, the official short-range BlazeFace bundle is downloaded and
        cached automatically (see module docstring). Short-range is the
        right choice for the target use case: a child seated close to a
        phone/tablet camera (~< 2m), which is exactly BlazeFace
        short-range's designed operating distance.
        """
        resolved_path = model_path or _ensure_model_downloaded(
            _DEFAULT_MODEL_PATH, _DEFAULT_MODEL_URL
        )
        # MediaPipe's native asset resolver doesn't recognize Windows
        # drive-letter absolute paths (C:\..., D:\...) as absolute — it
        # always treats model_asset_path as relative to its own install
        # dir, so any Windows absolute path here fails with errno=22
        # regardless of slash direction. Reading the bytes ourselves and
        # passing model_asset_buffer sidesteps that broken path resolution
        # entirely.
        with open(resolved_path, "rb") as f:
            model_bytes = f.read()
        options = vision.FaceDetectorOptions(
            # Force CPU explicitly. MediaPipe's GPU delegate on Windows
            # routes through ANGLE (OpenGL ES -> Direct3D translation),
            # which on this host hits the NVIDIA driver's media layer
            # (nvdxgdmal64.dll) and has been crashing the whole process
            # with an access violation. This server has no real GPU
            # acceleration benefit to lose here — these are small, fast
            # CPU-bound models anyway.
            base_options=BaseOptions(
                model_asset_buffer=model_bytes,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect_all(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """Run detection, return all detected faces as FaceBox (pixel coords)."""
        H, W = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._detector.detect(mp_image)

        if not result.detections:
            return []

        boxes = []
        for det in result.detections:
            bbox = det.bounding_box
            x0, y0 = max(0, bbox.origin_x), max(0, bbox.origin_y)
            x1 = min(W, bbox.origin_x + bbox.width)
            y1 = min(H, bbox.origin_y + bbox.height)
            w, h = x1 - x0, y1 - y0
            if w <= 0 or h <= 0:
                continue

            keypoints = {}
            for name, kp in zip(self._KEYPOINT_NAMES, det.keypoints or []):
                keypoints[name] = (kp.x * W, kp.y * H)

            confidence = det.categories[0].score if det.categories else 0.0

            boxes.append(FaceBox(
                x=x0, y=y0, w=w, h=h,
                confidence=confidence,
                keypoints=keypoints,
            ))
        return boxes

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[FaceBox]:
        """Detect all faces, return only the one with the largest area (per spec)."""
        boxes = self.detect_all(frame_bgr)
        if not boxes:
            return None
        return max(boxes, key=lambda b: b.area)

    @staticmethod
    def crop(frame_bgr: np.ndarray, face: FaceBox,
             margin_ratio: float = FACE_MARGIN_RATIO) -> np.ndarray:
        """Crop the face region from the frame, with a margin, clamped to frame bounds."""
        expanded = face.expanded(margin_ratio, frame_bgr.shape)
        return frame_bgr[expanded.y: expanded.y + expanded.h,
                          expanded.x: expanded.x + expanded.w].copy()

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
