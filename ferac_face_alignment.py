"""
face_alignment.py
=================
Eye-horizontal face alignment using MediaPipe Tasks API
(mp.tasks.vision.FaceLandmarker), per task spec.

IMPORTANT — API version note: see face_detector.py's docstring. The
legacy mp.solutions.face_mesh API is removed in current mediapipe
releases; this module uses the modern FaceLandmarker Task, which is also
what MediaPipe's official Flutter/Android/iOS plugins bind to.

Rotates the cropped face so the line between the two eyes is horizontal,
correcting head tilt before the image reaches preprocessing.py.

FaceLandmarker outputs the same 468-point canonical face mesh topology as
the legacy FaceMesh solution, so the landmark indices for eye centers are
unchanged:
    Left eye outer/inner corners average:  landmarks 33, 133
    Right eye outer/inner corners average: landmarks 362, 263

Model bundle: downloaded automatically on first use to
~/.cache/ferac_mediapipe/face_landmarker.task. Pre-download and pass a
local path via FaceAligner(model_path=...) for offline deployment.
"""

import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ferac_mediapipe")
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_DEFAULT_MODEL_PATH = os.path.join(_CACHE_DIR, "face_landmarker.task")


def _ensure_model_downloaded(model_path: str, url: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading MediaPipe face landmarker model bundle to {model_path} ...")
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")
    return model_path


class FaceAligner:
    """
    Usage:
        aligner = FaceAligner()
        aligned = aligner.align(face_crop_bgr)   # rotated np.ndarray, same size
        # if no mesh landmarks found, returns the original crop unchanged
    """

    LEFT_EYE_IDX = (33, 133)
    RIGHT_EYE_IDX = (362, 263)

    def __init__(self, min_confidence: float = 0.5, model_path: Optional[str] = None):
        resolved_path = model_path or _ensure_model_downloaded(
            _DEFAULT_MODEL_PATH, _DEFAULT_MODEL_URL
        )
        # See ferac_face_detector.py: MediaPipe's native asset resolver
        # doesn't recognize Windows drive-letter absolute paths, so we read
        # the bytes ourselves and pass model_asset_buffer instead.
        with open(resolved_path, "rb") as f:
            model_bytes = f.read()
        options = vision.FaceLandmarkerOptions(
            # Force CPU explicitly — see ferac_face_detector.py: MediaPipe's
            # GPU delegate on Windows has been crashing the process via the
            # NVIDIA driver's media layer (nvdxgdmal64.dll).
            base_options=BaseOptions(
                model_asset_buffer=model_bytes,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_confidence,
            min_face_presence_confidence=min_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _eye_centers(self, face_bgr: np.ndarray):
        H, W = face_bgr.shape[:2]
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]  # list of NormalizedLandmark for the first face

        def avg_point(idx_pair):
            pts = [lm[i] for i in idx_pair]
            x = sum(p.x for p in pts) / len(pts) * W
            y = sum(p.y for p in pts) / len(pts) * H
            return np.array([x, y], dtype=np.float32)

        left_eye = avg_point(self.LEFT_EYE_IDX)
        right_eye = avg_point(self.RIGHT_EYE_IDX)
        return left_eye, right_eye

    def align(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Rotate `face_bgr` so the eyes are horizontal. Returns the original
        crop unchanged if no face landmarks could be detected on it (e.g.
        crop too small/blurry) — alignment is a best-effort refinement,
        never a hard requirement for the pipeline to proceed.
        """
        eyes = self._eye_centers(face_bgr)
        if eyes is None:
            return face_bgr

        left_eye, right_eye = eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))

        H, W = face_bgr.shape[:2]
        center = ((left_eye[0] + right_eye[0]) / 2.0,
                  (left_eye[1] + right_eye[1]) / 2.0)

        rot_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        aligned = cv2.warpAffine(
            face_bgr, rot_matrix, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
