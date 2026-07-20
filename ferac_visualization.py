"""
visualization.py
=================
Drawing utilities: bounding box, emotion label, confidence, FPS, and
current timestamp, per task spec. Pure OpenCV — no model logic here.
"""

import time
from typing import Optional

import cv2
import numpy as np

from ferac_face_detector import FaceBox

# BGR colors (OpenCV convention)
COLOR_BOX = (60, 200, 60)
COLOR_TEXT_BG = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_SUPPRESSED = (60, 60, 220)

EMOTION_COLORS = {
    "joy": (60, 200, 60),
    "anger": (40, 40, 220),
    "fear": (200, 140, 30),
    "neutral": (180, 180, 180),
}


class FPSCounter:
    """Simple exponential-moving-average FPS counter."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._last_t: Optional[float] = None
        self._fps: float = 0.0

    def tick(self) -> float:
        now = time.time()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = (self.alpha * inst_fps + (1 - self.alpha) * self._fps
                            if self._fps > 0 else inst_fps)
        self._last_t = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def draw_face_result(frame_bgr: np.ndarray, face: Optional[FaceBox],
                     label: Optional[str], confidence: float,
                     fps: float = None, timestamp_str: str = None,
                     expected_emotion: str = None) -> np.ndarray:
    """
    Draw the full annotation set onto a copy of frame_bgr and return it.
    Safe to call every frame even if face is None or label is None
    (suppressed-confidence case) — draws whatever is available.
    """
    out = frame_bgr.copy()
    H, W = out.shape[:2]

    if face is not None:
        color = EMOTION_COLORS.get(label, COLOR_BOX) if label else COLOR_SUPPRESSED
        cv2.rectangle(out, (face.x, face.y), (face.x + face.w, face.y + face.h),
                      color, 2)

        if label is not None:
            text = f"{label}  {confidence:.2f}"
        else:
            text = f"uncertain ({confidence:.2f})"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_y = max(face.y - 10, th + 10)
        cv2.rectangle(out, (face.x, text_y - th - 8), (face.x + tw + 10, text_y + 4),
                      COLOR_TEXT_BG, -1)
        cv2.putText(out, text, (face.x + 5, text_y - 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # ── HUD: FPS + timestamp + expected emotion (top-left) ──
    hud_lines = []
    if fps is not None:
        hud_lines.append(f"FPS: {fps:.1f}")
    if timestamp_str is not None:
        hud_lines.append(f"t = {timestamp_str}")
    if expected_emotion is not None:
        hud_lines.append(f"protocol expects: {expected_emotion}")

    y0 = 24
    for i, line in enumerate(hud_lines):
        y = y0 + i * 24
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (8, y - th - 6), (8 + tw + 10, y + 4), COLOR_TEXT_BG, -1)
        cv2.putText(out, line, (12, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   COLOR_TEXT, 1, cv2.LINE_AA)

    if face is None:
        msg = "no face detected"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, msg, (W // 2 - tw // 2, H - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SUPPRESSED, 2, cv2.LINE_AA)

    return out


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"
