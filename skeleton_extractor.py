"""
Skeleton Extractor — MediaPipe
==============================
بيستخرج 3D skeleton من فيديو باستخدام MediaPipe
وبيحوّله لـ format الموديل [150, 24, 3]

MediaPipe → 33 landmarks → mapping → 24 joints (MMASD format)

Uses the modern MediaPipe Tasks API (mp.tasks.vision.PoseLandmarker)
instead of the legacy mp.solutions.pose ("Solutions") API. The legacy
API always builds a full CalculatorGraph with a GPU/GL context at
construction time with no way to force CPU-only execution — on this
host that GPU context briefly touches the NVIDIA driver's media layer
(nvdxgdmal64.dll), which crashes the whole process with an access
violation some time later (the leftover driver thread fires after the
GPU context/module is unloaded). The Tasks API exposes an explicit
BaseOptions(delegate=Delegate.CPU), exactly like the fix already
applied in ferac_face_detector.py / ferac_face_alignment.py, which
avoids creating any GPU context in the first place.
"""

import os
import urllib.request
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ============================================================
# JOINT MAPPING — من MediaPipe (33) لـ MMASD (24)
# ============================================================
# MediaPipe Pose landmarks:
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

MEDIAPIPE_TO_MMASD = {
    # MMASD joint : MediaPipe landmark index
    0:  0,   # nose → nose
    1:  11,  # left_shoulder
    2:  12,  # right_shoulder
    3:  13,  # left_elbow
    4:  14,  # right_elbow
    5:  15,  # left_wrist
    6:  16,  # right_wrist
    7:  23,  # left_hip
    8:  24,  # right_hip
    9:  25,  # left_knee
    10: 26,  # right_knee
    11: 27,  # left_ankle
    12: 28,  # right_ankle
    13: 7,   # left_ear
    14: 8,   # right_ear
    15: 9,   # mouth_left
    16: 10,  # mouth_right
    17: 17,  # left_pinky
    18: 18,  # right_pinky
    19: 19,  # left_index
    20: 20,  # right_index
    21: 29,  # left_heel
    22: 30,  # right_heel
    23: 31,  # left_foot_index
}


_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ferac_mediapipe")
_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
_DEFAULT_MODEL_PATH = os.path.join(_CACHE_DIR, "pose_landmarker_full.task")


def _ensure_model_downloaded(model_path: str, url: str) -> str:
    if os.path.isfile(model_path):
        return model_path
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    print(f"Downloading MediaPipe pose landmarker model bundle to {model_path} ...")
    urllib.request.urlretrieve(url, model_path)
    print("Download complete.")
    return model_path


class SkeletonExtractor:
    """
    يستخرج 3D skeleton من فيديو باستخدام MediaPipe Pose

    مثال:
        extractor = SkeletonExtractor()
        skeleton = extractor.extract('video.mp4')
        # skeleton shape: [150, 24, 3]
    """

    def __init__(self,
                 max_frames: int = 150,
                 num_joints: int = 24,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 model_path: Optional[str] = None):

        self.max_frames = max_frames
        self.num_joints = num_joints

        resolved_path = model_path or _ensure_model_downloaded(
            _DEFAULT_MODEL_PATH, _DEFAULT_MODEL_URL
        )
        # MediaPipe's native asset resolver doesn't recognize Windows
        # drive-letter absolute paths as absolute — read the bytes
        # ourselves and pass model_asset_buffer instead (see
        # ferac_face_detector.py for the same workaround).
        with open(resolved_path, "rb") as f:
            model_bytes = f.read()

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_buffer=model_bytes,
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def extract(self, video_path: str) -> Optional[np.ndarray]:
        """
        يستخرج skeleton من فيديو

        Args:
            video_path: path للفيديو

        Returns:
            numpy array [150, 24, 3] أو None لو فشل
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)

        if not cap.isOpened():
            print(f"❌ مش قادر يفتح الفيديو: {video_path}")
            return None

        frames_skeletons = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"   📹 Video: {total_frames} frames @ {fps:.1f} FPS")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # استخراج الـ landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(frame_idx * (1000.0 / fps))
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_world_landmarks:
                skeleton_frame = self._landmarks_to_array(result.pose_world_landmarks[0])
                frames_skeletons.append(skeleton_frame)
            else:
                # لو مفيش detection في الـ frame دي، ضيف zeros
                frames_skeletons.append(np.zeros((self.num_joints, 3)))

            frame_idx += 1

        cap.release()

        if not frames_skeletons:
            print("❌ مش قادر يستخرج أي skeleton")
            return None

        skeleton_sequence = np.array(frames_skeletons)  # [frames, 24, 3]

        # تحقق إن في frames فيها data فعلاً
        valid_frames = np.any(skeleton_sequence != 0, axis=(1, 2)).sum()
        print(f"   ✅ Valid frames: {valid_frames}/{len(frames_skeletons)}")

        if valid_frames < 10:
            print("❌ مش كفاية frames فيها skeleton — تأكد من وضوح الطفل في الفيديو")
            return None

        # Resample لـ 150 frames
        skeleton_sequence = self._resample(skeleton_sequence)

        # Normalize
        skeleton_sequence = self._normalize(skeleton_sequence)

        return skeleton_sequence.astype(np.float32)

    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """بيحول MediaPipe landmarks لـ numpy array بـ 24 joints"""
        # استخرج كل الـ 33 landmarks
        all_landmarks = np.array([
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ])  # shape: [33, 3]

        # عمل mapping لـ 24 joints
        skeleton_24 = np.zeros((self.num_joints, 3))
        for mmasd_joint, mediapipe_idx in MEDIAPIPE_TO_MMASD.items():
            if mediapipe_idx < len(all_landmarks):
                skeleton_24[mmasd_joint] = all_landmarks[mediapipe_idx]

        return skeleton_24

    def _resample(self, skeleton: np.ndarray) -> np.ndarray:
        """Resample لـ 150 frames"""
        current_frames = skeleton.shape[0]

        if current_frames == self.max_frames:
            return skeleton

        if current_frames > self.max_frames:
            indices = np.linspace(0, current_frames - 1, self.max_frames, dtype=int)
            return skeleton[indices]
        else:
            # Interpolation
            indices = np.linspace(0, current_frames - 1, self.max_frames)
            resampled = np.zeros((self.max_frames, self.num_joints, 3))
            for i, idx in enumerate(indices):
                low = int(idx)
                high = min(low + 1, current_frames - 1)
                alpha = idx - low
                resampled[i] = (1 - alpha) * skeleton[low] + alpha * skeleton[high]
            return resampled

    def _normalize(self, skeleton: np.ndarray) -> np.ndarray:
        """Normalize لـ [-1, 1]"""
        non_zero = np.any(skeleton != 0, axis=(1, 2))

        if non_zero.sum() == 0:
            return skeleton

        frames = skeleton[non_zero].copy()
        center = np.mean(frames, axis=(0, 1), keepdims=True)
        frames -= center

        max_val = np.abs(frames).max()
        if max_val > 0:
            frames /= max_val

        skeleton[non_zero] = frames
        return skeleton

    def close(self):
        self.landmarker.close()

    def __del__(self):
        """cleanup"""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
