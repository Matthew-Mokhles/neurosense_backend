"""
app_inference.py
=================
Top-level orchestrator. Wires together:

    Camera/Video frame
        -> face_detector.FaceDetector        (largest face, per spec)
        -> face_alignment.FaceAligner        (eye-horizontal rotation)
        -> preprocessing.preprocess_face     (resize/normalize, training-identical)
        -> inference.FERACInferenceModel     (convnext_small, softmax, argmax)
        -> temporal_smoothing.TemporalSmoother (deque(10), moving avg + majority vote)
        -> visualization.draw_face_result    (bbox, label, confidence, FPS, timestamp)

Exposes exactly the four entry points requested:
    infer_image(path_or_array)
    infer_video(path, ...)
    infer_webcam(...)
    infer_stream(frame_source, ...)

...and the mobile-facing API:
    predict_emotion(frame) -> {"emotion": "joy", "confidence": 0.93}

This file contains NO duplicated logic from the other modules — it is
purely orchestration plus the public entry points.
"""

import time
from typing import Optional, Callable, Generator, Union

import cv2
import numpy as np

from ferac_config import (
    DEFAULT_VIDEO_PATH, CONFIDENCE_THRESHOLD, SMOOTHING_WINDOW,
    expected_emotion_at,
)
from ferac_face_detector import FaceDetector, FaceBox
from ferac_face_alignment import FaceAligner
from ferac_inference import FERACInferenceModel
from ferac_temporal_smoothing import TemporalSmoother
from ferac_visualization import draw_face_result, format_timestamp, FPSCounter
from ferac_gradcam import GradCAM, overlay_heatmap
from ferac_preprocessing import preprocess_face


class FERACPipeline:
    """
    Holds all stateful components (model, detector, aligner, smoother) so
    they're constructed ONCE and reused across frames — re-instantiating
    a MediaPipe detector or reloading the checkpoint per-frame would be
    needlessly slow and is exactly the kind of mistake this class exists
    to prevent.
    """

    def __init__(self, device: str = None, fp16: bool = False,
                 smoothing_window: int = SMOOTHING_WINDOW,
                 confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 enable_alignment: bool = True):
        self.model = FERACInferenceModel(device=device, fp16=fp16)
        self.detector = FaceDetector()
        self.aligner = FaceAligner() if enable_alignment else None
        self.smoother = TemporalSmoother(window=smoothing_window,
                                         confidence_threshold=confidence_threshold)
        self._gradcam: Optional[GradCAM] = None  # lazy-created, see explain()

    # ── Core per-frame step (no smoothing — used by predict_emotion) ──

    def predict_single_frame(self, frame_bgr: np.ndarray):
        """
        Runs detection -> alignment -> classification on ONE frame, with
        NO temporal smoothing applied. Returns:
            (face: FaceBox|None, label: str|None, confidence: float, probs: np.ndarray|None)
        label is None if no face was found OR confidence < threshold.
        """
        face = self.detector.detect_largest(frame_bgr)
        if face is None:
            return None, None, 0.0, None

        crop = self.detector.crop(frame_bgr, face)
        if crop.size == 0:
            return face, None, 0.0, None

        if self.aligner is not None:
            crop = self.aligner.align(crop)

        label, confidence, probs = self.model.predict(crop)
        if confidence < CONFIDENCE_THRESHOLD:
            return face, None, confidence, probs
        return face, label, confidence, probs

    # ── Core per-frame step WITH smoothing (used by video/webcam/stream loops) ──

    def predict_smoothed(self, frame_bgr: np.ndarray):
        """
        Same as predict_single_frame, but pushes the raw probability
        vector through self.smoother first. Returns:
            (face, label, confidence, is_stable)
        `label` here reflects the smoothed/majority-voted decision, not
        the single-frame raw argmax — this is what should be DISPLAYED
        to the user; predict_single_frame's raw output is what you'd use
        for offline analysis where smoothing isn't wanted.
        """
        face, raw_label, raw_confidence, probs = self.predict_single_frame(frame_bgr)
        if probs is None:
            # No face / empty crop this frame — still advance nothing,
            # let the smoother's existing history persist (a brief
            # detection dropout shouldn't reset stability).
            return face, None, 0.0, self.smoother.is_full

        label, confidence, is_stable = self.smoother.update(probs)
        return face, label, confidence, is_stable

    # ── Optional Grad-CAM ──

    def explain(self, face_bgr_crop: np.ndarray, class_idx: Optional[int] = None) -> np.ndarray:
        """
        Returns a BGR overlay image showing Grad-CAM attention for the
        given (already cropped+aligned) face. Lazily constructs the
        GradCAM wrapper on first call. Optional — only call this if you
        want the heatmap; normal inference never touches this path, so
        there's no performance cost when explainability isn't requested.
        """
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model.model)

        tensor = preprocess_face(face_bgr_crop)
        tensor = tensor.to(self.model.device)
        heatmap = self._gradcam.generate(tensor, class_idx=class_idx)
        return overlay_heatmap(face_bgr_crop, heatmap)

    def close(self):
        self.detector.close()
        if self.aligner is not None:
            self.aligner.close()
        if self._gradcam is not None:
            self._gradcam.remove_hooks()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ─────────────────────────────────────────────────────────────────────────
# Module-level singleton — lazily constructed so importing this file
# doesn't immediately load the model/MediaPipe graphs (useful for the
# Flutter-facing predict_emotion() API, which should pay that cost once
# on first call, not at import time).
# ─────────────────────────────────────────────────────────────────────────

_pipeline: Optional[FERACPipeline] = None


def _get_pipeline() -> FERACPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = FERACPipeline()
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────
# Required entry points
# ─────────────────────────────────────────────────────────────────────────

def infer_image(image: Union[str, np.ndarray], draw: bool = True,
                explain: bool = False):
    """
    Run the full pipeline on a single image (file path or BGR np.ndarray).
    Returns a dict:
        {
          "emotion": str | None,
          "confidence": float,
          "annotated_frame": np.ndarray | None (if draw=True),
          "gradcam_overlay": np.ndarray | None (if explain=True and a face was found),
        }
    No temporal smoothing is applied (a single image has no history).
    """
    pipeline = _get_pipeline()

    if isinstance(image, str):
        frame = cv2.imread(image)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        frame = image

    face, label, confidence, probs = pipeline.predict_single_frame(frame)

    result = {"emotion": label, "confidence": confidence,
             "annotated_frame": None, "gradcam_overlay": None}

    if draw:
        result["annotated_frame"] = draw_face_result(frame, face, label, confidence)

    if explain and face is not None:
        crop = pipeline.detector.crop(frame, face)
        if pipeline.aligner is not None:
            crop = pipeline.aligner.align(crop)
        class_idx = int(np.argmax(probs)) if probs is not None else None
        result["gradcam_overlay"] = pipeline.explain(crop, class_idx=class_idx)

    return result


def infer_video(video_path: str = None, display: bool = True,
                output_path: Optional[str] = None,
                max_frames: Optional[int] = None) -> Generator[dict, None, None]:
    """
    Process a video file frame-by-frame with full temporal smoothing.
    Yields one result dict per frame:
        {"timestamp": float, "emotion": str|None, "confidence": float,
         "expected_emotion": str|None, "clip_name": str|None}

    If display=True, shows an annotated window live (press 'q' to stop).
    If output_path is given, writes an annotated copy of the video there.

    `video_path` defaults to config.DEFAULT_VIDEO_PATH (the Tom & Jerry
    clip already present in the project's inference_videos/ folder).
    """
    video_path = video_path or DEFAULT_VIDEO_PATH
    pipeline = _get_pipeline()

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if output_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_source, (w, h))

    fps_counter = FPSCounter()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            timestamp_sec = frame_idx / fps_source
            expected, clip_name = expected_emotion_at(timestamp_sec)

            face, label, confidence, is_stable = pipeline.predict_smoothed(frame)
            current_fps = fps_counter.tick()

            if display or writer is not None:
                annotated = draw_face_result(
                    frame, face, label, confidence,
                    fps=current_fps,
                    timestamp_str=format_timestamp(timestamp_sec),
                    expected_emotion=expected,
                )
                if display:
                    cv2.imshow("FERAC Inference", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if writer is not None:
                    writer.write(annotated)

            yield {
                "timestamp": timestamp_sec,
                "emotion": label,
                "confidence": confidence,
                "expected_emotion": expected,
                "clip_name": clip_name,
                "is_stable": is_stable,
            }
            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()


def infer_webcam(camera_index: int = 0, display: bool = True,
                 on_result: Optional[Callable[[dict], None]] = None) -> None:
    """
    Live webcam inference loop with full temporal smoothing. Runs until
    'q' is pressed (if display=True) or indefinitely if display=False and
    no on_result callback ever returns a stop signal — in that case the
    caller is expected to run this in a thread and manage lifecycle
    externally (see infer_stream for a more composable alternative).

    on_result: optional callback invoked with the same per-frame result
    dict that infer_video yields, useful for logging/telemetry without
    needing the live display window.
    """
    pipeline = _get_pipeline()
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {camera_index}")

    fps_counter = FPSCounter()
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            elapsed = time.time() - start_time
            face, label, confidence, is_stable = pipeline.predict_smoothed(frame)
            current_fps = fps_counter.tick()

            result = {
                "timestamp": elapsed,
                "emotion": label,
                "confidence": confidence,
                "is_stable": is_stable,
            }
            if on_result is not None:
                on_result(result)

            if display:
                annotated = draw_face_result(
                    frame, face, label, confidence,
                    fps=current_fps, timestamp_str=format_timestamp(elapsed),
                )
                cv2.imshow("FERAC Webcam Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if display:
            cv2.destroyAllWindows()


def infer_stream(frame_source: Callable[[], Optional[np.ndarray]],
                 on_result: Callable[[dict], None],
                 should_stop: Optional[Callable[[], bool]] = None) -> None:
    """
    Generic streaming entry point for non-OpenCV-VideoCapture sources —
    this is the shape a mobile camera stream bridge would use: each call
    to `frame_source()` should return the next BGR np.ndarray frame (or
    None when the stream ends), and `on_result(dict)` is called once per
    processed frame with the same result schema as infer_video.

    `should_stop`, if given, is checked each iteration to allow external
    cancellation (e.g. a Flutter-side "stop session" button setting a
    flag this callable reads).

    This function deliberately has NO knowledge of cv2.VideoCapture,
    Flutter platform channels, or any specific transport — it's the
    common core that infer_webcam/infer_video are thin wrappers around
    for their respective sources, and that a custom mobile bridge can
    reuse directly.
    """
    pipeline = _get_pipeline()
    fps_counter = FPSCounter()
    start_time = time.time()

    while True:
        if should_stop is not None and should_stop():
            break

        frame = frame_source()
        if frame is None:
            break

        elapsed = time.time() - start_time
        face, label, confidence, is_stable = pipeline.predict_smoothed(frame)
        current_fps = fps_counter.tick()

        on_result({
            "timestamp": elapsed,
            "emotion": label,
            "confidence": confidence,
            "is_stable": is_stable,
            "fps": current_fps,
        })


# ─────────────────────────────────────────────────────────────────────────
# Mobile / Flutter-facing API
# ─────────────────────────────────────────────────────────────────────────

def predict_emotion(frame: np.ndarray) -> dict:
    """
    The exact API requested for Flutter integration:

        predict_emotion(frame) -> {"emotion": "joy", "confidence": 0.93}

    `frame` is a single BGR np.ndarray (e.g. decoded from a camera frame
    on the Flutter side and passed across the platform channel/FFI bridge
    as raw bytes -> numpy array on the Python inference-service side).

    This function applies temporal smoothing via the shared pipeline
    singleton's smoother — so repeated calls across consecutive frames
    from the same session will stabilize exactly like infer_video/
    infer_webcam do. If your Flutter integration calls this from
    independent, non-sequential frames (e.g. one-shot snapshots), call
    reset_emotion_smoothing() between unrelated sessions to avoid
    carrying over stale history.

    Returns {"emotion": None, "confidence": 0.0} if no face is detected
    or confidence is below threshold — this is the "suppressed
    prediction" case from the spec, and the Flutter UI should treat
    emotion=None as "nothing to show right now", not as an error.
    """
    pipeline = _get_pipeline()
    _, label, confidence, _ = pipeline.predict_smoothed(frame)
    return {"emotion": label, "confidence": round(float(confidence), 4)}


def reset_emotion_smoothing() -> None:
    """Clear the temporal smoothing history — call this when starting a
    new viewing session (e.g. a new video in the protocol) so the
    previous session's predictions don't bleed into the new one."""
    pipeline = _get_pipeline()
    pipeline.smoother.reset()
