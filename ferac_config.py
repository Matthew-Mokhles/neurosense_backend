"""
config.py
=========
Single source of truth for everything that MUST match training exactly.

Every value here was read directly from the training pipeline
(ferac_train_v2.py) and the actual checkpoint on disk — none of it is
assumed. See the inline citation comment above each constant.

If you retrain a new checkpoint, update CHECKPOINT_PATH and re-verify
these constants against the new training run's printed dataset report.
"""

import os

# ─────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Best checkpoint confirmed via results_v2/convnext_small_classification_report.csv
# (macro F1 = 0.7245, accuracy = 0.845 on the 155-image FERAC test set).
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "saved_models_v2", "convnext_small_best.pth")

# Default sample video already present in the project (used by infer_video()
# when no path is given).
DEFAULT_VIDEO_PATH = os.path.join(
    PROJECT_ROOT, "inference_videos",
    "Tom & Jerry - Try Not to Laugh Challenge - Classic Cartoon Compilation - @WB Kids.mp4",
)

EXPORT_DIR = os.path.join(PROJECT_ROOT, "exported_models")

# ─────────────────────────────────────────────────────────────────────────
# Model identity — confirmed via inspect_checkpoint.py against the actual
# .pth state_dict (key prefix "m.", stem shape (96,3,4,4), head.fc shape
# (4,768)) and against build_experiment()/TIMM_MAP in ferac_train_v2.py.
# ─────────────────────────────────────────────────────────────────────────

TIMM_MODEL_NAME = "convnext_small.fb_in22k_ft_in1k"
NUM_CLASSES = 4

# ─────────────────────────────────────────────────────────────────────────
# Preprocessing — confirmed via prepare_dataloaders()'s `test_tf` branch
# in ferac_train_v2.py (the clean, non-augmented eval transform):
#   transforms.Resize((img_size, img_size))
#   transforms.ToTensor()
#   transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
# convnext_small is NOT in MODEL_IMG_SIZE, so it uses the pipeline default
# IMG_SIZE = 224 (NOT swin_v2's 256 override).
# ─────────────────────────────────────────────────────────────────────────

IMG_SIZE = 224
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────────────────
# Class mapping — confirmed via inspect_checkpoint.py reading the live
# train/ directory through torchvision.datasets.ImageFolder, the EXACT
# mechanism prepare_dataloaders() uses to build id2label during training:
#   class_to_idx: {'Natural': 0, 'anger': 1, 'fear': 2, 'joy': 3}
#
# IMPORTANT: the checkpoint's output index 0 corresponds to the literal
# folder name "Natural" (capital N), not "neutral". The mobile app's
# requested label set is anger/fear/joy/neutral — DISPLAY_LABEL_MAP below
# performs that renaming ONLY at the presentation layer. The underlying
# index order (0,1,2,3) must never be changed, reordered, or guessed —
# doing so silently breaks compatibility with the trained checkpoint.
# ─────────────────────────────────────────────────────────────────────────

ID2LABEL_TRAINED = {0: "Natural", 1: "anger", 2: "fear", 3: "joy"}

# Cosmetic rename for the app-facing API / on-screen text only.
DISPLAY_LABEL_MAP = {
    "Natural": "neutral",
    "anger": "anger",
    "fear": "fear",
    "joy": "joy",
}

ID2LABEL_DISPLAY = {i: DISPLAY_LABEL_MAP[name] for i, name in ID2LABEL_TRAINED.items()}

# ─────────────────────────────────────────────────────────────────────────
# Inference-time behaviour (new — not training-time facts, but explicit
# product decisions per the task brief)
# ─────────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.6          # suppress predictions below this
SMOOTHING_WINDOW = 10               # deque length for temporal smoothing
FACE_DETECTION_MIN_CONFIDENCE = 0.5
FACE_MARGIN_RATIO = 0.25            # expand detected face box by this fraction before crop

# Video protocol — for annotating which expected emotion the app is
# eliciting at a given playback timestamp (used for logging / evaluation
# overlays, NOT fed into the model).
VIDEO_PROTOCOL = [
    {"start": 0,   "end": 120, "clip": "Tom and Jerry Funny Moments",  "expected": "joy"},
    {"start": 120, "end": 240, "clip": "PAW Patrol Rescue Scene",      "expected": "fear"},
    {"start": 240, "end": 360, "clip": "Inside Out Anger Scene",       "expected": "anger"},
    {"start": 360, "end": 480, "clip": "Aquarium Relaxation Video",    "expected": "neutral"},
    {"start": 480, "end": 600, "clip": "Minions Funny Moments",        "expected": "joy"},
]


def expected_emotion_at(timestamp_sec: float):
    """Return the protocol's expected emotion label for a given playback time, or None."""
    for seg in VIDEO_PROTOCOL:
        if seg["start"] <= timestamp_sec < seg["end"]:
            return seg["expected"], seg["clip"]
    return None, None
