"""
inference.py
=============
Core model wrapper: loads the EXACT architecture used at training time
(TimmModel wrapping convnext_small.fb_in22k_ft_in1k, per ferac_train_v2.py
and the inspected checkpoint), loads the trained checkpoint, and exposes
predict() returning (label, confidence, full_probs).

This file intentionally re-declares TimmModel rather than importing it
from ferac_train_v2.py, because that training script pulls in heavy
training-only dependencies (matplotlib, seaborn, sklearn, tqdm,
transformers' SegformerModel for the unrelated improved_segformer
architecture) that have no business being imported into a mobile-facing
inference path. The class body below is copied verbatim from
ferac_train_v2.py's TimmModel — same forward() logic, same state_dict
key structure (confirmed against the checkpoint's "m." prefix via
inspect_checkpoint.py) — so the trained weights load with zero
modification and zero risk of architecture drift.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from ferac_config import (
    CHECKPOINT_PATH, TIMM_MODEL_NAME, NUM_CLASSES,
    ID2LABEL_TRAINED, ID2LABEL_DISPLAY, IMG_SIZE,
)
from ferac_preprocessing import preprocess_face


class TimmModel(nn.Module):
    """Verbatim copy of ferac_train_v2.py's TimmModel — DO NOT modify the
    forward() logic or attribute name (`self.m`) without re-verifying the
    checkpoint still loads; the state_dict keys are literally "m.<rest>"."""

    def __init__(self, name, num_classes, pretrained=False, drop_rate=0.0):
        super().__init__()
        self.m = timm.create_model(name, pretrained=pretrained,
                                   num_classes=num_classes, drop_rate=drop_rate)

    def forward(self, x):
        return self.m(x)


class FERACInferenceModel:
    """
    Loads the trained convnext_small checkpoint and exposes a single
    predict() method. Device-aware (CUDA if available, else CPU — both
    work, CPU is what most Flutter/mobile deployments will end up
    approximating via ONNX/TFLite, see export.py).

    Usage:
        model = FERACInferenceModel()
        label, confidence, probs = model.predict(face_bgr_crop)
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH,
                 device: str = None, fp16: bool = False):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fp16 = fp16 and self.device.type == "cuda"  # FP16 only meaningful on CUDA

        # pretrained=False: we are about to overwrite every weight with the
        # checkpoint anyway; downloading ImageNet weights first would be
        # pure waste (and the offline/mobile dev box may have no internet).
        self.model = TimmModel(TIMM_MODEL_NAME, NUM_CLASSES,
                               pretrained=False, drop_rate=0.0)

        # weights_only=False: this checkpoint is a local, trusted file we
        # produced ourselves. The default weights_only=True path uses
        # torch's newer _weights_only_unpickler, whose persistent_load/
        # load_tensor has been the exact crash site (access violation) on
        # this host — falling back to the classic pickle loader avoids
        # that code path entirely.
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=True)
        # strict=True deliberately — any key mismatch here means the
        # architecture assumption is wrong and must fail loudly, not
        # silently degrade to random weights on missing keys.

        self.model.eval()
        self.model.to(self.device)
        if self.fp16:
            self.model.half()

        self.id2label_trained = ID2LABEL_TRAINED
        self.id2label_display = ID2LABEL_DISPLAY

    @torch.no_grad()
    def predict_tensor(self, input_tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
        """
        input_tensor: [1, 3, H, W] preprocessed tensor (see preprocessing.py).
        Returns (class_idx, confidence, full_probs[num_classes]) — uses the
        TRAINED index space (0=Natural/neutral, 1=anger, 2=fear, 3=joy),
        NOT yet mapped to display labels. Use predict() for the full,
        display-ready pipeline.
        """
        input_tensor = input_tensor.to(self.device)
        if self.fp16:
            input_tensor = input_tensor.half()

        outputs = self.model(input_tensor)
        probs = F.softmax(outputs.float(), dim=1)  # softmax in fp32 regardless of fp16 inference
        probs_np = probs.squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs_np))
        confidence = float(probs_np[idx])
        return idx, confidence, probs_np

    def predict(self, face_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Full single-image inference: face_bgr (cropped+aligned face, BGR
        uint8) -> (display_label, confidence, full_probs).

        This is the function referenced by the task brief's required
        pattern:
            with torch.no_grad():
                outputs = model(image)
                probs = softmax(outputs)
                prediction = argmax(probs)
        — implemented exactly that way inside predict_tensor() above.
        """
        tensor = preprocess_face(face_bgr, img_size=IMG_SIZE)
        idx, confidence, probs_np = self.predict_tensor(tensor)
        display_label = self.id2label_display[idx]
        return display_label, confidence, probs_np

    def predict_raw_label(self, face_bgr: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Same as predict() but returns the TRAINED label ('Natural', not
        'neutral') — useful for cross-checking against the original
        classification reports in results_v2/."""
        tensor = preprocess_face(face_bgr, img_size=IMG_SIZE)
        idx, confidence, probs_np = self.predict_tensor(tensor)
        return self.id2label_trained[idx], confidence, probs_np
