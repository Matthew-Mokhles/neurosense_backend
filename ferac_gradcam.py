"""
gradcam.py
==========
Optional Grad-CAM explainability, per task spec ("Add Grad-CAM support
... Make it optional").

ConvNeXt has no classic "last conv layer feeding global pool" structure
in the ResNet sense, but timm's convnext_small still exposes a clear
final-stage feature map before the head's norm+pool+fc, which is the
correct hook point for Grad-CAM: m.stages[-1] (the last ConvNeXt stage,
output shape [B, 768, H', W'] before the head). This is verified against
the checkpoint inspection (m.stages.3.blocks.2... are the last-stage
keys, feeding into m.head.norm -> m.head.fc).

Usage:
    cam = GradCAM(inference_model.model)          # wraps the loaded model
    heatmap = cam.generate(input_tensor, class_idx)  # HxW float32 in [0,1]
    overlay = overlay_heatmap(face_bgr, heatmap)      # blended BGR image
    cam.remove_hooks()                                 # cleanup when done
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM for the convnext_small TimmModel wrapper. Hooks the output
    of the last ConvNeXt stage (model.m.stages[-1]) as the target
    activation layer.
    """

    def __init__(self, timm_model_wrapper):
        """
        timm_model_wrapper: an instance of inference.TimmModel (has a
        `.m` attribute which is the actual timm ConvNeXt model).
        """
        self.model = timm_model_wrapper
        self.target_layer = self.model.m.stages[-1]

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self._activations = out.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        """
        input_tensor: [1, 3, H, W], on the same device as the model.
        class_idx: which class's CAM to compute; if None, uses the
        model's own top prediction.

        Returns an HxW float32 heatmap, normalized to [0, 1], at the
        SAME spatial resolution as input_tensor (resized up from the
        target layer's native feature-map resolution via bilinear
        interpolation) — ready to directly overlay on the original crop.
        """
        was_training = self.model.training
        self.model.eval()

        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        activations = self._activations          # [1, C, h, w]
        gradients = self._gradients               # [1, C, h, w]

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # [1, C, 1, 1] — GAP of grads
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if was_training:
            self.model.train()

        return cam.astype(np.float32)

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove_hooks()


def overlay_heatmap(face_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Blend a [0,1] heatmap (same resolution model saw, e.g. 224x224) onto
    the ORIGINAL-resolution face_bgr crop. Resizes the heatmap to match
    face_bgr's resolution before blending.
    """
    H, W = face_bgr.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(face_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
