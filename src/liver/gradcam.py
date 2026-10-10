"""Grad-CAM and occlusion testing (Objective 5, findings F-11, F-12).

Objective 5 asks whether predictions are driven by the lesion rather than by
background or lighting. That is a claim which can be false, so it is implemented as
a measurement rather than an illustration:

  * `attention_in_border` -- how much of the model's attention lands in the outer
    frame of the image, where the lesion almost never is.
  * `occlusion_map`       -- slide a patch over the image and record how much the
                             prediction moves. Regions that matter show up directly.

A model reading the table rather than the liver produces high border attention, and
the occlusion map lights up around the edges.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import models
from .config import IMAGE_SIZE


def grad_cam(
    model: nn.Module, layer: nn.Module, x: torch.Tensor
) -> tuple[np.ndarray, float]:
    """Class-activation map for the positive logit. Returns (cam 0-1, probability)."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    fwd = layer.register_forward_hook(
        lambda _m, _i, out: activations.__setitem__("v", out)
    )
    bwd = layer.register_full_backward_hook(
        lambda _m, _gi, gout: gradients.__setitem__("v", gout[0])
    )
    # A frozen trunk has no parameters requiring grad, so autograd would stop before
    # reaching the target layer and the backward hook would never fire. Requiring
    # grad on the input keeps the graph alive all the way down.
    x = x.clone().requires_grad_(True)
    try:
        model.zero_grad()
        logit = model(x)
        logit[:, 0].backward()
        acts = activations["v"][0]
        grads = gradients["v"][0]
    finally:
        fwd.remove()
        bwd.remove()

    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(
        cam[None, None], size=(x.shape[-2], x.shape[-1]),
        mode="bilinear", align_corners=False,
    )[0, 0]
    return cam.detach().cpu().numpy(), float(torch.sigmoid(logit).item())


def attention_in_border(cam: np.ndarray, frac: float = 0.20) -> float:
    """Share of total activation falling in the outer frame.

    A uniform map gives roughly 1 - (1-2*frac)^2, which is 0.64 at frac=0.20.
    Substantially above that means the model is looking outside the organ.
    """
    n = cam.shape[0]
    k = int(n * frac)
    centre = cam[k:-k, k:-k].sum()
    return float(1.0 - centre / (cam.sum() + 1e-9))


@torch.no_grad()
def occlusion_map(
    model: nn.Module, x: torch.Tensor, patch: int = 48, stride: int = 16
) -> tuple[np.ndarray, float]:
    """Slide a grey patch across the image; record the drop in positive probability.

    Unlike Grad-CAM this needs no gradients and cannot be fooled by a sharp but
    irrelevant activation -- if hiding a region does not change the prediction,
    that region was not being used.
    """
    model.eval()
    base = float(torch.sigmoid(model(x)).item())
    _, _, h, w = x.shape
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    for top in range(0, h - patch + 1, stride):
        for left in range(0, w - patch + 1, stride):
            occluded = x.clone()
            occluded[:, :, top:top + patch, left:left + patch] = 0.0
            prob = float(torch.sigmoid(model(occluded)).item())
            heat[top:top + patch, left:left + patch] += base - prob
            counts[top:top + patch, left:left + patch] += 1

    heat = heat / np.maximum(counts, 1)
    return heat, base


def load_image(path, size: int = IMAGE_SIZE) -> tuple[torch.Tensor, Image.Image]:
    with Image.open(path) as opened:
        img = opened.convert("RGB").resize((size, size), Image.LANCZOS)
    tensor = models.eval_transform(size)(img).unsqueeze(0)
    return tensor, img


def overlay(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Heatmap composited over the original, for figures."""
    import matplotlib.cm as cm

    heat = cm.jet(np.clip(cam, 0, 1))[:, :, :3]
    return np.clip((1 - alpha) * (np.asarray(img) / 255.0) + alpha * heat, 0, 1)
