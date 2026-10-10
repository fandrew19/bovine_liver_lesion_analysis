"""Classical colour + texture features for the benchmark arm (proposal section 2.6).

Kept as the honest yardstick: at 132 animals a well-tuned classical model can match
a CNN, and the gap between them says how much the network is really adding beyond
colour and texture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import graycomatrix, graycoprops

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
GLCM_LEVELS = 64


def extract(path: Path, size: int = 256) -> np.ndarray:
    """Colour histograms + summary statistics + GLCM texture descriptors.

    Runs on the harmonised PNGs, so every image arrives already EXIF-corrected,
    square-cropped and at one resolution.
    """
    with Image.open(path) as opened:
        img = opened.convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.LANCZOS)
        arr = np.asarray(img)

    features: list[float] = []

    # Colour distribution in RGB and HSV. Hue and saturation carry the pale bile-duct
    # tracts of fascioliasis and the white foci of abscessation.
    hsv = rgb2hsv(arr)
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 255), density=True)
        features.extend(hist)
        features.extend([arr[:, :, channel].mean(), arr[:, :, channel].std()])

        hist_hsv, _ = np.histogram(hsv[:, :, channel], bins=12, range=(0, 1), density=True)
        features.extend(hist_hsv)

    grey = (rgb2gray(arr) * 255).astype(np.uint8) // (256 // GLCM_LEVELS)
    glcm = graycomatrix(
        grey,
        distances=[1, 3, 5],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    for prop in GLCM_PROPS:
        features.extend(graycoprops(glcm, prop).ravel())

    return np.asarray(features, dtype=np.float32)


def extract_all(paths: list[Path], size: int = 256) -> np.ndarray:
    return np.vstack([extract(p, size) for p in paths])
