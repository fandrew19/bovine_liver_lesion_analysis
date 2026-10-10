"""Shared paths and constants."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "Project Data"
HARMONISED_ROOT = PROJECT_ROOT / "data_harmonised"
OUTPUTS = PROJECT_ROOT / "outputs"

MANIFEST_CSV = OUTPUTS / "manifest.csv"
DROPPED_CSV = OUTPUTS / "dropped_duplicates.csv"

SEED = 42

# Harmonisation target. Every image is decoded, EXIF-rotated and cropped through one
# path, then stored LOSSLESSLY as PNG.
#
# Re-encoding as JPEG was tried and rejected: a second compression interacts with the
# first, and because the two class folders arrived at different original qualities the
# double-compression artefact was more class-predictive than the original
# (block-grid probe AUC 0.61 -> 0.98). Gaussian pre-blur was also rejected; a fixed
# sigma applied to differently-sized originals leaves a size-dependent, and therefore
# class-dependent, residual.
HARMONISE_SIZE = 256

# Model input.
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def ensure_dirs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "models").mkdir(parents=True, exist_ok=True)
