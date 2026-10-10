"""Transfer-learning backbones (proposal section 2.6, finding F-07).

Objective 2 asks for three backbones in two regimes. The frozen regime is treated
separately from fine-tuning because it can be evaluated far more cheaply: with the
convolutional trunk fixed, embeddings are computed once and only a linear head is
trained. That matters on a 4-thread CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

BACKBONES = ("mobilenet_v2", "efficientnet_b0", "resnet18")

_BUILDERS = {
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
}


def build(name: str, pretrained: bool = True, freeze: bool = False) -> nn.Module:
    """Single-logit binary classifier on a pretrained trunk."""
    builder, weights = _BUILDERS[name]
    model = builder(weights=weights if pretrained else None)

    if name == "resnet18":
        model.fc = nn.Linear(model.fc.in_features, 1)
        trunk = [p for n_, p in model.named_parameters() if not n_.startswith("fc")]
    else:
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        trunk = list(model.features.parameters())

    if freeze:
        for p in trunk:
            p.requires_grad = False
    return model


def target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block -- where Grad-CAM is hooked."""
    return model.layer4[-1] if name == "resnet18" else model.features[-1]


def embedding_dim(name: str) -> int:
    return {"mobilenet_v2": 1280, "efficientnet_b0": 1280, "resnet18": 512}[name]


def as_feature_extractor(name: str) -> nn.Module:
    """Pretrained trunk with the classifier replaced by identity."""
    model = build(name, pretrained=True, freeze=True)
    if name == "resnet18":
        model.fc = nn.Identity()
    else:
        model.classifier = nn.Identity()
    return model.eval()


def train_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation per proposal section 2.6.

    RandomErasing is deliberately omitted: on 132 animals it removes the lesion from
    an already tiny positive class and teaches the model to guess from context.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageListDataset(Dataset):
    """Images are pre-harmonised on disk, so loading is a plain open+transform."""

    def __init__(self, paths, labels, transform):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as opened:
            img = opened.convert("RGB")
        return self.transform(img), self.labels[i]


@torch.no_grad()
def compute_embeddings(
    name: str, paths, n_views: int = 1, seed: int = 42, batch_size: int = 16
) -> np.ndarray:
    """Embed every image `n_views` times under augmentation.

    Returns shape (n_views, n_images, dim). View 0 always uses the deterministic
    eval transform, so a clean embedding is available for scoring.
    """
    torch.manual_seed(seed)
    extractor = as_feature_extractor(name)
    views = []
    for v in range(n_views):
        tf = eval_transform() if v == 0 else train_transform()
        loader = DataLoader(
            ImageListDataset(paths, np.zeros(len(paths)), tf),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        chunks = [extractor(x).flatten(1).cpu().numpy() for x, _ in loader]
        views.append(np.vstack(chunks))
    return np.stack(views)
