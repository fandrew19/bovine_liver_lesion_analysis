"""Fold training for the deep arm (findings F-04, F-07, F-13).

The key change from the original notebook: early stopping watches the INNER
validation split carved from the training fold, never the outer test fold. The
outer fold is scored exactly once, by a model that never saw it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from . import models
from .config import SEED
from .splits import Fold


@dataclass
class FoldResult:
    inner_prob: np.ndarray
    test_prob: np.ndarray
    epochs_run: int
    state_dict: dict | None = None


def seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_frozen(
    embeddings: np.ndarray, y: np.ndarray, fold: Fold, C: float = 0.1
) -> FoldResult:
    """Linear head on precomputed embeddings.

    Augmented views of the training rows are stacked as extra training examples;
    inner and test rows use view 0 (the deterministic transform) only.
    """
    clean = embeddings[0]
    X_train = np.vstack([view[fold.train_idx] for view in embeddings])
    y_train = np.tile(y[fold.train_idx], embeddings.shape[0])

    scaler = StandardScaler().fit(X_train)
    head = LogisticRegression(
        max_iter=3000, C=C, class_weight="balanced", random_state=SEED
    )
    head.fit(scaler.transform(X_train), y_train)

    return FoldResult(
        inner_prob=head.predict_proba(scaler.transform(clean[fold.inner_idx]))[:, 1],
        test_prob=head.predict_proba(scaler.transform(clean[fold.test_idx]))[:, 1],
        epochs_run=0,
    )


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    out = [torch.sigmoid(model(x)).flatten().cpu().numpy() for x, _ in loader]
    return np.concatenate(out) if out else np.array([])


def train_finetune(
    backbone: str,
    paths: list,
    y: np.ndarray,
    fold: Fold,
    epochs: int = 12,
    lr: float = 3e-4,
    batch_size: int = 16,
    patience: int = 4,
    keep_state: bool = False,
) -> FoldResult:
    """Fine-tune the whole trunk, early-stopping on the inner split."""
    seed_everything(SEED + fold.repeat * 10 + fold.fold)

    def loader(idx, train: bool):
        tf = models.train_transform() if train else models.eval_transform()
        return DataLoader(
            models.ImageListDataset([paths[i] for i in idx], y[idx], tf),
            batch_size=batch_size, shuffle=train, num_workers=0,
        )

    train_loader = loader(fold.train_idx, True)
    inner_loader = loader(fold.inner_idx, False)
    test_loader = loader(fold.test_idx, False)

    model = models.build(backbone, pretrained=True, freeze=False)

    n_pos = int(y[fold.train_idx].sum())
    n_neg = len(fold.train_idx) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_loss, best_state, stale, ran = float("inf"), None, 0, 0

    for epoch in range(epochs):
        model.train()
        for x, target in train_loader:
            optimiser.zero_grad()
            loss = criterion(model(x), target.unsqueeze(1))
            loss.backward()
            optimiser.step()
        scheduler.step()
        ran = epoch + 1

        # Selection signal comes from the inner split only.
        model.eval()
        with torch.no_grad():
            losses = [
                criterion(model(x), t.unsqueeze(1)).item() for x, t in inner_loader
            ]
        inner_loss = float(np.mean(losses))

        if inner_loss < best_loss - 1e-4:
            best_loss = inner_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return FoldResult(
        inner_prob=_predict(model, inner_loader),
        test_prob=_predict(model, test_loader),
        epochs_run=ran,
        state_dict=best_state if keep_state else None,
    )
