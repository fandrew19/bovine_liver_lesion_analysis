"""Group-aware nested cross-validation (findings F-02, F-04, F-08).

Two rules the previous protocol broke:

  1. Every image of one liver stays inside one fold. 159 images come from 132
     animals, and repeat views are near-identical -- scattering them across folds
     tests the model on livers it trained on.
  2. Nothing touching the outer validation fold may influence the model. Early
     stopping and threshold choice happen on an inner split of the training fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train_idx: np.ndarray   # outer training rows, minus the inner validation slice
    inner_idx: np.ndarray   # inner validation: early stopping + threshold selection
    test_idx: np.ndarray    # outer validation: scored once, never selected on

    @property
    def name(self) -> str:
        return f"r{self.repeat}f{self.fold}"


def _inner_split(
    y: np.ndarray, groups: np.ndarray, idx: np.ndarray, seed: int, n_splits: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Carve one grouped, stratified slice out of the outer training fold."""
    inner = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_pos, inner_pos = next(inner.split(np.zeros(len(idx)), y[idx], groups[idx]))
    return idx[train_pos], idx[inner_pos]


def nested_folds(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    seed: int = 42,
) -> list[Fold]:
    """Build every (repeat, fold) split up front so all models see identical data.

    Sharing one fold list across the deep and classical arms is what makes the
    comparison in proposal section 2.6 fair.
    """
    folds: list[Fold] = []
    for repeat in range(n_repeats):
        outer = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        for fold, (train_idx, test_idx) in enumerate(
            outer.split(np.zeros(len(y)), y, groups), start=1
        ):
            inner_train, inner_val = _inner_split(
                y, groups, train_idx, seed=seed + repeat
            )
            folds.append(
                Fold(
                    repeat=repeat,
                    fold=fold,
                    train_idx=inner_train,
                    inner_idx=inner_val,
                    test_idx=test_idx,
                )
            )
    return folds


def describe(folds: list[Fold], y: np.ndarray, groups: np.ndarray) -> str:
    lines = [
        f"{'split':>8s} {'train':>12s} {'inner':>12s} {'test':>12s}  "
        f"{'test livers':>11s}  overlap"
    ]
    for f in folds:
        leaked = set(groups[f.test_idx]) & set(groups[np.r_[f.train_idx, f.inner_idx]])
        def part(idx):
            return f"{len(idx):3d} ({int(y[idx].sum()):2d}+)"
        lines.append(
            f"{f.name:>8s} {part(f.train_idx):>12s} {part(f.inner_idx):>12s} "
            f"{part(f.test_idx):>12s}  {len(set(groups[f.test_idx])):11d}  "
            f"{'LEAK' if leaked else 'none'}"
        )
    return "\n".join(lines)
