"""Metrics, operating-point selection and pooled out-of-fold reporting (F-08, F-12).

Two departures from the original notebook:

  * The decision threshold is chosen on the inner fold to meet a sensitivity target,
    because proposal section 2.7 argues false negatives are the costlier error in
    meat inspection. A fixed 0.5 ignores that argument.
  * Alongside per-fold mean +/- SD, out-of-fold predictions are pooled into one
    curve with a confidence interval bootstrapped over LIVERS, not images -- at 132
    animals, per-fold SD over 5 folds is a very noisy summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)

METRIC_COLUMNS = ["sensitivity", "specificity", "precision", "f1", "auc"]


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:                      # a fold with one class present
        auc = float("nan")
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "threshold": threshold,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y_true)),
    }


def threshold_for_sensitivity(
    y_true: np.ndarray, y_prob: np.ndarray, target: float = 0.90
) -> float:
    """Lowest-specificity-cost threshold that still reaches `target` sensitivity.

    Chosen on inner-fold predictions only. Falls back to the best Youden's J point
    when the target is unreachable (a small inner fold may have few positives).
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = tpr >= target
    if ok.any():
        # Among thresholds meeting the target, take the one with least fall-out.
        best = np.argmin(np.where(ok, fpr, np.inf))
    else:
        best = np.argmax(tpr - fpr)
    thr = thresholds[best]
    return float(np.clip(thr, 1e-6, 1 - 1e-6))


def summarise_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD across folds, as proposal section 2.7 specifies."""
    rows = []
    for model, group in df.groupby("model", sort=False):
        row = {"model": model, "folds": len(group)}
        for col in METRIC_COLUMNS:
            row[col] = f"{np.nanmean(group[col]):.3f} ± {np.nanstd(group[col]):.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Pooled out-of-fold AUC with a 95% CI resampled over livers.

    Resampling images would treat three photographs of one animal as three
    independent observations and produce an interval that is too narrow.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique}

    point = roc_auc_score(y_true, y_prob)
    scores = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in picked])
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def pooled_report(
    oof: pd.DataFrame, seed: int = 42
) -> pd.DataFrame:
    """One honest row per model, computed on pooled out-of-fold predictions."""
    rows = []
    for model, group in oof.groupby("model", sort=False):
        y = group["y_true"].to_numpy()
        p = group["y_prob"].to_numpy()
        g = group["liver_id"].to_numpy()
        auc, lo, hi = bootstrap_auc_ci(y, p, g, seed=seed)
        thr = float(group["threshold"].median())
        m = compute_metrics(y, p, thr)
        rows.append({
            "model": model,
            "auc": auc,
            "auc_lo": lo,
            "auc_hi": hi,
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
            "precision": m["precision"],
            "f1": m["f1"],
            "threshold": thr,
            "n": len(group),
        })
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
