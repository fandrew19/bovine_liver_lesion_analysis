"""Negative controls -- tests the pipeline is supposed to FAIL.

Three probe families, deliberately kept apart because they mean different things:

  capture     -- properties of the FILE (resolution, compression settings, EXIF).
                 Carries no image content at all, so any skill here is pure leakage.
  artefact    -- JPEG 8-pixel block structure in the PIXELS. A compression
                 fingerprint. Also pure leakage.
  content     -- overall contrast and entropy of the image. NOT leakage in the same
                 sense: a liver with white abscess foci genuinely differs in contrast
                 from a uniform one. Reported for information, never gated on.

Only `capture` and `artefact` gate the pipeline. Bundling a content statistic into
an artefact probe overstates the leak -- it was the original mistake in this audit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED

GATE_AUC = 0.65


def capture_metadata_features(paths: list[Path]) -> np.ndarray:
    """File properties only -- no pixel reaches the classifier."""
    rows = []
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
            tables = getattr(img, "quantization", None)
            qt = float(np.mean(tables[0])) if tables else -1.0
            exif = img.getexif() or {}
            orientation = float(exif.get(274) or 0)
        rows.append([w, h, w / h, w * h, qt, orientation])
    return np.asarray(rows, dtype=float)


def compression_artefact_features(paths: list[Path], size: int = 256) -> np.ndarray:
    """JPEG 8-pixel block structure alone -- a compression fingerprint.

    Gradient energy on the block grid divided by gradient energy everywhere else.
    A ratio near 1.0 means no detectable blocking. Deliberately excludes any
    global contrast term, which would be a content feature.
    """
    rows = []
    for p in paths:
        with Image.open(p) as opened:
            img = ImageOps.exif_transpose(opened).convert("L").resize((size, size))
        a = np.asarray(img, dtype=float)
        dh = np.abs(np.diff(a, axis=1))
        dv = np.abs(np.diff(a, axis=0))
        rows.append(
            [
                dh[:, 7::8].mean() / (np.delete(dh, np.s_[7::8], axis=1).mean() + 1e-9),
                dv[7::8, :].mean() / (np.delete(dv, np.s_[7::8], axis=0).mean() + 1e-9),
            ]
        )
    return np.asarray(rows, dtype=float)


def content_statistics_features(paths: list[Path], size: int = 256) -> np.ndarray:
    """Global contrast and compressibility. Diagnostic only -- not a gate."""
    rows = []
    for p in paths:
        with Image.open(p) as opened:
            img = ImageOps.exif_transpose(opened).convert("L").resize((size, size))
        a = np.asarray(img, dtype=float)
        rows.append([a.std(), a.mean(), p.stat().st_size / (size * size)])
    return np.asarray(rows, dtype=float)


def _score(X, y, groups, estimator, n_splits: int = 5) -> tuple[float, float]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    auc = cross_val_score(estimator, X, y, groups=groups, cv=cv, scoring="roc_auc")
    return float(auc.mean()), float(auc.std())


def run_probe(
    name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, gated: bool = True
) -> dict:
    """Score with both a tree and a margin model, and keep the stronger result --
    a leak found by either model is still a leak."""
    rf = RandomForestClassifier(400, random_state=SEED, class_weight="balanced")
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=SEED),
    )
    rf_mean, _ = _score(X, y, groups, rf)
    svm_mean, _ = _score(X, y, groups, svm)
    worst = max(rf_mean, svm_mean)
    if not gated:
        verdict = "content (not gated)"
    else:
        verdict = "LEAKS" if worst >= GATE_AUC else "clean"
    return {
        "probe": name,
        "rf_auc": rf_mean,
        "svm_auc": svm_mean,
        "worst_auc": worst,
        "gated": gated,
        "verdict": verdict,
    }


def shuffled_label_probe(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_repeats: int = 5
) -> dict:
    """Permute labels inside the real CV protocol. Above chance here means the
    evaluation is broken regardless of what the images contain."""
    rng = np.random.default_rng(SEED)
    rf = RandomForestClassifier(200, random_state=SEED, class_weight="balanced")
    scores = []
    for _ in range(n_repeats):
        mean, _ = _score(X, rng.permutation(y), groups, rf)
        scores.append(mean)
    mean = float(np.mean(scores))
    return {
        "probe": "shuffled labels (protocol sanity)",
        "rf_auc": mean,
        "svm_auc": float("nan"),
        "worst_auc": mean,
        "gated": True,
        "verdict": "BROKEN" if abs(mean - 0.5) > 0.15 else "clean",
    }


def report(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[
        ["probe", "rf_auc", "svm_auc", "worst_auc", "gated", "verdict"]
    ]
